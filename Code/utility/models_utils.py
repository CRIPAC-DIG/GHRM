import torch
import math
import torch.nn.functional as F
import torch.nn as nn

class PACRRConvMax2dModule(torch.nn.Module):
    def __init__(self, shape, n_filters, k, channels):
        super().__init__()
        self.shape = shape
        if shape != 1:
            self.pad = torch.nn.ConstantPad2d((0, shape-1, 0, shape-1), 0)
        else:
            self.pad = None
        self.conv = torch.nn.Conv2d(channels, n_filters, shape)
        self.activation = torch.nn.ReLU()
        self.k = k
        self.shape = shape
        self.channels = channels

    def forward(self, simmat):
        BATCH, CHANNEL,  QLEN, DLEN = simmat.shape
        if self.pad:
            simmat = self.pad(simmat)
        conv = self.activation(self.conv(simmat))#batch, channels, q_len, d_len
        top_filters, _ = conv.max(dim=1) #batch, q_len, d_len
        top_toks, _ = top_filters.topk(self.k, dim=2) # batch, q_len, k
        result = top_toks.reshape(BATCH, QLEN, self.k)
        return result
class DRMMLogCountHistogram(torch.nn.Module):
    def __init__(self, bins, pad_idx):
        super().__init__()
        self.bins = bins
        self.pad_idx = pad_idx

    def forward(self, simmat, dtoks, qtoks):
        # THIS IS SLOW ... Any way to make this faster? Maybe it's not worth doing on GPU?
        BATCH, QLEN, DLEN = simmat.shape
        # +1e-5 to nudge scores of 1 to above threshold
        bins = ((simmat + 1.000001) / 2. * (self.bins - 1)).int()
        # set weights of 0 for padding (in both query and doc dims)
        weights = ((dtoks != self.pad_idx).reshape(BATCH, 1, DLEN).expand(BATCH, QLEN, DLEN) * \
                  (qtoks != self.pad_idx).reshape(BATCH, QLEN, 1).expand(BATCH, QLEN, DLEN)).float()

        # no way to batch this... loses gradients here. https://discuss.pytorch.org/t/histogram-function-in-pytorch/5350
        bins, weights = bins.cpu(), weights.cpu()
        histogram = []
        for superbins, w in zip(bins, weights):
            result = []
            # for b in superbins:
            #     result.append(torch.stack([torch.bincount(q, x, self.bins) for q, x in zip(b, w)], dim=0))
            for q, x in zip(superbins, w):
                result.append(torch.bincount(q, x, self.bins))
            result = torch.stack(result, dim=0)
            histogram.append(result)
        histogram = torch.stack(histogram, dim=0)

        # back to GPU
        histogram = histogram.to(simmat.device)
        return (histogram.float() + 1e-5).log()


class KNRMRbfKernelBank(torch.nn.Module):
    def __init__(self, mus=None, sigmas=None, dim=1, requires_grad=True):
        super().__init__()
        self.dim = dim
        kernels = [KNRMRbfKernel(m, s, requires_grad=requires_grad) for m, s in zip(mus, sigmas)]
        self.kernels = torch.nn.ModuleList(kernels)

    def count(self):
        return len(self.kernels)

    def forward(self, data):
        return torch.stack([k(data) for k in self.kernels], dim=self.dim)


class KNRMRbfKernel(torch.nn.Module):
    def __init__(self, initial_mu, initial_sigma, requires_grad=True):
        super().__init__()
        self.mu = torch.nn.Parameter(torch.tensor(initial_mu), requires_grad=requires_grad)
        self.sigma = torch.nn.Parameter(torch.tensor(initial_sigma), requires_grad=requires_grad)

    def forward(self, data):
        adj = data - self.mu
        return torch.exp(-0.5 * adj * adj / self.sigma / self.sigma)


class GaussianKernel(torch.nn.Module):
    def __init__(self, mu: float = 1., sigma: float = 1.):
        """Gaussian kernel constructor."""
        super().__init__()
        self.mu = mu
        self.sigma = sigma

    def forward(self, x):
        """Forward."""
        return torch.exp(
            -0.5 * ((x - self.mu) ** 2) / (self.sigma ** 2)
        )

class SpatialGRU(nn.Module):
    """
    Spatial GRU Module.

    :param channels: Number of word interaction tensor channels.
    :param units: Number of SpatialGRU units.
    :param activation: Activation function to use, one of:
            - String: name of an activation
            - Torch Modele subclass
            - Torch Module instance
            Default: hyperbolic tangent (`tanh`).
    :param recurrent_activation: Activation function to use for
        the recurrent step, one of:
            - String: name of an activation
            - Torch Modele subclass
            - Torch Module instance
            Default: sigmoid activation (`sigmoid`).
    :param direction: Scanning direction. `lt` (i.e., left top)
        indicates the scanning from left top to right bottom, and
        `rb` (i.e., right bottom) indicates the scanning from
        right bottom to left top.

    Examples:
        >>> import matchzoo as mz
        >>> channels, units= 4, 10
        >>> spatial_gru = mz.modules.SpatialGRU(channels, units)

    """

    def __init__(
        self,
        channels: int = 4,
        units: int = 10,
        direction: str = 'lt'
    ):
        """:class:`SpatialGRU` constructor."""
        super().__init__()
        self._units = units
        self._activation = torch.nn.Tanh()
        self._recurrent_activation = torch.nn.Sigmoid()
        self._direction = direction
        self._channels = channels

        if self._direction not in ('lt', 'rb'):
            raise ValueError(f"Invalid direction. "
                             f"`{self._direction}` received. "
                             f"Must be in `lt`, `rb`.")

        self._input_dim = self._channels + 3 * self._units

        self._wr = nn.Linear(self._input_dim, self._units * 3)
        self._wz = nn.Linear(self._input_dim, self._units * 4)
        self._w_ij = nn.Linear(self._channels, self._units)
        self._U = nn.Linear(self._units * 3, self._units, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters."""
        nn.init.xavier_normal_(self._wr.weight)
        nn.init.xavier_normal_(self._wz.weight)
        nn.init.orthogonal_(self._w_ij.weight)
        nn.init.orthogonal_(self._U.weight)

    def softmax_by_row(self, z: torch.tensor) -> tuple:
        """Conduct softmax on each dimension across the four gates."""

        # z_transform: [B, 4, U]
        z_transform = z.reshape((-1, 4, self._units))
        # zi, zl, zt, zd: [B, U]
        zi, zl, zt, zd = F.softmax(z_transform, dim=1).unbind(dim=1)
        return zi, zl, zt, zd

    def calculate_recurrent_unit(
        self,
        inputs: torch.tensor,
        states: list,
        i: int,
        j: int
    ):
        """
        Calculate recurrent unit.

        :param inputs: A tensor which contains interaction
            between left text and right text.
        :param states: An array of tensors which stores the hidden state
            of every step.
        :param i: Recurrent row index.
        :param j: Recurrent column index.

        """

        # Get hidden state h_diag, h_top, h_left
        # h = [B, U]
        h_diag = states[i][j]
        h_top = states[i][j + 1]
        h_left = states[i + 1][j]

        # Get interaction between word i, j: s_ij
        # s = [B, C]
        s_ij = inputs[i][j]

        # Concatenate h_top, h_left, h_diag, s_ij
        # q = [B, 3*U+C]
        q = torch.cat([torch.cat([h_top, h_left], 1), torch.cat([h_diag, s_ij], 1)], 1)

        # Calculate reset gate
        # r = [B, 3*U]
        r = self._recurrent_activation(self._wr(q))

        # Calculate updating gate
        # z: [B, 4*U]
        z = self._wz(q)

        # Perform softmax
        # zi, zl, zt, zd: [B, U]
        zi, zl, zt, zd = self.softmax_by_row(z)

        # Get h_ij_
        # h_ij_ = [B, U]
        h_ij_l = self._w_ij(s_ij)
        h_ij_r = self._U(r * (torch.cat([h_left, h_top, h_diag], 1)))
        h_ij_ = self._activation(h_ij_l + h_ij_r)

        # Calculate h_ij
        # h_ij = [B, U]
        h_ij = zl * h_left + zt * h_top + zd * h_diag + zi * h_ij_

        return h_ij

    def forward(self, inputs):
        """
        Perform SpatialGRU on word interation matrix.

        :param inputs: input tensors.
        """

        batch_size, channels, left_length, right_length = inputs.shape

        # inputs = [L, R, B, C]
        inputs = inputs.permute([2, 3, 0, 1])
        if self._direction == 'rb':
            # inputs = [R, L, B, C]
            inputs = torch.flip(inputs, [0, 1])

        # states = [L+1, R+1, B, U]
        states = [
            [torch.zeros([batch_size, self._units]).type_as(inputs)
             for j in range(right_length + 1)] for i in range(left_length + 1)
        ]

        # Calculate h_ij
        # h_ij = [B, U]
        for i in range(left_length):
            for j in range(right_length):
                states[i + 1][j + 1] = self.calculate_recurrent_unit(inputs, states, i, j)
        return states[left_length][right_length]