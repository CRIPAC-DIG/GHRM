from utility.parser import parse_args
from utility.load_data import *
import subprocess
import torch

args = parse_args()


data_generator = Data(path=args.data_path + args.dataset, batch_size=args.batch_size)
doc_dict, doc_dict_rev = data_generator.doc_dict, data_generator.doc_dict_rev
qrl_dict, qrl_dict_rev = data_generator.qrl_dict, data_generator.qrl_dict_rev
qrelf = args.data_path + args.dataset + '/qrels'

def pad_sequences(items, maxlen, value=data_generator.n_words):
    result = []
    for item in items:
        if len(item) < maxlen:
            item = item + [value] * (maxlen - len(item))
        if len(item) > maxlen:
            item = item[:maxlen]
        result.append(item)
    return result

def words_lookup(docs):
    if args.model == 'GRMM':
        return [data_generator.doc_unqiue_word_list[i] for i in docs]
    else:
        return [data_generator.doc_word_list[i] for i in docs]

def test(model, qrls_to_test, drop_flag=False, batch_test_flag=False):
    test_qrls = qrls_to_test
    n_test_qrls = len(test_qrls)

    rate_batches = np.zeros(shape=(n_test_qrls, 150))
    count = 0

    for qrl_id in test_qrls: 
        doc_ids = data_generator.test_set[qrl_id]

        docs_words = pad_sequences(words_lookup(doc_ids), maxlen=args.doc_len, value=data_generator.n_words)

        qrls_words = [data_generator.qrl_word_list[qrl_id]]
        qrls_words = pad_sequences(qrls_words, maxlen=args.qrl_len, value=data_generator.n_words)
        qrls_words = np.tile(qrls_words, [1, len(doc_ids)])
        qrls_words = np.reshape(qrls_words, [-1, args.qrl_len])
        with torch.no_grad():
            model.eval()
            rate_batch = model(qrls_words, docs_words, doc_ids , test=True)
            rate_batch = rate_batch.cpu()
        rate_batches[count, :rate_batch.shape[1]] = rate_batch
        count += 1


    ######################
    # trec_eval
    runf = './test.run'
    rerank_run = {}

    for i_qrl in range(len(test_qrls)):
        qn = test_qrls[i_qrl]
        for i_doc in range(len(data_generator.test_set[qn])):
            dn = data_generator.test_set[qn][i_doc]
            qid = qrl_dict_rev[qn]
            did = doc_dict_rev[dn]
            if dn not in data_generator.test_set[qn]:
                continue
            rerank_run.setdefault(qid, {})[did] = rate_batches[i_qrl, i_doc]

    with open(runf, 'wt') as runfile:
        for qid in rerank_run:
            scores = list(sorted(rerank_run[qid].items(), key=lambda x: (x[1], x[0]), reverse=True))
            for i, (did, score) in enumerate(scores):
                runfile.write(f'{qid} 0 {did} {i+1} {score} run \n')            

    trec_eval_f = 'bin/trec_eval' 
    output = subprocess.check_output([trec_eval_f, '-m', 'ndcg_cut.20', '-m', 'P.20', qrelf, runf]).decode().rstrip()
    if args.report:
        print(output) 

    assert count == n_test_qrls
    return eval(output.split()[2]), eval(output.split()[-1])
