import subprocess
import numpy as np

precision = []
ndcg =[]

dataset = 'clueweb09'
#dataset = 'robust04'
qrelf = '../Data/{}/qrels'.format(dataset)
trec_eval_f = 'bin/trec_eval'
for i in range(1,6):
    runf = './{}.f{}.best.run'.format(dataset, str(i))
    output = subprocess.check_output([trec_eval_f, '-m', 'ndcg_cut.20', '-m', 'P.20', qrelf, runf]).decode().rstrip()
    precision.append(eval(output.split()[2]))
    ndcg.append(eval(output.split()[-1]))

precision = np.array(precision)
ndcg = np.array(ndcg)

print(dataset)
print("p@20:", precision.mean())
print("ndcg@20:", ndcg.mean())
