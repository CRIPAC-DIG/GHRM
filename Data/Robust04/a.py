import subprocess
trec_eval_f = '../../HGCF/bin/trec_eval'
VALIDATION_METRIC = 'ndcg_cut.20'
#runf = 'run.robust04.bm25.topics.robust04.txt'
for i in range(1,6):
	runf = 'test_run/f{}.test.run'.format(str(i))
	qrelf = 'qrels'
	output = subprocess.check_output([trec_eval_f, '-m', VALIDATION_METRIC, qrelf, runf]).decode().rstrip()
	print(output)
