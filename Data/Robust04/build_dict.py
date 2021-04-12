with open('clean.documents.txt', 'r') as f1, open('clean.queries.txt', 'r') as f2:
  docs = f1.readlines()
  qrls = f2.readlines()


remap_docs, remap_words, remap_qrls = {}, {}, {}
new_docs, new_qrls = [], []


for line in docs:
  did, sent = line.split('\t')
  words = sent.split()
  n = ''
  if did not in remap_docs:
    remap_docs[did] = len(remap_docs)
  n += str(remap_docs[did]) + '\t'
  for w in words:
    if w not in remap_words:
      remap_words[w] = len(remap_words)
    n += str(remap_words[w]) + ' '
  new_docs.append(n)


for line in qrls:
  qid, sent = line.split('\t')
  words = sent.split()
  n = ''
  if qid not in remap_qrls:
    remap_qrls[qid] = len(remap_qrls)
  n += str(remap_qrls[qid]) + '\t'
  for w in words:
    if w not in remap_words:
      remap_words[w] = len(remap_words)
    n += str(remap_words[w]) + ' '
  new_qrls.append(n)


with open('map.documents.txt', 'w') as f1, open('map.queries.txt', 'w') as f2:
  for line in new_docs:
    f1.writelines(line + '\n')
  for line in new_qrls:
    f2.writelines(line + '\n')


with open('doc_dict.txt', 'w') as f1, open('qrl_dict.txt', 'w') as f2, \
     open('word_dict.txt', 'w') as f3:
  for key in remap_docs.keys():
    f1.writelines(str(key) + '\t' + str(remap_docs[key]) + '\n')
  for key in remap_qrls.keys():
    f2.writelines(str(key) + '\t' + str(remap_qrls[key]) + '\n')
  for key in remap_words.keys():
    f3.writelines(str(key) + '\t' + str(remap_words[key]) + '\n')
