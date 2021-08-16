import nltk
from nltk.corpus import stopwords
import re, string
from nltk.stem import WordNetLemmatizer
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='robust04', help='dataset name: robust04/clueweb09')
args = parser.parse_args()
path = './{}/'.format(args.dataset)

stemmer = WordNetLemmatizer()
stop = set(stopwords.words('english'))

with open(path + 'queries.tsv', 'r') as f:
  queries = f.readlines()
with open(path + 'documents.tsv', 'r') as f:
  docs = f.readlines()


id_words_pairs = []
freq = {}
id_list = set()
for line in queries:
  _, sid, sentence = line.split('\t')
  if sid in id_list:
    continue
  id_list.add(sid)
  sentence = sentence.lower()
  sentence = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", sentence)
  sentence = sentence.translate(str.maketrans('','', string.punctuation))
  words = [word.strip(string.digits) for word in sentence.split() if word.strip(string.digits) not in stop]
  id_words_pairs.append((sid, words))
  for word in words:
    if word in freq:
      freq[word] += 100
    else:
      freq[word] = 100

sent = []
for (sid, row) in id_words_pairs:
  sequ = ''
  for word in row:
      sequ = sequ + ' ' + stemmer.lemmatize(word)
  sent.append((sid, sequ))

with open(path + 'clean.queries.txt', 'w') as f:
  for (sid, s) in sent:
    f.writelines(sid + '\t' + s +'\n')


id_words_pairs = []
id_list = set()
for line in docs:
  _, sid, sentence = line.split('\t')
  if sid in id_list:
    continue
  id_list.add(sid)
  sentence = sentence.lower()
  sentence = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", sentence)
  sentence = sentence.translate(str.maketrans('','', string.punctuation))
  words = [word.strip(string.digits) for word in sentence.split() if word.strip(string.digits) not in stop]
  id_words_pairs.append((sid, words))
  for word in words:
    if word in freq:
      freq[word] += 1
    else:
      freq[word] = 1

sent = []
for (sid, row) in id_words_pairs:
  sequ = ''
  for word in row:
    if freq[word] >= 10:
      sequ = sequ + ' ' + stemmer.lemmatize(word)
  sent.append((sid, sequ))

with open(path + 'clean.documents.txt', 'w') as f:
  for (sid, s) in sent:
    f.writelines(sid + '\t' + s +'\n')