import nltk
from nltk.corpus import stopwords
import re, string
from nltk.stem import WordNetLemmatizer

stemmer = WordNetLemmatizer()
nums = [str(i) for i in range (10)]
with open('queries.tsv', 'r') as f:
  x = f.readlines()

def h_num(word):
  for i in word:
    if i in nums:
      return True
  return False
stop = set(stopwords.words('english'))

temp = []
freq = {}
id_list = set()
for line in x:
  sid = line.split('\t')[1]
  if sid in id_list:
    continue
  id_list.add(sid)
  sentence = line.split('\t')[-1]
  sentence = sentence.lower()
  sentence = sentence.replace('-',' ')
  sentence = sentence.replace('--',' ')
  #cleanr = re.compile('<.*?>')
  #sentence = re.sub(cleanr, ' ', sentence)
  #sentence = re.sub('[?|!|\'|"|#]', '', sentence)
  #sentence = re.sub('[.|,|\\\\|/]', '', sentence)
  sentence = sentence.translate(str.maketrans('','', string.punctuation))
  words = [word for word in sentence.split() if word not in stop and not h_num(word)]
  temp.append((sid, words))
  for word in words:
    if word in freq:
      freq[word] += 1
    else:
      freq[word] = 1

sent = []
for (sid, row) in temp:
  sequ = ''
  for word in row:
    if freq[word] >= 5:
      sequ = sequ + ' ' + stemmer.lemmatize(word)
  sent.append((sid, sequ))

with open('clean.queries.txt', 'w') as f:
  for (sid, s) in sent:
    f.writelines(sid + '\t' + s +'\n')
