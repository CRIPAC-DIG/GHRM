import nltk
from nltk.corpus import stopwords
import re, string

stemmer = nltk.stem.SnowballStemmer('english')

with open('documents.tsv', 'r') as f:
  x = f.readlines()

stop = set(stopwords.words('english'))

temp = []
freq = {}
for line in x:
  sid = line.split('\t')[1]
  sentence = line.split('\t')[-1]
  sentence = sentence.lower()
  #cleanr = re.compile('<.*?>')
  #sentence = re.sub(cleanr, ' ', sentence)
  #sentence = re.sub('[?|!|\'|"|#]', '', sentence)
  #sentence = re.sub('[.|,|\\\\|/]', '', sentence)
  sentence = sentence.translate(str.maketrans('','', string.punctuation))

  words = [word for word in sentence.split() if word not in stop]
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
    if freq[word] >= 1:
      sequ = sequ + ' ' + stemmer.stem(word)
  sent.append((sid, sequ))

with open('clean.documents.txt', 'w') as f:
  for (sid, s) in sent:
    f.writelines(sid + '\t' + s +'\n')
