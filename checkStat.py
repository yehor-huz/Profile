import json
import re
import sys
import math
from collections import Counter, defaultdict
import matplotlib.pyplot as plt

data_file = sys.argv[1]
stopwords_file = sys.argv[2]

with open(stopwords_file, "r", encoding="utf-8") as f:
    stopwords = set(w.strip().lower() for w in f if w.strip())

with open(data_file, "r", encoding="utf-8") as f:
    data = json.load(f)

documents = []

for item in data:
    text = item.get("description", "")
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"0-9", "", text)
    text = text.lower()
    tokens = re.findall(r"\b\w+\b", text)
    tokens = [t for t in tokens if t not in stopwords]
    documents.append(tokens)

N = len(documents)

tf = []
df = defaultdict(int)

for doc in documents:
    counts = Counter(doc)
    total = len(doc)
    tf_doc = {}
    for word, c in counts.items():
        tf_doc[word] = c / total if total > 0 else 0
    tf.append(tf_doc)
    for word in counts:
        df[word] += 1

idf = {}

for word, d in df.items():
    idf[word] = math.log(N / d) if d > 0 else 0

tfidf_scores = defaultdict(float)

for i in range(N):
    for word, val in tf[i].items():
        tfidf_scores[word] += val * idf[word]

percent = 0.2
uncommonWords = sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)[int(len(tfidf_scores.items()) * percent):]
top = uncommonWords[-20:]
print(top)
