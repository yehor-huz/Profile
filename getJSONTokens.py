import re
import pandas as pd
import math
import json
from collections import Counter, defaultdict


def cleanText(text):
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[\(\)\[\]\{\}<>]", " ", text)
    text = re.sub(r"[^\w\s\.\+#]", " ", text)
    text = re.sub(r"_", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def tokenizeRow(text, corpus_words, stopwords):
    text = cleanText(text)

    tokens = text.split()

    tokens = [
        t for t in tokens
        if t not in stopwords and t in corpus_words
    ]

    return tokens


def build_tfidf(docs):
    N = len(docs)

    tf = []
    df = defaultdict(int)

    for doc in docs:
        counts = Counter(doc)
        total = len(doc)

        tf_doc = {}

        for w, c in counts.items():
            tf_doc[w] = c / total if total > 0 else 0

        tf.append(tf_doc)

        for w in counts:
            df[w] += 1

    idf = {}

    for w, d in df.items():
        idf[w] = math.log(N / d) if d > 0 else 0

    tfidf = []

    for i in range(N):
        doc_scores = {}

        for w, v in tf[i].items():
            doc_scores[w] = v * idf[w]

        tfidf.append(doc_scores)

    return tfidf


PATH_JSON = "C:\\University\\4_course\\ІнфоПлюс\\Profile\\tasks_output.json"
PATH_STOPWORDS = "C:\\University\\4_course\\ІнфоПлюс\\Profile\\stopwords.txt"
OUTPUT_JSON = "C:\\University\\4_course\\ІнфоПлюс\\Profile\\tasks_output_with_tokens.json"


data = pd.read_json(PATH_JSON)

stopwords = set(
    line.strip().lower()
    for line in open(PATH_STOPWORDS, encoding="utf8")
)


corpus = " ".join(data["description"].dropna().astype(str))

corpus = cleanText(corpus)

corpus_words = set(corpus.split())
corpus_words = {w for w in corpus_words if w not in stopwords}


docs = []

for text in data["description"].fillna(""):
    tokens = tokenizeRow(text, corpus_words, stopwords)
    docs.append(tokens)


tfidf_scores = build_tfidf(docs)


THRESHOLD = 0.8
filtered_docs = []

for i in range(len(docs)):
    new_tokens = set()

    for w in docs[i]:
        score = tfidf_scores[i].get(w, 0)

        if score <= THRESHOLD:
            new_tokens.add(w)

    filtered_docs.append(list(new_tokens))


data["tokens"] = filtered_docs


result = data.to_dict(orient="records")

with open(OUTPUT_JSON, "w", encoding="utf8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)


print("Saved to:", OUTPUT_JSON)
