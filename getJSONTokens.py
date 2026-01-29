import re
import pandas as pd
import math
import json
from collections import Counter, defaultdict
from uk_stemmer import UkStemmer


stemmer = UkStemmer()


def cleanText(text):
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[\(\)\[\]\{\}<>]", " ", text)
    text = re.sub(r"[^\w\s\.\+#]", " ", text)
    text = re.sub(r"_", " ", text)
    text = re.sub(r"[^\w\s]{2,}", " ", text)
    text = re.sub(r"\b\w+['’`]\w+\b", " ", text)
    text = re.sub(r"\b\w\.{1,}\b", " ", text)
    text = re.sub(r"\b[a-zа-яіїє]\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip().lower()

    return text


def tokenizeRow(text, corpus_words, stopwords):
    text = cleanText(text)

    tokens = [
        t for t in text.split()
        if t not in stopwords
    ]

    result = []

    for t in tokens:
        s = stemmer.stem_word(t)

        if s in corpus_words:
            result.append(s)

    return result


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
        idf[w] = math.log((N + 1) / (d + 1)) + 1


    tfidf = []

    for i in range(N):
        doc_scores = {}

        for w, v in tf[i].items():
            doc_scores[w] = v * idf[w]

        tfidf.append(doc_scores)

    return tfidf


PATH_JSON = "C:\\University\\4_course\\ІнфоПлюс\\Profile\\tasks_output.json"
PATH_STOPWORDS = "C:\\University\\4_course\\ІнфоПлюс\\Profile\\stopwords_ua.txt"
OUTPUT_JSON = "C:\\University\\4_course\\ІнфоПлюс\\Profile\\tasks_output_with_tokens.json"


data = pd.read_json(PATH_JSON)

stopwords = set(
    line.strip().lower()
    for line in open(PATH_STOPWORDS, encoding="utf8")
)


corpus = " ".join(data["description"].dropna().astype(str))

corpus = cleanText(corpus)


corpus_tokens = [
    stemmer.stem_word(w)
    for w in corpus.split()
    if w not in stopwords
]

corpus_words = set(corpus_tokens)


docs = []

for text in data["description"].fillna(""):
    docs.append(tokenizeRow(text, corpus_words, stopwords))


tfidf_scores = build_tfidf(docs)


global_scores = defaultdict(float)

for doc in tfidf_scores:
    for w, v in doc.items():
        global_scores[w] += v

sorted_words = sorted(
    global_scores.items(),
    key=lambda x: x[1],
    reverse=True
)

PERCENT = 0.2
cut_index = int(len(sorted_words) * PERCENT)

top_words = set(w for w, _ in sorted_words[:cut_index])


REMOVE_DUPLICATES = True

def remove_duplicates(tokens):
    return list(dict.fromkeys(tokens))

filtered_docs = []

for doc in docs:
    new_tokens = [
        w for w in doc
        if w not in top_words
    ]

    if REMOVE_DUPLICATES:
        new_tokens = remove_duplicates(new_tokens)

    filtered_docs.append(new_tokens)


data["tokens"] = filtered_docs


result = data.to_dict(orient="records")

with open(OUTPUT_JSON, "w", encoding="utf8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)


print("Saved to:", OUTPUT_JSON)
