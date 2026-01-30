import json
import math
import pandas as pd
import re
import stanza

def clearText(text):
    if not isinstance(text, str): return ""
    text = re.sub(r"http(s)?:\/\/[a-zA-Z0-9\-\/.\-]+", "", text)
    text = re.sub(r"[0-9.]+", "", text)
    text = re.sub(r"\b\\[a-z]\b", " ", text)
    text = re.sub(r"\\", " ", text)
    text = re.sub(r"[`'\"]", "", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text

def get_lemmas(text):
    if not text:
        return []
    
    doc = nlp(text) 
    
    lemmas = []
    for sent in doc.sentences:
        for word in sent.words:
            if word.lemma:
                lemmas.append(word.lemma)
    
    return lemmas

JSON_PATH = "C:\\University\\4_course\\ІнфоПлюс\\Profile\\tasks_output.json"
OUTPUT_JSON = "C:\\University\\4_course\\ІнфоПлюс\\Profile\\tasks_output_tokenized.json"

try:
    data = pd.read_json(JSON_PATH)
except ValueError:
    print("Помилка зчитування JSON.")
    exit()

# stanza.download('uk', processors='tokenize,lemma')
nlp = stanza.Pipeline('uk', processors='tokenize,lemma', use_gpu=False, verbose=False)

data["cleaned_description"] = data["description"].apply(clearText)

data["tokens"] = data["cleaned_description"].apply(get_lemmas)
executors_tokens_dict = data.groupby("executor")["tokens"].apply(list).to_dict()

print("Збереження у JSON")
result = data.to_dict(orient="records")

with open(OUTPUT_JSON, "w", encoding="utf8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2, default=str)
