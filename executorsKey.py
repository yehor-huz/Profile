import json
import math
import pandas as pd

PATH_JSON = "C:\\University\\4_course\\ІнфоПлюс\\Profile\\tasks_output_tokenized.json"
PATH_STOPWORDS = "C:\\University\\4_course\\ІнфоПлюс\\Profile\\stopwords_ua.txt"
OUTPUT_RESULT_JSON = "C:\\University\\4_course\\ІнфоПлюс\\Profile\\executor_filtered_idf.json"

data = pd.read_json(PATH_JSON)

try:
    with open(PATH_STOPWORDS, encoding="utf8") as f:
        stopwords = set(line.strip().lower() for line in f)
except FileNotFoundError:
    stopwords = set()

def filter_tokens(token_list):
    clean_list = []
    for token in token_list:
        token_lower = token 
        
        if not token.isalnum():
            continue            
        if token_lower in stopwords:
            continue          
        clean_list.append(token_lower)       
    return clean_list

print("Фільтрація токенів...")
data["tokens"] = data["tokens"].apply(filter_tokens)
data = data[data["tokens"].map(len) > 0]

executor_docs = {}
grouped = data.groupby("executor")

for executor, group in grouped:
    all_tokens_of_executor = set()
    for token_list in group['tokens']:
        all_tokens_of_executor.update(token_list)
    
    executor_docs[executor] = all_tokens_of_executor

N = len(executor_docs) 
all_words_idf = {}

all_unique_words = set().union(*executor_docs.values())

for word in all_unique_words:
    doc_freq = 0
    for doc_tokens in executor_docs.values():
        if word in doc_tokens:
            doc_freq += 1
    
    if doc_freq > 0:
        idf = math.log(N / doc_freq)
    else:
        idf = 0
        
    all_words_idf[word] = idf

print("Фільтрація за перцентилями (20%-80%)...")

sorted_words_by_idf = sorted(all_words_idf.items(), key=lambda x: x[1])
total_unique_words = len(sorted_words_by_idf)

if total_unique_words > 0:
    lower_cutoff = int(total_unique_words * 0.2) 
    upper_cutoff = int(total_unique_words * 0.8) 

    valid_range_words = sorted_words_by_idf[lower_cutoff:upper_cutoff]
    
    valid_words_map = dict(valid_range_words)
    
    print(f"Всього унікальних слів: {total_unique_words}")
    print(f"Залишено слів (діапазон 20%-80%): {len(valid_words_map)}")
else:
    valid_words_map = {}

final_result = {}

for executor, tokens_set in executor_docs.items():
    executor_valid_words = []
    
    for word in tokens_set:

        if word in valid_words_map:
            score = valid_words_map[word]
            executor_valid_words.append((word, score))

    sorted_executor_words = sorted(executor_valid_words, key=lambda x: x[1], reverse=True)

    final_result[executor] = {word: round(score, 4) for word, score in sorted_executor_words}

with open(OUTPUT_RESULT_JSON, "w", encoding="utf8") as f:
    json.dump(final_result, f, ensure_ascii=False, indent=2)

print(f"Результати збережено в {OUTPUT_RESULT_JSON}")