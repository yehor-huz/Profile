import pandas as pd
import matplotlib.pyplot as plt
import re

def clearText(text):
    text = re.sub(r"http(s)?:\/\/[a-zA-Z0-9\-\/.\-]+", "", text)
    text = re.sub(r"[0-9.]+", "", text)
    text = re.sub(r"\\[a-zA-Z]+", "", text)
    text = re.sub(r"[`'\"]", "", text)
    text = re.sub(r"[а-яА-ЯіїІЇґҐ]+\s[А-ЯI.]+")

    return text

JSON_PATH = "C:\\University\\4_course\\ІнфоПлюс\\Profile\\tasks_output.json"

data = pd.read_json(JSON_PATH)

#executors = pd.unique(data["executor"])





 