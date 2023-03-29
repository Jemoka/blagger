import requests
import json

import pandas as pd

from transformers import pipeline

import re

def _tex_sanitize(s):
    """Sanitizes a string so that it can be properly compiled in TeX.
    Escapes the most common TeX special characters: ~^_#%${}
    Removes backslashes.
    """
    s = re.sub(r'\\\w+\{.*?\}', '', s)
    s = re.sub(r'([_^$%&#{}])', r'\\\1', s)
    s = re.sub(r'\~', r'\\~{}', s)
    s = re.sub(r'\{', r'', s)
    s = re.sub(r'\}', r'', s)
    s = re.sub(r'\\', r'', s)
    s = re.sub(r'&\w+;', r'', s)
    return s

headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}
result = requests.get("https://www.jemoka.com/index.json", headers=headers)
df = pd.read_json(result.content.decode())
df.index = df.title
df.drop(columns=["categories", "tags", "title"], inplace=True)

# generative = pipeline("multitask-qa-qg", model="valhalla/distilt5-qa-qg-hl-6-4")
extractive = pipeline("question-answering", model='deepset/roberta-base-squad2')
# generative = pipeline("text2text-generation", model = "bigscience/T0")
generative = pipeline("text2text-generation", model="google/t5-v1_1-base")


text = _tex_sanitize(df.loc['Linear Map'].contents)

lines = text.split("\n")
lines[0]


# > 40% confidence is a reasonable answer metric
sorted([(j["score"], j["answer"]) for j in [extractive(context=i, question="What is a linear map?") for i in lines[:10]]], key=lambda x:x[0], reverse=True)


