"""qa.py

Question-Answering Pipeline Cache
"""

import torch
from transformers import pipeline

# get device
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# cache so that globally only one pipeline gets initialized
PIPELINE = None

if not PIPELINE:
    PIPELINE = pipeline("question-answering", model='deepset/roberta-base-squad2')

def QA(**payload):
    return PIPELINE(context=payload["context"],
                    question=payload["question"],
                    device=DEVICE)
