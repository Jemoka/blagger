"""nli.py

Natural Language Inference Pipeline Cache
"""

import torch
from torch.nn import Softmax
from transformers import AutoTokenizer, RobertaForSequenceClassification

# get device
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# cache so that globally only one pipeline gets initialized
PIPELINE = None
TOKENIZER = None
SOFTMAX = Softmax(dim=1)

if not PIPELINE:
    PIPELINE = RobertaForSequenceClassification.from_pretrained("roberta-large-mnli").to(DEVICE)
if not TOKENIZER:
    TOKENIZER = AutoTokenizer.from_pretrained("roberta-large-mnli")

LABELS = { 0: "contradiction",
           1: "neutral",
           2: "entailment" }

def NLI(**payload):
    sentence1 = payload["sentence1"]
    sentence2 = payload["sentence2"]

    output = SOFTMAX(
        PIPELINE(**TOKENIZER(sentence1, sentence2,
                            return_tensors="pt").to(DEVICE),
                 labels=torch.LongTensor([1])).logits).squeeze()

    prediction = LABELS[torch.argmax(output).item()]
    confidence = torch.max(output).item()

    return { "reasoning": prediction,
             "score": confidence }
