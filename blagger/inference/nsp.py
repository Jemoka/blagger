"""qa.py

Question-Answering Pipeline Cache
"""

import torch
from torch.nn import Softmax
from transformers import AutoTokenizer, BertForNextSentencePrediction

# get device
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# cache so that globally only one pipeline gets initialized
PIPELINE = None
TOKENIZER = None
SOFTMAX = Softmax(dim=1)

if not PIPELINE:
    PIPELINE = BertForNextSentencePrediction.from_pretrained("bert-base-uncased").to(DEVICE)
if not TOKENIZER:
    TOKENIZER = AutoTokenizer.from_pretrained("bert-base-uncased")

def NSP(**payload):
    sentence1 = payload["sentence1"]
    sentence2 = payload["sentence2"]

    output = SOFTMAX(
        PIPELINE(**TOKENIZER(sentence1, sentence2,
                            return_tensors="pt").to(DEVICE)
                ).logits).squeeze()

    follows = not bool(torch.argmax(output).item())
    confidence = torch.max(output).item()

    return { "follow": follows,
             "score": confidence }

# PIPELINE(" [SEP] Yes, I think so.")
# PIPELINE("Does this sentence follow the next one? [SEP] ")


