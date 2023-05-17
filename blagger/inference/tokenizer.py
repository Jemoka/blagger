"""t2t.py

General purpose text2text task
"""

from enum import Enum
import torch
from torch.nn import Softmax
from transformers import GPT2Tokenizer

# get device
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# cache so that globally only one pipeline gets initialized
PIPELINE = None
SOFTMAX = Softmax(dim=1)

if not PIPELINE:
    PIPELINE = GPT2Tokenizer.from_pretrained("gpt2")

def TOKENIZER(**payload):
    payload = payload["payload"]

    return { "output": [i.replace("Ġ", "") for i in
                        filter(lambda i:i!="", PIPELINE.tokenize(payload.lower()))] }


