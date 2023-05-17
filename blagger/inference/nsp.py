"""nsp.py

Next Sentence Prediction pipeline cache
"""

import torch
from torch.nn import Softmax
from transformers import AutoTokenizer, FNetForNextSentencePrediction

# get device
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# cache so that globally only one pipeline gets initialized
PIPELINE = None
TOKENIZER = None
SOFTMAX = Softmax(dim=1)

if not PIPELINE:
    PIPELINE = FNetForNextSentencePrediction.from_pretrained("google/fnet-base").to(DEVICE)
if not TOKENIZER:
    TOKENIZER = AutoTokenizer.from_pretrained("google/fnet-base")

def NSP(**payload):
    sentence1 = payload["sentence1"]
    sentence2 = payload["sentence2"]

    output = SOFTMAX(
        PIPELINE(**TOKENIZER(sentence1, sentence2,
                            return_tensors="pt").to(DEVICE),
                 labels=torch.LongTensor([1])).logits).squeeze()

    follows = not bool(torch.argmax(output).item())
    confidence = torch.max(output).item()

    return { "follow": follows,
             "score": confidence }

# PIPELINE(" [SEP] Yes, I think so.")
NSP(sentence1="What's the defintion of linear map?",
    sentence2="The answer can be found in small chicken characterization.")


