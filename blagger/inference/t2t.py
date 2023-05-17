"""t2t.py

General purpose text2text task
"""

from enum import Enum
import torch
from torch.nn import Softmax
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# get device
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# cache so that globally only one pipeline gets initialized
PIPELINE = None
TOKENIZER = None

if not PIPELINE:
    PIPELINE = AutoModelForSeq2SeqLM.from_pretrained("t5-base").to(DEVICE)
if not TOKENIZER:
    TOKENIZER = AutoTokenizer.from_pretrained("t5-base")

class T2TTask(Enum):
    ACCEPTABILITY = "cola sentence"
    SUMMARIZE = "summarize"

def T2T(**payload):
    task = payload.get("task", T2TTask.SUMMARIZE) # sumarize by default
    payload = payload["payload"]

    assert type(task) == T2TTask, "Unexpected task. Please pass instance of T2TTask."
    task = task.value

    output = PIPELINE.generate(**TOKENIZER(f"{task}: ", payload,
                                           return_tensors="pt").to(DEVICE),
                               max_length=1000)
    output = TOKENIZER.decode(output[0], skip_special_tokens=True).strip()

    return { "output": output }
