"""qa.py

Question-Answering Pipeline Cache
"""

import torch
from sentence_transformers import SentenceTransformer, util

# get device
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# cache so that globally only one pipeline gets initialized
PIPELINE = None

if not PIPELINE:
    PIPELINE = SentenceTransformer('sentence-transformers/msmarco-distilbert-base-tas-b', device=DEVICE)

def P_RANK(**payload):
    docs = payload["documents"]
    query = payload["question"]

    query_emb = PIPELINE.encode(query)
    doc_emb = PIPELINE.encode(docs)

    scores = util.dot_score(query_emb, doc_emb)[0].cpu().tolist()

    #Combine docs & scores
    doc_score_pairs = list(zip(docs, scores))

    return [{"document": i[0],
             "score": i[1]} for i in doc_score_pairs]




