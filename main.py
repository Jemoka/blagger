import requests
import json
import pandas as pd
from transformers import pipeline
import re
import torch
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from dataclasses import dataclass
from rank_bm25 import BM25Okapi

from bpe import Encoder

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device('cpu')

@dataclass
class EngineResponse:
    extractive_answer: str
    extractive_padding: list[str]
    extractive_score: float
    permalink: str

class Engine:
    """Corpus objects holds the query api and the pipeline

    Parameters
    ----------
    data : pd.DataFrame
        The dataframe with columns ["contents", "permalink"] and the index
        as titles to search on.
    threshold : float, optional
        The confidence threshold to return a result.
    """

    def __init__(self, data, threshold=0.4):
        self.__raw = data

        t = Encoder(200, pct_bpe=0.88, EOW="", SOW="", UNK="", PAD="")
        t.fit(self.__raw.contents.apply(lambda x:x.lower()).tolist())

        self.__tf = lambda x:list(filter(lambda i:i!="", t.tokenize(x.lower())))

        self.__content_engine = BM25Okapi(data.contents.apply(lambda x:self.__tf(x)).tolist())
        self.__title_engine = BM25Okapi(data.index.to_series().apply(lambda x:self.__tf(x)).tolist())

        self.__extractive_pipeline = pipeline("question-answering", model='deepset/roberta-base-squad2')
        self.__threshold = threshold

        self.tokenizer = t

    @staticmethod
    def __tex_sanitize(s):
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

    @staticmethod
    def __clean_query_for_bm25(query:str):
        """Cleans query of unimportant stopwords
        """

        stopword_list = stopwords.words('english')
        question_list = ["who", "whose", "what", "where", "when", "how", "why", "which"]

        tokenizer = RegexpTokenizer(r'\w+')
        tokenized = tokenizer.tokenize(query)

        # filter, join, and return
        return " ".join(list(filter((lambda x : (x not in stopword_list) and
                                    (x not in question_list)), tokenized)))

    def __assemble_result(self, title, content, link, qa):
        """Assemble `EngineResponse` given a qa answer

        Parameters
        ----------
        title : str
            The title `index` of the article.
        content : str
            The text of the response article.
        link : str
            The permalink to the response article.
        qa : dict
            The output from Huggingface QA.

        Returns
        -------
        EngineResponse
            The response from the QA Engine.
        """

        # the window to return padding
        PADDING_SIZE = 100

        # calculate padding text
        padding_start = content[max(0, qa["start"]-PADDING_SIZE):qa["start"]]
        padding_end = content[qa["end"]+1:min(len(content), qa["end"]+PADDING_SIZE)]

        return EngineResponse(qa["answer"], [padding_start, padding_end],
                              qa["score"], link)

    def __run_qa(self, index, query):
        """util function to run QA.

        Parameters
        ----------
        index : str
            The data index to query.
        query : str
            The string query.

        Returns
        -------
        EngineResponse, optional
             Results (if above threashold) the response from the QA engine.
        """

        entry = self.__raw.loc[index]
        text = self.__tex_sanitize(entry.contents).lower()
        qa = self.__extractive_pipeline(context=text, question=query.lower(), device=DEVICE)

        if text == "":
            return None # empty page

        if qa["score"] > self.__threshold:
            return self.__assemble_result(index, text, entry.permalink, qa)

    def query(self, query):
        """Queries the engine for the best-match solution.

        Parameters
        ----------
        query : str
            The string query to search the database with.

        Returns
        -------
        EngineResponse
            The response from the query engine.
        """

        # clean and split query into words to search for TF-IDF
        cleaned_query = self.__tf(self.__clean_query_for_bm25(query))

        # get the search result on title and on full-text
        title_result = self.__title_engine.get_top_n(cleaned_query, self.__raw.index, 1)[0]
        full_text_result = self.__content_engine.get_top_n(cleaned_query, self.__raw.index, 1)[0]

        # check score on title and return
        result = self.__run_qa(title_result, query.lower())
        if result: return result

        # otherwise check score on full text and return
        result = self.__run_qa(full_text_result, query.lower())
        if result: return result

def get_data(uri="https://www.jemoka.com/index.json"):
    """Use requests to get the data needed to perform the search task
    """
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}
    result = requests.get(uri, headers=headers)
    df = pd.read_json(result.content.decode())
    df.index = df.title
    df.drop(columns=["categories", "tags", "title"], inplace=True)
    return df

df = get_data()
e = Engine(df, 0.1)
e.query("what is rosettafold?")

