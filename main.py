import requests
import json
import pandas as pd
from transformers import pipeline
import re
import torch
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from dataclasses import dataclass, asdict
from rank_bm25 import BM25Okapi
from transformers import GPT2Tokenizer
from nltk import sent_tokenize
from flask_cors import CORS

from flask import Flask, request
from flask_json import FlaskJSON, JsonError, json_response, as_json


DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device('cpu')

@dataclass
class EngineResponse:
    # abstractive_answer: str
    extractive_answer: str
    extractive_padding: list[str]
    extractive_score: float
    permalink: str
    title: str

class Engine:
    """Corpus objects holds the query api and the pipeline

    Parameters
    ----------
    data : pd.DataFrame
        The dataframe with columns ["html", "permalink"] and the index
        as titles to search on.
    threshold : float, optional
        The confidence threshold to return a result.
    """

    def __init__(self, data, threshold=0.4):
        self.__raw = data

        t = GPT2Tokenizer.from_pretrained("gpt2")

        self.__tf = lambda x:[i.replace("Ġ", "") for i in
                              filter(lambda i:i!="", t.tokenize(x.lower()))]

        self.__content_engine = BM25Okapi(data.html.apply(lambda x:self.__tf(x)).tolist())
        self.__title_engine = BM25Okapi(data.index.to_series().apply(lambda x:self.__tf(x)).tolist())

        self.__extractive_pipeline = pipeline("question-answering", model='deepset/roberta-base-squad2')
        # self.__abstractive_pipeline = pipeline("text2text-generation", model='vblagoje/bart_lfqa')
        self.__threshold = threshold

        self.tokenizer = t

    @staticmethod
    def __tex_sanitize(s):
        """Sanitizes a string so that it can be properly compiled in TeX.
        Escapes the most common TeX special characters: ~^_#%${}
        Removes backslashes.
        """

        # sanitiz
        s = re.sub(r'\\\w+\{.*?\}', '', s)
        s = re.sub(r'([_^$%&#{}])', r'\\\1', s)
        s = re.sub(r'\~', r'\\~{}', s)
        s = re.sub(r'\{', r'', s)
        s = re.sub(r'\}', r'', s)
        s = re.sub(r'<.*?>', r'', s)
        s = re.sub(r'\\', r'', s)
        s = re.sub(r'&\w+;', r'', s)
        s = re.sub(r' ?\n+ ?', '. ', s)
        s = re.sub(r'\. ?\.', '.', s)

        return s

    @staticmethod
    def __clean_query_for_bm25(query:str):
        """Cleans query of unimportant stopwords
        """

        stopword_list = stopwords.words('english')
        question_list = ["who", "whose", "what", "where", "when",
                         "how", "why", "which"]

        tokenizer = RegexpTokenizer(r'\w+')
        tokenized = tokenizer.tokenize(query.lower())

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
        # asbt: dict
        #     The output from Huggingface summary.

        Returns
        -------
        EngineResponse
            The response from the QA Engine.
        """

        # the window to return padding
        PADDING_SIZE = 500

        # calculate padding text
        padding_start = content[max(0, qa["start"]-PADDING_SIZE):qa["start"]]
        padding_end = content[qa["end"]+1:min(len(content), qa["end"]+PADDING_SIZE)]

        # we take 1 sentence because the results is kinda odd
        return EngineResponse(qa["answer"], [padding_start, padding_end], qa["score"], link, title)

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
        text = self.__tex_sanitize(entry.html).lower()
        if text ==  "":
            return None # empty page

        qa = self.__extractive_pipeline(context=text, question=query.lower(), device=DEVICE)

        if qa["score"] > self.__threshold:
            # # also run abstractive QA
            # # the window to return padding
            # PADDING_SIZE = 300

            # # calculate padding text
            # padding_start = text[max(0, qa["start"]-PADDING_SIZE):qa["start"]]
            # padding_end = text[qa["end"]+1:min(len(text), qa["end"]+PADDING_SIZE)]

            # # create the right input
            # context = padding_start+qa["answer"]+padding_end
            # # query!
            # prompt = f"question: {query} context: {context}"
            # abstr = self.__abstractive_pipeline(prompt)

            return self.__assemble_result(index, text, entry.permalink, qa)

    def quick_query(self, query):
        """Runs a quick query of the database.

        Parameters
        ----------
        query : str
            The query string.

        Returns
        -------
        title_result : str
            The result of a quick query on the title.
        full_text_result : str
            The result of a full text query.
        """
        # clean and split query into words to search for TF-IDF
        cleaned_query = self.__tf(self.__clean_query_for_bm25(query))

        # get the search result on title and on full-text
        title_result = self.__title_engine.get_top_n(cleaned_query, self.__raw.index, 1)[0]
        full_text_result = self.__content_engine.get_top_n(cleaned_query, self.__raw.index, 1)[0]

        return title_result, full_text_result
        

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

        title_result, full_text_result = self.quick_query(query)

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

# start a flask app
app = Flask(__name__)
FlaskJSON(app)
CORS(app)

# perform query
@app.route('/query')
@as_json
def query():
    q = request.args.get('q')

    result = e.query(q)
    
    if result:
        return {"result": "ok",
                "payload": asdict(result)}
    else:
        return {"result": "failed",
                "payload": "Could not find suitable answer!"}

# yipee
if __name__ == '__main__':
    app.run(host='0.0.0.0')
