"""string.py

String manipulation utilities.
"""

import re

from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords

def tex_sanitize(s):
    """Sanitizes a string so that it can be properly compiled in TeX.

    Parameters
    ----------
    s : str
        String to sanitize. 

    Returns
    -------
    str
        Sanitized string

    Notes
    -----
    Escapes the most common TeX special characters: ~^_#%${}.
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

def tighten_query(query:str):
    """Removes unnecessary stopwords from a query. 

    Parameters
    ----------
    query : str
        String to extract keywords from. 

    Returns
    -------
    str
        Sanitized string

    Notes
    -----
    Prox good to come back and think about this
    and whether or not better tooling exists.
    """

    stopword_list = stopwords.words('english')
    question_list = ["who", "whose", "what", "where", "when",
                     "how", "why", "which", "'s"]

    tokenizer = RegexpTokenizer(r'\w')
    tokenized = word_tokenize(query.lower())

    # filter, join
    cleaned = " ".join(list(
        filter((lambda x : (x not in stopword_list) and
                (x not in question_list)), tokenized)))

    # remove punctuation and return
    return re.sub(r"\W", ' ', cleaned).strip()
    

