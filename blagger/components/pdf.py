"""pdf.py

PDF Operations
"""

import os
from glob import glob
import json
from tempfile import TemporaryDirectory
import shutil
import subprocess
import re
from tika import parser

from nltk import sent_tokenize

import numpy as np

from PIL import Image

from ..inference.p_rank import P_RANK

# FOR DEBUG:
# import sys
# sys.path.append("..")
# FILEDIR = os.path.dirname(
#     os.path.abspath("./figures.py"))

# constants to identify the pdffigures executable
FILEDIR = os.path.dirname(os.path.abspath(__file__))
# path to java
# TODO change this at will or put in .env 
JARDIR = os.path.abspath( 
    os.path.join(FILEDIR, "../../opt/pdffigures2.jar"))
JAVADIR = os.path.realpath(shutil.which("java"))

def extract_fig_mention(s):
    """Get the figure/table mentions from a caption string

    Parameters
    ----------
    s : str
        String to extract info from.

    Note
    ----
    We can only extract one of these per string

    Returns
    -------
    list 
        [['f', ID], ['t', ID]] etc.
    """

    res = re.search(r"([f|t][i|a][g|b][A-Z]*)\.? ?(\d*).|:\W+", s, flags=re.IGNORECASE)

    if res and res.group(2):
        fig_type = res.group(1)[0].lower()
        fig_num = int(res.group(2))
        return fig_num, fig_type
    else: return None

def clean_label(s):
    """Clean the figure/table labels from a caption string.

    Parameters
    ----------
    s : str
        String to clean.

    Returns
    -------
    str
        The cleaned string.
    """

    return re.sub(r"([f|t][i|a][g|b][A-Z]*)\.? ?(\d*).|:\W+", "", s, flags=re.IGNORECASE)

def extract_text(target):
    """Extract text from PDF file.

    Parameters
    ----------
    target : str
        The file to get figures from.

    Returns
    -------
    dict
        {"raw": raw text, "sents": sentences}
    """

    # get full path of target
    target_path = os.path.abspath(target)

    # extract
    text = parser.from_file(target_path)["content"].strip()

    # replace
    cleaned = re.sub("\n+", " ", text)
    cleaned = re.sub("([f|t][i|a][g|b][A-Z]*)\.", r"\1", cleaned, flags=re.IGNORECASE)
    sents = sent_tokenize(cleaned)

    # filter result
    return sents


def extract_figures(target):
    """Extract figures from PDF file with pdffigures2.

    Parameters
    ----------
    target : str
        The file to get figures from.

    Returns
    -------
    list
        A list dictionaries containing figures, their captions, and a numpy array for the figure.
    """

    # get full path of target
    target_path = os.path.abspath(target)

    # store temporary directory
    wd = os.getcwd()
    # create and change to temporary directory
    with TemporaryDirectory() as tmpdir:
        # change into temproary directory and extract figures
        os.chdir(tmpdir)
        subprocess.check_output(f"java -jar {JARDIR} -g meta -m fig {target_path} -q", shell=True)

        # read the metadata file
        meta_path = glob("meta*.json")[0]
        with open(meta_path, 'r') as df:
            meta = json.load(df)

        # open each of the images as numpy
        for figure in meta["figures"]:
            img = Image.open(figure["renderURL"])
            figure["render"] = np.array(img)
            img.close()

    # change directory back
    os.chdir(wd)

    return meta["figures"]

def select(figures, sents, query, threshold=100, topn=5):
    """Select the best sentences + figures, if any, that would respond to text query with QA.

    Parameters
    ----------
    figures : list
        The output of extract_figures() of the figures of the paper.
    sents : list
        The output of extract_text() on the paper.
    query : str
        The text query to search on.
    threshold : float, optional
        The threshold to return a result.
    topn : int, optional
        The top n of text identification keep.

    Results
    -------
    List[dict], optional
        If the result crosses the threshold, return the relavent figure(s).
    """

    # extract figure ids and mentions
    fig_ids = [extract_fig_mention(i["caption"]) for i in figures]

    # extract best text scores
    text_scores = P_RANK(documents=sents, question=query)
    best_text_scores = sorted(filter(lambda x:x["score"] > threshold, text_scores),
                            key=lambda x:x["score"], reverse=True)[:topn]
    fig_mentions = [i for i in
                    [extract_fig_mention(i["document"]) for i in best_text_scores] if i]
    best_text = [i["document"] for i in best_text_scores]

    # get captions and text scores for caption
    captions = [clean_label(i["caption"]) for i in figures]
    fig_scores = P_RANK(documents=captions, question=query)
    best_fig_scores = sorted(filter(lambda x:x[1]["score"] > threshold, enumerate(fig_scores)),
                            key=lambda x:x[1]["score"], reverse=True)[:topn]
    fig_rels = [fig_ids[i[0]] for i in best_fig_scores]

    # combine final relavent figures
    rel_fig_indicies = list(set(fig_mentions+fig_rels))
    # and get actual index
    fig_indicies = [fig_ids.index(i) for i in rel_fig_indicies]

    figs = [figures[i] for i in fig_indicies]

    return best_text, figs

