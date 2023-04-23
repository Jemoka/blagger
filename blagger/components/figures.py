"""figures.py

Thin wrapper around the wonderful ../../opt/pdffigures2.jar
to extract figures from a PDF file.
"""

import os
from glob import glob
import json
from tempfile import TemporaryDirectory
import shutil
import subprocess
import re

import numpy as np

from PIL import Image

# from ..inference.nsp import NSP
# from ..inference.qa import QA
# from ..inference.nli import NLI
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

    return re.sub(r"(f|t)(i|a)(g|b)\w+ ?\d*(.|:)\W+", "", s, flags=re.IGNORECASE)

def extract(target):
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

def select_figure(figures, query, threshold=95):
    """Select the best figure, if any, that would respond to text query with QA.

    Parameters
    ----------
    figures : list
        The output of extract() of the figures of the paper.
    query : str
        The text query to search on.
    threshold : float, optional
        The threshold to return a result.

    Results
    -------
    List[dict], optional
        If the result crosses the threshold, return the relavent figure(s).
    """
    # to score each elements based an a query
    result = P_RANK(documents=[clean_label(i["caption"]) for i in figures], question=query)

    # get best results
    best_results = list(filter(lambda x:x[1]["score"] >= threshold, enumerate(result)))

    # sort
    result_sorted = sorted(best_results, key=lambda x:x[1]["score"], reverse=True)

    # get the actual figures
    answer_figures = [figures[i[0]] for i in result_sorted]

    return answer_figures
