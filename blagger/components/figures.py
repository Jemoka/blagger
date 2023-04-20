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

import numpy as np

from PIL import Image

from ..inference.qa import QA

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

def select_figure(figures, query, threshold=0.5):
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
    dict, optional
        If the result crosses the threshold, return the relavent figure.
    """
    # to score each elements based an a query
    qa_results = []
    for fig in figures:
        qa_results.append(QA(context=fig["caption"], question=query))
    best_result = sorted(enumerate(qa_results), key=lambda x:x[1]["score"], reverse=True)[0]

    # if we are over threshold, return
    if best_result[1]["score"] >= threshold:
        return figures[best_result[0]]
