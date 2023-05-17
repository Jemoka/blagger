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
