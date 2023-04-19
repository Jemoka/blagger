"""figures.py

Thin wrapper around the wonderful ../../opt/pdffigures2.jar
to extract figures from a PDF file.
"""

import os
import json
from tempfile import TemporaryDirectory
import shutil

# FOR DEBUG:
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
    dict
        TODO not sure.
    """

# get full path of target
target_path = os.path.abspath(target)

    # store temporary directory
wd = os.getcwd()
    # create and change to temporary directory
with TemporaryDirectory() as tmpdir:
    os.chdir(tmpdir)

    os.system(f"java -jar {JARDIR} -g meta -m fig {target_path}")
    print(tmpdir) # TODO

# change directory back
os.chdir(wd)



target = "../../data/paper.pdf" 
target_path

os.getcwd()

