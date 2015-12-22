from parallel_svmlight_loader import load_svmlight_file

import numpy as np
import os


currdir = os.path.dirname(os.path.abspath(__file__))
datafile = os.path.join(currdir, "data", "svmlight_file.txt")


def test_load_svmlight_file():
    X, y = load_svmlight_file(datafile, n_jobs=1)
