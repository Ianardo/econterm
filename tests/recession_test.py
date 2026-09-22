import numpy as np
from econterm.viz import recession_spans

D = np.array(["2020-01-01", "2020-02-01", "2020-03-01", "2020-04-01", "2020-05-01"],
             dtype="datetime64[D]")


def test_recession_in_middle():
    flags = np.array([0.0, 1.0, 1.0, 0.0, 0.0])
    assert recession_spans(D, flags) == [(D[1], D[3])]


def test_starts_mid_recession():
    flags = np.array([1.0, 1.0, 0.0, 0.0, 0.0])
    assert recession_spans(D, flags) == [(D[0], D[2])]


def test_ends_mid_recession():
    flags = np.array([0.0, 0.0, 0.0, 1.0, 1.0])
    assert recession_spans(D, flags) == [(D[3], D[4])]


def test_no_recession():
    flags = np.zeros(5)
    assert recession_spans(D, flags) == []
    
