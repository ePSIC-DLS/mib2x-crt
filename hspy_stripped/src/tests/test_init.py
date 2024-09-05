import hyperspy.api as hs
import numpy as np
import pytest

from hspy_stripped.signal import BaseSignal, Signal2D


def test_base_init():
    data = np.arange(2*3*4*5).reshape(2, 3, 4, 5)

    # authentic HyperSpy
    s_original = hs.signals.BaseSignal(data)
    do = s_original._to_dictionary()

    # stripped version
    s_strip = BaseSignal(data)
    ds = s_strip._to_dictionary()

    assert do.keys() == ds.keys()

    # assert same data
    assert np.isclose(do["data"], ds["data"]).all()

    # assert same axes
    for oax, sax in zip(do["axes"], ds["axes"]):
        assert oax == sax

    # assert same metadata
    for omd, smd in zip(do["metadata"], ds["metadata"]):
        assert omd == smd

    # assert same attribute
    for oa, sa in zip(do["attributes"], ds["attributes"]):
        assert oa == sa


def test_signal2d_init():
    data = np.arange(2*3*4*5).reshape(2, 3, 4, 5)

    # authentic HyperSpy
    s_original = hs.signals.Signal2D(data)
    do = s_original._to_dictionary()

    # stripped version
    s_strip = Signal2D(data)
    ds = s_strip._to_dictionary()

    assert s_original._signal_dimension == s_strip._signal_dimension

    assert do.keys() == ds.keys()

    # assert same data
    assert np.isclose(do["data"], ds["data"]).all()

    # assert same axes
    for oax, sax in zip(do["axes"], ds["axes"]):
        assert oax == sax

    # assert same metadata
    for omd, smd in zip(do["metadata"], ds["metadata"]):
        assert omd == smd

    # assert same attribute
    for oa, sa in zip(do["attributes"], ds["attributes"]):
        assert oa == sa
