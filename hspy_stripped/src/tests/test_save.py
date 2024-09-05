import h5py
import numpy as np
import pytest

import hyperspy.api as hs

from hspy_stripped.signal import BaseSignal, Signal2D


def test_base_save(tmpdir):
    data = np.arange(2*3*4*5).reshape(2, 3, 4, 5)

    # authentic HyperSpy
    s_original = hs.signals.BaseSignal(data)
    s_original.save(f"{tmpdir}/original.hspy")

    # stripped version
    s_strip = BaseSignal(data)
    s_strip.save(f"{tmpdir}/strip.hspy")

    with (h5py.File(f"{tmpdir}/original.hspy", "r") as fo,
          h5py.File(f"{tmpdir}/strip.hspy", "r") as fs):

        assert fo.attrs["file_format"] == fs.attrs["file_format"]
        assert fo.attrs["file_format_version"] == fs.attrs["file_format_version"]

        assert "/Experiments/__unnamed__" in fo
        assert "/Experiments/__unnamed__" in fs

        foexp = fo["/Experiments/__unnamed__"]
        fsexp = fs["/Experiments/__unnamed__"]

        assert foexp.attrs["package"] == fsexp.attrs["package"]
        assert foexp.attrs["package_version"] == fsexp.attrs["package_version"]

        assert foexp["attributes"].attrs["_lazy"] == fsexp["attributes"].attrs["_lazy"]
        assert foexp["attributes"].attrs["ragged"] == fsexp["attributes"].attrs["ragged"]

        for ax in ("axis-0", "axis-1", "axis-2", "axis-3"):
            assert foexp[ax].attrs["_type"] == fsexp[ax].attrs["_type"]
            assert foexp[ax].attrs["is_binned"] == fsexp[ax].attrs["is_binned"]
            assert foexp[ax].attrs["name"] == fsexp[ax].attrs["name"]
            assert foexp[ax].attrs["navigate"] == fsexp[ax].attrs["navigate"]
            assert foexp[ax].attrs["offset"] == fsexp[ax].attrs["offset"]
            assert foexp[ax].attrs["scale"] == fsexp[ax].attrs["scale"]
            assert foexp[ax].attrs["size"] == fsexp[ax].attrs["size"]

        assert np.isclose(foexp["data"], fsexp["data"]).all()

        assert "learning_results" in foexp
        assert "learning_results" in fsexp

        assert foexp["metadata/General"].attrs["title"] == fsexp["metadata/General"].attrs["title"]
        # time stamp won't be the same
        for attr in ("hyperspy_version", "io_plugin", "operation"):
            assert foexp[f"metadata/General/FileIO/0"].attrs[attr] == fsexp["metadata/General/FileIO/0"].attrs[attr]

        assert foexp["metadata/Signal"].attrs["signal_type"] == fsexp["metadata/Signal"].attrs["signal_type"]

        for attr in ("original_axes_manager", "original_shape", "signal_unfolded", "unfolded"):
            assert foexp[f"metadata/_HyperSpy/Folding"].attrs[attr] == fsexp["metadata/_HyperSpy/Folding"].attrs[attr]

        assert "original_metadata" in foexp
        assert "original_metadata" in fsexp


def test_signal2d_save(tmpdir):
    data = np.arange(2*3*4*5).reshape(2, 3, 4, 5)

    # authentic HyperSpy
    s_original = hs.signals.Signal2D(data)
    s_original.save(f"{tmpdir}/original.hspy")

    # stripped version
    s_strip = Signal2D(data)
    s_strip.save(f"{tmpdir}/strip.hspy")

    with (h5py.File(f"{tmpdir}/original.hspy", "r") as fo,
          h5py.File(f"{tmpdir}/strip.hspy", "r") as fs):

        assert fo.attrs["file_format"] == fs.attrs["file_format"]
        assert fo.attrs["file_format_version"] == fs.attrs["file_format_version"]

        assert "/Experiments/__unnamed__" in fo
        assert "/Experiments/__unnamed__" in fs

        foexp = fo["/Experiments/__unnamed__"]
        fsexp = fs["/Experiments/__unnamed__"]

        assert foexp.attrs["package"] == fsexp.attrs["package"]
        assert foexp.attrs["package_version"] == fsexp.attrs["package_version"]

        assert foexp["attributes"].attrs["_lazy"] == fsexp["attributes"].attrs["_lazy"]
        assert foexp["attributes"].attrs["ragged"] == fsexp["attributes"].attrs["ragged"]

        for ax in ("axis-0", "axis-1", "axis-2", "axis-3"):
            assert foexp[ax].attrs["_type"] == fsexp[ax].attrs["_type"]
            assert foexp[ax].attrs["is_binned"] == fsexp[ax].attrs["is_binned"]
            assert foexp[ax].attrs["name"] == fsexp[ax].attrs["name"]
            assert foexp[ax].attrs["navigate"] == fsexp[ax].attrs["navigate"]
            assert foexp[ax].attrs["offset"] == fsexp[ax].attrs["offset"]
            assert foexp[ax].attrs["scale"] == fsexp[ax].attrs["scale"]
            assert foexp[ax].attrs["size"] == fsexp[ax].attrs["size"]

        assert np.isclose(foexp["data"], fsexp["data"]).all()

        assert "learning_results" in foexp
        assert "learning_results" in fsexp

        assert foexp["metadata/General"].attrs["title"] == fsexp["metadata/General"].attrs["title"]
        # time stamp won't be the same
        for attr in ("hyperspy_version", "io_plugin", "operation"):
            assert foexp[f"metadata/General/FileIO/0"].attrs[attr] == fsexp["metadata/General/FileIO/0"].attrs[attr]

        assert foexp["metadata/Signal"].attrs["signal_type"] == fsexp["metadata/Signal"].attrs["signal_type"]

        for attr in ("original_axes_manager", "original_shape", "signal_unfolded", "unfolded"):
            assert foexp[f"metadata/_HyperSpy/Folding"].attrs[attr] == fsexp["metadata/_HyperSpy/Folding"].attrs[attr]

        assert "original_metadata" in foexp
        assert "original_metadata" in fsexp
