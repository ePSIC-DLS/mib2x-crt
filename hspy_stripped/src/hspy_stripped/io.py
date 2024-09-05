from collections.abc import MutableMapping
from datetime import datetime
from pathlib import Path

from .rsciio.hspy._api import file_writer
from .rsciio._hierarchical import HSPY_VERSION as hs_version


def save(filename, signal, overwrite=None, file_format=None, **kwds):
    """Only hspy"""

    extension = ".hspy"

    filename = Path(filename)

    write = True  # just write, no checking

    if write:
        # Pass as a string for now, pathlib.Path not
        # properly supported in io_plugins
        signal = _add_file_load_save_metadata("save", signal)

        signal_dic = signal._to_dictionary(add_models=True)

        # do this in hspy file writer
        # signal_dic["package_info"] = get_object_package_info(signal)

        if not isinstance(filename, MutableMapping):
            file_writer(str(filename), signal_dic, **kwds)

            signal.tmp_parameters.set_item("folder", filename.parent)
            signal.tmp_parameters.set_item("filename", filename.stem)
            signal.tmp_parameters.set_item("extension", extension)
        else:
            file_writer(filename, signal_dic, **kwds)

            if hasattr(filename, "path"):
                file = Path(filename.path).resolve()
                signal.tmp_parameters.set_item("folder", file.parent)
                signal.tmp_parameters.set_item("filename", file.stem)
                signal.tmp_parameters.set_item("extension", extension)


def _add_file_load_save_metadata(operation, signal):
    mdata_dict = {
        "operation": operation,
        "io_plugin": "rsciio.hspy",
        "hyperspy_version": hs_version,
        "timestamp": datetime.now().astimezone().isoformat(),
    }
    # get the largest integer key present under General.FileIO, returning 0
    # as default if none are present
    largest_index = max(
        [
            int(i.replace("Number_", "")) + 1
            for i in signal.metadata.get_item("General.FileIO", {}).keys()
        ]
        + [0]
    )

    signal.metadata.set_item(f"General.FileIO.{largest_index}", mdata_dict)

    return signal
