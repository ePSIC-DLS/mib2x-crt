from packaging.version import Version

import h5py
import numpy as np

version = "3.3"
default_version = Version(version)
HSPY_VERSION = "2.0.1"


def flatten_data(x, is_hdf5=False):
    new_data = np.empty(shape=x.shape, dtype=object)
    shapes = np.empty(shape=x.shape, dtype=object)
    for i in np.ndindex(x.shape):
        data_ = np.array(x[i]).ravel()
        if np.issubdtype(data_.dtype, np.dtype("U")):
            if is_hdf5:
                # h5py doesn't support numpy unicode dtype, convert to
                # compatible dtype
                new_data[i] = data_.astype(h5py.string_dtype())
            else:
                # Convert to list to save ragged array of array with string dtype
                new_data[i] = data_.tolist()
        else:
            new_data[i] = data_
        shapes[i] = np.array(np.array(x[i]).shape)
    return new_data, shapes


def get_signal_chunks(shape, dtype, signal_axes=None, target_size=1e6):
    """
    Function that calculates chunks for the signal, preferably at least one
    chunk per signal space.

    Parameters
    ----------
    shape : tuple
        The shape of the dataset to be stored / chunked.
    dtype : {dtype, string}
        The numpy dtype of the data.
    signal_axes : {None, iterable of ints}
        The axes defining "signal space" of the dataset. If None, the default
        h5py chunking is performed.
    target_size : int
        The target number of bytes for one chunk
    """
    typesize = np.dtype(dtype).itemsize
    if shape == (0,) or signal_axes is None:
        # enable autochunking from h5py
        return True

    # largely based on the guess_chunk in h5py
    bytes_per_signal = np.prod([shape[i] for i in signal_axes]) * typesize
    signals_per_chunk = int(np.floor_divide(target_size, bytes_per_signal))
    navigation_axes = tuple(i for i in range(len(shape)) if i not in signal_axes)
    num_nav_axes = len(navigation_axes)
    num_signals = np.prod([shape[i] for i in navigation_axes])
    if signals_per_chunk < 2 or num_nav_axes == 0:
        # signal is larger than chunk max
        chunks = [s if i in signal_axes else 1 for i, s in enumerate(shape)]
        return tuple(chunks)
    elif signals_per_chunk > num_signals:
        return shape
    else:
        # signal is smaller than chunk max
        # Index of axes with size smaller than required to make all chunks equal
        small_idx = []
        # Sizes of axes with size smaller than required to make all chunks equal
        small_sizes = []
        iterate = True
        while iterate:
            iterate = False
            # Calculate the size of the chunks of the axes not in `small_idx`
            # The process is iterative because `nav_axes_chunks` can be bigger
            # than some axes sizes. If that is the case, the value must be
            # recomputed at the next iteration after having added the "offending"
            # axes to `small_idx`
            nav_axes_chunks = int(
                np.floor(
                    (signals_per_chunk / np.prod(small_sizes))
                    ** (1 / (num_nav_axes - len(small_sizes)))
                )
            )
            for index, size in enumerate(shape):
                if (
                    index not in (list(signal_axes) + small_idx)
                    and size < nav_axes_chunks
                ):
                    small_idx.append(index)
                    small_sizes.append(size)
                    iterate = True
        chunks = [
            s if i in signal_axes or i in small_idx else nav_axes_chunks
            for i, s in enumerate(shape)
        ]
        return tuple(int(x) for x in chunks)



class HierarchicalWriter:
    """
    An object used to simplify and organize the process for writing a
    Hierarchical signal, such as hspy/zspy format.
    """

    target_size = 1e6
    _unicode_kwds = None
    _is_hdf5 = False

    def __init__(self, file, signal, group, **kwds):
        """Initialize a generic file writer for hierachical data storage types.

        Parameters
        ----------
        file: str
            The file where the signal is to be saved
        signal: BaseSignal
            A BaseSignal to be saved
        group: Group
            A group to where the experimental data will be saved.
        kwds:
            Any additional keywords used for saving the data.
        """
        self.file = file
        self.signal = signal
        self.group = group
        self.Dataset = None
        self.Group = None
        self.kwds = kwds

    @staticmethod
    def _get_object_dset(*args, **kwargs):  # pragma: no cover
        raise NotImplementedError("This method must be implemented by subclasses.")

    @staticmethod
    def _store_data(*arg):  # pragma: no cover
        raise NotImplementedError("This method must be implemented by subclasses.")

    @classmethod
    def overwrite_dataset(
        cls,
        group,
        data,
        key,
        signal_axes=None,
        chunks=None,
        show_progressbar=True,
        **kwds,
    ):
        if chunks is None:
            chunks = get_signal_chunks(
                data.shape, data.dtype, signal_axes, cls.target_size
            )
        if np.issubdtype(data.dtype, np.dtype("U")):
            # Saving numpy unicode type is not supported in h5py
            data = data.astype(np.dtype("S"))

        if data.dtype != np.dtype("O"):
            got_data = False
            while not got_data:
                try:
                    these_kwds = kwds.copy()
                    these_kwds.update(
                        dict(
                            shape=data.shape,
                            dtype=data.dtype,
                            exact=True,
                            chunks=chunks,
                        )
                    )

                    # If chunks is True, the `chunks` attribute of `dset` below
                    # contains the chunk shape guessed by h5py
                    dset = group.require_dataset(key, **these_kwds)
                    got_data = True
                except TypeError:
                    # if the shape or dtype/etc do not match,
                    # we delete the old one and create new in the next loop run
                    del group[key]

        if data.dtype == np.dtype("O"):
            new_data, shapes = flatten_data(data, is_hdf5=cls._is_hdf5)

            dset = cls._get_object_dset(group, new_data, key, chunks, **kwds)
            shape_dset = cls._get_object_dset(
                group, shapes, f"_ragged_shapes_{key}", chunks, dtype=int, **kwds
            )

            cls._store_data(
                (new_data, shapes),
                (dset, shape_dset),
                group,
                (key, f"_ragged_shapes_{key}"),
                (chunks, chunks),
                show_progressbar,
            )
        else:
            cls._store_data(data, dset, group, key, chunks, show_progressbar)

    def write(self):
        self.write_signal(self.signal, self.group, **self.kwds)

    def write_signal(
        self,
        signal,
        group,
        write_dataset=True,
        chunks=None,
        show_progressbar=True,
        **kwds,
    ):
        """Writes a signal dict to a hdf5/zarr group"""
        # group.attrs.update(signal["package_info"])
        group.attrs.update({'package': 'hyperspy',
                            'package_version': HSPY_VERSION})

        for i, axis_dict in enumerate(signal["axes"]):
            group_name = f"axis-{i}"
            # delete existing group in case the file have been open in 'a' mode
            # and we are saving a different type of axis, to avoid having
            # incompatible axis attributes from previously saved axis.
            if group_name in group.keys():
                del group[group_name]
            coord_group = group.create_group(group_name)
            self.dict2group(axis_dict, coord_group, **kwds)

        mapped_par = group.require_group("metadata")
        metadata_dict = signal["metadata"]

        if write_dataset:
            self.overwrite_dataset(
                group,
                signal["data"],
                "data",
                signal_axes=[
                    idx
                    for idx, axis in enumerate(signal["axes"])
                    if not axis["navigate"]
                ],
                chunks=chunks,
                show_progressbar=show_progressbar,
                **kwds,
            )

        if default_version < Version("1.2"):
            metadata_dict["_internal_parameters"] = metadata_dict.pop("_HyperSpy")

        self.dict2group(metadata_dict, mapped_par, **kwds)
        original_par = group.require_group("original_metadata")
        self.dict2group(signal["original_metadata"], original_par, **kwds)
        learning_results = group.require_group("learning_results")
        self.dict2group(signal["learning_results"], learning_results, **kwds)
        attributes = group.require_group("attributes")
        self.dict2group(signal["attributes"], attributes, **kwds)

        if signal["models"]:
            model_group = self.file.require_group("Analysis/models")
            self.dict2group(signal["models"], model_group, **kwds)
            for model in model_group.values():
                model.attrs["_signal"] = group.name

    def dict2group(self, dictionary, group, **kwds):
        "Recursive writer of dicts and signals"
        for key, value in dictionary.items():
            if isinstance(value, dict):
                self.dict2group(value, group.require_group(key), **kwds)
            elif isinstance(value, (np.ndarray, self.Dataset)):
                self.overwrite_dataset(group, value, key, **kwds)

            elif value is None:
                group.attrs[key] = "_None_"

            elif isinstance(value, bytes):
                try:
                    # binary string if has any null characters (otherwise not
                    # supported by hdf5)
                    value.index(b"\x00")
                    group.attrs["_bs_" + key] = np.void(value)
                except ValueError:
                    group.attrs[key] = value.decode()

            elif isinstance(value, str):
                group.attrs[key] = value

            elif isinstance(value, list):
                if len(value):
                    self.parse_structure(key, group, value, "_list_", **kwds)
                else:
                    group.attrs["_list_empty_" + key] = "_None_"

            elif isinstance(value, tuple):
                if len(value):
                    self.parse_structure(key, group, value, "_tuple_", **kwds)
                else:
                    group.attrs["_tuple_empty_" + key] = "_None_"

            else:
                try:
                    group.attrs[key] = value
                except Exception:
                    pass

    def parse_structure(self, key, group, value, _type, **kwds):
        try:
            # Here we check if there are any signals in the container, as
            # casting a long list of signals to a numpy array takes a very long
            # time. So we check if there are any, and save numpy the trouble
            if np.any([isinstance(t, dict) and "_sig_" in t for t in value]):
                tmp = np.array([[0]])
            else:
                tmp = np.array(value)
        except ValueError:
            tmp = np.array([[0]])

        if np.issubdtype(tmp.dtype, object) or tmp.ndim != 1:
            self.dict2group(
                dict(zip([str(i) for i in range(len(value))], value)),
                group.require_group(_type + str(len(value)) + "_" + key),
                **kwds,
            )
        elif np.issubdtype(tmp.dtype, np.dtype("U")):
            if _type + key in group:
                del group[_type + key]
            group.create_dataset(
                _type + key, shape=tmp.shape, **self._unicode_kwds, **kwds
            )
            group[_type + key][:] = tmp[:]
        else:
            if _type + key in group:
                del group[_type + key]
            group.create_dataset(_type + key, data=tmp, **kwds)
