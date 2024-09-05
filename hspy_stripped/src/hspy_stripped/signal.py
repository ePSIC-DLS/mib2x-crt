from collections.abc import MutableMapping
import copy
from pathlib import Path
import types
import unicodedata

import numpy as np
import traits.api as t

from .axes import AxesManager, BaseDataAxis
from .io import save as io_save


class BaseSignal:
    """

    Attributes
    ----------
    ragged : bool
        Whether the signal is ragged or not.
    isig
        Signal indexer/slicer.
    inav
        Navigation indexer/slicer.
    metadata : hyperspy.misc.utils.DictionaryTreeBrowser
        The metadata of the signal structured as documented in
        :ref:`metadata_structure`.
    original_metadata : hyperspy.misc.utils.DictionaryTreeBrowser
        All metadata read when loading the data.


    Examples
    --------
    General signal created from a numpy or cupy array.

    >>> data = np.ones((10, 10))
    >>> s = hs.signals.BaseSignal(data)

    """

    _dtype = "real"
    # When _signal_dimension=-1, the signal dimension of BaseSignal is defined
    # by the dimension of the array, and this is implemented by the default
    # value of navigate=False in BaseDataAxis
    _signal_dimension = -1
    _signal_type = ""
    _lazy = False
    _alias_signal_types = []
    _additional_slicing_targets = [
        "metadata.Signal.Noise_properties.variance",
    ]

    def __init__(self, data, **kwds):
        """
        Create a signal instance.

        Parameters
        ----------
        data : numpy.ndarray
           The signal data. It can be an array of any dimensions.
        axes : [dict/axes], optional
            List of either dictionaries or axes objects to define the axes (see
            the documentation of the :class:`~hyperspy.axes.AxesManager`
            class for more details).
        attributes : dict, optional
            A dictionary whose items are stored as attributes.
        metadata : dict, optional
            A dictionary containing a set of parameters
            that will to stores in the ``metadata`` attribute.
            Some parameters might be mandatory in some cases.
        original_metadata : dict, optional
            A dictionary containing a set of parameters
            that will to stores in the ``original_metadata`` attribute. It
            typically contains all the parameters that has been
            imported from the original data file.
        ragged : bool or None, optional
            Define whether the signal is ragged or not. Overwrite the
            ``ragged`` value in the ``attributes`` dictionary. If None, it does
            nothing. Default is None.
        """
        # the 'full_initialisation' keyword is private API to be used by the
        # _assign_subclass method. Purposely not exposed as public API.
        # Its purpose is to avoid creating new attributes, which breaks events
        # and to reduce overhead when changing 'signal_type'.
        if kwds.get("full_initialisation", True):
            self._create_metadata()

            self.models = ModelManager(self)

            self.learning_results = LearningResults()

            kwds["data"] = data
            self._plot = None

            self.inav = SpecialSlicersSignal(self, True)
            self.isig = SpecialSlicersSignal(self, False)
            self._load_dictionary(kwds)

        if self._signal_dimension >= 0:
            # We don't explicitly set the signal_dimension of ragged because
            # we can't predict it in advance
            self.axes_manager._set_signal_dimension(self._signal_dimension)

    def _create_metadata(self):
        self._metadata = DictionaryTreeBrowser()
        mp = self.metadata
        mp.add_node("_HyperSpy")
        mp.add_node("General")
        mp.add_node("Signal")
        mp._HyperSpy.add_node("Folding")
        folding = mp._HyperSpy.Folding
        folding.unfolded = False
        folding.signal_unfolded = False
        folding.original_shape = None
        folding.original_axes_manager = None
        self._original_metadata = DictionaryTreeBrowser()
        self.tmp_parameters = DictionaryTreeBrowser()

    def _load_dictionary(self, file_data_dict):
        """Load data from dictionary.

        Parameters
        ----------
        file_data_dict : dict
            A dictionary containing at least a 'data' keyword with an array of
            arbitrary dimensions. Additionally the dictionary can contain the
            following items:

            * data: the signal data. It can be an array of any dimensions.

            * axes: a dictionary to define the axes (see the documentation of
              the :class:`~hyperspy.axes.AxesManager` class for more details).
            * attributes: a dictionary whose items are stored as attributes.

            * metadata: a dictionary containing a set of parameters that will
              to stores in the `metadata` attribute. Some parameters might be
              mandatory in some cases.
            * original_metadata: a dictionary containing a set of parameters
              that will to stores in the `original_metadata` attribute. It
              typically contains all the parameters that has been
              imported from the original data file.
            * ragged: a bool, defining whether the signal is ragged or not.
              Overwrite the attributes['ragged'] entry

        """
        self.data = file_data_dict["data"]
        oldlazy = self._lazy
        attributes = file_data_dict.get("attributes", {})
        ragged = file_data_dict.get("ragged")
        if ragged is not None:
            attributes["ragged"] = ragged
        if "axes" not in file_data_dict:
            file_data_dict["axes"] = self._get_undefined_axes_list(
                attributes.get("ragged", False)
            )
        self.axes_manager = AxesManager(file_data_dict["axes"])
        # Setting `ragged` attributes requires the `axes_manager`
        for key, value in attributes.items():
            if hasattr(self, key):
                if isinstance(value, dict):
                    for k, v in value.items():
                        setattr(getattr(self, key), k, v)
                else:
                    setattr(self, key, value)
        if "models" in file_data_dict:
            self.models._add_dictionary(file_data_dict["models"])
        if "metadata" not in file_data_dict:
            file_data_dict["metadata"] = {}
        else:
            # Get all hspy object back from their dictionary representation
            _obj_in_dict2hspy(file_data_dict["metadata"], lazy=self._lazy)
        if "original_metadata" not in file_data_dict:
            file_data_dict["original_metadata"] = {}

        self.original_metadata.add_dictionary(file_data_dict["original_metadata"])
        self.metadata.add_dictionary(file_data_dict["metadata"])

        if "title" not in self.metadata.General:
            self.metadata.General.title = ""
        if self._signal_type or not self.metadata.has_item("Signal.signal_type"):
            self.metadata.Signal.signal_type = self._signal_type
        if "learning_results" in file_data_dict:
            self.learning_results.__dict__.update(file_data_dict["learning_results"])
        if self._lazy is not oldlazy:
            self._assign_subclass()

    def _get_undefined_axes_list(self, ragged=False):
        """Returns default list of axes construct from the data array shape."""
        axes = []
        for s in self.data.shape:
            axes.append(
                {
                    "size": int(s),
                }
            )
        # With ragged signal with navigation dimension 0 and signal dimension 0
        # we return an empty list to avoid getting a navigation axis of size 1,
        # which is incorrect, because it corresponds to the ragged dimension
        if ragged and len(axes) == 1 and axes[0]["size"] == 1:
            axes = []
        return axes

    def _to_dictionary(
        self, add_learning_results=True, add_models=False, add_original_metadata=True
    ):
        """Returns a dictionary that can be used to recreate the signal.

        All items but `data` are copies.

        Parameters
        ----------
        add_learning_results : bool, optional
            Whether or not to include any multivariate learning results in
            the outputted dictionary. Default is True.
        add_models : bool, optional
            Whether or not to include any models in the outputted dictionary.
            Default is False
        add_original_metadata : bool
            Whether or not to include the original_medata in the outputted
            dictionary. Default is True.

        Returns
        -------
        dic : dict
            The dictionary that can be used to recreate the signal

        """
        dic = {
            "data": self.data,
            "axes": self.axes_manager._get_axes_dicts(),
            "metadata": copy.deepcopy(self.metadata.as_dictionary()),
            "tmp_parameters": self.tmp_parameters.as_dictionary(),
            "attributes": {"_lazy": self._lazy, "ragged": self.axes_manager._ragged},
        }
        if add_original_metadata:
            dic["original_metadata"] = copy.deepcopy(
                self.original_metadata.as_dictionary()
            )
        if add_learning_results and hasattr(self, "learning_results"):
            dic["learning_results"] = copy.deepcopy(self.learning_results.__dict__)
        if add_models:
            dic["models"] = self.models._models.as_dictionary()
        return dic

    def save(
        self, filename=None, overwrite=None, extension=None, file_format=None, **kwds
    ):
        if filename is None:
            if self.tmp_parameters.has_item(
                "filename"
            ) and self.tmp_parameters.has_item("folder"):
                filename = Path(
                    self.tmp_parameters.folder, self.tmp_parameters.filename
                )
                extension = (
                    self.tmp_parameters.extension if not extension else extension
                )
            elif self.metadata.has_item("General.original_filename"):
                filename = self.metadata.General.original_filename
            else:
                raise ValueError("File name not defined")

        if not isinstance(filename, MutableMapping):
            filename = Path(filename)
            if extension is not None:
                filename = filename.with_suffix(f".{extension}")
        io_save(filename, self, overwrite=overwrite, file_format=file_format, **kwds)

    @property
    def data(self):
        """The underlying data structure as a :class:`numpy.ndarray` (or
        :class:`dask.array.Array`, if the Signal is lazy)."""
        return self._data

    @data.setter
    def data(self, value):
        # Object supporting __array_function__ protocol (NEP-18) or the
        # array API standard doesn't need to be cast to numpy array
        if not (
            hasattr(value, "__array_function__")
            or hasattr(value, "__array_namespace__")
        ):
            value = np.asanyarray(value)
        self._data = np.atleast_1d(value)

    @property
    def metadata(self):
        """The metadata of the signal."""
        return self._metadata

    @property
    def original_metadata(self):
        """The original metadata of the signal."""
        return self._original_metadata


class Signal2D(BaseSignal):
    """General 2D signal class."""

    _signal_dimension = 2

    def __init__(self, *args, **kwargs):
        if kwargs.get("ragged", False):
            raise ValueError("Signal2D can't be ragged.")
        super().__init__(*args, **kwargs)


class ModelManager(object):
    """Container for models"""

    def __init__(self, signal, dictionary=None):
        self._signal = signal
        self._models = DictionaryTreeBrowser()
        self._add_dictionary(dictionary)

    def _add_dictionary(self, dictionary=None):
        if dictionary is not None:
            for k, v in dictionary.items():
                if k.startswith("_") or k in ["restore", "remove"]:
                    raise KeyError("Can't add dictionary with key '%s'" % k)
                k = slugify(k, True)
                self._models.set_item(k, v)
                setattr(self, k, self.ModelStub(self, k))


class LearningResults(object):
    """Stores the parameters and results from a decomposition."""

    # Decomposition
    factors = None
    loadings = None
    explained_variance = None
    explained_variance_ratio = None
    number_significant_components = None
    decomposition_algorithm = None
    poissonian_noise_normalized = None
    output_dimension = None
    mean = None
    centre = None
    # Clustering values
    cluster_membership = None
    cluster_labels = None
    cluster_centers = None
    cluster_centers_estimated = None
    cluster_algorithm = None
    number_of_clusters = None
    estimated_number_of_clusters = None
    cluster_metric_data = None
    cluster_metric_index = None
    cluster_metric = None
    # Unmixing
    bss_algorithm = None
    unmixing_matrix = None
    bss_factors = None
    bss_loadings = None
    # Shape
    unfolded = None
    original_shape = None
    # Masks
    navigation_mask = None
    signal_mask = None


class SpecialSlicers(object):
    def __init__(self, obj, isNavigation):
        """Create a slice of the signal. The indexing supports integer,
        decimal numbers or strings (containing a decimal number and an units).

        >>> s = hs.signals.Signal1D(np.arange(10))
        >>> s
        <Signal1D, title: , dimensions: (|10)>
        >>> s.data
        array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        >>> s.axes_manager[0].scale = 0.5
        >>> s.axes_manager[0].axis
        array([0. , 0.5, 1. , 1.5, 2. , 2.5, 3. , 3.5, 4. , 4.5])
        >>> s.isig[0.5:4.].data
        array([1, 2, 3, 4, 5, 6, 7])
        >>> s.isig[0.5:4].data
        array([1, 2, 3])
        >>> s.isig[0.5:4:2].data
        array([1, 3])
        >>> s.axes_manager[0].units = 'µm'
        >>> s.isig[:'2000 nm'].data
        array([0, 1, 2, 3])
        """
        self.isNavigation = isNavigation
        self.obj = obj

    def __getitem__(self, slices, out=None):
        return self.obj._slicer(slices, self.isNavigation, out=out)


class SpecialSlicersSignal(SpecialSlicers):
    def __setitem__(self, i, j):
        """x.__setitem__(i, y) <==> x[i]=y"""
        if isinstance(j, BaseSignal):
            j = j.data
        array_slices = self.obj._get_array_slices(i, self.isNavigation)
        self.obj.data[array_slices] = j

    def __len__(self):
        return self.obj.axes_manager.signal_shape[0]


def _serpentine_iter(shape):
    """Similar to np.ndindex, but yields indices
    in serpentine pattern, like snake game.
    Takes shape in hyperspy order, not numpy order.

    Code by Stackoverflow user Paul Panzer,
    from https://stackoverflow.com/questions/57366966/

    Note that the [::-1] reversing is necessary to iterate first along
    the x-direction on multidimensional navigators.
    """
    shape = shape[::-1]
    N = len(shape)
    idx = N * [0]
    drc = N * [1]
    while True:
        yield (*idx,)[::-1]
        for j in reversed(range(N)):
            if idx[j] + drc[j] not in (-1, shape[j]):
                idx[j] += drc[j]
                break
            drc[j] *= -1
        else:  # pragma: no cover
            break


def _flyback_iter(shape):
    "Classic flyback scan pattern generator which yields indices in similar fashion to np.ndindex. Takes shape in hyperspy order, not numpy order."
    shape = shape[::-1]

    class ndindex_reversed(np.ndindex):
        def __next__(self):
            next(self._it)
            return self._it.multi_index[::-1]

    return ndindex_reversed(shape)


class DictionaryTreeBrowser:

    def __init__(self, dictionary=None, double_lines=False, lazy=True):
        self._lazy_attributes = {}
        self._double_lines = double_lines

        if dictionary is None:
            dictionary = {}

        if lazy:
            self._lazy_attributes.update(dictionary)
        else:
            self._process_dictionary(dictionary, double_lines)

    def _process_dictionary(self, dictionary, double_lines):
        """Process the provided dictionary to set the attributes"""
        for key, value in dictionary.items():
            if key == "_double_lines":
                value = double_lines
            self._setattr(key, value, keep_existing=True)

    def process_lazy_attributes(self):
        """Run the DictionaryTreeBrowser machinery for the lazy attributes."""
        if len(self._lazy_attributes) > 0:
            _logger.debug("Processing lazy attributes DictionaryBrowserTree")
            self._process_dictionary(self._lazy_attributes, self._double_lines)
        self._lazy_attributes = {}

    def add_dictionary(self, dictionary, double_lines=False):
        """Add new items from dictionary."""
        if len(self._lazy_attributes) > 0:
            # To simplify merging lazy and non lazy attribute, we get self
            # as a dictionary and update the dictionary with the attributes
            d = self.as_dictionary()
            nested_dictionary_merge(d, dictionary)
            self.__init__(d, double_lines=double_lines, lazy=True)
        else:
            self._process_dictionary(dictionary, double_lines)

    def set_item(self, item_path, value):
        if not self.has_item(item_path):
            self.add_node(item_path)
        if isinstance(item_path, str):
            item_path = item_path.split(".")
        if len(item_path) > 1:
            self.__getattribute__(item_path.pop(0)).set_item(item_path, value)
        else:
            self.__setattr__(item_path.pop(), value)

    def add_node(self, node_path):
        """Adds all the nodes in the given path if they don't exist.

        Parameters
        ----------
        node_path: str
            The nodes must be separated by full stops (periods).

        Examples
        --------

        >>> dict_browser = DictionaryTreeBrowser({})
        >>> dict_browser.add_node('First.Second')
        >>> dict_browser.First.Second = 3
        >>> dict_browser
        └── First
            └── Second = 3

        """
        keys = node_path.split(".")
        dtb = self
        for key in keys:
            if dtb.has_item(key) is False:
                dtb[key] = DictionaryTreeBrowser(lazy=False)
            dtb = dtb[key]

    def has_item(
        self, item_path, default=None, full_path=True, wild=False, return_path=False
    ):
        """
        Given a path, return True if it exists. May also perform a search
        whether an item exists and optionally returns the full path instead of
        boolean value.

        The nodes of the path are separated using periods.

        Parameters
        ----------
        item_path : str
            A string describing the path with each item separated by
            full stops (periods).
        full_path : bool, default True
            If True, the full path to the item has to be given. If
            False, a search for the item key is performed (can include
            additional nodes preceding they key separated by full stops).
        wild : bool, default True
            Only applies if ``full_path=False``. If True, searches for any items
            where ``item`` matches a substring of the item key (case insensitive).
            Default is ``False``.
        return_path : bool, default False
            Only applies if ``full_path=False``. If False, a boolean
            value is returned. If True, the full path to the item is returned,
            a list of paths for multiple matches, or default value if it does
            not exist.
        default :
            The value to return for path if the item does not exist (default is ``None``).

        Examples
        --------

        >>> dict = {'To' : {'be' : True}}
        >>> dict_browser = DictionaryTreeBrowser(dict)
        >>> dict_browser.has_item('To')
        True
        >>> dict_browser.has_item('To.be')
        True
        >>> dict_browser.has_item('To.be.or')
        False
        >>> dict_browser.has_item('be', full_path=False)
        True
        >>> dict_browser.has_item('be', full_path=False, return_path=True)
        'To.be'

        """
        if full_path:
            if isinstance(item_path, str):
                item_path = item_path.split(".")
            else:
                item_path = copy.copy(item_path)
            attrib = item_path.pop(0)
            if hasattr(self, attrib):
                if len(item_path) == 0:
                    return True
                else:
                    item = self[attrib]
                    if isinstance(item, type(self)):
                        return item.has_item(item_path)
                    else:
                        return False
            else:
                return False
        else:
            if not return_path:
                return self._nested_get(item_path, wild=wild) != []
            else:
                result = self._nested_get(item_path, wild=wild, return_path=True)
                if len(result) == 0:
                    return default
                elif len(result) == 1:
                    return result[0][0]
                return [i[0] for i in result]

    def get_item(
        self, item_path, default=None, full_path=True, wild=False, return_path=False
    ):
        if full_path:
            if isinstance(item_path, str):
                item_path = item_path.split(".")
            else:
                item_path = copy.copy(item_path)
            attrib = item_path.pop(0)
            if hasattr(self, attrib):
                if len(item_path) == 0:
                    return self[attrib]
                else:
                    item = self[attrib]
                    if isinstance(item, type(self)):
                        return item.get_item(item_path, default=default)
                    else:
                        return default
            else:
                return default
        else:
            result = self._nested_get(item_path, wild=wild, return_path=return_path)
            if len(result) == 0:
                return default
            elif len(result) == 1:
                if return_path:
                    return result[0][1], result[0][0]
                else:
                    return result[0]
            else:
                if return_path:
                    return [i[1] for i in result], [i[0] for i in result]
                else:
                    return result

    def _nested_get_iter(self, item, wild=False):
        """Recursive function to search for an item key in a nested
        DictionaryTreeBrowser."""
        self.process_lazy_attributes()
        for key_, item_ in self.__dict__.items():
            if not isinstance(item_, types.MethodType) and not key_.startswith("_"):
                key = item_["key"]
                if isinstance(item_["_dtb_value_"], DictionaryTreeBrowser):
                    for result in item_["_dtb_value_"]._nested_get_iter(item, wild):
                        yield key + "." + result[0], result[1]
                else:
                    if key == item or (
                        wild and (str(item).lower() in str(key).lower())
                    ):
                        yield key, item_["_dtb_value_"]

    def _nested_get(self, item_path, wild=False, return_path=False):
        """Search for an item key in a nested DictionaryTreeBrowser and yield a
        list of values. If `wild` is `True`, looks for any key that contains
        the string `item` (case insensitive). If part of a path is given,
        search for matching items and then make sure that the full path is
        contained."""
        if "." in item_path:
            item = item_path.split(".").pop(-1)
        else:
            item = item_path
        result = list(self._nested_get_iter(item, wild))
        # remove item where item matches, but not additional elements of item_path
        if return_path:
            return [i for i in result if item_path in i[0]]
        else:
            return [i[1] for i in result if item_path in i[0]]

    def __getitem__(self, key):
        self.process_lazy_attributes()
        return self.__getattribute__(key)

    def __setitem__(self, key, value):
        self.__setattr__(key, value)

    def __getattr__(self, name):
        """__getattr__ is called when the default attribute access (
        __getattribute__) fails with an AttributeError.

        """
        # Skip the attribute we are not interested in. This is also necessary
        # to recursive loops.
        if name.startswith("__"):
            raise AttributeError(name)

        # Attribute name are been slugified, so we need to do the same for
        # the dictionary keys. Also check with `_sig_` prefix for signal attributes.
        keys = [slugify(k) for k in self._lazy_attributes.keys()]
        if name in keys or f"_sig_{name}" in keys:
            # It is a lazy attribute, we need to process the lazy attribute
            self.process_lazy_attributes()
            return self.__dict__[name]["_dtb_value_"]
        else:
            raise AttributeError(name)

    def __getattribute__(self, name):
        if isinstance(name, bytes):
            name = name.decode()
        name = slugify(name, valid_variable_name=True)
        item = super().__getattribute__(name)

        if isinstance(item, dict) and "_dtb_value_" in item and "key" in item:
            return item["_dtb_value_"]
        else:
            return item

    def __setattr__(self, key, value):
        self._setattr(key, value, keep_existing=False)

    def _setattr(self, key, value, keep_existing=False):
        if key in ["_double_lines", "_lazy_attributes"]:
            super().__setattr__(key, value)
            return

        if key.startswith("_sig_"):
            key = key[5:]

            value = BaseSignal(**value)

        slugified_key = str(slugify(key, valid_variable_name=True))
        if isinstance(value, dict):
            if slugified_key in self.__dict__.keys() and keep_existing:
                self.__dict__[slugified_key]["_dtb_value_"].add_dictionary(
                    value, double_lines=self._double_lines
                )
                return
            else:
                value = DictionaryTreeBrowser(
                    value, double_lines=self._double_lines, lazy=False
                )
        super().__setattr__(slugified_key, {"key": key, "_dtb_value_": value})

    def __contains__(self, item):
        return self.has_item(item_path=item)

    def as_dictionary(self):
        """Returns its dictionary representation."""

        if len(self._lazy_attributes) > 0:
            return copy.deepcopy(self._lazy_attributes)

        par_dict = {}

        # from hyperspy.axes import AxesManager, BaseDataAxis
        # from hyperspy.signal import BaseSignal

        for key_, item_ in self.__dict__.items():
            if not isinstance(item_, types.MethodType):
                if key_ in ["_db_index", "_double_lines", "_lazy_attributes"]:
                    continue
                key = item_["key"]
                if isinstance(item_["_dtb_value_"], DictionaryTreeBrowser):
                    item = item_["_dtb_value_"].as_dictionary()
                elif isinstance(item_["_dtb_value_"], BaseSignal):
                    item = item_["_dtb_value_"]._to_dictionary()
                    key = "_sig_" + key
                elif hasattr(item_["_dtb_value_"], "_to_dictionary"):
                    item = item_["_dtb_value_"]._to_dictionary()
                elif isinstance(item_["_dtb_value_"], AxesManager):
                    item = item_["_dtb_value_"]._get_axes_dicts()
                    key = "_hspy_AxesManager_" + key
                elif isinstance(item_["_dtb_value_"], BaseDataAxis):
                    item = item_["_dtb_value_"].get_axis_dictionary()
                    key = "_hspy_Axis_" + key
                elif type(item_["_dtb_value_"]) in (list, tuple):
                    signals = []
                    container = item_["_dtb_value_"]
                    # Support storing signals in containers
                    for i, item in enumerate(container):
                        if isinstance(item, BaseSignal):
                            signals.append(i)
                    if signals:
                        to_tuple = False
                        if type(container) is tuple:
                            container = list(container)
                            to_tuple = True
                        for i in signals:
                            container[i] = {"_sig_": container[i]._to_dictionary()}
                        if to_tuple:
                            container = tuple(container)
                    item = container
                else:
                    item = item_["_dtb_value_"]
                par_dict.update({key: item})
        return par_dict


_slugify_strip_re_data = "".join(
    c for c in map(chr, np.delete(np.arange(256), [95, 32])) if not c.isalnum()
).encode()

def slugify(value, valid_variable_name=False):
    """
    Normalizes string, converts to lowercase, removes non-alpha characters,
    and converts spaces to hyphens.

    Adapted from Django's "django/template/defaultfilters.py".

    """
    if not isinstance(value, str):
        try:
            # Convert to unicode using the default encoding
            value = str(value)
        except BaseException:
            # Try latin1. If this does not work an exception is raised.
            value = str(value, "latin1")
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore")
    value = value.translate(None, _slugify_strip_re_data).decode().strip()
    value = value.replace(" ", "_")
    if valid_variable_name and not value.isidentifier():
        value = "Number_" + value
    return value


