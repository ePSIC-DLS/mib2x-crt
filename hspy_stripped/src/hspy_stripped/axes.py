import copy
import inspect
from collections.abc import Iterable
from contextlib import contextmanager

import numpy as np
import traits.api as t


def isfloat(number):
    """Check if a number or array is of float type.

    This is necessary because e.g. isinstance(np.float32(2), float) is False.

    """
    if hasattr(number, "dtype"):
        return np.issubdtype(number, np.floating)
    else:
        return isinstance(number, float)

def isiterable(obj):
    return isinstance(obj, Iterable)

class TupleSA(tuple):
    """A tuple that can set the attributes of its items"""

    def __getitem__(self, *args, **kwargs):
        item = super().__getitem__(*args, **kwargs)
        try:
            return type(self)(item)
        except TypeError:
            # When indexing, the returned object is not a tuple
            return item

    def set(self, **kwargs):
        """Set the attributes of its items

        Parameters
        ----------
        kwargs : dict
            The name of the attributes and their values. If a value is iterable,
            then attribute of each item of the tuple will be set to each of the values.
        """
        for key, value in kwargs.items():
            no_name = [item for item in self if not hasattr(item, key)]
            if no_name:
                raise AttributeError(f"'The items {no_name} have not attribute '{key}'")
            else:
                if isiterable(value) and not isinstance(value, str):
                    for item, value_ in zip(self, value):
                        setattr(item, key, value_)
                else:
                    for item in self:
                        setattr(item, key, value)

    def get(self, *args):
        """Get the attributes of its items

        Parameters
        ----------
        args : list
            The names of the attributes to get.

        Returns
        -------
        output : dict
            The name of the attributes and their values.
        """
        output = dict()
        for key in args:
            values = list()
            for item in self:
                if not hasattr(item, key):
                    raise AttributeError(f"'The item {item} has not attribute '{key}'")
                else:
                    values.append(getattr(item, key))
            output[key] = tuple(values)
        return output

    def __add__(self, *args, **kwargs):
        return type(self)(super().__add__(*args, **kwargs))

    def __mul__(self, *args, **kwargs):
        return type(self)(super().__mul__(*args, **kwargs))

def ordinal(value):
    """
    Converts zero or a *postive* integer (or their string
    representations) to an ordinal value.

    >>> for i in range(1,13):
    ...     ordinal(i)
    ...
    '1st'
    '2nd'
    '3rd'
    '4th'
    '5th'
    '6th'
    '7th'
    '8th'
    '9th'
    '10th'
    '11th'
    '12th'

    >>> for i in (100, '111', '112',1011):
    ...     ordinal(i)
    ...
    '100th'
    '111th'
    '112th'
    '1011th'

    Notes
    -----
    Author:  Serdar Tumgoren
    https://code.activestate.com/recipes/576888-format-a-number-as-an-ordinal/
    MIT license
    """
    try:
        value = int(value)
    except ValueError:
        return value

    if value % 100 // 10 != 1:
        if value % 10 == 1:
            ordval = "%d%s" % (value, "st")
        elif value % 10 == 2:
            ordval = "%d%s" % (value, "nd")
        elif value % 10 == 3:
            ordval = "%d%s" % (value, "rd")
        else:
            ordval = "%d%s" % (value, "th")
    else:
        ordval = "%d%s" % (value, "th")

    return ordval


class ndindex_nat(np.ndindex):
    def __next__(self):
        return super().__next__()[::-1]


def create_axis(**kwargs):
    """Creates a uniform, a non-uniform axis or a functional axis depending on
    the kwargs provided. If `axis` or  `expression` are provided, a non-uniform
    or a functional axis is created, respectively. Otherwise a uniform axis is
    created, which can be defined by `scale`, `size` and `offset`.

    Alternatively, the offset_index of the offset channel can be specified.

    Parameters
    ----------
    axis : iterable of values (list, tuple or 1D numpy array) (optional)
    expression : Component function in SymPy text expression format (str) (optional)
    offset : float (optional)
    scale : float (optional)
    size : number of channels (optional)

    Returns
    -------
    A DataAxis, FunctionalDataAxis or a UniformDataAxis

    """
    if "axis" in kwargs.keys():  # non-uniform axis
        axis_class = DataAxis
    elif "expression" in kwargs.keys():  # Functional axis
        axis_class = FunctionalDataAxis
    else:  # if not argument is provided fall back to uniform axis
        axis_class = UniformDataAxis
    return axis_class(**kwargs)


class UnitConversion:
    """
    Parent class containing unit conversion functionalities of
    Uniform Axis.

    Parameters
    ----------
    offset : float
        The first value of the axis vector.
    scale : float
        The spacing between axis points.
    size : int
        The number of points in the axis.
    """

    def __init__(self, units=None, scale=1.0, offset=0.0):
        if units is None:
            units = t.Undefined
        self.units = units
        self.scale = scale
        self.offset = offset

    @property
    def units(self):
        return self._units

    @units.setter
    def units(self, s):
        if s == "":
            self._units = t.Undefined
        self._units = s


class BaseDataAxis(t.HasTraits):
    """Parent class defining common attributes for all DataAxis classes.

    Parameters
    ----------
    name : str, optional
        Name string by which the axis can be accessed. `<undefined>` if not set.
    units : str, optional
         String for the units of the axis vector. `<undefined>` if not set.
    navigate : bool, optional
        True for a navigation axis. Default False (signal axis).
    is_binned : bool, optional
        True if data along the axis is binned. Default False.
    """

    name = t.Str()
    units = t.Str()
    size = t.CInt()
    low_value = t.Float()
    high_value = t.Float()
    value = t.Range("low_value", "high_value")
    low_index = t.Int(0)
    high_index = t.Int()
    slice = t.Instance(slice)
    navigate = t.Bool(False)
    is_binned = t.Bool(False)
    index = t.Range("low_index", "high_index")
    axis = t.Array()

    def __init__(
        self,
        index_in_array=None,
        name=None,
        units=None,
        navigate=False,
        is_binned=False,
        **kwargs,
    ):
        super().__init__()
        if name is None:
            name = t.Undefined
        if units is None:
            units = t.Undefined

        if "_type" in kwargs:
            _type = kwargs.get("_type")
            if _type != self.__class__.__name__:
                raise ValueError(
                    f"The passed `_type` ({_type}) of axis is "
                    "inconsistent with the given attributes."
                )
        _name = self.__class__.__name__

        self._suppress_value_changed_trigger = False
        self._suppress_update_value = False
        self.name = name
        self.units = units
        self.low_index = 0
        self.on_trait_change(self._update_slice, "navigate")
        self.on_trait_change(self.update_index_bounds, "size")
        self.on_trait_change(self._update_bounds, "axis")

        self.index = 0
        self.navigate = navigate
        self.is_binned = is_binned
        self.axes_manager = None
        self._is_uniform = False

        # The slice must be updated even if the default value did not
        # change to correctly set its value.
        self._update_slice(self.navigate)

    @property
    def is_uniform(self):
        return self._is_uniform

    def _index_changed(self, name, old, new):
        pass

    def _value_changed(self, name, old, new):
        pass

    @property
    def index_in_array(self):
        if self.axes_manager is not None:
            return self.axes_manager._axes.index(self)
        else:
            raise AttributeError(
                "This {} does not belong to an AxesManager"
                " and therefore its index_in_array attribute "
                " is not defined".format(self.__class__.__name__)
            )

    @property
    def index_in_axes_manager(self):
        if self.axes_manager is not None:
            return self.axes_manager._get_axes_in_natural_order().index(self)
        else:
            raise AttributeError(
                "This {} does not belong to an AxesManager"
                " and therefore its index_in_array attribute "
                " is not defined".format(self.__class__.__name__)
            )

    def _get_positive_index(self, index):
        # To be used with re
        if index < 0:
            index = self.size + index
            if index < 0:
                raise IndexError("index out of bounds")
        return index

    def _get_index(self, value):
        if isfloat(value):
            return self.value2index(value)
        else:
            return value

    def _get_array_slices(self, slice_):
        """Returns a slice to slice the corresponding data axis.

        Parameters
        ----------
        slice_ : {float, int, slice}

        Returns
        -------
        my_slice : slice

        """
        if isinstance(slice_, slice):
            if not self.is_uniform and isfloat(slice_.step):
                raise ValueError("Float steps are only supported for uniform axes.")

        v2i = self.value2index

        if isinstance(slice_, slice):
            start = slice_.start
            stop = slice_.stop
            step = slice_.step
        else:
            if isfloat(slice_):
                start = v2i(slice_)
            else:
                start = self._get_positive_index(slice_)
            stop = start + 1
            step = None

        start = self._parse_value(start)
        stop = self._parse_value(stop)
        step = self._parse_value(step)

        if isfloat(step):
            step = int(round(step / self.scale))

        if isfloat(start):
            try:
                start = v2i(start)
            except ValueError:
                if start > self.high_value:
                    # The start value is above the axis limit
                    raise IndexError(
                        "Start value above axis high bound for  axis %s."
                        "value: %f high_bound: %f"
                        % (repr(self), start, self.high_value)
                    )
                else:
                    # The start value is below the axis limit,
                    # we slice from the start.
                    start = None
        if isfloat(stop):
            try:
                stop = v2i(stop)
            except ValueError:
                if stop < self.low_value:
                    # The stop value is below the axis limits
                    raise IndexError(
                        "Stop value below axis low bound for  axis %s."
                        "value: %f low_bound: %f" % (repr(self), stop, self.low_value)
                    )
                else:
                    # The stop value is below the axis limit,
                    # we slice until the end.
                    stop = None

        if step == 0:
            raise ValueError("slice step cannot be zero")

        return slice(start, stop, step)

    def _slice_me(self, slice_):
        raise NotImplementedError("This method must be implemented by subclasses")

    def _get_name(self):
        name = (
            self.name
            if self.name is not t.Undefined
            else ("Unnamed " + ordinal(self.index_in_axes_manager))
            if self.axes_manager is not None
            else "Unnamed"
        )
        return name

    def __repr__(self):
        text = "<%s axis, size: %i" % (
            self._get_name(),
            self.size,
        )
        if self.navigate is True:
            text += ", index: %i" % self.index
        text += ">"
        return text

    def __str__(self):
        return self._get_name() + " axis"

    def update_index_bounds(self):
        self.high_index = self.size - 1

    def _update_bounds(self):
        if len(self.axis) != 0:
            self.low_value, self.high_value = (self.axis.min(), self.axis.max())

    def _update_slice(self, value):
        if value is False:
            self.slice = slice(None)
        else:
            self.slice = None

    def get_axis_dictionary(self):
        return {
            "_type": self.__class__.__name__,
            "name": _parse_axis_attribute(self.name),
            "units": _parse_axis_attribute(self.units),
            "navigate": self.navigate,
            "is_binned": self.is_binned,
        }

    def copy(self):
        return self.__class__(**self.get_axis_dictionary())

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, memo):
        cp = self.copy()
        return cp

    def _parse_value_from_string(self, value):
        """Return calibrated value from a suitable string"""
        if len(value) == 0:
            raise ValueError("Cannot index with an empty string")
        # Starting with 'rel', it must be relative slicing
        elif value.startswith("rel"):
            try:
                relative_value = float(value[3:])
            except ValueError:
                raise ValueError("`rel` must be followed by a number in range [0, 1].")
            if relative_value < 0 or relative_value > 1:
                raise ValueError("Relative value must be in range [0, 1]")
            value = self.low_value + relative_value * (self.high_value - self.low_value)
        # if first character is a digit, try unit conversion
        # otherwise we don't support it
        elif value[0].isdigit():
            if self.is_uniform:
                value = self._get_value_from_value_with_units(value)
            else:
                raise ValueError(
                    "Unit conversion is only supported for " "uniform axis."
                )
        else:
            raise ValueError(f"`{value}` is not a suitable string for slicing.")

        return value

    def _parse_value(self, value):
        """Convert the input to calibrated value if string, otherwise,
        return the same value."""
        if isinstance(value, str):
            value = self._parse_value_from_string(value)
        elif isinstance(value, (list, tuple, np.ndarray)):
            value = np.asarray(value)
            if value.dtype.type is np.str_:
                value = np.array([self._parse_value_from_string(v) for v in value])
        return value

    def value2index(self, value, rounding=round):
        pass

    def index2value(self, index):
        pass

    def value_range_to_indices(self, v1, v2):
        """Convert the given range to index range.

        When a value is out of the axis limits, the endpoint is used instead.
        `v1` must be preceding `v2` in the axis values

          - if the axis scale is positive, it means v1 < v2
          - if the axis scale is negative, it means v1 > v2

        Parameters
        ----------
        v1, v2 : float
            The end points of the interval in the axis units.

        Returns
        -------
        i2, i2 : float
            The indices corresponding to the interval [v1, v2]
        """
        i1, i2 = 0, self.size - 1
        error_message = "Wrong order of the values: for axis with"
        if self._is_increasing_order:
            if v1 is not None and v2 is not None and v1 > v2:
                raise ValueError(
                    f"{error_message} increasing order, v2 ({v2}) "
                    f"must be greater than v1 ({v1})."
                )

            if v1 is not None and self.low_value < v1 <= self.high_value:
                i1 = self.value2index(v1)
            if v2 is not None and self.high_value > v2 >= self.low_value:
                i2 = self.value2index(v2)
        else:
            if v1 is not None and v2 is not None and v1 < v2:
                raise ValueError(
                    f"{error_message} decreasing order: v1 ({v1}) "
                    f"must be greater than v2 ({v2})."
                )

            if v1 is not None and self.high_value > v1 >= self.low_value:
                i1 = self.value2index(v1)
            if v2 is not None and self.low_value < v2 <= self.high_value:
                i2 = self.value2index(v2)
        return i1, i2

    def update_from(self, axis, attributes):
        """Copy values of specified axes fields from the passed AxesManager.

        Parameters
        ----------
        axis : :class:`~hyperspy.axes.BaseDataAxis`
            The instance to use as a source for values.
        attributes : iterable of str
            The name of the attribute to update. If the attribute does not
            exist in either of the AxesManagers, an AttributeError will be
            raised.

        Returns
        -------
        bool
            True if any changes were made, otherwise False.

        """
        any_changes = False
        changed = {}
        for f in attributes:
            a, b = getattr(self, f), getattr(axis, f)
            cond = np.allclose(a, b) if isinstance(a, np.ndarray) else a == b
            if not cond:
                changed[f] = getattr(axis, f)
        if len(changed) > 0:
            self.trait_set(**changed)
            any_changes = True
        return any_changes

    def convert_to_uniform_axis(self, keep_bounds=True, log_scale_error=True):
        """
        Convert to an uniform axis.

        Parameters
        ----------
        keep_bounds : bool
            If ``True``, the first and last value of the axis will not be changed.
            The new scale is calculated by substracting the last value by the first
            value and dividing by the number of intervals.
            If ``False``, the scale and offset are calculated using
            :meth:`numpy.polynomial.polynomial.Polynomial.fit`, which minimises
            the scale difference over the whole axis range but the bounds of
            the axis can change (in some cases quite significantly, in particular when the
            interval width is changing continuously). Default is ``True``.
        log_scale_error : bool
            If ``True``, the maximum scale error will be logged as INFO.
            Default is ``True``.

        Examples
        --------
        Using ``keep_bounds=True`` (default):

        >>> s = hs.data.luminescence_signal(uniform=False)
        >>> print(s.axes_manager)
        <Axes manager, axes: (|1024)>
                    Name |   size |  index |  offset |   scale |  units
        ================ | ====== | ====== | ======= | ======= | ======
        ---------------- | ------ | ------ | ------- | ------- | ------
                  Energy |   1024 |      0 | non-uniform axis |     eV
        >>> s.axes_manager[-1].convert_to_uniform_axis(keep_bounds=True)
        >>> print(s.axes_manager)
        <Axes manager, axes: (|1024)>
                    Name |   size |  index |  offset |   scale |  units
        ================ | ====== | ====== | ======= | ======= | ======
        ---------------- | ------ | ------ | ------- | ------- | ------
                  Energy |   1024 |      0 |     1.6 |  0.0039 |     eV

        Using ``keep_bounds=False``:

        >>> s = hs.data.luminescence_signal(uniform=False)
        >>> print(s.axes_manager)
        <Axes manager, axes: (|1024)>
                    Name |   size |  index |  offset |   scale |  units
        ================ | ====== | ====== | ======= | ======= | ======
        ---------------- | ------ | ------ | ------- | ------- | ------
                  Energy |   1024 |      0 | non-uniform axis |     eV
        >>> s.axes_manager[-1].convert_to_uniform_axis(keep_bounds=False)
        >>> print(s.axes_manager)
        <Axes manager, axes: (|1024)>
                    Name |   size |  index |  offset |   scale |  units
        ================ | ====== | ====== | ======= | ======= | ======
        ---------------- | ------ | ------ | ------- | ------- | ------
                  Energy |   1024 |      0 |     1.1 |  0.0033 |     eV


        See Also
        --------
        hyperspy.api.signals.BaseSignal.interpolate_on_axis

        Notes
        -----
        The function only converts the axis type and doesn't interpolate
        the data itself - see :meth:`~.api.signals.BaseSignal.interpolate_on_axis`
        to interpolate data on a uniform axis.

        """
        indices = np.arange(self.size)
        if keep_bounds:
            scale = (self.axis[-1] - self.axis[0]) / (self.size - 1)
            offset = self.axis[0]
        else:
            # polyfit minimize the error over the whole axis
            offset, scale = np.polynomial.Polynomial.fit(
                indices, self.axis, deg=1
            ).convert()
        d = self.get_axis_dictionary()
        axes_manager = self.axes_manager
        if "axis" in d:
            del d["axis"]
        if len(self.axis) > 1 and log_scale_error:
            scale_err = np.max(self.axis - (scale * indices + offset))
        d["_type"] = "UniformDataAxis"
        d["size"] = self.size
        self.__class__ = UniformDataAxis
        self.__init__(**d, scale=scale, offset=offset)
        self.axes_manager = axes_manager

    @property
    def _is_increasing_order(self):
        """
        Determine if the axis has an increasing, decreasing order or no order
        at all.

        Returns
        -------
        True if order is increasing, False if order is decreasing, None
        otherwise.

        """
        steps = self.axis[1:] - self.axis[:-1]
        if np.all(steps > 0):
            return True
        elif np.all(steps < 0):
            return False
        else:
            # the axis is not ordered
            return None


class DataAxis(BaseDataAxis):
    pass


class FunctionalDataAxis(BaseDataAxis):
    pass


class UniformDataAxis(BaseDataAxis, UnitConversion):
    """DataAxis class for a uniform axis defined through a ``scale``, an
    ``offset`` and a ``size``.

    The most common type of axis. It is defined by the ``offset``, ``scale``
    and ``size`` parameters, which determine the `initial value`, `spacing` and
    `length` of the axis, respectively. The actual ``axis`` array is
    automatically calculated from these three values. The ``UniformDataAxis``
    is a special case of the ``FunctionalDataAxis`` defined by the function
    ``scale * x + offset``.

    Parameters
    ----------
    offset : float
        The first value of the axis vector.
    scale : float
        The spacing between axis points.
    size : int
        The number of points in the axis.

    Examples
    --------
    Sample dictionary for a `UniformDataAxis`:

    >>> dict0 = {'offset': 300, 'scale': 1, 'size': 500}
    >>> s = hs.signals.Signal1D(np.ones(500), axes=[dict0])
    >>> s.axes_manager[0].get_axis_dictionary() # doctest: +SKIP
    {'_type': 'UniformDataAxis',
     'name': <undefined>,
     'units': <undefined>,
     'navigate': False,
     'size': 500,
     'scale': 1.0,
     'offset': 300.0}
    """

    def __init__(
        self,
        index_in_array=None,
        name=None,
        units=None,
        navigate=False,
        size=1,
        scale=1.0,
        offset=0.0,
        is_binned=False,
        **kwargs,
    ):
        super().__init__(
            index_in_array=index_in_array,
            name=name,
            units=units,
            navigate=navigate,
            is_binned=is_binned,
            **kwargs,
        )
        # These traits need to added dynamically to be removed when necessary
        self.add_trait("scale", t.CFloat)
        self.add_trait("offset", t.CFloat)
        self.scale = scale
        self.offset = offset
        self.size = size
        self.update_axis()
        self._is_uniform = True
        self.on_trait_change(self.update_axis, ["scale", "offset", "size"])

    def _slice_me(self, _slice):
        """Returns a slice to slice the corresponding data axis and
        change the offset and scale of the UniformDataAxis accordingly.

        Parameters
        ----------
        _slice : {float, int, slice}

        Returns
        -------
        my_slice : slice
        """
        my_slice = self._get_array_slices(_slice)
        start, step = my_slice.start, my_slice.step

        if start is None:
            if step is None or step > 0:
                start = 0
            else:
                start = self.size - 1
        self.offset = self.index2value(start)
        if step is not None:
            self.scale *= step
        self.size = len(self.axis[my_slice])
        return my_slice

    def get_axis_dictionary(self):
        d = super().get_axis_dictionary()
        d.update({"size": self.size, "scale": self.scale, "offset": self.offset})
        return d

    def value2index(self, value, rounding=round):
        pass

    def update_axis(self):
        self.axis = self.offset + self.scale * np.arange(self.size)

    def calibrate(self, value_tuple, index_tuple, modify_calibration=True):
        scale = (value_tuple[1] - value_tuple[0]) / (index_tuple[1] - index_tuple[0])
        offset = value_tuple[0] - scale * index_tuple[0]
        if modify_calibration is True:
            self.offset = offset
            self.scale = scale
        else:
            return offset, scale

    def update_from(self, axis, attributes=None):
        """Copy values of specified axes fields from the passed AxesManager.

        Parameters
        ----------
        axis : :class:`~hyperspy.axes.UniformDataAxis`
            The UniformDataAxis instance to use as a source for values.
        attributes : iterable of str or None
            The name of the attribute to update. If the attribute does not
            exist in either of the AxesManagers, an AttributeError will be
            raised. If `None`, `scale`, `offset` and `units` are updated.
        Returns
        -------
        A boolean indicating whether any changes were made.

        """
        if attributes is None:
            attributes = ["scale", "offset", "size"]
        return super().update_from(axis, attributes)

    def crop(self, start=None, end=None):
        """Crop the axis in place.
        Parameters
        ----------
        start : int, float, or None
            The beginning of the cropping interval. If type is ``int``,
            the value is taken as the axis index. If type is ``float`` the index
            is calculated using the axis calibration. If `start`/`end` is
            ``None`` the method crops from/to the low/high end of the axis.
        end : int, float, or None
            The end of the cropping interval. If type is ``int``,
            the value is taken as the axis index. If type is ``float`` the index
            is calculated using the axis calibration. If `start`/`end` is
            ``None`` the method crops from/to the low/high end of the axis.
        """
        if start is None:
            start = 0
        if end is None:
            end = self.size
        # Use `_get_positive_index` to support reserved indexing
        i1 = self._get_positive_index(self._get_index(start))
        i2 = self._get_positive_index(self._get_index(end))

        self.offset = self.index2value(i1)
        self.size = i2 - i1
        self.update_axis()


    @property
    def scale_as_quantity(self):
        return self._get_quantity("scale")

    @scale_as_quantity.setter
    def scale_as_quantity(self, value):
        self._set_quantity(value, "scale")

    @property
    def offset_as_quantity(self):
        return self._get_quantity("offset")

    @offset_as_quantity.setter
    def offset_as_quantity(self, value):
        self._set_quantity(value, "offset")

    def convert_to_functional_data_axis(
        self, expression, units=None, name=None, **kwargs
    ):
        d = super().get_axis_dictionary()
        axes_manager = self.axes_manager
        if units:
            d["units"] = units
        if name:
            d["name"] = name
        d.update(kwargs)
        this_kwargs = self.get_axis_dictionary()
        self.remove_trait("scale")
        self.remove_trait("offset")
        self.__class__ = FunctionalDataAxis
        d["_type"] = "FunctionalDataAxis"
        self.__init__(expression=expression, x=UniformDataAxis(**this_kwargs), **d)
        self.axes_manager = axes_manager

    def convert_to_non_uniform_axis(self):
        d = super().get_axis_dictionary()
        axes_manager = self.axes_manager
        self.__class__ = DataAxis
        d["_type"] = "DataAxis"
        self.remove_trait("scale")
        self.remove_trait("offset")
        self.__init__(**d, axis=self.axis)
        self.axes_manager = axes_manager


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


class AxesManager(t.HasTraits):
    """Contains and manages the data axes.

    It supports indexing, slicing, subscripting and iteration. As an iterator,
    iterate over the navigation coordinates returning the current indices.
    It can only be indexed and sliced to access the DataAxis objects that it
    contains. Standard indexing and slicing follows the "natural order" as in
    Signal, i.e. [nX, nY, ...,sX, sY,...] where `n` indicates a navigation axis
    and `s` a signal axis. In addition, AxesManager supports indexing using
    complex numbers a + bj, where b can be one of 0, 1, 2 and 3 and a valid
    index. If b is 3, AxesManager is indexed using the order of the axes in the
    array. If b is 1(2), indexes only the navigation(signal) axes in the
    natural order. In addition AxesManager supports subscription using
    axis name.

    Attributes
    ----------
    signal_axes, navigation_axes : list
        Contain the corresponding DataAxis objects

    coordinates, indices, iterpath

    Examples
    --------

    Create a spectrum with random data

    >>> s = hs.signals.Signal1D(np.random.random((2,3,4,5)))
    >>> s.axes_manager
    <Axes manager, axes: (4, 3, 2|5)>
                Name |   size |  index |  offset |   scale |  units
    ================ | ====== | ====== | ======= | ======= | ======
         <undefined> |      4 |      0 |       0 |       1 | <undefined>
         <undefined> |      3 |      0 |       0 |       1 | <undefined>
         <undefined> |      2 |      0 |       0 |       1 | <undefined>
    ---------------- | ------ | ------ | ------- | ------- | ------
         <undefined> |      5 |      0 |       0 |       1 | <undefined>
    >>> s.axes_manager[0]
    <Unnamed 0th axis, size: 4, index: 0>
    >>> s.axes_manager[3j]
    <Unnamed 2nd axis, size: 2, index: 0>
    >>> s.axes_manager[1j]
    <Unnamed 0th axis, size: 4, index: 0>
    >>> s.axes_manager[2j]
    <Unnamed 3rd axis, size: 5>
    >>> s.axes_manager[1].name = "y"
    >>> s.axes_manager["y"]
    <y axis, size: 3, index: 0>
    >>> for i in s.axes_manager:
    ...     print(i, s.axes_manager.indices)
    (0, 0, 0) (0, 0, 0)
    (1, 0, 0) (1, 0, 0)
    (2, 0, 0) (2, 0, 0)
    (3, 0, 0) (3, 0, 0)
    (3, 1, 0) (3, 1, 0)
    (2, 1, 0) (2, 1, 0)
    (1, 1, 0) (1, 1, 0)
    (0, 1, 0) (0, 1, 0)
    (0, 2, 0) (0, 2, 0)
    (1, 2, 0) (1, 2, 0)
    (2, 2, 0) (2, 2, 0)
    (3, 2, 0) (3, 2, 0)
    (3, 2, 1) (3, 2, 1)
    (2, 2, 1) (2, 2, 1)
    (1, 2, 1) (1, 2, 1)
    (0, 2, 1) (0, 2, 1)
    (0, 1, 1) (0, 1, 1)
    (1, 1, 1) (1, 1, 1)
    (2, 1, 1) (2, 1, 1)
    (3, 1, 1) (3, 1, 1)
    (3, 0, 1) (3, 0, 1)
    (2, 0, 1) (2, 0, 1)
    (1, 0, 1) (1, 0, 1)
    (0, 0, 1) (0, 0, 1)

    """

    _axes = t.List(BaseDataAxis)
    _step = t.Int(1)

    def __init__(self, axes_list):
        super().__init__()

        # Remove all axis for cases, we reinitiliase the AxesManager
        if self._axes:
            self.remove(self._axes)
        self.create_axes(axes_list)

        self._update_attributes()
        self._update_trait_handlers()
        self.iterpath = "serpentine"
        self._ragged = False

    @property
    def ragged(self):
        return self._ragged

    def _update_trait_handlers(self, remove=False):
        things = {
            self._on_index_changed: "_axes.index",
            self._on_slice_changed: "_axes.slice",
            self._on_size_changed: "_axes.size",
            self._on_scale_changed: "_axes.scale",
            self._on_offset_changed: "_axes.offset",
        }

        for k, v in things.items():
            self.on_trait_change(k, name=v, remove=remove)

    def _get_positive_index(self, axis):
        if axis < 0:
            axis += len(self._axes)
            if axis < 0:
                raise IndexError("index out of bounds")
        return axis

    def _array_indices_generator(self):
        shape = (
            self.navigation_shape[::-1]
            if self.navigation_size > 0
            else [
                1,
            ]
        )
        return np.ndindex(*shape)

    def _am_indices_generator(self):
        shape = (
            self.navigation_shape
            if self.navigation_size > 0
            else [
                1,
            ]
        )[::-1]
        return ndindex_nat(*shape)

    def __getitem__(self, y):
        """x.__getitem__(y) <==> x[y]"""
        if isinstance(y, str) or not np.iterable(y):
            if y == "nav":
                axes = self.navigation_axes
            elif y == "sig":
                axes = self.signal_axes
            else:
                return self[(y,)][0]
        else:
            axes = [self._axes_getter(ax) for ax in y]
        _, indices = np.unique([_id for _id in map(id, axes)], return_index=True)
        ans = tuple(axes[i] for i in sorted(indices))
        return ans

    def _axes_getter(self, y):
        if isinstance(y, BaseDataAxis):
            if y in self._axes:
                return y
            else:
                raise ValueError(f"{y} is not in {self}")
        if isinstance(y, str):
            axes = list(self._get_axes_in_natural_order())
            while axes:
                axis = axes.pop()
                if y == axis.name:
                    return axis
            raise ValueError("There is no DataAxis named %s" % y)
        elif (
            isfloat(y.real)
            and not y.real.is_integer()
            or isfloat(y.imag)
            and not y.imag.is_integer()
        ):
            raise TypeError(
                "axesmanager indices must be integers, " "complex integers or strings"
            )
        if y.imag == 0:  # Natural order
            return self._get_axes_in_natural_order()[y]
        elif y.imag == 3:  # Array order
            # Array order
            return self._axes[int(y.real)]
        elif y.imag == 1:  # Navigation natural order
            #
            return self.navigation_axes[int(y.real)]
        elif y.imag == 2:  # Signal natural order
            return self.signal_axes[int(y.real)]
        else:
            raise IndexError(
                "axesmanager imaginary part of complex indices " "must be 0, 1, 2 or 3"
            )

    def __getslice__(self, i=None, j=None):
        """x.__getslice__(i, j) <==> x[i:j]"""
        return self._get_axes_in_natural_order()[i:j]

    def _get_axes_in_natural_order(self):
        return self.navigation_axes + self.signal_axes

    @property
    def _navigation_shape_in_array(self):
        return self.navigation_shape[::-1]

    @property
    def _signal_shape_in_array(self):
        return self.signal_shape[::-1]

    @property
    def shape(self):
        nav_shape = self.navigation_shape if self.navigation_shape != (0,) else tuple()
        sig_shape = self.signal_shape if self.signal_shape != (0,) else tuple()
        return nav_shape + sig_shape

    @property
    def signal_extent(self):
        """The low and high values of the signal axes."""
        signal_extent = []
        for signal_axis in self.signal_axes:
            signal_extent.append(signal_axis.low_value)
            signal_extent.append(signal_axis.high_value)
        return tuple(signal_extent)

    @property
    def navigation_extent(self):
        """The low and high values of the navigation axes."""
        navigation_extent = []
        for navigation_axis in self.navigation_axes:
            navigation_extent.append(navigation_axis.low_value)
            navigation_extent.append(navigation_axis.high_value)
        return tuple(navigation_extent)

    @property
    def all_uniform(self):
        if any([axis.is_uniform is False for axis in self._axes]):
            return False
        else:
            return True

    def remove(self, axes):
        """Remove one or more axes"""
        axes = self[axes]
        if not np.iterable(axes):
            axes = (axes,)
        for ax in axes:
            self._remove_one_axis(ax)

    def _remove_one_axis(self, axis):
        """Remove the given Axis.

        Raises
        ------
        ValueError
            If the Axis is not present.

        """
        axis = self._axes_getter(axis)
        axis.axes_manager = None
        self._axes.remove(axis)

    def __delitem__(self, i):
        self.remove(self[i])

    def _get_data_slice(self, fill=None):
        """Return a tuple of slice objects to slice the data.

        Parameters
        ----------
        fill: None or iterable of (int, slice)
            If not None, fill the tuple of index int with the given
            slice.

        """
        cslice = [
            slice(None),
        ] * len(self._axes)
        if fill is not None:
            for index, slice_ in fill:
                cslice[index] = slice_
        return tuple(cslice)

    def create_axes(self, axes_list):
        """Given a list of either axes dictionaries, these are
        added to the AxesManager. In case dictionaries defining the axes
        properties are passed, the
        :class:`~hyperspy.axes.DataAxis`,
        :class:`~hyperspy.axes.UniformDataAxis`,
        :class:`~hyperspy.axes.FunctionalDataAxis` instances are first
        created.

        The index of the axis in the array and in the `_axes` lists
        can be defined by the index_in_array keyword if given
        for all axes. Otherwise, it is defined by their index in the
        list.

        Parameters
        ----------
        axes_list : list of dict
            The list of axes to create.

        """
        # Reorder axes_list using index_in_array if it is defined
        # for all axes and the indices are not repeated.
        indices = set(
            [
                axis["index_in_array"]
                for axis in axes_list
                if hasattr(axis, "index_in_array")
            ]
        )
        if len(indices) == len(axes_list):
            axes_list.sort(key=lambda x: x["index_in_array"])
        for axis_dict in axes_list:
            if isinstance(axis_dict, dict):
                self._append_axis(**axis_dict)
            else:
                self._axes.append(axis_dict)

    def set_axis(self, axis, index_in_axes_manager):
        """Replace an axis of current signal with one given in argument.

        Parameters
        ----------
        axis : :class:`~hyperspy.axes.BaseDataAxis`
            The axis to replace the current axis with.
        index_in_axes_manager : int
            The index of the axis in current signal to replace
            with the axis passed in argument.

        """
        self._axes[index_in_axes_manager] = axis

    def _update_max_index(self):
        self._max_index = 1
        for i in self.navigation_shape:
            self._max_index *= i
        if self._max_index != 0:
            self._max_index -= 1

    @property
    def iterpath(self):
        """Sets the order of iterating through the indices in the navigation
        dimension. Can be either "flyback" or "serpentine", or an iterable
        of navigation indices.
        """
        return self._iterpath

    @iterpath.setter
    def iterpath(self, path):
        if isinstance(path, str):
            if path == "serpentine":
                self._iterpath = "serpentine"
                self._iterpath_generator = _serpentine_iter(self.navigation_shape)
            elif path == "flyback":
                self._iterpath = "flyback"
                self._iterpath_generator = _flyback_iter(self.navigation_shape)
            else:
                raise ValueError(
                    f'The iterpath scan pattern is set to `"{path}"`. '
                    'It must be either "serpentine" or "flyback", or an iterable '
                    "of navigation indices, and is set either as multifit "
                    "`iterpath` argument or `axes_manager.iterpath`"
                )
        else:
            # Passing a custom indices iterator
            try:
                iter(path)  # If this fails, its not an iterable and we raise TypeError
            except TypeError as e:
                raise TypeError(
                    f"The iterpath `{path}` is not an iterable. "
                    "Ensure it is an iterable like a list, array or generator."
                ) from e
            try:
                if not (inspect.isgenerator(path) or type(path) is GeneratorLen):
                    # If iterpath is a generator, then we can't check its first value, have to trust it
                    first_indices = path[0]
                    if not isinstance(first_indices, Iterable):
                        raise TypeError
                    assert len(first_indices) == self.navigation_dimension
            except TypeError as e:
                raise TypeError(
                    f"Each set of indices in the iterpath should be an iterable, e.g. `(0,)` or `(0,0,0)`. "
                    f"The first entry currently looks like: `{first_indices}`, and does not satisfy this requirement."
                ) from e
            except AssertionError as e:
                raise ValueError(
                    f"The current iterpath yields indices of length "
                    f"{len(path)}. It should deliver incides with length "
                    f"equal to the navigation dimension, which is {self.navigation_dimension}."
                ) from e
            else:
                self._iterpath = path
                self._iterpath_generator = iter(self._iterpath)

    def _get_iterpath_size(self, masked_elements=0):
        "Attempts to get the iterpath size, returning None if it is unknown"
        if isinstance(self.iterpath, str):
            # flyback and serpentine have well-defined lengths <- navigation_size
            maxval = self.navigation_size - masked_elements
        else:
            try:
                maxval = len(self.iterpath)
                if masked_elements:
                    # Checking if mask indices exist in the iterpath could take a long time,
                    # or may not be possible in the case of a generator.
                    pass
            except TypeError:
                # progressbar is shown, so user can monitor "iterations per second"
                # but the length of the bar is unknown
                maxval = None
        return maxval

    def __next__(self):
        """
        Standard iterator method, returns the current coordinates

        Returns
        -------
        self.indices : tuple of ints
            Returns a tuple containing the coordinates of the current
            iteration.

        """
        self.indices = next(self._iterpath_generator)
        return self.indices

    def __iter__(self):
        # re-initialize iterpath as it is set before correct data shape
        # is created before data shape is known
        self.iterpath = self._iterpath
        return self

    @contextmanager
    def switch_iterpath(self, iterpath=None):
        """
        Context manager to change iterpath. The original iterpath is restored
        when exiting the context.

        Parameters
        ----------
        iterpath : str, optional
            The iterpath to use. The default is None.

        Yields
        ------
        None.

        Examples
        --------
        >>> s = hs.signals.Signal1D(np.arange(2*3*4).reshape([3, 2, 4]))
        >>> with s.axes_manager.switch_iterpath('serpentine'):
        ...     for indices in s.axes_manager:
        ...         print(indices)
        (0, 0)
        (1, 0)
        (1, 1)
        (0, 1)
        (0, 2)
        (1, 2)
        """
        if iterpath is not None:
            original_iterpath = self._iterpath
            self._iterpath = iterpath
        try:
            yield
        finally:
            # if an error is raised when using this context manager, we
            # reset the original value of _iterpath
            if iterpath is not None:
                self.iterpath = original_iterpath

    def _append_axis(self, **kwargs):
        axis = create_axis(**kwargs)
        axis.axes_manager = self
        self._axes.append(axis)

    def _on_index_changed(self):
        pass

    def _on_slice_changed(self):
        pass

    def _on_size_changed(self):
        pass

    def _on_scale_changed(self):
        pass

    def _on_offset_changed(self):
        pass

    def convert_units(self, axes=None, units=None, same_units=True, factor=0.25):
        """Convert the scale and the units of the selected axes. If the unit
        of measure is not supported by the pint library, the scale and units
        are not changed.

        Parameters
        ----------
        axes : int, str, iterable of :class:`~hyperspy.axes.DataAxis` or None, default None
            Convert to a convenient scale and units on the specified axis.
            If int, the axis can be specified using the index of the
            axis in ``axes_manager``.
            If string, argument can be ``"navigation"`` or ``"signal"`` to
            select the navigation or signal axes. The axis name can also be
            provided. If ``None``, convert all axes.
        units : list of str, str or None, default None
            If list, the selected axes will be converted to the provided units.
            If string, the navigation or signal axes will be converted to the
            provided units.
            If ``None``, the scale and the units are converted to the appropriate
            scale and units to avoid displaying scalebar with >3 digits or too
            small number. This can be tweaked by the ``factor`` argument.
        same_units : bool
            If ``True``, force to keep the same units if the units of
            the axes differs. It only applies for the same kind of axis,
            ``"navigation"`` or ``"signal"``. By default the converted unit
            of the first axis is used for all axes. If ``False``, convert all
            axes individually.
        %s

        Notes
        -----
        Requires a uniform axis.
        """
        convert_navigation = convert_signal = True

        if axes is None:
            axes = self.navigation_axes + self.signal_axes
            convert_navigation = len(self.navigation_axes) > 0
        elif axes == "navigation":
            axes = self.navigation_axes
            convert_signal = False
            convert_navigation = len(self.navigation_axes) > 0
        elif axes == "signal":
            axes = self.signal_axes
            convert_navigation = False
        elif isinstance(axes, (UniformDataAxis, int, str)):
            if not isinstance(axes, UniformDataAxis):
                axes = self[axes]
            axes = (axes,)
            convert_navigation = axes[0].navigate
            convert_signal = not convert_navigation
        else:
            raise TypeError("Axes type `{}` is not correct.".format(type(axes)))

        for axis in axes:
            if not axis.is_uniform:
                raise NotImplementedError(
                    "This operation is not implemented for non-uniform axes "
                    f"such as {axis}"
                )

        if isinstance(units, str) or units is None:
            units = [units] * len(axes)
        elif isinstance(units, list):
            if len(units) != len(axes):
                raise ValueError(
                    "Length of the provided units list {} should "
                    "be the same than the length of the provided "
                    "axes {}.".format(units, axes)
                )
        else:
            raise TypeError(
                "Units type `{}` is not correct. It can be a "
                "`string`, a `list` of string or `None`."
                "".format(type(units))
            )

        if same_units:
            if convert_navigation:
                units_nav = units[: self.navigation_dimension]
                self._convert_axes_to_same_units(
                    self.navigation_axes, units_nav, factor
                )
            if convert_signal:
                offset = self.navigation_dimension if convert_navigation else 0
                units_sig = units[offset:]
                self._convert_axes_to_same_units(self.signal_axes, units_sig, factor)
        else:
            for axis, unit in zip(axes, units):
                axis.convert_to_units(unit, factor=factor)


    def _convert_axes_to_same_units(self, axes, units, factor=0.25):
        # Check if the units are supported
        for axis in axes:
            if axis._ignore_conversion(axis.units):
                return

        # Set the same units for all axes, use the unit of the first axis
        # as reference
        axes[0].convert_to_units(units[0], factor=factor)
        unit = axes[0].units  # after conversion, in case units[0] was None.
        for axis in axes[1:]:
            # Convert only the units have the same dimensionality
            if _ureg(axis.units).dimensionality == _ureg(unit).dimensionality:
                axis.convert_to_units(unit, factor=factor)

    def update_axes_attributes_from(self, axes, attributes=None):
        pass


    def _update_attributes(self):
        getitem_tuple = []
        values = []
        signal_axes = ()
        navigation_axes = ()
        for axis in self._axes:
            # Until we find a better place, take property of the axes
            # here to avoid difficult to debug bugs.
            axis.axes_manager = self
            if axis.slice is None:
                getitem_tuple += (axis.index,)
                values.append(axis.value)
                navigation_axes += (axis,)
            else:
                getitem_tuple += (axis.slice,)
                signal_axes += (axis,)
        if not signal_axes and navigation_axes:
            getitem_tuple[-1] = slice(axis.index, axis.index + 1)

        self._signal_axes = signal_axes[::-1]
        self._navigation_axes = navigation_axes[::-1]
        self._getitem_tuple = tuple(getitem_tuple)

        if len(self.signal_axes) == 1 and self.signal_axes[0].size == 1:
            self._signal_dimension = 0
        else:
            self._signal_dimension = len(self.signal_axes)
        self._navigation_dimension = len(self.navigation_axes)

        self._signal_size = np.prod(self.signal_shape) if self.signal_shape else 0
        self._navigation_size = (
            np.prod(self.navigation_shape) if self.navigation_shape else 0
        )

        self._update_max_index()

    @property
    def signal_axes(self):
        """The signal axes as a TupleSA.

        A TupleSA object is a tuple with a `set` method
        to easily set the attributes of its items.
        """
        return TupleSA(self._signal_axes)

    @property
    def navigation_axes(self):
        """The navigation axes as a TupleSA.

        A TupleSA object is a tuple with a `set` method
        to easily set the attributes of its items.
        """
        return TupleSA(self._navigation_axes)

    @property
    def signal_shape(self):
        """The shape of the signal space."""
        return tuple([axis.size for axis in self._signal_axes])

    @property
    def navigation_shape(self):
        """The shape of the navigation space."""
        if self.navigation_dimension != 0:
            return tuple([axis.size for axis in self._navigation_axes])
        else:
            return ()

    @property
    def signal_size(self):
        """The size of the signal space."""
        return self._signal_size

    @property
    def navigation_size(self):
        """The size of the navigation space."""
        return self._navigation_size

    @property
    def navigation_dimension(self):
        """The dimension of the navigation space."""
        return self._navigation_dimension

    @property
    def signal_dimension(self):
        """The dimension of the signal space."""
        return self._signal_dimension

    def _set_signal_dimension(self, value):
        if len(self._axes) == 0 or self._signal_dimension == value:
            # Nothing to be done
            return
        elif self.ragged and value > 0:
            raise ValueError(
                "Signal containing ragged array " "must have zero signal dimension."
            )
        elif value > len(self._axes):
            raise ValueError(
                "The signal dimension cannot be greater "
                f"than the number of axes which is {len(self._axes)}"
            )
        elif value < 0:
            raise ValueError("The signal dimension must be a positive integer")

        # Figure out which axis needs navigate=True
        tl = [True] * len(self._axes)
        if value != 0:
            tl[-value:] = (False,) * value
        for axis in self._axes:
            # Changing navigate attribute will update the axis._slice
            # which in turn will trigger _on_slice_changed and call
            # _update_attribute
            axis.navigate = tl.pop(0)

    def key_navigator(self, event):
        """Set hotkeys for controlling the indices of the navigator plot"""
        pass

    def copy(self):
        return copy.copy(self)

    def deepcopy(self):
        return copy.deepcopy(self)

    def __deepcopy__(self, *args):
        return AxesManager(self._get_axes_dicts())

    def _get_axes_dicts(self, axes=None):
        if axes is None:
            axes = self._axes
        axes_dicts = []
        for axis in axes:
            axes_dicts.append(axis.get_axis_dictionary())
        return axes_dicts

    def as_dictionary(self):
        am_dict = {}
        for i, axis in enumerate(self._axes):
            am_dict["axis-%i" % i] = axis.get_axis_dictionary()
        return am_dict

    def _get_signal_axes_dicts(self):
        return [axis.get_axis_dictionary() for axis in self.signal_axes[::-1]]

    def _get_navigation_axes_dicts(self):
        return [axis.get_axis_dictionary() for axis in self.navigation_axes[::-1]]

    def _get_dimension_str(self):
        string = "("
        for axis in self.navigation_axes:
            string += str(axis.size) + ", "
        string = string.rstrip(", ")
        string += "|"
        for axis in self.signal_axes:
            string += str(axis.size) + ", "
        string = string.rstrip(", ")
        if self.ragged:
            string += "ragged"
        string += ")"
        return string

    def __repr__(self):
        text = "<Axes manager, axes: %s>\n" % self._get_dimension_str()
        ax_signature_uniform = "% 16s | %6g | %6s | %7.2g | %7.2g | %6s "
        ax_signature_non_uniform = "% 16s | %6g | %6s | non-uniform axis | %6s "
        signature = "% 16s | %6s | %6s | %7s | %7s | %6s "
        text += signature % ("Name", "size", "index", "offset", "scale", "units")
        text += "\n"
        text += signature % ("=" * 16, "=" * 6, "=" * 6, "=" * 7, "=" * 7, "=" * 6)

        def axis_repr(ax, ax_signature_uniform, ax_signature_non_uniform):
            if ax.is_uniform:
                return ax_signature_uniform % (
                    str(ax.name)[:16],
                    ax.size,
                    str(ax.index),
                    ax.offset,
                    ax.scale,
                    ax.units,
                )
            else:
                return ax_signature_non_uniform % (
                    str(ax.name)[:16],
                    ax.size,
                    str(ax.index),
                    ax.units,
                )

        for ax in self.navigation_axes:
            text += "\n"
            text += axis_repr(ax, ax_signature_uniform, ax_signature_non_uniform)
        text += "\n"
        text += signature % ("-" * 16, "-" * 6, "-" * 6, "-" * 7, "-" * 7, "-" * 6)
        for ax in self.signal_axes:
            text += "\n"
            text += axis_repr(ax, ax_signature_uniform, ax_signature_non_uniform)
        if self.ragged:
            text += "\n"
            text += "     Ragged axis |               Variable length"

        return text

    def _repr_html_(self):
        text = (
            "<style>\n"
            "table, th, td {\n\t"
            "border: 1px solid black;\n\t"
            "border-collapse: collapse;\n}"
            "\nth, td {\n\t"
            "padding: 5px;\n}"
            "\n</style>"
        )
        text += (
            "\n<p><b>< Axes manager, axes: %s ></b></p>\n" % self._get_dimension_str()
        )

        def format_row(*args, tag="td", bold=False):
            if bold:
                signature = "\n<tr class='bolder_row'> "
            else:
                signature = "\n<tr> "
            signature += " ".join(("{}" for _ in args)) + " </tr>"
            return signature.format(
                *map(lambda x: "\n<" + tag + ">{}</".format(x) + tag + ">", args)
            )

        def axis_repr(ax):
            index = ax.index if ax.navigate else ""
            if ax.is_uniform:
                return format_row(
                    ax.name, ax.size, index, ax.offset, ax.scale, ax.units
                )
            else:
                return format_row(
                    ax.name,
                    ax.size,
                    index,
                    "non-uniform axis",
                    "non-uniform axis",
                    ax.units,
                )

        if self.navigation_axes:
            text += "<table style='width:100%'>\n"
            text += format_row(
                "Navigation axis name",
                "size",
                "index",
                "offset",
                "scale",
                "units",
                tag="th",
            )
            for ax in self.navigation_axes:
                text += axis_repr(ax)
            text += "</table>\n"
        if self.signal_axes:
            text += "<table style='width:100%'>\n"
            text += format_row(
                "Signal axis name", "size", "", "offset", "scale", "units", tag="th"
            )
            for ax in self.signal_axes:
                text += axis_repr(ax)
            text += "</table>\n"
        return text

    @property
    def coordinates(self):
        pass

    @coordinates.setter
    def coordinates(self, coordinates):
        pass

    @property
    def indices(self):
        pass

    @indices.setter
    def indices(self, indices):
        pass

    def _get_axis_attribute_values(self, attr):
        return [getattr(axis, attr) for axis in self._axes]

    def _set_axis_attribute_values(self, attr, values):
        """Set the given attribute of all the axes to the given
        value(s)

        Parameters
        ----------
        attr : string
            The DataAxis attribute to set.
        values : any
            If iterable, it must have the same number of items
            as axes are in this AxesManager instance. If not iterable,
            the attribute of all the axes are set to the given value.

        """
        if not isiterable(values):
            values = [
                values,
            ] * len(self._axes)
        elif len(values) != len(self._axes):
            raise ValueError(
                "Values must have the same number"
                "of items are axes are in this AxesManager"
            )
        for axis, value in zip(self._axes, values):
            setattr(axis, attr, value)

    @property
    def navigation_indices_in_array(self):
        return tuple([axis.index_in_array for axis in self.navigation_axes])

    @property
    def signal_indices_in_array(self):
        return tuple([axis.index_in_array for axis in self.signal_axes])

    @property
    def axes_are_aligned_with_data(self):
        """Verify if the data axes are aligned with the signal axes.

        When the data are aligned with the axes the axes order in `self._axes`
        is [nav_n, nav_n-1, ..., nav_0, sig_m, sig_m-1 ..., sig_0].

        Returns
        -------
        aligned : bool

        """
        nav_iia_r = self.navigation_indices_in_array[::-1]
        sig_iia_r = self.signal_indices_in_array[::-1]
        iia_r = nav_iia_r + sig_iia_r
        aligned = iia_r == tuple(range(len(iia_r)))
        return aligned

    def _sort_axes(self):
        """Sort _axes to align them.

        When the data are aligned with the axes the axes order in `self._axes`
        is [nav_n, nav_n-1, ..., nav_0, sig_m, sig_m-1 ..., sig_0]. This method
        sort the axes in this way. Warning: this doesn't sort the `data` axes.

        """
        am = self
        new_axes = am.navigation_axes[::-1] + am.signal_axes[::-1]
        self._axes = list(new_axes)

    def gui_navigation_sliders(self, title="", display=True, toolkit=None):
        # With traits 6.1 and traitsui 7.0, we have this deprecation warning,
        # which is fine to filter
        # https://github.com/enthought/traitsui/issues/883
        pass



class GeneratorLen:
    """
    Helper class for creating a generator-like object with a known length.
    Useful when giving a generator as input to the AxesManager iterpath, so that the
    length is known for the progressbar.

    Found at: https://stackoverflow.com/questions/7460836/how-to-lengenerator/7460986

    Parameters
    ----------
    gen : generator
        The Generator containing hyperspy navigation indices.
    length : int
        The manually-specified length of the generator.
    """

    def __init__(self, gen, length):
        self.gen = gen
        self.length = length

    def __len__(self):
        return self.length

    def __iter__(self):
        return self.gen


def _parse_axis_attribute(value):
    """Parse axis attribute"""
    if value is t.Undefined:
        return None
    else:
        return value
