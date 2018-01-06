from functools import wraps

import numpy as np


class RingBuffer(object):
    """A statically typed, fixed size ring buffer for performing rolling
    reductions.

    Parameters
    ----------
    shape : int or tuple[int]
        The shape of the underlying array.
    fill_value : any
        The default fill value of the buffer.
    dtype : np.dtype
        The numpy dtype of the buffer.

    Notes
    -----
    The buffer rotates along axis 0.

    Examples
    --------
    >>> # create a ring buffer of 5 integers, defaulting to 0
    >>> rb = RingBuffer(5, 0, 'i8')
    >>> for n in range(5):
    ...     rb.push(n)
    >>> rb.sum()
    10
    >>> rb.mean()
    2.0

    >>> # push 5, evicting 0, buffer semantically holds: [1, 2, 3, 4, 5]
    >>> rb.push(5)
    >> rb.sum()
    15
    >>> rb.mean()
    3.0

    >>> # create a ring buffer of 5 integer arrays of length 2, defaulting to 0
    >>> rb2d = RingBuffer((5, 2), 0, 'i8')
    >>> for n in range(5):
    ...     rb2d.push([10 * n, 10 * n + 1])
    >>> rb2d.sum()  # sums all elements
    205
    >>> rb2d.sum(axis=0)  # sum along axis 0
    array([100, 105])

    >>> rb2d.sum(axis=1)
    Traceback (most recent call last):
       ...
    ValueError: cannot perform reduction along non-primary axis
    """
    def __init__(self, shape, fill_value, dtype):
        self._buffer = np.full(shape, fill_value, dtype)
        self._capacity = self._buffer.shape[0]
        self._ix = 0

    def __len__(self):
        return self._capacity

    def fill(self, value):
        """Fill the buffer with a single value. This is useful for resetting a
        RingBuffer.

        Parameters
        ----------
        value : any
            The value to fill the buffer with.
        """
        self._buffer[:] = value

    def push(self, element):
        """Push an element into the ring buffer.

        Parameters
        ----------
        element : any
            The element to insert.
        """
        ix = self._ix
        self._buffer[ix] = element
        self._ix = (ix + 1) % self._capacity

    def _numpy_reduction(name):
        """Define a method which forwards to a method of the underlying buffer.
        """
        @wraps(getattr(np.ndarray, name), assigned=('__name__', '__doc__'))
        def method(self, axis=None, *args, **kwargs):
            if axis not in (None, 0):
                raise ValueError(
                    'cannot perform reduction along non-primary axis',
                )

            return getattr(self._buffer, name)(axis, *args, **kwargs)
        return method

    for reduction in 'all', 'any', 'max', 'mean', 'min', 'prod', 'std', 'sum':
        locals()[reduction] = _numpy_reduction(reduction)
        del reduction

    del _numpy_reduction
