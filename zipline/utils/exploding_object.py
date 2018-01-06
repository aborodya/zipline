class ExplodingObject(object):
    """An attribute which has no attributes but produces a more informative
    error message when accessed.

    Parameters
    ----------
    name : str
        The name of the object. This will appear in the error messages.

    Notes
    -----
    One common use for this object is so ensure that an attribute always exists
    even if sometimes it should not be used.
    """
    def __init__(self, name):
        self._name = name

    def __getattr__(self, attr):
        raise AttributeError(
            'attempted to access attribute %r of ExplodingObject %r' % (
                attr,
                self._name,
            ),
        )
