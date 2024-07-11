def is_datasetlike(obj):
    """Check if an object looks like a Dataset."""
    if hasattr(obj, 'samples') and \
       hasattr(obj, 'sa') and \
       hasattr(obj, 'fa') and \
       hasattr(obj, 'a'):
        return True

    return False


def accepts_dataset_as_samples(fx):
    """Decorator to extract samples from Datasets.

    Little helper to allow methods to be written for plain data (if they
    don't need information from a Dataset), but at the same time also
    accept whole Datasets as input.
    """
    def extract_samples(obj, data):
        if is_datasetlike(data):
            return fx(obj, data.samples)
        else:
            return fx(obj, data)
    return extract_samples


def _str(obj, *args, **kwargs):
    """Helper to get a structured __str__ for all objects.

    If an object has a `descr` attribute, its content will be used instead of
    an auto-generated description.

    Optional additional information might be added under certain debugging
    conditions (e.g. `id(obj)`).

    Parameters
    ----------
    obj : object
      This will typically be `self` of the to be documented object.
    *args, **kwargs : str
      An arbitrary number of additional items. All of them must be of type
      `str`. All items will be appended comma separated to the class name.
      Keyword arguments will be appended as `key`=`value.

    Returns
    -------
    str
    """
    truncate = None

    s = None
    # don't do descriptions for dicts like our collections as they might contain
    # an actual item 'descr'
    if hasattr(obj, 'descr') and not isinstance(obj, dict):
        s = obj.descr
    if s is None:
        s = obj.__class__.__name__
        auto_descr = ', '.join(list(args)
                       + ["%s=%s" % (k, v) for k, v in kwargs.items()])
        if len(auto_descr):
            s = s + ': ' + auto_descr

    if truncate is not None and len(s) > truncate - 5:
        # -5 to take <...> into account
        s = s[:truncate-5] + '...'

    return '<' + s + '>'


def is_sequence_type(inst):
    """Return True if an instance is of an iterable type

    Verified by wrapping with iter() call
    """
    try:
        _ = iter(inst)
        return True
    except TypeError:
        return False