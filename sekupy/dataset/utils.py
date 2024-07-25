import re
import numpy as np
from io import StringIO


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

def _repr_attrs(obj, attrs, default=None, error_value='ERROR'):
    """Helper to obtain a list of formatted attributes different from
    the default
    """
    out = []
    for a in attrs:
        v = getattr(obj, a, error_value)
        if not (v is default or isinstance(v, str) and v == default):
            out.append('%s=%s' % (a, v))
    return out


def table2string(table, out=None):
    """Given list of lists figure out their common widths and print to out

    Parameters
    ----------
    table : list of lists of strings
      What is aimed to be printed
    out : None or stream
      Where to print. If None -- will print and return string

    Returns
    -------
    string if out was None
    """

    print2string = out is None
    if print2string:
        out = StringIO()

    # equalize number of elements in each row
    Nelements_max = len(table) \
                    and max(len(x) for x in table)

    for i, table_ in enumerate(table):
        table[i] += [''] * (Nelements_max - len(table_))

    # figure out lengths within each column
    atable = np.asarray(table).astype(str)
    # eat whole entry while computing width for @w (for wide)
    markup_strip = re.compile('^@([lrc]|w.*)')
    col_width = [ max( [len(markup_strip.sub('', x))
                        for x in column] ) for column in atable.T ]
    string = ""
    for i, table_ in enumerate(table):
        string_ = ""
        for j, item in enumerate(table_):
            item = str(item)
            if item.startswith('@'):
                align = item[1]
                item = item[2:]
                if not align in ['l', 'r', 'c', 'w']:
                    raise ValueError('Unknown alignment %s. Known are l,r,c' % align)
            else:
                align = 'c'

            NspacesL = max(np.ceil((col_width[j] - len(item))/2.0), 0)
            NspacesR = max(col_width[j] - NspacesL - len(item), 0)

            if align in ['w', 'c']:
                pass
            elif align == 'l':
                NspacesL, NspacesR = 0, NspacesL + NspacesR
            elif align == 'r':
                NspacesL, NspacesR = NspacesL + NspacesR, 0
            else:
                raise RuntimeError('Should not get here with align=%s' % align)

            string_ += "%%%ds%%s%%%ds " \
                       % (NspacesL, NspacesR) % ('', item, '')
        string += string_.rstrip() + '\n'
    out.write(string)

    if print2string:
        value = out.getvalue()
        out.close()
        return value

def is_sequence_type(inst):
    """Return True if an instance is of an iterable type

    Verified by wrapping with iter() call
    """
    try:
        _ = iter(inst)
        return True
    except TypeError:
        return False