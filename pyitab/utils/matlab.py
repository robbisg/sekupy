def h5py2string(referenced):
    string = "".join("".join([u''.join(chr(c)) for c in referenced]))

    return string