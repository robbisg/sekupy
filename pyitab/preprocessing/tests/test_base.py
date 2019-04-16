from pyitab.tests import fetch_ds
from pyitab.preprocessing.memory import MemoryReducer


def test_memoryreducer(fetch_ds):

    ds = fetch_ds
    n_bytes_old = ds.samples.itemsize * ds.samples.size

    ds = MemoryReducer().transform(ds)

    assert n_bytes_old > ds.samples.itemsize * ds.samples.size

    last_item = list(ds.a.prepro[-1].keys())[0]
    assert last_item == 'memory_reducer' 