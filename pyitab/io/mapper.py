from pyitab.io.bids import load_bids_dataset
from pyitab.io.connectivity import load_mat_ds
from pyitab.io.base import load_dataset


def get_loader(name):

    mapper = {
        'bids': load_bids_dataset,
        'base': load_dataset,
        'mat': load_mat_ds

    }

    return mapper[name]