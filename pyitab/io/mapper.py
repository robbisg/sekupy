from pyitab.io.bids import load_bids_dataset
from pyitab.io.connectivity import load_mat_ds
from pyitab.io.base import load_dataset
from pyitab.simulation.loader import load_simulations
from pyitab.io.mambo import load_bids_mambo_dataset


def get_loader(name):

    mapper = {
        'bids': load_bids_dataset,
        'base': load_dataset,
        'mat': load_mat_ds,
        'simulations': load_simulations,
        'bids-meg': load_bids_mambo_dataset
    }

    return mapper[name]