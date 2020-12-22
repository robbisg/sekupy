import pytest
import pandas as pd
import os

from pyitab.analysis.results.base import filter_dataframe
from pyitab.analysis.results.simulations import get_results

currdir = os.path.dirname(os.path.abspath(__file__))
currdir = os.path.abspath(os.path.join(currdir, os.pardir))
datadir = os.path.join(currdir, "tests", "data")


@pytest.fixture
def dataframe():
    print(datadir)
    return pd.read_csv(os.path.join(datadir, "test.csv"))

@pytest.fixture
def directory():
    return os.path.join(datadir, "derivatives")


def test_filter(dataframe):

    algorithms = {
        "KMeans": 101,
        "SpectralClustering": 89,
        "GaussianMixture": 105
    }

    for k, v in algorithms.items():
        df_ = filter_dataframe(dataframe, algorithm=[k])
        assert len(df_) == v

    with pytest.raises(Exception):
        df_ = filter_dataframe(dataframe, algorithm=["FooAlgorithm"])


def test_load_simulations(directory):
    print(directory)
    dataframe = get_results(directory, pipeline="c2b+real")
    assert len(dataframe) == 4

    dataframe = get_results(directory,
                            field_list=['algorithm'],
                            pipeline="c2b+real", 
                            filter={"algorithm":["SpectralClustering"]})
    
    assert len(dataframe) == 2


    