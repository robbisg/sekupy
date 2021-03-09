from .base import Transformer
from ..utils.dataset import get_ds_data
from mvpa2.datasets import Dataset

import logging
logger = logging.getLogger(__name__)


class ScikitWrapper(Transformer):
    """Transformer to be used with scikit-learn transformers.
    They must implement fit_transform method, one application
    can be Principal Component decomposition.

    Parameters
    ----------
    estimator : [type], optional
        [description], by default None
    """

    def __init__(self, estimator=None, **kwargs):
        self.node = estimator
        Transformer.__init__(self, name='scikit-transfomer')


    def transform(self, ds):
        logger.info('Dataset preprocessing: Transforming using scikit-learn...')

        X, y = get_ds_data(ds)
        X_ = self.node.fit_transform(X, y)

        if X_.shape[1] == ds.shape[1]:
            fa = ds.fa
        else:
            fa = None

        ds_ = Dataset(X_, sa=ds.sa, a=ds.a, fa=fa)

        return Transformer.transform(self, ds_)
