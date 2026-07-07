import logging
from sekupy.preprocessing.base import Transformer

logger = logging.getLogger(__name__)


class MneTransformer(Transformer):
    """Base class for transformers operating on MNE Raw or Epochs objects."""

    def __init__(self, name='mne_transformer', **kwargs):
        Transformer.__init__(self, name=name, **kwargs)

    def map_transformer(self, ds):
        pass

    def transform(self, ds):
        return ds
