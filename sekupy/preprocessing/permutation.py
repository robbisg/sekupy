from sekupy.preprocessing.base import Transformer
from sekupy.utils.dataset import get_ds_data
from sklearn.utils import shuffle

class Permutator(Transformer):
    def __init__(self, n=100, name='permutator', **kwargs):
        
        self.n = n + 1
        self.i = 0
        super().__init__(name=name, **kwargs)

    def transform(self, ds):
        """Transform function shuffles only samples, without
        modifying any sample attribute.

        Parameters
        ----------
        ds : [type]
            [description]

        Returns
        -------
        [type]
            [description]

        Raises
        ------
        StopIteration
            [description]
        """

        while self.i < self.n:
            if self.i != 0:

                indices = range(ds.shape[0])
                idx = shuffle(indices)
                ds_ = ds.copy()
                ds_.samples = ds.samples[idx]
            else:
                ds_ = ds
                
            self.i += 1
            return super().transform(ds_)

        raise StopIteration()
