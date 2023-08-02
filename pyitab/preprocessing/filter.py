from scipy import signal
from pyitab.preprocessing.base import Transformer



class ButterFilter(Transformer):
    # TODO: Then use a superclass Filter
    # It uses scipy filters (maybe can be used with MNE)

    def __init__(self, order=4, min_freq=None, max_freq=None, btype='bandpass', **kwargs):
        self.order = order

        # TODO: make a check function
        if (min_freq is not None) and (max_freq is not None):
            self.freq = [min_freq, max_freq]
        elif (max_freq is None) and (min_freq is not None):
            self.freq = min_freq
        elif (max_freq is not None) and (min_freq is None):
            self.freq = max_freq

        self.btype = btype

        Transformer.__init__(self, name='butter', **kwargs)


    def transform(self, ds):

        data = ds.samples
        
        if not hasattr(ds.a, 'sample_frequency'):
            raise Exception("dataset must have a sample frequency attribute in ds.a")

        b, a = signal.butter(self.order, self.freq, btype=self.btype, fs=ds.a.sample_frequency)
        data = signal.filtfilt(b, a, data.T)
        ds.samples = data.T

        return ds