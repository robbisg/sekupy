import numpy as np
import scipy as sp
from scipy import signal


class SimulationModel(object):
    def __init__(self, order=None, noise=None, delay=None, snr=1e6, **kwargs):

        self.order = order
        self.noise = noise
        self.delay = delay
        self.snr = snr

    
    def fit(self, bs_sequence, bs_length):
        



class AutoRegressiveModel(SimulationModel):
    
    def __init__(self, order=10, noise=0.01, **kwargs):        
        SimulationModel.__init__(self, order, noise, **kwargs)

    def fit(self, matrices, bs_sequence, bs_lenght):
        
        data = []

        for i, state in enumerate(bs_sequence):

            length = bs_lenght[i]

            # TODO: move data_bs in superclass and build create data in subclasses
            data_bs = self.noise * np.random.randn(length, matrices.shape[1])

            for t in np.arange(self.order, length):
                for d in range(self.order):
                    data_bs[t,:] = data_bs[t,:] + np.dot(data_bs[t-d,:], matrices[state,:,:,d])

            
            data.append(data_bs)

        data = np.hstack(data)
        
        random_noise = np.random.randn(data.shape)
        data_noise = data + np.sqrt(1/self.snr)*(random_noise/np.std(random_noise))


        self.data = data_noise


        return data


class DelayedModel(SimulationModel):
    def __init__(self, order=5, noise=1, delay=0.0195, **kwargs):
        SimulationModel.__init__(self, order, noise, delay, **kwargs)

    
    def fit(self, matrices, bs_sequence, bs_lenght):
    
        data = []

        for i, state in enumerate(bs_sequence):

            m = matrices[state]
            adjacency_matrix = np.int_(np.mean(np.abs(m), axis=2) != 0) - np.eye(m.shape[1])
            diagonal = np.dstack([np.diag(np.diag(m[...,i])) for i in range(m.shape[-1])])

            length = bs_lenght[i]

            # TODO: move data_bs in superclass and build create data in subclasses
            remove_idx = np.int_(np.sum(adjacency_matrix, axis=0) == 0)
            data_bs = self.noise * np.random.randn(length, matrices.shape[1]) * remove_idx

            # TODO: move in superclass, remember diagonal!
            for t in np.arange(self.order, length):
                for d in range(self.order):
                    data_bs[t,:] = data_bs[t,:] + np.dot(data_bs[t-d,:], diagonal[:,:,d])

            leading, following = np.nonzero(adjacency_matrix)
            for l, f in zip(leading, following):
                data_bs[:,f] = data_bs[:,f] + self._get_delayed_signal(data_bs[:,l])


            data.append(data_bs)

        data = np.vstack(data)

        random_noise = np.random.randn(data.shape)
        data_noise = data + np.sqrt(1/self.snr)*(random_noise/np.std(random_noise))

        self.data = data_noise

        return data



class TimeDelayedModel(DelayedModel):

    def __init__(self, order=5, noise=1, delay=0.0195, fsample=125, **kwargs):
        self.delay = delay * fsample
        DelayedModel.__init__(self, order, noise, delay, **kwargs)


        def _get_delayed_signal(self, leading_signal):
            return np.hstack((np.random.randn(self.delay), leading_signal[self.delay:]))

        

class PhaseDelayedModel(DelayedModel):

    def _get_delayed_signal(self, leading_signal):

        hilbert_l = signal.hilbert(leading_signal)
        hilbert_f = signal.hilbert(np.random.randn(*leading_signal.shape))
        angle = (np.angle(hilbert_l)+self.delay)
        shifted_signal = np.abs(hilbert_f) * np.exp(angle*1j)
        return np.real(shifted_signal)