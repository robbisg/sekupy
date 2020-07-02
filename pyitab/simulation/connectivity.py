import numpy as np
import itertools
from pyitab.preprocessing.base import Transformer

import logging
logger = logging.getLogger(__name__)


class ConnectivityStateSimulator(Transformer):

    def __init__(self, n_nodes=10, max_edges=5, fsamp=128,
                 n_brain_states=6, length_dynamics=100, 
                 state_duration={'min_time': 2.5, 'max_time': 3.5}, 
                 method='random'):
        """This class is used to simulate the connectivity state dynamics.

        Parameters
        ----------
        n_nodes : int, optional
            Number of nodes used in the simulation (e.g. brain sources), by default 10
        max_edges : int, optional
            Maximum number of simultaneous connections, by default 5
        fsamp : int, optional
            Sampling frequency of the brain nodes signal, by default 128
        n_brain_states : int, optional
            Number of brain states in the dynamics, by default 6
        length_dynamics : int, optional
            Length of the dynamics (e.g. number of brain state events), by default 100
        state_duration : dict, optional
            Duration of a state:
            - it can be a dictionary with max_time and min_time, and duration is uniformly
            sampled between min and max
            - it can be a dictionary with distribution and parameters keywords to specify
            how state duration should be generated (see np.random)
        method : str, optional
            method used to generate brain dynamics, by default 'random'
        """
    
        edges = [e for e in itertools.combinations(np.arange(n_nodes), 2)]
        n_edges = len(edges)

        states = []
        for i in range(max_edges):
            states += [e for e in itertools.combinations(np.arange(n_edges), i+1)]

        states = np.array(states)

        self._edges = edges
        self._n_edges = len(edges)
        self._edges_states = states
        self._n_states = len(states)
        self._method = method

        self._n_nodes = n_nodes
        self._max_edges = max_edges
        self._fs = fsamp

        self._n_brain_states = n_brain_states
        self._duration = state_duration
        self._length_dynamics = length_dynamics

        Transformer.__init__(self, name='connectivity_state_simulator')


    def transform(self, ds):
        return self.fit()
        

    def fit(self):

        # Randomize random stuff
        states_idx = np.random.randint(0, self._n_states, self._n_brain_states)        
        selected_states = self._edges_states[states_idx]

        bs_length = self.generate_duration()

        # This is done using Hidden Markov Models but since 
        # Transition matrix is uniformly distributed we can use random sequency
        #     mc = mcmix(nBS,'Fix',ones(nBS)*(1/nBS));
        #     seqBS= simulate(mc,nbs_sequence-1);
        bs_sequence = self.simulate_dynamics()

        bs_dynamics = [] 
        for i, time in enumerate(bs_length):
            for _ in range(time):
                bs_dynamics.append(bs_sequence[i])
        bs_dynamics = np.array(bs_dynamics)

        brain_matrices = []
        for j in range(self._n_brain_states):
            matrix = np.eye(self._n_nodes)
            selected_edges = selected_states[j]
            for edge in selected_edges:
                matrix[self._edges[np.array(edge)]] = 1
            brain_matrices.append(matrix)

        brain_matrices = np.array(brain_matrices)

        self._states = brain_matrices
        self._state_length = bs_length
        self._state_sequence = bs_sequence
        self._dynamics = bs_dynamics
        self._time = self._duration['params']

        return self


    def simulate_dynamics(self, method='random'):

        if self._method == 'random':
            return np.random.randint(0, 
                                    self._n_brain_states, 
                                    self._length_dynamics)


    def generate_duration(self):
        import matplotlib.pyplot as pl
        duration = {}
        if ('max_time' in self._duration.keys()) or \
            ('min_time' in self._duration.keys()):
            duration['distribution'] = np.random.uniform
            duration['params'] = {'low': self._duration['min_time'], 
                                  'high': self._duration['max_time']}
        
        # TODO: check that distribution is numpy random style
        else:
            duration = self._duration.copy()

        logger.debug(duration)
        duration['params'].update({'size': self._length_dynamics})

        data = duration['distribution'](**duration['params'])        
        data = np.int_(self._fs * np.abs(data))
        
        self._duration = duration

        return data




