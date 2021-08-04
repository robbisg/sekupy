import numpy as np
import itertools
from pyitab.preprocessing.base import Transformer

import logging
logger = logging.getLogger(__name__)


class BoldStateSimulator(Transformer):

    def __init__(self, ntime=200, nvox=50, nstates=15,
                 nsub=1, group_std=0, sub_std=0.1, 
                 sub_evprob=0., length_std=1, peak_delay=6, 
                 peak_disp=1, extime=2, TR=2.47, TRfactor=1,  
                 seed=500):
        """
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
        """The state simulator using transform generates the dynamics
        of the system. 
        """
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

        has_zero = True
        while has_zero:
            data = duration['distribution'](**duration['params'])
            data = np.int_(self._fs * np.abs(data))
            # logger.info(data)
            if np.count_nonzero(data) == 0:
                has_zero = False
        
        self._duration = duration

        return data




