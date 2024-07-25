import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr, ttest_ind, zscore
from typing import Optional, Tuple

from sekupy.analysis.base import Analyzer
from sekupy.utils.dataset import get_ds_data

class _GSBS(Analyzer):
    """
    This class uses a greedy search algorithm to segment timeseries
    into neural state with stable activity patterns.
    The algorithm identifies the timepoint of state transition and 
    the best number of states using t-statistics

    You can find more information about the method here:
    Geerligs L., van Gerven M., Güçlü U (2020) 
    Detecting neural state transitions underlying event segmentation
    biorXiv. https://doi.org/10.1101/2020.04.30.069989

    Parameters
    ----------
    kmax : int
        Maximum number of neural states to be estimated.
        (a reasonable choice is t/2) (maybe should be included in fit)
    tr_tuning : int, optional
        Number of timepoints to be included in the finetuning, it 
        optimizes the transition time around a boundary.
        If = 0 no fine tuning is performed, if < 0 all timepoints are 
        included, by default 1.
    blocksize : int, optional
        Minimal number that constitues a block, this is used to speed up the 
        computation when the number of tp is large, it finds local optimun 
        within a block of blocksize, by default 50.
    dmin : int, optional
        Number of timepoints around the diagonal that are not taken into 
        account in the computation of the t-distance metric, by default 1.
    """

    def __init__(self, tr_tuning=1, blocksize=50, dmin=1, **kwargs):


        self.dmin = dmin
        self.finetune = tr_tuning
        self.blocksize = blocksize

        Analyzer.__init__(self, name='gsbs', **kwargs)

    
    def get_bounds(self, k):
        """
        Keyword Arguments:
            k {Optional[int]} -- number of states
                By default the function returns the boundaries for the optimal number of states (k=nstates).
                When k is given, the boundaries for k states are returned.
        Returns:
            ndarray -- array with length == number of timepoints, where a zero indicates no state transition
            at a particular timepoint and a higher number indicates a state transition. State transitions
            are numbered in the order in which they are detected in GSBS (stronger boundaries tend
            to be detected first).
        """
        assert self._argmax is not None
        if k is None:
            k = self._argmax
        if self.finetune != 0:
            return self.all_bounds[k]
        else:
            return self._bounds * self.get_deltas(k)


    def get_deltas(self, k=None):
        """
        Keyword Arguments:
            k {Optional[int]} -- number of states
                By default the function returns the deltas for the optimal number of states (k=nstates).
                When k is given, the deltas for k states are returned.
        Returns:
            ndarray -- array with length == number of timepoints, where a zero indicates no state transition
            at a particular timepoint and a one indicates a state transition.
        """
        assert self._argmax is not None

        if k is None:
            k = self._argmax

        if self.finetune!=0:
            deltas = np.logical_and(self.all_bounds[k] <= k, self.all_bounds[k] > 0)
        else:
            deltas = np.logical_and(self._bounds <= k, self._bounds > 0)
        deltas = deltas * 1

        return deltas


    def get_states(self, X, k=None):
        """
        Keyword Arguments:
            k {Optional[int]} -- number of states
                By default the function returns the states for the optimal number of states (k=nstates).
                When k is given, k states are returned.

        Returns:
            ndarray -- array with length == number of timepoints,
            where each timepoint is numbered according to the neural state it is in.
        """
        assert self._argmax is not None
        if k is None:
            k = self._argmax
        states = self.assign_states(X, self.get_deltas(k)) + 1
        return states



    def get_state_patterns(self, X, k=None):
        """
        Keyword Arguments:
            k {Optional[int]} -- number of states
                By default the function returns the state patterns for the optimal number of states (k=nstates).
                When k is given, the state patterns for k states are returned.

        Returns:
            ndarray -- timepoint by nstates array
            Contains the average voxel activity patterns for each of the estimates neural states
        """
        assert self._argmax is not None
        if k is None:
            k = self._argmax
        deltas = self.get_deltas(k)
        states = self.assign_states(X, deltas)
        states_unique = np.unique(states)
        xmeans = np.zeros((len(states_unique), X.shape[1]), float)

        for state in states_unique:
            xmeans[state] = X[state == states].mean(0)

        return xmeans


    def get_strengths(self, X, k=None):
        """
        Keyword Arguments:
            k {Optional[int]} -- number of states
                By default the function returns the transition strengths for the optimal number of states (k=nstates).
                When k is given, the transition strengths for k states are returned.

        Returns:
            ndarray -- array with length == number of timepoints, where a zero indicates no state transition
            at a particular timepoint and another value indicates a state transition. The numbers indicate
            the strength of a state transition, as indicated by the Pearson correlation-distance between neural
            activity patterns in consecutive states.
        """
        assert self._argmax is not None
        if k is None:
            k = self._argmax
        deltas = self.get_deltas(k)
        states = self.assign_states(X, deltas)
        states_unique = np.unique(states)
        pcorrs = np.zeros(len(states_unique) - 1, float)
        xmeans = np.zeros((len(states_unique), X.shape[1]), float)

        for state in states_unique:
            xmeans[state] = X[state == states].mean(0)
            if state > 0:
                pcorrs[state - 1] = pearsonr(xmeans[state], xmeans[state - 1])[0]

        strengths = np.zeros(deltas.shape, float)
        strengths[deltas == 1] = 1 - pcorrs

        return strengths



    def fit(self, X, y=None, kmax=None):
        """This function performs the GSBS and t-distance computations to determine
        the location of state boundaries and the optimal number of states.
        """

        if kmax is None:
            kmax = np.int16(X.shape[0] * .5)

        ind = np.triu(np.ones(X.shape[0], bool), self.dmin)
        Z = zscore(X, axis=1, ddof=1)

        self.all_bounds = np.zeros((kmax + 1, X.shape[0]), int)
        self._bounds = np.zeros(X.shape[0], int)
        self._deltas = np.zeros(X.shape[0], bool)
        self._tdists = np.zeros(kmax + 1, float)

        if y is None:
            t = np.cov(Z)[ind]
        else:
            t = np.cov(zscore(y, axis=1, ddof=1))[ind]

        for k in range(2, kmax + 1):
            states = self.assign_states(X, self._deltas)
            wdists = self.calculate_wdists_blocks(X, Z, states)
            argmax = wdists.argmax()

            self._bounds[argmax] = k
            self._deltas[argmax] = True

            if self.finetune != 0 and k > 2:
                self._bounds = self.finetuning(X, Z)
                self._deltas = self._bounds > 0
                self.all_bounds[k] = self._bounds

            states = self.assign_states(X, self._deltas)[:,None]
            diff, same, alldiff = (lambda c: (c == 1, c == 0, c > 0))(cdist(states, states, "cityblock")[ind])
            if sum(same) < 2:
                self._tdists[k] = 0
            else:
                self._tdists[k] = ttest_ind(t[same], t[diff], equal_var=False)[0]

        self._argmax = self._tdists.argmax()

        self.scores = dict()
        self.scores['labels'] = self.get_states(X)
        self.scores['states'] = self.get_state_patterns(X)





    def finetuning(self, X, Z):

        bounds = self._bounds
        finetune = self.finetune

        finebounds = np.copy(bounds.astype(int))
        for kk in np.unique(bounds[bounds > 0]):
            ind = (finebounds == kk).nonzero()[0][0]
            finebounds[ind] = 0
            deltas = finebounds > 0
            states = self.assign_states(X, self._deltas)
            if finetune < 0:
                boundopt = np.arange(1, states.shape[0])
            else:
                boundopt = np.arange(max((1, ind-finetune)), min((states.shape[0],ind+finetune+1)), 1)
            wdists = self.calculate_wdists(X, Z, deltas, boundopt, states)
            argmax = wdists.argmax()
            finebounds[argmax] = kk

        return finebounds


    def assign_states(self, X, deltas):
        states = np.zeros(len(deltas), int)
        for i, delta in enumerate(deltas[1:]):
            states[i + 1] = states[i] + np.int16(delta)

        return states


    def calculate_wdists(self, Xt, Zt, deltas, boundopt, states):

        xmeans = np.zeros(Xt.shape, float)
        wdists = -1 * np.ones(Xt.shape[0], float)

        if boundopt is None:
            boundopt = np.arange(1, Xt.shape[0])

        for state in np.unique(states):
            mask_state = states == state
            xmeans[mask_state] = Xt[mask_state].mean(0)

        for i in boundopt:
            if deltas[i] == False:
                state_idx = np.nonzero(states[i] == states)[0]
                xmean = np.copy(xmeans[state_idx])
                xmeans[state_idx[0]: i] = Xt[state_idx[0]: i].mean(0)
                xmeans[i: state_idx[-1] + 1] = Xt[i: state_idx[-1] + 1].mean(0)
                zmeans = zscore(xmeans, axis=1, ddof=1)
                wdists[i] = xmeans.shape[1] * (zmeans * Zt).mean() / (xmeans.shape[1] - 1)
                xmeans[state_idx] = xmean

        return wdists


    def calculate_wdists_blocks(self, X, Z, states):

        deltas = self._deltas
        blocksize = self.blocksize

        if len(np.unique(states)) > 1:
            boundopt = np.zeros(max(states)+1)
            prevstate = -1
            for s in np.unique(states):
                state = np.where((states > prevstate) & (states <= s))[0]
                numt = state.shape[0]
                if numt > blocksize or s == max(states):
                    xt = X[state]
                    zt = Z[state]
                    wdists = self.calculate_wdists(xt, zt, deltas[state], None, states[state])
                    boundopt[s] = wdists.argmax() + state[0]
                    prevstate = s

            boundopt = boundopt[boundopt > 0]
            boundopt = boundopt.astype(int)

        else:
            boundopt = None

        wdists = self.calculate_wdists(X, Z, deltas, boundopt, states)

        return wdists



class GSBS(_GSBS):
    """
    This class uses a greedy search algorithm to segment timeseries
    into neural state with stable activity patterns.
    The algorithm identifies the timepoint of state transition and 
    the best number of states using t-statistics

    You can find more information about the method here:
    Geerligs L., van Gerven M., Güçlü U (2020) 
    Detecting neural state transitions underlying event segmentation
    biorXiv. https://doi.org/10.1101/2020.04.30.069989

    Parameters
    ----------
    kmax : int
        Maximum number of neural states to be estimated.
        (a reasonable choice is t/2) (maybe should be included in fit)
    tr_tuning : int, optional
        Number of timepoints to be included in the finetuning, it 
        optimizes the transition time around a boundary.
        If = 0 no fine tuning is performed, if < 0 all timepoints are 
        included, by default 1.
    blocksize : int, optional
        Minimal number that constitues a block, this is used to speed up the 
        computation when the number of tp is large, it finds local optimun 
        within a block of blocksize, by default 50.
    dmin : int, optional
        Number of timepoints around the diagonal that are not taken into 
        account in the computation of the t-distance metric, by default 1.
    """

    def fit(self, ds, kmax=None):
        X, _ = get_ds_data(ds)
        super().fit(X, kmax=kmax)


