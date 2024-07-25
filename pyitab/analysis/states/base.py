import numpy as np
import os
from pyitab.utils.dataset import get_ds_data
from pyitab.analysis.base import Analyzer
from pyitab.analysis.states.subsamplers import VarianceSubsampler
from sklearn import cluster
from sklearn.pipeline import Pipeline
from scipy.spatial.distance import euclidean
import mne.time_frequency as freq

import logging
logger = logging.getLogger(__name__)


class Clustering(Analyzer):
    # TODO: Wrapper for sklearn clustering??

    def __init__(self, 
                 estimator=cluster.KMeans(),
                 name='state',
                 **kwargs):

        self.estimator = estimator
        self._est_params = self._get_estimator_params(estimator)

        Analyzer.__init__(self, name=name)

    
    def fit(self, ds,
            prepro=VarianceSubsampler(),
            **kwargs):
        """This method fits the dataset using the clustering algorithm

        Parameters
        ----------
        ds : [type]
            [description]
        prepro : [type], optional
            [description], by default VarianceSubsampler()

        Attributes
        -------
        scores: dict
            The results of the state identification
            'labels': array
                array with the assigned cluster for the 
                subsampled set of samples
            'states': array
                the centroids of the clustered set using
                the subsampled dataset.
            'dynamics': array
                the predicted labels of the full dataset using
                the fitted algorithm 
            'X': array
                the subsampled dataset
            'targets': array
                the targets of the dataset (if applicable)
            'state_similarity': array
                the similarity of the dataset with the most
                similar centroid
        """


        # Check if estimator needs n_clusters
        ds_ = prepro.transform(ds)
        logger.info("Dataset shape %s" % (str(ds_.shape)))
            
        X, _ = get_ds_data(ds_)
        
        logger.debug(isinstance(self.estimator, Pipeline))
        logger.debug(self.estimator)
        if isinstance(self.estimator, Pipeline):
            name, estimator = self.estimator.steps[0]
        else:
            estimator = self.estimator
        
        estimator = estimator.fit(X)

        self.scores = dict()
        
        if hasattr(estimator, 'labels_'):
            self.scores['labels'] = estimator.labels_

        elif hasattr(estimator, 'predict'):
            # Gaussian Mixture
            self.scores['labels'] = estimator.predict(X)

        if hasattr(estimator, 'cluster_centers_'):
            self.scores['states'] = estimator.cluster_centers_
        elif hasattr(estimator, 'means_'):
            # Gaussian Mixture
            self.scores['states'] = estimator.means_
        elif hasattr(estimator, 'labels_'):
            # DBSCAN
            self.scores['states'] = self.get_centroids(X, self.scores['labels'])

        if hasattr(estimator, 'predict'):
            self.scores['dynamics'] = estimator.predict(ds.samples)
        else:
            # DBSCAN
            self.scores['dynamics'] = self._predict(ds.samples, self.scores['states'])

        _ = self._predict(ds.samples, self.scores['states'])

        self.scores['X'] = X
        self.scores['targets'] = ds.sa.targets

        self._info = self._store_info(ds)

        return self    

    

    def get_centroids(self, X, labels):
        """
        Returns the centroid of a clustering experiment
        
        Parameters
        ----------
        X : n_samples x n_features array
            The full dataset used for clustering
        
        labels : n_samples array
            The clustering labels for each sample.
            
            
        Returns
        -------
        centroids : n_cluster x n_features shaped array
            The centroids of the clusters.
        """
        
        return np.array([X[labels == l].mean(0) for l in np.unique(labels)])


    def _predict(self, X, centroids, measure=euclidean, **kwargs):
        """
        Returns the similarity of the dataset to each centroid,
        using a dissimilarity distance function.
        
        Parameters
        ----------
        X : n_samples x n_features array
            The full dataset used for clustering
        
        centroids : n_cluster x n_features array
            The cluster centroids.
            
        measure : a scipy.spatial.distance function | default: euclidean
            This is the dissimilarity measure, this should be a python
            function, see scipy.spatial.distance.
            
        
        Returns
        -------
        labels : n_samples array
            The array indicating the most similar cluster center for
            each sample.
        
        """
        from pyitab.utils.math import similiarity

        dist = similiarity(centroids, X, measure=measure, **kwargs)
        labels, order = np.nonzero(dist.min(0) == dist)

        self.scores['state_similarity'] = dist

        labels_ = np.zeros_like(labels)
        labels_[order] = labels
        
        return labels_


    def get_state_frequencies(self, dynamics=None, sfreq=128,
                              method=freq.psd_array_welch, **kwargs):
        """
        Returns the spectrum of the state occurence for each subject.
        
        Parameters
        ----------
        state_dynamics :    n_states x n_timepoints array
                            The state dynamics output from fit_states
                            function.
                            
        method : function from mne.time_frequency package

        X : n_timepoinst x n_features array
            This is the array on which we need to build the state_dynamics
            whether not provided nor calculated.
        
        
        Returns
        -------
        results : tuple,
                first element is the array of frequencies,
                second element is the array n_states x frequencies
                of the spectrum.
        
        """

        if (dynamics is None):
            dynamics = self.scores['state_similarity']

        return method(dynamics, sfreq, **kwargs)


    def get_transition_matrix(self, dynamics=None):
        """
        Extracts the probability transition matrix given the
        fitted timecourse for each subject.
        This is the output of fit_centroids function
        
        Parameters
        ----------
        group_fitted_timecourse :   n_subjects x n_timepoints array
                                    The state dynamics output from fit_centroids
                                    function.
                            
        
        
        Returns
        -------
        transition_p : n_states x n_states matrix,
                    The number of transition from x state to y state, stored
                        in matrix[x,y]. The syntax is matrix[starting_state, ending_state].
        """
        
        if dynamics is None:
            dynamics = self.scores['state_similarity']
         
        n_states = dynamics.max()
        transitions = np.zeros((n_states+1, n_states+1))

        for i, state in enumerate(dynamics):
            if (i+1) < len(dynamics):
                transitions[state, dynamics[i+1]] += 1

        # transitions /= len(self._dynamics) - 1
        
        return transitions   


    def get_state_duration(self, dynamics=None):
        """
        Extracts the average of state duration given the
        fitted timecourse for each subject.
        This is the output of fit_centroids function
        
        Parameters
        ----------
        dynamics :   n_timepoints array
                The state dynamics output 
                from _predict function.
                            

        Returns
        -------
        state_duration : vector n_transition x 2 matrix,
                    The duration of each state for each subject/session.
                    first element is the state number, second the duration
                    in points
        """

        if dynamics is None:
            dynamics = self.scores['state_similarity']
        
        counter = 0
        
        state_duration = []
        for i, state in enumerate(dynamics):
            if (i+1) < len(dynamics):
                if state == dynamics[i+1]:
                    counter += 1
                else:
                    counter += 1
                    state_duration.append([state, float(counter)])
                    counter = 0
        
        return np.array(state_duration)


    def save(self, path=None, **kwargs):

        from scipy.io import savemat

        params = dict()
        params.update(kwargs)
        params.update(self._est_params)

        path, prefix = Analyzer.save(self, path=path, **params)

        savemat(os.path.join(path, "results-%s.mat" % (prefix)), {'data': self.scores})
        return path


    def _get_estimator_params(self, estimator):

        if isinstance(self.estimator, Pipeline):
            name, estimator = self.estimator.steps[0]
        else:
            estimator = self.estimator

        params = estimator.__dict__.copy()
        params['algorithm'] = str(estimator).split('(')[0]

        return params


    def _check_fields(self):
        pass




def get_extrema_histogram(arg_extrema, n_timepoints):
    
    hist_arg = np.zeros(n_timepoints)
    n_subjects = len(np.unique(arg_extrema[0]))
    
    for i in range(n_subjects):
        sub_max_arg = arg_extrema[1][arg_extrema[0] == i]
        hist_arg[sub_max_arg] += 1
        
    return hist_arg

       

