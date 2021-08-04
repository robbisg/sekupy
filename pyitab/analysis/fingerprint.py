
from pyitab.analysis.base import Analyzer
from pyitab.preprocessing import SampleSlicer
import numpy as np
import itertools

class Identifiability(Analyzer):

    def fit(self, ds, attr='targets'):

        unique = np.unique(ds.sa[attr].value)
        row, col = np.triu_indices(len(unique), k=0)
        
        identifiability_matrix = np.zeros((len(unique), len(unique)))
        accuracy_matrix = np.zeros((len(unique), len(unique)))

        task_combinations = itertools.combinations_with_replacement(unique, 2)
        for j, (t1, t2) in enumerate(task_combinations):
            ds1 = SampleSlicer(**{attr:[t1]}).transform(ds)
            ds2 = SampleSlicer(**{attr:[t2]}).transform(ds)

            s1 = ds1.samples - ds1.samples.mean(1)[:, np.newaxis]
            s2 = ds2.samples - ds2.samples.mean(1)[:, np.newaxis]

            dot = np.dot(s1, s2.T)

            n1 = np.sqrt(np.diag(np.dot(s1, s1.T)))[:, np.newaxis]
            n2 = np.sqrt(np.diag(np.dot(s2, s2.T)))[:, np.newaxis]

            r = np.dot(s1 / n1, (s2 / n2).T)
            
            i_self = np.mean(np.diag(r))

            id1 = np.triu_indices(r.shape[0], k=1)
            id2 = np.tril_indices(r.shape[0], k=-1)

            i_diff1 = np.mean(r[id1])
            i_diff2 = np.mean(r[id2])


            identifiability_matrix[row[j], col[j]] = i_self - i_diff1
            identifiability_matrix[col[j], row[j]] = i_self - i_diff2

            prediction1 = np.argmax(r, axis=0)
            prediction2 = np.argmax(r, axis=1)

            accuracy1 = np.count_nonzero(prediction1 == np.arange(r.shape[0]))/r.shape[0]
            accuracy2 = np.count_nonzero(prediction2 == np.arange(r.shape[0]))/r.shape[0]
            
            accuracy_matrix[row[j], col[j]] = accuracy1
            accuracy_matrix[col[j], row[j]] = accuracy2


        self.scores = dict()
        self.scores['matrix'] = identifiability_matrix
        self.scores['variable'] = unique
        self.scores['accuracy'] = accuracy_matrix

        return


class TaskPredictionTavor(Analyzer):
    def fit(self, ds, target_attr, regressor_attr, prepro=None):




class BehaviouralFingerprint():
    """This analysis is based on the paper  `Shen et al. 2017, Nature Protocol 
    <http://dx.doi.org/10.1038/nprot.2016.178>`_

    The pipeline is used to predict individual behaviour from brain connectivity.

    """

    def __init__():
        return

    def fit(self, ds):       


        return