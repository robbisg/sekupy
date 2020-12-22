import numpy as np

from sklearn.model_selection import BaseCrossValidator
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.utils.validation import _num_samples

class DoubleGroupCrossValidator(BaseCrossValidator):
    """[summary]
    
    Parameters
    ----------
    BaseCrossValidator : [type]
        [description]
    
    Raises
    ------
    NotImplementedError
        [description]
    NotImplementedError
        [description]
    
    Returns
    -------
    [type]
        [description]
    """
    def __init__(self, cv_test=LeaveOneGroupOut()):
        self.cv_test = cv_test
        return BaseCrossValidator.__init__(self)


    def _iter_test_masks(self, X=None, y=None, groups=None):
        """Generates boolean masks corresponding to test sets.
        By default, delegates to _iter_test_indices(X, y, groups)
        """
        raise NotImplementedError

    def _iter_test_indices(self, X=None, y=None, groups=None):
        """Generates integer indices corresponding to test sets."""
        raise NotImplementedError

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like, of length n_samples
            The target variable for supervised learning problems.
        groups : array-like, with shape (n_samples, 2), optional
            Group labels for the samples used while splitting the dataset into
            train/test set.
        Returns
        -------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """

        #X, y, groups = indexable(X, y, *groups)
        indices = np.arange(_num_samples(X))
        
        train_group, test_group = groups.T

        for train_g in np.unique(train_group):
            train_mask = train_group == train_g

            train_index = indices[train_mask]
            test_mask = np.logical_not(train_mask)

            rest_test = test_group[test_mask]

            for _, test_index in self.cv_test.split(X[test_mask],
                                                    y[test_mask],
                                                    test_group[test_mask]):
                test_index = indices[test_mask][test_index]
                yield train_index, test_index


    def get_n_splits(self, X=None, y=None, groups=None):
        _, test_group = groups
        
        return len(np.unique(test_group))