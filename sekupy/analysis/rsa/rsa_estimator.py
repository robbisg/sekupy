"""RSA Estimator wrapper for use with SearchLight and other sklearn-compatible frameworks."""

import numpy as np
from sklearn.base import BaseEstimator
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import make_scorer


class RSAEstimator(BaseEstimator):
    """Estimator wrapper for RSA that can be used within SearchLight.
    
    This class wraps RSA functionality to make it compatible with 
    scikit-learn's estimator interface, allowing it to be used as 
    an estimator parameter in SearchLight analysis.
    
    Parameters
    ----------
    metric : str, default='euclidean'
        The distance metric to use for computing dissimilarities.
        Any metric supported by scipy.spatial.distance.pdist can be used.
        
    Attributes
    ----------
    distance_matrix_ : ndarray
        The computed distance matrix after fitting.
    """
    
    def __init__(self, metric='euclidean'):
        self.metric = metric
        
    def fit(self, X, y=None):
        """Fit the RSA estimator.
        
        Computes the distance matrix for the training data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,), optional
            Target values (used for compatibility, but not required for RSA).
            
        Returns
        -------
        self : RSAEstimator
            Fitted estimator.
        """
        # Compute pairwise distances
        self.distance_matrix_ = pdist(X, metric=self.metric)
        # Store y for predict method compatibility
        self.y_ = y
        return self
    
    def predict(self, X):
        """Predict method for compatibility with sklearn scoring.
        
        For RSA, we return the condensed distance matrix as "predictions".
        This allows the estimator to work with various sklearn scorers.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to predict on.
            
        Returns
        -------
        distances : ndarray of shape (n_samples * (n_samples - 1) / 2,)
            Condensed distance matrix.
        """
        return pdist(X, metric=self.metric)
    
    def score(self, X, y=None):
        """Compute a score for the RSA analysis.
        
        This computes the negative mean distance as a score metric.
        In RSA, we want to capture the representational structure,
        so we return a metric based on the distance matrix.
        
        For compatibility with regression scoring metrics like R², 
        we can interpret the score as how well the representations 
        are structured (lower distance = better structure).
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data.
        y : array-like of shape (n_samples,), optional
            True labels (used for compatibility).
            
        Returns
        -------
        score : float
            Negative mean distance (higher is better for more similar representations).
        """
        # Compute distance for test data
        distance = pdist(X, metric=self.metric)
        # Return negative mean distance (so higher score = more similar)
        # Scale to a more reasonable range for comparison with other metrics
        return -np.mean(distance)
    
    def transform(self, X):
        """Transform data to distance matrix representation.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to transform.
            
        Returns
        -------
        distances : ndarray of shape (n_samples * (n_samples - 1) / 2,)
            Condensed distance matrix.
        """
        return pdist(X, metric=self.metric)


def rsa_scorer(metric='euclidean'):
    """Create a scorer function for RSA that can be used with cross-validation.
    
    Parameters
    ----------
    metric : str, default='euclidean'
        The distance metric to use.
        
    Returns
    -------
    scorer : callable
        A scorer function compatible with sklearn's scoring parameter.
    """
    def _score(estimator, X, y=None):
        """Score function for RSA."""
        distance = pdist(X, metric=metric)
        # Return negative mean distance
        return -np.mean(distance)
    
    return make_scorer(_score, greater_is_better=True)
