"""RSA Estimator wrapper for use with SearchLight and other sklearn-compatible frameworks."""

import numpy as np
from sklearn.base import BaseEstimator
from scipy.spatial.distance import pdist
from sklearn.metrics import make_scorer


def _compute_rsa_score(X, metric='euclidean'):
    """Helper function to compute RSA score from data.
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Data to compute score for.
    metric : str, default='euclidean'
        Distance metric to use.
        
    Returns
    -------
    score : float
        Negative mean distance.
    """
    distance = pdist(X, metric=metric)
    return -np.mean(distance)


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
        Common options include: 'euclidean', 'correlation', 'cosine', 
        'cityblock', 'hamming'.
        
    Attributes
    ----------
    distance_matrix_ : ndarray
        The computed distance matrix after fitting.
        
    Examples
    --------
    >>> from sekupy.analysis.rsa import RSAEstimator
    >>> from sekupy.analysis.searchlight import SearchLight
    >>> from sklearn.model_selection import StratifiedShuffleSplit
    >>> 
    >>> # Create RSA estimator
    >>> rsa_estimator = RSAEstimator(metric='euclidean')
    >>> 
    >>> # Define a custom scorer for RSA
    >>> class RSAScorer:
    ...     def __call__(self, estimator, X, y=None):
    ...         return estimator.score(X, y)
    >>> 
    >>> # Use RSA within SearchLight
    >>> analysis = SearchLight(
    ...     estimator=rsa_estimator,
    ...     radius=9.0,
    ...     scoring={'rsa': RSAScorer()},
    ...     cv=StratifiedShuffleSplit(n_splits=2, test_size=0.2),
    ...     verbose=0
    ... )
    >>> 
    >>> # Fit on your dataset
    >>> # analysis.fit(ds)
    
    Notes
    -----
    The RSAEstimator computes pairwise distances between samples, making it 
    suitable for representational similarity analysis. When used with 
    SearchLight, it analyzes the representational structure in local 
    neighborhoods of voxels/features.
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
        return _compute_rsa_score(X, metric=self.metric)
    
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
        return _compute_rsa_score(X, metric=metric)
    
    return make_scorer(_score, greater_is_better=True)
