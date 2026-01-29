"""RSA Estimator wrapper for use with SearchLight and other sklearn-compatible frameworks."""

import numpy as np
from sklearn.base import BaseEstimator
from scipy.spatial.distance import pdist
from sklearn.metrics import make_scorer


def _compute_condition_averages(X, y):
    """Helper function to compute condition-averaged representations.
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Data to average by condition.
    y : array-like of shape (n_samples,)
        Condition labels for each sample.
        
    Returns
    -------
    condition_averages : ndarray of shape (n_conditions, n_features)
        Averaged representations for each condition.
    unique_conditions : ndarray of shape (n_conditions,)
        Unique condition labels.
        
    Raises
    ------
    ValueError
        If y is None, or if fewer than 2 unique conditions are present,
        or if any condition has no samples.
    """
    if y is None:
        raise ValueError("y cannot be None for RSA. Condition labels are required.")
    
    X = np.asarray(X)
    y = np.asarray(y)
    
    # Get unique conditions
    unique_conditions = np.unique(y)
    
    # Validate that we have at least 2 conditions
    if len(unique_conditions) < 2:
        raise ValueError(
            f"RSA requires at least 2 unique conditions, but only {len(unique_conditions)} found. "
            "Cannot compute pairwise distances with fewer than 2 conditions."
        )
    
    # Compute condition averages
    condition_averages = []
    for condition in unique_conditions:
        mask = y == condition
        n_samples = np.sum(mask)
        
        # Validate that condition has at least one sample
        if n_samples == 0:
            raise ValueError(f"Condition {condition} has no samples.")
        
        avg = X[mask].mean(axis=0)
        condition_averages.append(avg)
    
    condition_averages = np.array(condition_averages)
    
    return condition_averages, unique_conditions


def _compute_rsa_score(X, y, metric='euclidean'):
    """Helper function to compute RSA score from data.
    
    RSA computes distances between condition-averaged representations.
    This function groups samples by condition (y), averages within each
    condition, then computes pairwise distances between conditions.
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Data to compute score for.
    y : array-like of shape (n_samples,)
        Condition labels for each sample.
    metric : str, default='euclidean'
        Distance metric to use.
        
    Returns
    -------
    score : float
        Negative mean distance between conditions.
    """
    condition_averages, _ = _compute_condition_averages(X, y)
    
    # Compute pairwise distances between condition averages
    distance = pdist(condition_averages, metric=metric)
    return -np.mean(distance)


class RSAEstimator(BaseEstimator):
    """Estimator wrapper for RSA that can be used within SearchLight.
    
    This class wraps RSA (Representational Similarity Analysis) functionality 
    to make it compatible with scikit-learn's estimator interface. RSA computes
    distances between condition-averaged representations, making it suitable for
    analyzing how different experimental conditions are represented.
    
    The estimator groups samples by their condition labels (y), averages samples
    within each condition, then computes pairwise distances between these
    condition averages.
    
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
        The computed distance matrix between condition averages after fitting.
    condition_averages_ : ndarray
        The averaged representations for each condition.
    unique_conditions_ : ndarray
        The unique condition labels.
        
    Examples
    --------
    >>> from sekupy.analysis.rsa import RSAEstimator
    >>> from sekupy.analysis.searchlight import SearchLight
    >>> from sklearn.model_selection import StratifiedShuffleSplit
    >>> import numpy as np
    >>> 
    >>> # Create RSA estimator
    >>> rsa_estimator = RSAEstimator(metric='euclidean')
    >>> 
    >>> # Fit with condition labels (y is required)
    >>> X = np.random.randn(10, 5)  # 10 samples, 5 features
    >>> y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2])  # 3 conditions
    >>> rsa_estimator.fit(X, y)
    >>> 
    >>> # Define a custom scorer for RSA
    >>> class RSAScorer:
    ...     def __call__(self, estimator, X, y):
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
    The RSAEstimator computes pairwise distances between condition-averaged
    representations. This is the standard approach in Representational Similarity
    Analysis, where the goal is to understand how different experimental conditions
    are represented, not individual trials. When used with SearchLight, it analyzes 
    the representational structure in local neighborhoods of voxels/features.
    
    References
    ----------
    Kriegeskorte, N., Mur, M., & Bandettini, P. (2008). Representational 
    similarity analysis - connecting the branches of systems neuroscience. 
    Frontiers in Systems Neuroscience, 2, 4.
    """
    
    def __init__(self, metric='euclidean'):
        self.metric = metric
        
    def fit(self, X, y=None):
        """Fit the RSA estimator.
        
        Computes the distance matrix for condition-averaged training data.
        RSA groups samples by their condition labels (y), averages samples
        within each condition, then computes pairwise distances between
        these condition averages.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Condition labels for each sample. Required for RSA.
            Must contain at least 2 unique conditions.
            
        Returns
        -------
        self : RSAEstimator
            Fitted estimator.
            
        Raises
        ------
        ValueError
            If y is None, or if fewer than 2 unique conditions are present.
        """
        self.condition_averages_, self.unique_conditions_ = _compute_condition_averages(X, y)
        
        # Compute pairwise distances between condition averages
        self.distance_matrix_ = pdist(self.condition_averages_, metric=self.metric)
        
        return self
    
    def predict(self, X, y):
        """Predict method for compatibility with sklearn scoring.
        
        For RSA, we compute distances between condition-averaged samples.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to predict on.
        y : array-like of shape (n_samples,)
            Condition labels for each sample. Required for RSA.
            Must contain at least 2 unique conditions.
            
        Returns
        -------
        distances : ndarray of shape (n_conditions * (n_conditions - 1) / 2,)
            Condensed distance matrix between condition averages.
            
        Raises
        ------
        ValueError
            If y is None, or if fewer than 2 unique conditions are present.
        """
        condition_averages, _ = _compute_condition_averages(X, y)
        return pdist(condition_averages, metric=self.metric)
    
    def score(self, X, y=None):
        """Compute a score for the RSA analysis.
        
        This computes the negative mean distance between condition averages.
        RSA groups samples by condition labels, averages within each condition,
        then computes pairwise distances between conditions.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data.
        y : array-like of shape (n_samples,)
            Condition labels. Required for RSA.
            Must contain at least 2 unique conditions.
            
        Returns
        -------
        score : float
            Negative mean distance between conditions (higher is better).
            
        Raises
        ------
        ValueError
            If y is None, or if fewer than 2 unique conditions are present.
        """
        return _compute_rsa_score(X, y, metric=self.metric)
    
    def transform(self, X, y):
        """Transform data to distance matrix representation.
        
        Computes distances between condition-averaged samples.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to transform.
        y : array-like of shape (n_samples,)
            Condition labels for each sample. Required for RSA.
            Must contain at least 2 unique conditions.
            
        Returns
        -------
        distances : ndarray of shape (n_conditions * (n_conditions - 1) / 2,)
            Condensed distance matrix between condition averages.
            
        Raises
        ------
        ValueError
            If y is None, or if fewer than 2 unique conditions are present.
        """
        condition_averages, _ = _compute_condition_averages(X, y)
        return pdist(condition_averages, metric=self.metric)


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
        return _compute_rsa_score(X, y, metric=metric)
    
    return make_scorer(_score, greater_is_better=True)
