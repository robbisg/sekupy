import numpy as np
import os

from nilearn.image.resampling import coord_transform
from nilearn import masking

from sklearn.metrics._scorer import _check_multimetric_scoring
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection._split import LeaveOneGroupOut
from sklearn.pipeline import Pipeline

from pyitab.ext.nilearn.searchlight import search_light
from pyitab.ext.nilearn.utils import _get_affinity, check_proximity
from pyitab.ext.nilearn.utils import load_proximity, save_proximity
from pyitab.analysis.base import Analyzer

from pyitab.utils.dataset import get_ds_data
from pyitab.utils.image import save_map
from pyitab.utils.files import make_dir
from pyitab.analysis.utils import get_params

import logging
logger = logging.getLogger(__name__)



def get_seeds(ds, radius):
    """This function is used to load the affinity matrix used
    for searchlight analysis. Given the mask it loads the affinity
    matrix, if mask and radius are the same, or builds and
    saves the matrix on disk if not.
    
    Parameters
    ----------
    ds : [type]
        [description]
    radius : [type]
        [description]
    
    Returns
    -------
    [type]
        [description]
    """
    
    if check_proximity(ds, radius):
        return load_proximity(ds, radius)
    
    
    # Get the seeds
    process_mask_coords = ds.fa.voxel_indices.T
    process_mask_coords = coord_transform(
        process_mask_coords[0], process_mask_coords[1],
        process_mask_coords[2], ds.a.imgaffine)
    process_mask_coords = np.asarray(process_mask_coords).T
    
    seeds = process_mask_coords
    coords = ds.fa.voxel_indices
    logger.info("Building proximity matrix...")
    A = _get_affinity(seeds, coords, radius, allow_overlap=True, affine=ds.a.imgaffine)
    
    save_proximity(ds, radius, A)
    
    return A



class SearchLight(Analyzer):
    """Implement search_light analysis using an arbitrary type of classifier.
    This is a wrapper of the nilearn algorithm to work with pymvpa dataset.

    Parameters
    -----------

    radius : float, optional
        radius of the searchlight ball, in millimeters. Defaults to 2.

    estimator : 'svr', 'svc', or an estimator object implementing 'fit'
        The object to use to fit the data

    n_jobs : int, optional. Default is -1.
        The number of CPUs to use to do the computation. -1 means
        'all CPUs'.

    scoring : string or callable, optional
        The scoring strategy to use. See the scikit-learn documentation
        If callable, takes as arguments the fitted estimator, the
        test data (X_test) and the test target (y_test) if y is
        not None.

    cv : cross-validation generator, optional
        A cross-validation generator. If None, a 3-fold cross
        validation is used or 3-fold stratified cross-validation
        when y is supplied.

    verbose : int, optional
        Verbosity level. Defaut is False

    save_partial : bool, default is False
        A boolean indicating whether to save partial maps or not.
        (Not well tested!)


    """

    def __init__(self, 
                 radius=9.,
                 estimator=Pipeline(steps=[('clf', SVC(C=1, kernel='linear'))]),
                 n_jobs=1, 
                 scoring='accuracy', 
                 cv=LeaveOneGroupOut(), 
                 permutation=0,
                 verbose=1,
                 save_partial=False,
                 **kwargs,
                 ):

        if not isinstance(estimator, Pipeline):
            estimator = Pipeline(steps=[('clf', estimator)])
        
        self.radius = radius
        self.estimator = estimator
        self.n_jobs = n_jobs
        self.permutation = permutation
        self.save_partial = save_partial
        
        self.scoring = scoring
        
        self.cv = cv
        self.verbose = verbose
        
        Analyzer.__init__(self, name='searchlight', **kwargs)



    def fit(self, ds, cv_attr='chunks'):
        """This function fit the searchlight given the dataset
        and the attribute used for cross_validation.
        
        Parameters
        ----------
        ds : pymvpa Dataset,
            The dataset used for classification
        cv_attr : str, optional
            The attribute used by cross-validation to chunk data,
            by default 'chunks'
        """
        
        A = get_seeds(ds, self.radius)
        
        estimator = self.estimator
        
        if isinstance(self.scoring, str):
            self.scoring = [self.scoring]
            
        self.scoring = _check_multimetric_scoring(estimator, 
                                                     scoring=self.scoring)
        
        X, y = get_ds_data(ds)
        y = LabelEncoder().fit_transform(y)

        if cv_attr is not None:
            if isinstance(cv_attr, list):
                groups = np.vstack([ds.sa[att].value for att in cv_attr]).T
            else:
                groups = ds.sa[cv_attr].value

        values = []
        indices = self._get_permutation_indices(len(y))
        
        save_partial = self.save_partial

        for n, idx in enumerate(indices):
            y_ = y[idx]
            
            # This is used to preserve partial good
            # files in case of permutations
            if n != 0: 
                save_partial = False

            scores = search_light(X, y_, estimator, A, groups,
                                  self.scoring, self.cv, self.n_jobs,
                                  self.verbose, save_partial)
            
            values.append(scores)
        
        self.scores = values
        splits = self._split_name(X, y, self.cv, groups)
        self._info = self._store_info(ds, cv_attr=cv_attr, test_order=splits)

        return self


    def _split_name(self, X, y, cv, groups):

        if len(groups.shape) == 1:
            groups = np.vstack((groups, groups)).T

        if len(X.shape) == 3:
            X = X[..., 0]

        # TODO: Bug if group is a list!
        split = [np.unique(groups[:,1][test])[0] for train, test in cv.split(X, y=y, groups=groups[:,1])]
        return split


    def _clean_partial(self, score):
        # TODO: I should only remove files or not?
        n_files = self.n_jobs
        if self.n_jobs == -1:
            from joblib import cpu_count
            n_files = cpu_count()
        
        for i in range(n_files):
            try:
                os.remove("%s_%4d.temp" %(score, i+1))
            except OSError:
                pass

    
    def _save_image(self, path, image, score, n_permutation, suffix, fx, **kwargs):

        reverse = self._info['a'].mapper.reverse1
        affine = self._info['a'].imgaffine
        
        params = {
              'score': score,
              'perm': "%04d" % (n_permutation),
              'suffix': suffix,
        }

        kwargs.update(params)
        filename = self._get_filename(**kwargs)

        #filename = "%s_perm_%04d_%s.nii.gz" %(score, n_permutation, suffix)
        logger.info("Saving %s" , filename)
        filename = os.path.join(path, filename)
        image = fx(image)
        save_map(filename, reverse(image), affine)


    def save(self, path=None, operations={"full": lambda x: x,
                                          "mean": lambda x: np.mean(x, axis=1)}, **kwargs):
        """This function is used to store searchlight images on disk.
        
        Parameters
        ----------
        path : string, optional
            destination path of files (the default is None, 
            but look at AnalysisPipeline documentation)
        operations : dict, optional
            List of operation to be performed on data.
            The default is {"full": lambda x: x,"avg": lambda x: np.mean(x, axis=1)}, 
            which imply that a full image with label "full" 
            and an across-folds average image with label "mean" are stored.
            You can do several operation by defining a function 
            and adding it to the dictionary
        """
   
        path, prefix = Analyzer.save(self, path, **kwargs)
        kwargs['prefix'] = prefix


        for i in range(len(self.scores)): # Permutation
            for score, image in self.scores[i].items():
                if self.save_partial:
                    self._clean_partial(score)

                for key, fx in operations.items():
                    self._save_image(path, image, score, i, key, fx, **kwargs)                  
    
    
    def _get_analysis_info(self):
        
        info = Analyzer._get_analysis_info(self)
        info['radius'] = self.radius
        
        return info
        

    def _get_filename(self, **kwargs):
        ""
        logger.debug(kwargs)
       
        params = dict()
        if len(kwargs.keys()) != 0:

            for keyword in ["sample_slicer", "target_transformer", "sample_transformer"]:
                if keyword in kwargs['prepro']:
                    params_ = get_params(kwargs, keyword)

                    if keyword == "sample_slicer":
                        params_ = {k: "+".join([str(v) for v in value]) 
                                   for k, value in params_.items()}
                    
                    if keyword == "sample_transformer":
                        params_ = {k: "+".join([str(v) for v in value]) 
                                   for k, value in params_['attr'].items()}                   
                    
                    params.update(params_)
        else:
            params['targets'] = "+".join(list(self._info['sa']['targets']))
        
        kwargs.update(params)

        logger.debug(params)
        
        # TODO: Solve empty prefix
        prefix = kwargs.pop('prefix')
        suffix = kwargs.pop('suffix')
        
        params_keys = list(params.keys())
        params_keys += ['score', 'perm']
        logger.debug(params_keys)
        midpart = "_".join(["%s-%s" % (k, str(kwargs[k]).replace("_", "+"))
                            for k in params_keys])
        
        filename = "%s.nii.gz" % ("_".join([prefix, midpart, suffix]))

        return filename
   