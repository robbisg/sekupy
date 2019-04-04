import numpy as np
import os

from nilearn.image.resampling import coord_transform
from nilearn import masking


from sklearn.metrics.scorer import _check_multimetric_scoring
from sklearn.svm import SVC
from sklearn.preprocessing.label import LabelEncoder
from sklearn.model_selection._split import LeaveOneGroupOut
from sklearn.pipeline import Pipeline

from pyitab.ext.nilearn.searchlight import search_light
from pyitab.ext.nilearn.utils import _get_affinity, check_proximity
from pyitab.ext.nilearn.utils import load_proximity, save_proximity
from pyitab.analysis.base import Analyzer

from pyitab.utils.dataset import get_ds_data
from pyitab.utils.image import save_map
from pyitab.utils.files import make_dir

import logging
logger = logging.getLogger(__name__)



def get_seeds(ds, radius):
    
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
    mask_img : Niimg-like object
        See http://nilearn.github.io/manipulating_images/input_output.html
        boolean image giving location of voxels containing usable signals.

    process_mask_img : Niimg-like object, optional
        See http://nilearn.github.io/manipulating_images/input_output.html
        boolean image giving voxels on which searchlight should be
        computed.

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
    """

    def __init__(self, 
                 radius=9.,
                 estimator=None,
                 n_jobs=1, 
                 scoring='accuracy', 
                 cv=LeaveOneGroupOut(), 
                 permutation=0,
                 verbose=1,
                 save_partial=False):

        if estimator is None:
            estimator = Pipeline(steps=[('clf', SVC(C=1, kernel='linear'))])

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
        
        Analyzer.__init__(self, name='searchlight')



    def fit(self, ds, cv_attr='chunks'):
        """
        Fit the searchlight
        """
        
        A = get_seeds(ds, self.radius)
        
        estimator = self.estimator
            
        self.scoring, _ = _check_multimetric_scoring(estimator, 
                                                     scoring=self.scoring)
        
        X, y = get_ds_data(ds)
        y = LabelEncoder().fit_transform(y)


        if cv_attr != None:
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

        self._info = self._store_ds_info(ds, cv_attr=cv_attr, test_order=splits)

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

    

    def _save_image(self, path, image, score, n_permutation, img_type, fx):

        reverse = self._info['a'].mapper.reverse1
        affine = self._info['a'].imgaffine

        filename = "%s_perm_%04d_%s.nii.gz" %(score, n_permutation, img_type)
        logger.info("Saving %s" , filename)
        filename = os.path.join(path, filename)
        image = fx(image)
        save_map(filename, reverse(image), affine)





    def save(self, path=None, operations={"cv": lambda x: x,
                                          "avg": lambda x: np.mean(x, axis=1)}):
        """This function is used to store searchlight images on disk.
        
        Parameters
        ----------
        path : string, optional
            destination path of files (the default is None, but look at AnalysisPipeline documentation)
        operations : dict, optional
            List of operation to be performed on data.
            The default is {"cv": lambda x: x,"avg": lambda x: np.mean(x, axis=1)}, which imply that
            a full image with label "cv" and an average image with label "avg" are stored.
            You can do several operation by defining a function and adding it to the dictionary
        """

        
        path = Analyzer.save(self, path)


        for i in range(len(self.scores)): # Permutation
            for score, image in self.scores[i].items():
                if self.save_partial:
                    self._clean_partial(score)

                for key, fx in operations.items():
                    self._save_image(path, image, score, i, key, fx)                  
    
    
    def _get_analysis_info(self):
        
        info = Analyzer._get_analysis_info(self)
        info['radius'] = self.radius
        
        return info
        
   