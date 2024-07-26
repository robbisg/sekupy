from sekupy.mixin import LinearModelMixin
from nilearn.glm.regression import OLSModel, ARModel
from scipy.io import savemat

import patsy
import pandas as pd
import numpy as np

import logging
logger = logging.getLogger(__name__)


class LinearModel(LinearModelMixin):

    def __init__(self, name='linear_model', model='ols', 
                 attr='all', **kwargs):
        
        return super().__init__(name=name, 
                                attr=attr,
                                model=model, 
                                **kwargs)


    def fit(self, ds, formula='targets', contrast=None, 
            full_model=False, prepro=None, **model_kwargs):
        # TODO: Add the possibility to use a X design matrix
        """[summary]

        Parameters
        ----------
        ds : [type]
            [description]
        formula : str, optional
            [description], by default 'targets'. Do not include y term.
        contrast : dict, optional
            a dictionary with the key starting with 'f+' or 't+' to indicate which test should be used
            and then a keyword representing the contrast (e.g. 'f+omnibus' or 't+agevsdexterity'), the value
            must be the contrast matrix. by default None
        full_model : boolean, optional
            Indicates whether to use patsy styled formula or pd.get_dummies() (default: False or use patsy)
        prepro : [type], optional
            [description], by default None
        """

        print(self.design_attr)
        if self.design_attr == 'all':
            self.design_attr = [k for k in ds.sa.keys()]
        
        data = {k: ds.sa[k].value for k in self.design_attr}

        # TODO : Remove mean from each feature using prepro
        Y = ds.samples

        if full_model:
            dm = pd.get_dummies(pd.DataFrame(data))
            self._formula = "+".join(self.design_attr)
            columns = dm.columns
        else:
            dm = patsy.dmatrix(formula, data)
            self._formula = formula
            columns = dm.design_info.column_names
        
        X = np.asarray(dm)
        model = self.get_model(X, **model_kwargs)

        results = model.fit(Y)

        self.scores = results
        self.scores.design_info = columns

        if contrast is not None:
            self._contrast(contrast)

        strp_formula = formula.replace(" ", "").replace("-", "+")

        return super().fit(ds, formula=strp_formula)



    def _contrast(self, contrast):
        """[summary]

        Parameters
        ----------
        contrast : dict, optional
            a dictionary with the key starting with 'f+' or 't+' to indicate which test should be used
            and then a keyword representing the contrast (e.g. 'f+omnibus' or 't+agevsdexterity'), the value
            must be the contrast matrix. by default None
        """
        from scipy.stats import f, t

        self.scores.stats_contrasts = dict()

        for c, (test, matrix) in enumerate(contrast.items()):
            if 't+' in test:
                stats = self.scores.Tcontrast(matrix=matrix)
                t_dist = t(stats.df_den)
                stats.p_values = 1 - t_dist.cdf(stats.t)

            elif 'f+' in test:
                stats = self.scores.Fcontrast(matrix=matrix)
                f_dist = f(stats.df_num, stats.df_den)
                stats.p_values = 1 - f_dist.cdf(stats.F)              

            else:
                continue

            self.scores.stats_contrasts[test+str(c+1)] = stats.__dict__

        return

    # Only in subclasses
    def _get_analysis_info(self):

        info = super()._get_analysis_info()
        info['formula'] = self._info['formula'].replace(" ", "").replace("-","+")

        return info


    def save(self, path=None, fields=None, **kwargs):
        """Save the results
        
        Parameters
        ----------
        path : str, optional
            path where to store files (the default is 
            set up by :class:`sekupy.analysis.Analyzer`)
        """
        
        import os

        if fields is None:
            fields = ['stats_contrasts', 'MSE', 'theta', 
                      'r_square', 'design_info']

        path, prefix = super().save(path=path, **kwargs)
        kwargs.update({'prefix': prefix})

        mat_score = dict()

        for field in fields:
            if hasattr(self.scores, field):
                mat_score[field] = getattr(self.scores, field)

        filename = self._get_filename(**kwargs)
        logger.info("Saving %s" % (filename))
            
        savemat(os.path.join(path, filename), mat_score)
                
        return


    def _get_filename(self, **kwargs):
        "target-<values>_id-<datetime>_mask-<mask>_value-<roi_value>_data.mat"
        params = {}

        # TODO: Solve empty prefix
        prefix = kwargs.pop('prefix')
        trailing = "stats-glm"
        filename = "%s_%s_data.mat" % (prefix, trailing)

        return filename
