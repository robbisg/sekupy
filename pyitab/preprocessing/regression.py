from pyitab.mixin import LinearModelMixin
import numpy as np

import patsy
import logging
logger = logging.getLogger(__name__)


class SampleResidualTransformer(LinearModelMixin):

    def __init__(self, name='sample_residual', **kwargs):
        super().__init__(name=name, **kwargs)

    def transform(self, ds, **model_kwargs):

        if self.design_attr == 'all':
            self.design_attr = [k for k in ds.sa.keys()]

        data = {k: ds.sa[k].value for k in self.design_attr}

        X = []
        for k in self.design_attr:
            x = np.asarray(patsy.dmatrix(k + ' - 1', data))
            X.append(x)

        X = np.hstack(X)
        model = self.get_model(X, **model_kwargs)
        
        Y = ds.samples

        self.scores = model.fit(Y)

        ds_ = ds.copy()
        ds_.samples = self.scores.resid
        logger.info("Residuals from GLM with %s attributes" % (', '.join(self.design_attr)))

        return super().transform(ds_)




class FeatureResidualTransformer(LinearModelMixin):
    
    def __init__(self, name='feature_residual', **kwargs):
        super().__init__(name=name, **kwargs)



    def transform(self, ds, **model_kwargs):

        if self.design_attr == 'all':
            self.design_attr = [k for k in ds.fa.keys()]

        data = {k: ds.fa[k].value for k in self.design_attr}

        X = []
        for k in self.design_attr:
            x = np.asarray(patsy.dmatrix(k + ' - 1', data))
            X.append(x)
        
        X = np.hstack(X)
        model = self.get_model(X, **model_kwargs)
        
        Y = ds.samples.T

        self.scores = model.fit(Y)

        ds_ = ds.copy()
        ds_.samples = self.scores.resid.T
        logger.info("Residuals from GLM with %s attributes" % (', '.join(self.design_attr)))

        return super().transform(ds_)
