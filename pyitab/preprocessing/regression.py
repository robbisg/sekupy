from nistats.regression import *
from pyitab.preprocessing.base import Transformer

import logging
logger = logging.getLogger(__name__)

class ResidualTransformer(Transformer):
    

    def __init__(self, name='residual', 
                 model='ols', attr='all', **kwargs):

        self.design_attr = attr
        self.model = model
        Transformer.__init__(self, name=name, model=model)

    
    def transform(self, ds, **model_kwargs):

        X = self.build_design_matrix(ds)
        # TODO: Center X columns?
        model = self.get_model(X.T, **model_kwargs)
        
        Y = ds.samples

        self.scores_ = model.fit(Y)

        ds_ = ds.copy()
        ds_.samples = self.scores_.resid
        logger.info("Residuals from GLM with %s attributes" % (', '.join(self.design_attr)))

        return ds_



    def build_design_matrix(self, ds):
        # Override in subclasses
        pass

    # TODO: Use another function / class
    def get_model(self, X, **kwargs):

        mapper = {'ols': OLSModel,
                  'ar' :  ARModel}

        return mapper[self.model](X, **kwargs)





class SampleResidualTransformer(ResidualTransformer):

    def __init__(self, name='sample_residual', **kwargs):
        ResidualTransformer.__init__(self, name=name, **kwargs)

    def build_design_matrix(self, ds):
        from sklearn.preprocessing import LabelEncoder

        if self.design_attr == 'all':
            self.design_attr = [k for k in ds.sa.keys()]

        X = []
        for k in self.design_attr:
            values = ds.sa[k].value
            if values.dtype.kind in ['U', 'S']:
                values = LabelEncoder().fit_transform(values)

            X.append(values)

        return np.vstack(X)




class FeatureResidualTransformer(ResidualTransformer):
    
    def __init__(self, name='feature_residual', **kwargs):
        ResidualTransformer.__init__(self, name=name, **kwargs)


    # TODO: Add fx and values?
    def build_design_matrix(self, ds):

        if self.design_attr == 'all':
            self.design_attr = [k for k in ds.fa.keys()]

        X = []
        for k in self.design_attr:
            mask = ds.fa[k].value != 0
            values = np.mean(ds[:, mask], axis=1)
        
            X.append(values)

        return np.vstack(X) 
