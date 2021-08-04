from pyitab.analysis.base import Analyzer
from pyitab.preprocessing.base import Transformer
from nilearn.glm.regression import OLSModel, ARModel
from pyitab.utils import get_id

class LinearModelMixin(Analyzer, Transformer):
    def __init__(self, name='residual', 
                 model='ols', attr='all', **kwargs):

        self.design_attr = attr
        self.model = model

        self.id = get_id()
        if 'id' in kwargs.keys():
            self.id = kwargs['id']

        self.num = 1
        if 'num' in kwargs.keys():
            self.num = kwargs['num']

        Transformer.__init__(self, name=name, model=model)


    def get_model(self, X, **kwargs):

        mapper = {'ols': OLSModel,
                  'ar' : ARModel}

        return mapper[self.model](X, **kwargs)