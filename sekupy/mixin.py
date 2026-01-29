from sekupy.analysis.base import Analyzer
from sekupy.preprocessing.base import Transformer
from nilearn.glm.regression import OLSModel, ARModel
from sekupy.utils import get_id

class LinearModelMixin(Analyzer, Transformer):
    """Mixin class for linear model functionality.
    
    This mixin provides linear modeling capabilities combining
    analysis and transformation functionality for neuroimaging data.
    It supports ordinary least squares (OLS) and autoregressive (AR) models.
    
    Parameters
    ----------
    name : str, optional
        Name of the linear model component, by default 'residual'
    model : str, optional
        Type of linear model ('ols' or 'ar'), by default 'ols'
    attr : str, optional
        Attribute specification for design matrix, by default 'all'
    **kwargs : dict
        Additional parameters including id and num
        
    Attributes
    ----------
    design_attr : str
        Attribute used for design matrix construction
    model : str
        Type of linear model being used
    """
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
        """Get the appropriate linear model based on configuration.
        
        This method returns a configured linear model instance based on
        the model type specified during initialization.
        
        Parameters
        ----------
        X : array-like
            Design matrix for the linear model
        **kwargs : dict
            Additional parameters for model initialization
            
        Returns
        -------
        model
            Configured linear model instance (OLSModel or ARModel)
        """

        mapper = {'ols': OLSModel,
                  'ar' : ARModel}

        return mapper[self.model](X, **kwargs)