from mvpa_itab.preprocessing.functions import Detrender, TargetTransformer,\
    FeatureWiseNormalizer, FeatureSlicer, SampleSlicer, SampleWiseNormalizer,\
    FeatureStacker
from mvpa_itab.preprocessing.balancing.base import Balancer
from mvpa_itab.preprocessing.balancing.imbalancer import Imbalancer
from mvpa_itab.preprocessing.math import ZFisherTransformer, AbsoluteValueTransformer



def function_mapper(name):

    mapper = {
              'detrending': Detrender,
              'target_trans': TargetTransformer,
              'feature_norm': FeatureWiseNormalizer,
              'feature_slicer': FeatureSlicer,
              'sample_slicer': SampleSlicer,
              'sample_norm': SampleWiseNormalizer,
              'sample_stacker': FeatureStacker,
              'balancer': Balancer,
              'imbalancer': Imbalancer,
              'abs': AbsoluteValueTransformer,
              'zfisher': ZFisherTransformer
              }
    
    return mapper[name]