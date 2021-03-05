from pyitab.preprocessing.base import Transformer
from pyitab.preprocessing import Detrender, TargetTransformer, \
     FeatureStacker, SampleTransformer, TemporalTransformer, Resampler

from pyitab.preprocessing.slicers import FeatureSlicer, SampleSlicer, DatasetMasker
from pyitab.preprocessing.normalizers import FeatureZNormalizer, \
    SampleZNormalizer, SampleSigmaNormalizer, FeatureSigmaNormalizer, \
    DatasetFxNormalizer, SampleFxNormalizer
from pyitab.preprocessing.balancing.base import Balancer
from pyitab.preprocessing.balancing.imbalancer import Imbalancer
from pyitab.preprocessing.math import ZFisherTransformer, \
    AbsoluteValueTransformer, SignTransformer
from pyitab.preprocessing.memory import MemoryReducer
from pyitab.preprocessing.regression import FeatureResidualTransformer, \
    SampleResidualTransformer
from pyitab.simulation.autoregressive import PhaseDelayedModel, AutoRegressiveModel, \
    TimeDelayedModel
from pyitab.simulation.connectivity import ConnectivityStateSimulator
from pyitab.preprocessing.filter import ButterFilter
from pyitab.preprocessing.connectivity import SlidingWindowConnectivity
from pyitab.preprocessing.sklearn import ScikitWrapper



def function_mapper(name):

    mapper = {
              'detrender': Detrender,
              'dataset_masker': DatasetMasker,
              'target_transformer': TargetTransformer,
              'sample_transformer': SampleTransformer,
              'feature_slicer': FeatureSlicer,
              'sample_slicer': SampleSlicer,
              'sample_znormalizer': SampleZNormalizer,
              'feature_znormalizer': FeatureZNormalizer,
              'sample_sigmanorm': SampleSigmaNormalizer,
              'feature_sigmanorm': FeatureSigmaNormalizer,
              'ds_normalizer': DatasetFxNormalizer,
              'sample_normalizer': SampleFxNormalizer,
              'feature_stacker': FeatureStacker,
              'balancer': Balancer,
              'imbalancer': Imbalancer,
              'abs': AbsoluteValueTransformer,
              'sign': SignTransformer,
              'zfisher': ZFisherTransformer,
              'none': Transformer,
              'memory_reducer': MemoryReducer,
              'feature_residual': FeatureResidualTransformer,
              'sample_residual': SampleResidualTransformer,
              'temporal_transformer': TemporalTransformer,
              'connectivity_state_simulator': ConnectivityStateSimulator,
              'autoregressive_model': AutoRegressiveModel,
              'phase_delayed_model': PhaseDelayedModel,
              'time_delayed_model': TimeDelayedModel,
              'sliding_window_connectivity': SlidingWindowConnectivity,
              'butter_filter': ButterFilter,
              'resampler': Resampler,
              'scikit_wrapper': ScikitWrapper,
              }
    
    return mapper[name]
