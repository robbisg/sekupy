"""
from mvpa_itab.io.loader import DataLoader
from mvpa_itab.preprocessing.pipelines import PreprocessingPipeline
from sklearn.model_selection._split import GroupShuffleSplit, StratifiedKFold
from mvpa_itab.pipeline.decoding.roi_decoding import Decoding
from sklearn.pipeline import Pipeline
from sklearn.feature_selection.univariate_selection import SelectKBest
from sklearn.svm.classes import SVC
from mvpa_itab.pipeline.iterator import AnalysisIterator
from mvpa_itab.pipeline.decoding import AnalysisPipeline
from mvpa_itab.pipeline.script import ScriptConfigurator
loader = DataLoader(configuration_file="/media/robbis/DATA/fmri/carlo_ofp/ofp.conf", task='OFP')
ds = loader.fetch()
_default_options = {
       
                       'sampleslicer__evidence' : [[1], [2], [3]],
                       'cv__n_splits': [3, 5],
                        }
_default_conf = {
       
                       'prepro':['targettrans', 'sampleslicer'],
                       #'sampleslicer__evidence': ['1'], 
                       'targettrans__target':"decision",
                       
                       'estimator': [('svr', SVC(C=1, kernel='linear'))],
                       
                       'cv': StratifiedKFold,
                       'cv__n_splits': 5,
                       
                       'scores' : ['accuracy'],
                       
                       'analysis': Decoding,
                       'cv_attr': 'subject',
                       'kwargs__roi':['lateral_ips'],
                       'kwargs__prepro':['featurenorm', 'samplenorm'],
                       'analysis__n_jobs':2,
                       'analysis__permutation':2,
                   
                   }


iterator = AnalysisIterator(_default_options, ScriptConfigurator(**_default_conf))

for conf in iterator:
    kwargs = conf._get_kwargs()
    a = AnalysisPipeline(conf, name="regression").fit(ds, **kwargs)
    a.save()


class TestAnalysis(unittest.TestCase):

    def test_iterator(self):
        return

    def test_configurator(self):
        return

    def test_pipeline(self):
        return
"""

