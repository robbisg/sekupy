from pyitab.analysis.base import Analyzer
from pyitab.analysis.states.base import Clustering
from pyitab.analysis.states.subsamplers import VarianceSubsampler
from pyitab.analysis.iterator import AnalysisIterator
from pyitab.analysis.configurator import AnalysisConfigurator
from pyitab.analysis.pipeline import AnalysisPipeline
from pyitab.analysis.utils import get_rois
from pyitab.preprocessing.base import Transformer
from pyitab.preprocessing import FeatureSlicer


from sklearn.cluster import KMeans

class StateAnalyzer(Analyzer):

    def __init__(self, estimator=KMeans(), **kwargs):

        self.estimator = estimator
        self._default_config = {
            'estimator': [('clf1', self.estimator)], 
            'analysis': Clustering,
        }

        Analyzer.__init__(self, name='state-analyzer')


    def fit(self, ds, n_clusters=range(2, 20), 
            est_keyword='n_clusters', **kwargs):

        options = {
            'estimator__clf1__%s' % (est_keyword) : n_clusters
        }

        iterator = AnalysisIterator(
                            options, 
                            AnalysisConfigurator,
                            config_kwargs=self._default_config,
                            )

        self.scores = list()

        for n, analysis in enumerate(iterator):

            a = AnalysisPipeline(analysis, name='state-analysis').fit(ds, **kwargs)
            self.scores.append([n_clusters[n], a._estimator.scores])

        return

    

    def score(self, metrics=['silhouette']):

        if not hasattr(self, 'scores'):
            raise Exception("Try to run fit before score.")

        from pyitab.analysis.states.metrics import metrics as mapper
        
        self.metrics = dict()

        for m in metrics:
            fx = mapper[m]
            self.metrics[m] = list()

            for scores in self.scores:
                k = scores[0]
                score = scores[1]
                labels = score['labels']
                X = score['X']
                # TODO: use a different approach for kl
                s = fx(X, labels)

                self.metrics[m].append([k, s])
        
        return


class RoiStateAnalyzer(StateAnalyzer):

    def __init__(self, estimator=KMeans(), **kwargs):

        self.estimator = estimator
        self._default_config = {
            'estimator': [('clf1', self.estimator)], 
            'analysis': Clustering,
        }

        StateAnalyzer.__init__(self, name='roi-state-analyzer')
    

    def fit(self, ds, roi='all', roi_values=None, 
            n_clusters=range(2, 20), est_keyword='n_clusters', 
            prepro=Transformer(), clustering_prepro=VarianceSubsampler(),
            **kwargs):

        if roi_values is None:
            roi_values = get_rois(ds, roi)

        kwargs['prepro'] = clustering_prepro
                
        scores = dict()
        self._subsampled = dict()
        # TODO: How to use multiple ROIs
        for r, value in roi_values:
            ds_ = FeatureSlicer(**{r: value}).transform(ds)
            ds_ = prepro.transform(ds_)

            super().fit(ds_, 
                        n_clusters=n_clusters,
                        est_keyword=est_keyword,
                        **kwargs)
            
            string_value = "+".join([str(v) for v in value])
            key = "mask-%s_value-%s" % (r, string_value)

            scores[key] = self.scores
        
        self._scores = scores
        
        return self

    
    def score(self, metrics=['silhouette']):
        
        roi_metrics = dict()
        for roi, scores in self._scores.items():

            self.scores = scores

            super().score(metrics=metrics)

            roi_metrics[roi] = self.metrics

        self._metrics = roi_metrics

        return