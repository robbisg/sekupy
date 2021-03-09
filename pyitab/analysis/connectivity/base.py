from pyitab.analysis.base import Analyzer


class Connectivity(Analyzer):
    def __init__(self, name='analyzer', **kwargs):
        return Analyzer.__init__(name, **kwargs)()

    def fit(self, ds, **kwargs):
        return Analyzer.fit(ds, **kwargs)()

    def save(self, path=None):
        return Analyzer.save(path)()
