# sekupy

![example workflow](https://github.com/robbisg/sekupy/actions/workflows/test.yaml/badge.svg)
[![codecov](https://codecov.io/gh/robbisg/sekupy/branch/master/graph/badge.svg)](https://codecov.io/gh/robbisg/sekupy)
[![Documentation Status](https://readthedocs.org/projects/sekupy/badge/?version=latest)](https://sekupy.readthedocs.io/en/latest/?badge=latest)
[![Project Status: WIP – Initial development is in progress, but there has not yet been a stable, usable release suitable for the public.](https://www.repostatus.org/badges/latest/wip.svg)](https://www.repostatus.org/#wip)
[![CodeFactor](https://www.codefactor.io/repository/github/robbisg/sekupy/badge)](https://www.codefactor.io/repository/github/robbisg/sekupy)

`sekupy` is a python-package created for deterging your (dirty) (and) (multivariate) neuroimaging analyses. The package has been thought for
decoding analyses but it includes also basic univariate analyses.

It has some utilities to vary sets of parameters of the analyses without struggling with `for` and `if` statements.

It deterges your results, by saving them in a safe manner, by also keeping in mind BIDS.

`sekupy` is the deterged version of `pyitab`.

# Documentation

The documention can be found [here](https://sekupy.readthedocs.io/).

# Install
The package isn't yet on `pip`.
You can install it by using:
```
python setup.py install
```

# Example
The main idea is to use a dictionary to configure all parameters of your analysis, feed the configuration into an ```AnalysisPipeline``` object, call ```fit``` to obtain results, then ```save``` to store in a ```BIDS```-ish way.

For example if we want to perform a ```RoiDecoding``` analysis using some preprocessing steps we will have a script like this (this is not a complete example):
```python
from sekupy.analysis.configurator import AnalysisConfigurator
from sekupy.analysis.pipeline import AnalysisPipeline
from sekupy.analysis.decoding.roi_decoding import RoiDecoding

_default_config = {
                    # Here we specifiy that we have to transform the dataset labels
                    # then select samples and then balance data
                    'prepro': ['target_transformer', 'sample_slicer', 'balancer'],
                    
                    # Here we set which attribute to choose (dataset is a pymvpa dataset)
                    'target_transformer__attr': "image_type",
                    # Here we select samples with a image_type equal to I or O and evidence equal to 1
                    'sample_slicer__attr': {'image_type':["I", "O"], 'evidence':[1]},
                    # Then we say that we want to balance image_type at subject-level
                    "balancer__attr": 'subject',

                    # We setup the estimator in a sklearn way
                    'estimator': [
                        ('fsel', SelectKBest(k=50)),
                        ('clf', SVC(C=1, kernel='linear'))],
                    'estimator__clf__C': 1,
                    'estimator__clf__kernel': 'linear',
                    
                    # Then the cross-validation object (also sklearn)
                    'cv': LeaveOneGroupOut,
                    
                    'scores': ['accuracy'],
                    
                    # Then the analysis
                    'analysis': RoiDecoding,
                    'analysis__n_jobs': -1,
                    
                    'analysis__permutation': 0,
                    
                    'analysis__verbose': 0,
                    
                    # Here we say that we want use the region with value 1 in image+type mask
                    'kwargs__roi_values': [('image+type', [1]), ('image+type', [2]), ('image+type', [3]),
                                            ('image+type', [4]), ('image+type', [5])],
                    
                    # We want to use subject for our cross-validation
                    'kwargs__cv_attr': 'subject'
                    }

configuration = AnalysisConfigurator(**_default_config), 
                                     kind='configuration') 
kwargs = configuration._get_kwargs() 
a = AnalysisPipeline(conf, name="roi_decoding_across_full").fit(ds, **kwargs) 
a.save() 
```
Surf the code, starting from classes used here!!
