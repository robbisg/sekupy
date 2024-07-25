Preprocessing
=============
This package is used to manipulate :class:`~mvpa2.datasets.base.Dataset`.


Base classes
------------

.. automodule:: sekupy.preprocessing.base
   :members:
   :undoc-members:



Transformers
------------

* :doc:`balancing` : balance samples in the Dataset.
* :doc:`connectivity` : transform dataset for connectivity analyses
* :doc:`filters` : time filtering.
* :doc:`functions` : 
* :doc:`mapper` : list of all :class:`~sekupy.preprocessing.base.Transformer` s 
* :doc:`math` : transformation based on mathematical formulas (e.g. fisher transformation)
* :doc:`memory` : for memory management
* :doc:`normalizers` : normalize features or samples.
* :doc:`pipelines` : used to concatenate different :class:`~sekupy.preprocessing.base.Transformer` s
* :doc:`regression` : transform based on regression and orthogonalization
* :doc:`sklearn` : wrapper for `sklearn <https://scikit-learn.org>`_ package
* :doc:`slicers` : used to slice the dataset based on attributes

