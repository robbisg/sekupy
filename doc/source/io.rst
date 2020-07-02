I/O package
=====================
These package is used to perform operation of reading and writing
information about data you need to use.

The data must be organized in a standard way.
```
experiment_folder/
-- 0_results/ where analysis results will be placed
-- derivatives/ same as above (for compatibility with BIDS data format)
-- subjN/ single subject folder
---- 
```

Loader
------
Main class for data loading is `DataLoader`

.. autoclass:: pyitab.io.loader.DataLoader
   :members:
   :inherited-members:

