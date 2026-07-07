:orphan:

Examples Gallery
================

The following examples demonstrate the main features of **sekupy** from data
loading to multivariate analysis and result visualisation.  Each example is
self-contained and uses a small synthetic dataset so that it can be run
locally without requiring external neuroimaging data.

.. note::

   To run the examples on your own data, replace the ``make_example_dataset``
   helper with a :class:`~sekupy.io.loader.DataLoader` call pointing at your
   BIDS-organised dataset (see *Example 1* for the pattern).



.. raw:: html

    <div class="sphx-glr-thumbnails">


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Sekupy&#x27;s DataLoader abstracts the process of locating, filtering, and reading neuroimaging file...">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_01_load_data_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_plot_01_load_data.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Loading neuroimaging data with DataLoader</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Permutation testing provides a non-parametric null distribution for decoding accuracy, allowing...">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_04_decoding_permutation_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_plot_04_decoding_permutation.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Decoding with permutation testing</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The sekupy.preprocessing module provides a collection of Transformer objects that follow a tran...">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_02_preprocessing_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_plot_02_preprocessing.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Building preprocessing pipelines</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The fingerprint or identifiability analysis (Finn et al., 2015) measures how uniquely each indi...">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_07_fingerprint_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_plot_07_fingerprint.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Fingerprint (Identifiability) Analysis</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Cross-decoding tests whether a classifier trained on one condition (or time point) generalises ...">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_05_cross_decoding_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_plot_05_cross_decoding.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Cross-condition decoding (temporal generalisation)</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Representational Similarity Analysis (RSA; Kriegeskorte et al., 2008) measures the pairwise dis...">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_06_rsa_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_plot_06_rsa.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Representational Similarity Analysis (RSA)</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Sensitivity and robustness analyses require running the same analysis multiple times with varyi...">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_09_parameter_sweep_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_plot_09_parameter_sweep.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Parameter sweep with AnalysisIterator</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="RoiDecoding iterates over brain regions encoded in dataset.fa and runs a cross-validated classi...">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_03_roi_decoding_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_plot_03_roi_decoding.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">ROI-based decoding</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Clustering groups time points (or trials) of neuroimaging data into a discrete set of brain sta...">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_08_brain_states_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_plot_08_brain_states.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Brain State Clustering</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="After running any sekupy analysis pipeline, the results can be saved to disk and later loaded i...">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_10_results_analysis_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_plot_10_results_analysis.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Results loading, statistics, and visualisation</div>
    </div>


.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/plot_01_load_data
   /auto_examples/plot_04_decoding_permutation
   /auto_examples/plot_02_preprocessing
   /auto_examples/plot_07_fingerprint
   /auto_examples/plot_05_cross_decoding
   /auto_examples/plot_06_rsa
   /auto_examples/plot_09_parameter_sweep
   /auto_examples/plot_03_roi_decoding
   /auto_examples/plot_08_brain_states
   /auto_examples/plot_10_results_analysis



.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
