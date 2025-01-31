---
title: 'Sekupy: a tool for build clean pipelines in neuroimaging.'
tags:
  - Python
  - Pipelines
  - Machine Learning 
authors:
  - name: Roberto Guidotti 
    orcid: 
    affiliation: 1
affiliations:
  - name: Department of Neuroscience, Imaging and Clinical Sciences, University "G. D'Annunzio" Chieti-Pescara, Italy 
date: 1 September 2021 
bibliography: paper.bib
---

# Abstract


# Introduction
Current neuroimaging research relies on a broad range of software packages that are developed by research groups and driven by their research goals. Moreover, pushed by the quest of reproducibility and open science lead to releasing software freely. A recent effort, towards reproducibility have been made to standardize data organization and build tools that offers the possibility to integrate classical tools and perform automatic analyses. In 
Python language fueled this process due to the ability to offer a solid scientific environment and the possibility to integrate software and automatize building and testing activities.
This advantages lead to a bloom of neuroimaging packages that allow researchers to read, process and run complicated analysis pipeline with few lines of code. Moreover, the possibility to use the Object Oriented Paradigm has lead to the possibility to reuse code and share effective and powerful APIs. In addition, there has been a huge advancement in the set of methods proposed by neuroimaging researchers that lead to release code with different design philosophy and APIs. 
Moreover, these packages implement effectively complicated analyses, while other aspects such as loading and storing results and easily access derivatives for further statistical tests are left to other packages.
Unfortunately, this led to a difficulty in sharing analysis pipeline which are clear to understand at a first sight and without going into the details of the released package. Indeed, despite many effective efforts have been made to reduce this gap, in some cases, it is difficult to integrate and keep the scripts and pipelines clean.
In the last few years, we tried to build a tool that tries to solve this task and reduce the gap from general purpose automatic pipelines that have a broader spectrum of possibility, to cutting-edge analyses that need technical skills to code them.

Sekupy is a Python package that aims at solving some of these problems. Indeed it allows the integration of various tools in neuroimaging analyses, such as nilearn, scikit-learn, mne-python. With Sekupy, users can easily combine different methods and techniques, allowing for a comprehensive and flexible analysis pipeline. 

One of the key features of Sekupy is its interface, based on scikit-learn fit-transform paradigm, which enables researchers to effortlessly write their own methods and algorithms. It also introduce the save function which offers a way to safely and easily save results to the disk. The realm of sekupy is the decoding analysis and it allows to perform several flavours of this analyses, but it includes different tools for other multivariate analyses such as RSA, Fingerprint and State analysis.

Furthermore, Sekupy allows for the execution of analyses with different parameters. This flexibility enables researchers to explore various settings and configurations, facilitating robustness and sensitivity analyses. 

Overall, Sekupy provides a comprehensive toolkit for neuroimaging analyses, offering integration of different tools, an easy interface for custom methods, efficient result storage, and the ability to run analyses with different parameters. With Sekupy, researchers can streamline their analyses, enhance reproducibility, and gain valuable insights from their neuroimaging data.


# Materials and methods

## Data loading

A main feature of sekupy is the possibility to load datasets easily. The users needs to write the reader for a single file and organizing data in a BIDS way and sekupy does the rest. The `DataLoader` provides the interface for loading data from disk and also filtering files that should not be loaded for further analyses.
Data can be organized in different ways, but BIDS format is strongly suggested. One of the main issues with data organization is that sekupy is suited for decoding analyses which are mainly made with derivatives datasets such as fMRI beta values from GLM (cit) or connectivity analyses (cit), in this framework given the lack of derivatives full standardization we allowed some detour from current version, but the philosophy is to use BIDS-ish filename and store it with the same rules.

Example of dirfile.

For example in figure, we showed how a dataset should be organized to then be digested by sekupy. 
Now that the dataset is organized we can use the `DataLoader` class to prepare loading.

```python
loader = DataLoader(configuration_file=conf_file,
                    data_path="./hcp-bids-connectivity/,
                    subjects="./participants.tsv",
                    loader='bids-meg',
                    task='blp',
                    bids_atlas="complete",
                    bids_correction="corr",
                    bids_derivatives='True',
                    load_fx='hcp-blp')
```
In the example, we build a loader that loads data from a dataset which was previously created uses the mapped function `hcp-blp` to load each single file and filters files that has `atlas-complete` and `correction-corr` pair in their filenames. Moreover it allows to search for files in the derivatives folder with a `taks-blp` keyword.

The dataset is loaded when `ds = loader.fetch()` is ran and it creates a `Dataset` object, which is a `pymvpa` class and stores several information about the dataset. The dataset is the key ingredient of `sekupy` since other analyses and preprocessing steps need a dataset.

The `fetch` function, loads the information contained in the `participants.csv` file or in another named file with subject information, then for each subject all the information and files are loaded. For fMRI data the function automatically loads the dataset and information regarding atlases, attributes from `events.csv` and so on. If you have a dataset in BIDS, but with different files you need to specifiy the loading function `load_fx`. For example, in Figure 2, we show a customized `load_fx` function.

**Specify how `fetch` performs the loading**
**Example of load_fx or loader**
**Recap of pymvpa dataset**

In Sekupy, data loading allows users to easily integrate their own data readers and enter the tool smoothly. By plugging in their own reader functions, users can convert their data into a single format, which is the pymvpa dataset. This standardized format ensures compatibility and consistency throughout the analysis pipeline.

Whether you are working with different file formats or data sources, Sekupy's flexible data loading capabilities enable you to effortlessly import and convert your data into the desired format. This eliminates the need for manual data conversion and streamlines the data loading process.

With Sekupy's data loading functionality, you can focus on your analysis without worrying about the intricacies of data conversion. By providing a smooth and efficient data loading experience, Sekupy empowers researchers to quickly and easily access their data and start performing neuroimaging analyses.


## Preprocessing

In the `preprocessing` module of sekupy, various techniques are employed to prepare neuroimaging data for further analysis. These techniques include data cleaning, normalization, and feature extraction. Sekupy provides a range of preprocessing functions and pipelines that can be easily customized to suit specific research needs. By incorporating these preprocessing steps, users can ensure the quality and reliability of their data, leading to more accurate and robust results in subsequent analyses.
In this stage the dataset is transformed in other dataset forms, so each preprocessing step needs to implement the function `transform`, which takes a `Dataset` and returns the transformed `Dataset`.

To facilitate the preprocessing workflow, Sekupy offers seamless integration with popular Python libraries such as NumPy, SciPy, and scikit-learn. This allows users to leverage the power of these libraries while benefiting from the simplicity and flexibility of Sekupy's preprocessing functionalities. Whether it's removing noise, standardizing data, or extracting relevant features, Sekupy provides a comprehensive set of tools to streamline the preprocessing stage of neuroimaging analyses.

By utilizing Sekupy's preprocessing capabilities, researchers can save valuable time and effort in preparing their data, enabling them to focus more on the core aspects of their analyses. With its user-friendly interface and extensive documentation, Sekupy empowers users to efficiently preprocess their neuroimaging data and pave the way for insightful and impactful research outcomes.

Sekupy is a python package that aims at integrating several useful tools and keep the pipelines and analyses clean


## Analyses

Sekupy is primarily focused on multivariate analyses, particularly decoding. Decoding is a powerful technique used to extract meaningful information from neuroimaging data, allowing researchers to decode patterns of brain activity and make predictions about cognitive states or stimuli. With Sekupy, researchers can easily implement and customize decoding algorithms, leveraging the package's extensive set of tools and functionalities.

In addition to decoding, Sekupy also offers several other analysis functions. One such function is fingerprint analysis, which allows researchers to identify unique patterns or signatures in their data. This can be particularly useful in studying individual differences or identifying biomarkers.

Another important analysis function provided by Sekupy is RSA (Representational Similarity Analysis). RSA enables researchers to compare and quantify the similarity between neural representations, providing insights into how the brain processes and represents information.

Lastly, Sekupy includes state analysis capabilities, which allow researchers to analyze and characterize different brain states or conditions. This can be done through techniques such as clustering, classification, or dimensionality reduction.

Overall, Sekupy provides a comprehensive suite of analysis functions, catering to a wide range of multivariate analyses such as decoding, fingerprint analysis, RSA, and state analysis. Researchers can leverage these functions to gain deeper insights into their neuroimaging data and uncover meaningful patterns and relationships.

## Results

Sekupy provides a seamless way to store results in a BIDS-compliant manner. By adhering to the BIDS format, researchers can ensure the reproducibility and compatibility of their results across different analyses and studies. This standardized approach also facilitates the sharing and collaboration of research findings.

Once the results are stored, Sekupy offers several functionalities to read and analyze the stored results. One such functionality is the ability to read the results into pandas dataframes. This allows researchers to easily manipulate and explore the results using the powerful data manipulation capabilities of pandas. With pandas, researchers can perform statistical analyses, generate summary statistics, and visualize the results using libraries like seaborn.

By leveraging the integration with pandas, Sekupy empowers researchers to gain deeper insights from their stored results. Whether it's running statistical tests, generating visualizations, or conducting exploratory data analysis, Sekupy provides the tools necessary to extract meaningful information from the stored results.

With Sekupy's support for storing results in a BIDS-compliant manner and the ability to read and analyze the results using pandas, researchers can streamline their analysis workflow and make informed decisions based on the stored results.

# Conclusion


# Acknowledgements


# References


The package is accompanied by documentation (https://sekupy.readthedocs.io/en/latest/index.html) and a number of tutorial notebooks which serve as both guides to the package as well as educational resources.

# Conclusion


# Acknowledgements


# References
