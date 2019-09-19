from setuptools import setup, find_packages
import codecs
import os
import re

here = os.path.abspath(os.path.dirname(__file__))
def find_version(*file_paths):
    # Open in Latin-1 so that we avoid encoding errors.
    # Use codecs.open for Python 2 compatibility
    with codecs.open(os.path.join(here, *file_paths), 'r', 'latin1') as f:
        version_file = f.read()

    # The version line must have the form
    # __version__ = 'ver'
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


setup(
    name="pyitab",
    version=find_version('pyitab', '__init__.py'),
    description="Neuroimaging pipelines with python @ITAB",

    # The project URL.
    url='https://github.com/robbisg/pyitab/',

    # Author details
    author='Roberto Guidotti',
    author_email='rob.guidotti@gmail.com',

    # Choose your license
    license='MIT',

    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: MIT License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3.6',
    ],

    # What does your project relate to?
    keywords='mvpa fmri neuroimaging',

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages.
    packages=find_packages(),

    # List run-time dependencies here.  These will be installed by pip when your
    # project is installed.
    install_requires = ['nitime',
                        'nipy',
                        'nibabel',
                        'mne',
                        'imbalanced-learn',
                        'tqdm',
                        'scikit-learn',
                        'pymvpa2',
                        'seaborn',
                        'pandas',
                        'sentry-sdk'
                        ],

    # If there are data files included in your packages that need to be
    # installed, specify them here.  If using Python 2.6 or less, then these
    # have to be included in MANIFEST.in as well.
    package_data={
       'pyitab.io': ['data/*/*/*.*'],
    },


    # Included in external package
    #dependency_links=['http://github.com/robbisg/scikit-learn/tarball/group_cv#egg=package-1.0',
    #                  'http://github.com/robbisg/nilearn/tarball/scores_sl#egg=package-1.0'],

    # Although 'package_data' is the preferred approach, in some case you may
    # need to place data files outside of your packages.
    # see http://docs.python.org/3.4/distutils/setupscript.html#installing-additional-files
    # In this case, 'data_file' will be installed into '<sys.prefix>/my_data'
    # data_files = [('data', ['./pyitab/io/data/*.*'])],

    # To provide executable scripts, use entry points in preference to the
    # "scripts" keyword. Entry points provide cross-platform support and allow
    # pip to create the appropriate form of executable for the target platform.
    # entry_points={
    #    'console_scripts': [
    #        'sample=sample:main',
    #    ],
    #},
)