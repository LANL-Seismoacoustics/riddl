# -*- coding: utf-8 -*-
# © 2024. Triad National Security, LLC. All rights reserved.
#
# Manual install
# conda create -n riddl_env numpy scipy obspy tensorflow matplotlib regex --channel conda-forge
# conda activate riddl_env
# pip install -U scikit-learn
# pip install -e .
#
# From .yml
# conda env create -f riddl_env.yml

try:
    import setuptools
except ImportError:
    pass

import os
import glob
from distutils.core import setup

setup(name = "riddl",
      license = "MIT",
      version = "0.1",
      description = "Machine learning tools for processing infrasound data.",
      keywords=["infrasound", "geophysics", "seismic", "array"],
      author = "LANL Geophysical Explosion Monitoring (LANL-GEM) Infrasound Team",
      author_email = "",
      packages = ["riddl",
                  "riddl.cli"],
      entry_points = {'console_scripts':['riddl=riddl.cli.__main__:main']},

    install_requires = ['click',
                'matplotlib',
                'numpy',
                'numpydoc',
                'scipy',
                'ipython',
                'sphinx',
                'sphinx_rtd_theme']
    )