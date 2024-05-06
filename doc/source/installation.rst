.. _installation:

=====================================
Installation
=====================================

-----------------
Operating Systems
-----------------

*riddl* has been installed on machines running newer versions of Linux or Apple OS X.  Installation on a Windows system has not been tested, but requires an Anaconda Python installation, so it should be reasonably straightforward.  Installation of propagation and signal analysis tools on such a system might be more challenging however. 

----------------------------------------
Dependencies
----------------------------------------


**Anaconda**

The installation of *riddl* is ideally completed using pip through `Anaconda <https://docs.anaconda.com/free/anaconda/install/index.html>`_ to resolve and download the correct python libraries. If you don't currently have Anaconda installed on your system, you will need to do that first.


**InfraPy Signal Analysis Methods**

The AI/ML-based signal analysis methods in *riddl* are intended to supplement more traditional analysis capabilities implementated in the LANL `InfraPy <https://github.com/LANL-Seismoacoustics/infrapy>`_ signal analysis software suite.

-----------------------------
riddl Installation
-----------------------------

**Stand Alone Install**

A stand alone Anaconda environment can be created with the *riddl* YML file,

.. code-block:: none

    conda env create -f riddl_env.yml

If this command executes correctly and finishes without errors, it should print out instructions on how to activate and deactivate the new environment:

To activate the environment, use:

.. code-block:: none

    conda activate riddl_env

To deactivate an active environment, use

.. code-block:: none

    conda deactivate

**Installing via PyGS**

A number of LANL infrasound software tools have been developed and made available to the community through the `LANL Seismoacoustics github page <https://github.com/LANL-Seismoacoustics/infrapy>`_.  Collectively, these tools are referred to as the Python Geophysics Suite (PyGS).
Due to linkages between the various PyGS packages, it's useful to install these various packages into a single Anaconda environment; however, package conflicts exist that can make a full installation of these tools difficult.  Resolving these conflicts is an ongoing effort.

**Testing riddl**

Once the installation is complete, you can test the methods by running the command line interface help.  Firstly, activate either the :code:`riddl_env` or :code:`pygs_env`, then run the :code:`--help` or :code:`-h` option for stochprop.

.. code-block:: none

    riddl -h

This command will show the general usage of the stochprop package:

.. code-block:: none

    Usage: riddl [OPTIONS] COMMAND [ARGS]...

      riddl
      -----

      Robust Infrasound Detection via Deep Learning (RIDDL)

      Python-based tools for AI/ML-based signal analysis of infrasound data

    Options:
      -h, --help  Show this message and exit.

    Commands:
      data    Methods to create and manipulate data
      models  Build, evaluate, and use models
      utils   Utility tools

Usage of the individual packages and sub-commands can be similarly displayed with the help flag (e.g., :code:`riddl models fk -h`).
