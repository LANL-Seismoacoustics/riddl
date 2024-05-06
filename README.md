# riddl

`riddl` (**r**obust **i**nfrasound **d**etection via **d**eep **l**earning) - machine learning tools to analyze infrasound data.

## Authorship

The `riddl` software package is developed by the Geophysical Explosion Monitoring Team at Los Alamos National Laboratory.

## Documentation

Additional documentation is in-development.

## Anaconda

The installation of `riddl` currently depends on Anaconda to resolve and download the correct python libraries. So if you don’t currently have anaconda installed on your system, please do that first.

Anaconda can be downloaded from https://www.anaconda.com/download/.

Installing `riddl` will create a new Conda environment with the version of Python that it needs in that environment.

## Downloading

In a terminal, navigate to a directory that you would like to put `riddl` in, then download the repository by either https:

    >> git clone  (url here)
    
or by ssh:

    >> git clone (url here)
    
This will create a folder named `riddl`. This will be the base directory for your installation.

## Installation

With Anaconda installed and the repository cloned, you can now install `riddl`. The command below will create an environment named `riddl_env`, install the necessary packages into it, and install `riddl` into that environment.  Navigate to the base directory of `riddl` (there will be a file there named `riddl_env.yml`), and run:

    >> conda env create -f riddl_env.yml

If this command executes correctly and finishes without errors, it should print out instructions on 
how to activate and deactivate the new environment:

    To activate the environment, use:

        >> conda activate riddl_env

    To deactivate an active environment, use

        >> conda deactivate

## Errors/Issues

If you have any errors or issues related to the installation or basic functionality, or if you would like to request additional features, **the best way to get them to us is by submitting a new issue in the Issues Tab above**. 

Questions and problems that might not rise to the level of an Issue can be directed to:

jwbishop@lanl.gov\
pblom@lanl.gov\
jwebster@lanl.gov

## Copyright
© 2024. Triad National Security, LLC. All rights reserved.
This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos
National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S.
Department of Energy/National Nuclear Security Administration. All rights in the program are
reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear
Security Administration. The Government is granted for itself and others acting on its behalf a
nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare
derivative works, distribute copies to the public, perform publicly and display publicly, and to permit
others to do so.