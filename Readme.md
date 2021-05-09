This repository contains code used for reproducing modelling results of Gillian et al. 2016


Usage
---
This code has only been tested on 64bit linux with conda
1. Open a terminal/Powershell and navigate into the folder containing this file: `cd <drag the folder here from a file explorer`
2. Create a conda environment using the file `conda_env.yml`: `conda env create -f conda_env.yml`
3. Activate the environment: `conda activate gma`
3. Run the script fitting functions with the folder containing input data as argument: `python fitting_functions.py <path to folder containing input csv files>`

References:

1.  Gillan et al. eLife 2016;5:e11305. DOI: [10.7554/eLife.11305](http://dx.doi.org/10.7554/eLife.11305)

