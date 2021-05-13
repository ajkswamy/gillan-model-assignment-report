Welcome!

This repository contains the code and results of fitting a model from Gillan et al. eLife 2016;5:e11305. DOI: 10.7554/eLife.11305, pg. 19-20 using implementational details from the supplementary information of Otto et al. PNAS 2013; DOI: 10.1073/pnas.1312011110.

License: CC-BY

This code has only been tested on 64bit linux with Anaconda.

Usage
---

1. Open a terminal/Powershell and navigate into the folder containing this file: `cd <drag the folder here from a file explorer`
2. Create a conda environment using the file `conda_env.yml`: `conda env create -f conda_env.yml`
3. Activate the environment: `conda activate gma`
4. Run the script fitting functions with the folder containing input data as argument: `python code/fit_multilevel_model.py <path to folder containing input csv files>`
> Note: the number of subjects and the number of trials to fit can be modified on lines 140 and 141 of `fit_multilevel_model.py`

Output
---
Outputs are written in a folder that is a sibling of the folder containing input CSVs and named "Results". 
Map estimates are written in text-based JSON format. Results of NUTS sampling are written in NetCDF format which can 
be read and analyzed using the package `arviz`. See the file `code/create_summary_table.py` for an example where
both kinds of output files were loaded to create a summary table






