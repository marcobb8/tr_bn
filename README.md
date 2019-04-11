## Introduction
Code of the methods proposed in the PhD thesis "Learning Tractable Bayesian Networks"

## Prerequirements and installing guide

This software has been developed as a Python 2.7.15 package and includes some functionalities in Cython and C++11 (version 5.4.0). Consequently, it is needed a Python environment and internet connectivity to download additional package dependencies. Python software can be downloaded from <https://www.python.org/downloads/>.

We provide the steps for a clean installation in Ubuntu 16.04. This software has not been tried under Windows.

The package also uses the following dependencies. 

|Library     |Version|   License|
|------------|-------|----------|
| pandas     |   0.23|     BSD 3|
|  numpy     | 1.14.3|       BSD|
| Cython     | 0.28.2|    Apache|
|cloudpickle |  0.5.3|     BSD 3|
|scikit-learn| 0.20.2|   New BSD|
| matplotlib |  1.5.1|Matplotlib|


They can be installed through the following sentence:
sudo pip install "Library"
where Library must be replaced by the library to be installed.

Open the folder where you have saved TSEM project files (e.g., "~/Downloads/TSEM") and compile Cython files running the following commands in the command console:

python2.7 setup_dt.py build_ext --inplace

python2.7 setup_tw.py build_ext --inplace

python2.7 setup_et.py build_ext --inplace

python2.7 setup_cplus.py build_ext --inplace

python2.7 setup_cplus_data.py build_ext --inplace

python2.7 setup_gs.py build_ext --inplace

python2.7 setup_etc.py build_ext --inplace

## Example.py

File "example_learn_et.py" provides a demo that shows how to learn a bounded treewidth Baysian network (Chapter 3).
File "example_tsem.py" provides a demo that shows how to use the code to learn Bayesian networks in the presence of missing values (Chapter 4). 
File "example_mbcs.py" shows examples of how to learn an MBC in a generative and discriminative way (Chapters 5 and 6).
File "epilepsy.py" contains the code used to compare an MBC with nomograms for predicting Engel outcome 1, 2 and 5 years after surgery (Chapter 7).