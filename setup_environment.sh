#!/bin/bash

# Environemnt for running ImageDoubler
conda create -n keras python=3.6
conda activate keras

# tensorflow, keras, and GPU support
pip install tensorflow-gpu==1.14.0
pip install keras==2.2.5 h5py==2.10.0
conda install cudatoolkit=10.0.130
conda install cudnn=7.6.5=cuda10.0_0
# other required packages
conda install -c conda-forge opencv
pip install scikit-learn pandas matplotlib ipykernel seaborn tqdm


# Environment for running the benchmark
conda create -n scrna python=3.10 r-base=4.3.2 
conda activate scrna

# for single-cell data analysis
pip install scanpy doubletdetection anndata scvi-tools scrublet scikit-misc
# other required packages
pip install ipykernel matplotlib seaborn
# R packages for differential expression analyses
conda install -c conda-forge r-dplyr r-stringr
conda install -c bioconda bioconductor-deseq2 bioconductor-tximport bioconductor-rhdf5


# to run SoCube
conda create -n socube python=3.9
conda activate socube
pip install socube
