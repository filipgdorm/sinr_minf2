#!/usr/bin/env bash

# Activate Conda Environment [assuming your Miniconda installation is in your root directory]
source ~/opt/anaconda3/bin/activate sinr_icml_og

python lpt.py
python masking.py
python rdm_sampling.py
python tgt_sampling.py
