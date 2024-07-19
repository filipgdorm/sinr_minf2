#!/usr/bin/env bash

# Activate Conda Environment [assuming your Miniconda installation is in your root directory]
source ~/opt/anaconda3/bin/activate sinr_icml_og

size_factors=(1.0 5.0 10.0)
for size_factor in "${size_factors[@]}"; do
    mkdir -p "rdm_factor_${size_factor}"
    python rdm_sampling.py --size_factor "${size_factor}"           
    python evaluation.py --size_factor "${size_factor}"
done