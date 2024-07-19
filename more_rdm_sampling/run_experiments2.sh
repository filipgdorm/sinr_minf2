#!/usr/bin/env bash

# Activate Conda Environment [assuming your Miniconda installation is in your root directory]
source ~/opt/anaconda3/bin/activate sinr_icml_og

size_factors=(100 1000 10000)
for size_factor in "${size_factors[@]}"; do
    mkdir -p "rdm_nabs_${size_factor}"
    python rdm_sampling2.py --num_absences "${size_factor}"           
    python evaluation2.py --num_absences "${size_factor}"
done