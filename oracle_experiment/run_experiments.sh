#!/usr/bin/env bash

# Activate Conda Environment [assuming your Miniconda installation is in your root directory]
source ~/opt/anaconda3/bin/activate sinr_icml_og

result_dir="result_prop_"

props=(0.01 0.05 0.1)
for prop in "${props[@]}"; do
    mkdir -p "result_prop_${prop}"
    python expert_subsample.py --prop_absence "${prop}"           
    
done