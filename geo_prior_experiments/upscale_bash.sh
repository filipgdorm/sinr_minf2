#!/usr/bin/env bash

# Activate Conda Environment [assuming your Miniconda installation is in your root directory]
source ~/opt/anaconda3/bin/activate sinr_icml_og

MODEL_PATH="1000_cap_models/final_loss_an_full_input_enc_sin_cos_hard_cap_num_per_class_-1/model.pt"
EXP_NAME="an_full_-1"

# Define the gammas you want to iterate over
scaling_factors=(1.5 2.0 5.0 10.0 50.0)

for scaling_factor in "${scaling_factors[@]}"; do
    # Convert gamma to a string for constructing file paths
    python geo_prior_upscale.py --exp_name "${EXP_NAME}" --model_path "${MODEL_PATH}" --gamma "$scaling_factor"
done
