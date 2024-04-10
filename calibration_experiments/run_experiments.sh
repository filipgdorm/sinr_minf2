#!/usr/bin/env bash

# Activate Conda Environment [assuming your Miniconda installation is in your root directory]
source ~/opt/anaconda3/bin/activate sinr_icml_og

# Define arrays for experiment names and model paths
experiment_names=("an_full_1000" "an_slds_1000" "an_ssdl_1000")
model_paths=("1000_cap_models/final_loss_an_full_input_enc_sin_cos_hard_cap_num_per_class_1000/model.pt" \
            "1000_cap_models/final_loss_an_slds_input_enc_sin_cos_hard_cap_num_per_class_1000/model.pt" \
            "1000_cap_models/final_loss_an_ssdl_input_enc_sin_cos_hard_cap_num_per_class_1000/model.pt")

## Loop over experiment names and model paths
for ((i=0; i<${#experiment_names[@]}; i++)); do
    EXP_NAME="${experiment_names[i]}"
    MODEL_PATH="${model_paths[i]}"

    python calibration_metrics_v2.py --exp_name "${EXP_NAME}" --model_path "${MODEL_PATH}"

done
