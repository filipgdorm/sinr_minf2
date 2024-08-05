#!/usr/bin/env bash

# Activate Conda Environment [assuming your Miniconda installation is in your root directory]
source ~/miniconda3/bin/activate sinr_icml

noise_levels=(0.0 2.5 5.0 7.5 10.0 12.5 15.0)
model_type=("an_full_1000_env" "an_slds_1000_env" "an_ssdl_1000_env")

for ((i=0; i<${#model_type[@]}; i++)); do
    for noise_level in "${noise_levels[@]}"; do
        # Create dir
        result_dir="${model_type[i]}/masking_level_${noise_level}"
        mkdir -p $result_dir
        # Initialize counter
        counter=0
        # Directory containing the model files
        MODEL_DIR="../five_models_env/${model_type[i]}"
        # Loop over subdirectories in the model directory
        for SUBDIR in "$MODEL_DIR"/*; do
            if [ -d "$SUBDIR" ]; then  # Check if it is a directory
                # Loop over files in the subdirectory
                for MODEL_PATH in "$SUBDIR"/*; do
                    if [ -f "$MODEL_PATH" ]; then  # Check if it is a file
                        # Run the Python script with the current model path and counter
                        python masking.py --noise_level "${noise_level}" --model_path "${MODEL_PATH}" --result_dir "${result_dir}" --counter "${counter}"   
                        python evaluation.py --noise_level "${noise_level}" --model_path "${MODEL_PATH}" --result_dir "${result_dir}" --counter "${counter}"
                        # Increment the counter
                        ((counter++))
                    fi
                done
            fi
        done
    done
done