#!/usr/bin/env bash

# Activate Conda Environment [assuming your Miniconda installation is in your root directory]
source ~/opt/anaconda3/bin/activate sinr_icml_og

size_factors=(100 1000 10000)
model_type=("an_full_1000_env" "an_slds_1000_env" "an_ssdl_1000_env")

for size_factor in "${size_factors[@]}"; do
    for ((i=0; i<${#model_type[@]}; i++)); do
        # Create dir
        result_dir="rdm_nabs_${size_factor}/${model_type[i]}"
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
                        python rdm_sampling2.py --num_absences "${size_factor}" --model_path "${MODEL_PATH}" --result_dir "${result_dir}" --counter "${counter}"           
                        python evaluation2.py --num_absences "${size_factor}" --model_path "${MODEL_PATH}" --result_dir "${result_dir}" --counter "${counter}"
                        # Increment the counter
                        ((counter++))
                    fi
                done
            fi
        done
    done
done