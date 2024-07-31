#!/usr/bin/env bash

# Activate Conda Environment [assuming your Miniconda installation is in your root directory]
source ~/opt/anaconda3/bin/activate sinr_icml_og

thresh_method="masking"

model_type=("an_full_1000_env" "an_slds_1000_env" "an_ssdl_1000_env")

for ((i=0; i<${#model_type[@]}; i++)); do
    # Create dir
    result_dir="${thresh_method}/${model_type[i]}"
    thres_dir="${result_dir}/thresholds"
    mkdir -p "$thres_dir"
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
                    python "${thresh_method}.py" --model_path "${MODEL_PATH}" --result_dir "${thres_dir}" --counter "${counter}"
                    # Increment the counter
                    ((counter++))
                fi
            done
        fi
    done
done
#added evaluation bit
for ((i=0; i<${#model_type[@]}; i++)); do
    # Create dir
    result_dir="${thresh_method}/${model_type[i]}"
    mkdir -p "${result_dir}/results"
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
                    python evaluation.py --model_path "${MODEL_PATH}" --result_dir "${result_dir}" --counter "${counter}"
                    # Increment the counter
                    ((counter++))
                fi
            done
        fi
    done
done