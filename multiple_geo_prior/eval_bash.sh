#!/usr/bin/env bash

# Activate Conda Environment [assuming your Miniconda installation is in your root directory]
source ~/opt/anaconda3/bin/activate sinr_icml_og

thresh_method="lpt"

model_type=("an_full_1000") # Add other models as needed: "an_slds_1000" "an_ssdl_1000"

for ((i=0; i<${#model_type[@]}; i++)); do
    # Create dir
    result_dir="${thresh_method}/${model_type[i]}/${threshold}"
    mkdir -p "${result_dir}/results"
    # Initialize counter
    counter=0
    # Directory containing the model files
    MODEL_DIR="../five_models/${model_type[i]}"
    # Loop over subdirectories in the model directory
    for SUBDIR in "$MODEL_DIR"/*; do
        if [ -d "$SUBDIR" ]; then  # Check if it is a directory
            # Loop over files in the subdirectory
            for MODEL_PATH in "$SUBDIR"/*; do
                if [ -f "$MODEL_PATH" ]; then  # Check if it is a file
                    # Run the Python script with the current model path and counter
                    python evaluation_w_thres_n_delta.py --model_path "${MODEL_PATH}" --result_dir "${result_dir}" --counter "${counter}"
                    # Increment the counter
                    ((counter++))
                fi
            done
        fi
    done
done
