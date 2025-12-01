#!/bin/bash
export CUDA_VISIBLE_DEVICES=2

# Configurable parameters
datasets=("xjtu")   # 修正这里，去掉逗号
training_mode="supervised_with_contrast"
base_model="cnn"
target_batch="1"          # Set to empty for all batches, or specify a batch number (e.g., 3)
use_small_sample=false    # Set to false to disable small sampling

# Batch number configuration
declare -A n_value_dict=(
    ["xjtu"]=6
    ["mit"]=3
    ["hust"]=10
    ["tongji"]=3  # For Tongji's 3 subsets
)

# Dataset-specific configurations
declare -A data_paths=(
    ["xjtu"]="data/XJTU/"
    ["mit"]="data/MIT/"
    ["hust"]="data/HUST/"
    ["tongji"]="data/Tongji/"
)
# Tongji special case subsets
tongji_subsets=("NCA" "NCM" "NCM_NCA")

# Function to get batch range based on target_batch

# Function to get sample numbers based on use_small_sample
get_sample_numbers() {
    if [[ "$use_small_sample" == true ]]; then
        echo "4"
    else
        echo "0"  # Using 0 as a flag for no small sampling
    fi
}

for dataset in "${datasets[@]}"; do
    echo ""
    echo "===================================================================="
    echo "Processing dataset: $dataset"
    echo "===================================================================="
    get_batch_range() {
        if [[ -z "$target_batch" ]]; then
            # Process all batches
            echo $(seq 1 ${n_value_dict[$dataset]})
        else
            # Process only specified batch
            if (( $target_batch >= 1 && $target_batch <= ${n_value_dict[$dataset]} )); then
                echo $target_batch
            else
                echo "Error: Invalid batch number for dataset $dataset (max ${n_value_dict[$dataset]})" >&2
                exit 1
            fi
        fi
    }

    for small_sample_num in $(get_sample_numbers); do
        for i in $(get_batch_range); do
            # Determine the selected_subset based on dataset
            case $dataset in
                "xjtu")
                    selected_subset="cell_XJTU_batch${i}_data"
                    ;;
                "mit")
                    selected_subset="cell_mit_batch${i}_data"
                    ;;
                "hust")
                    selected_subset="cell_HUST_batch${i}_data"
                    ;;
                "tongji")
                    selected_subset="cell_Tongji_${tongji_subsets[i-1]}_data"  # Arrays are 0-indexed
                    ;;
            esac
            
            # Set output directory based on random_select
            if [[ "$use_small_sample" != 0 ]]; then
                run_description="${base_model}_batch${i}_random_select_${small_sample_num}_bat"
            else
                run_description="4transfer_${base_model}_${dataset}_batch${i}"
            fi
            echo "saving to ${run_description}"
            python -m main \
                --dataset $dataset \
                --experiment_description $dataset \
                --selected_subset $selected_subset \
                --run_description $run_description \
                --base_model $base_model \
                --training_mode $training_mode \
                --data_path ${data_paths[$dataset]} \
                --small_sample_num $small_sample_num
        done
    done
done