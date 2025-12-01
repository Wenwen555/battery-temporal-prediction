#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

# Configurable parameters
datasets=("xjtu" "tongji" "mit" "hust" ) 
base_model="cnn"

tongji_subsets=("NCA" "NCM" "NCM_NCA")
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

declare -A batch_ranges=(
    ["xjtu"]="1"
    ["mit"]="1"
    ["hust"]="1"
    ["tongji"]="2"
)

# declare -A target_batch_ranges=(
#     ["xjtu"]="1"          # Only batch1 for xjtu
#     ["tongji"]="3"        # Only batch3 for tongji
#     ["mit"]="1 2 3"       # Batches 1-3 for mit
#     ["hust"]="$(seq 1 10)" # Batches 1-10 for hust
# )


for dataset in "${datasets[@]}"; do
    batch=${batch_ranges[$dataset]}
    # if [[ "$dataset" != "hust" ]]; then
    #     continue
    # fi
    if [ $dataset == "hust" ]; then
        ckpt_path="/mnt/wenjt5/project1/experiments_logs/hust/cnn_batch${batch}_random_select_0_bat/"
    elif [ $dataset == "xjtu" ]; then
        ckpt_path="/mnt/wenjt5/project1/experiments_logs/xjtu/cnn_batch${batch}_random_select_0_bat/"
    elif [ $dataset == "mit" ]; then
        ckpt_path="/mnt/wenjt5/project1/experiments_logs/mit/cnn_batch${batch}_random_select_0_bat/"
    elif [ $dataset == "tongji" ]; then
        ckpt_path="/mnt/wenjt5/project1/experiments_logs/tongji/cnn_batch${batch}_random_select_0_bat/"
    fi

    case $dataset in
        "xjtu")
            selected_subset="cell_XJTU_batch${batch}_data"
            ;;
        "mit")
            selected_subset="cell_mit_batch${batch}_data"
            ;;
        "hust")
            selected_subset="cell_HUST_batch${batch}_data"
            ;;
        "tongji")
            selected_subset="cell_Tongji_${tongji_subsets[batch-1]}_data"  # Arrays are 0-indexed
            ;;
    esac

    echo "Loading checkpoint from: ${ckpt_path}"
    echo "Using dataset from: ${dataset}-${batch}"

    python -m plot_class \
        --dataset $dataset \
        --batch $selected_subset \
        --data_path ${data_paths[$dataset]} \
        --selected_ckpt $ckpt_path
done
