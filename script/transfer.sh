#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

# Configurable parameters
datasets=("mit" "hust" )
# 此处的training_mode无需修改，因为其包含了finetuing, mixture-training and source-only.
training_mode="transfer"  
base_model="cnn"
source_batch="1"
target_batch="3"
# MIT and XJTU 的Source_batch 选取是我自己设定的
# Tongji special case subsets
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

declare -A source_batches=(
    ["xjtu"]="1"
    ["mit"]="1"
    ["hust"]="1"
    ["tongji"]="3"
)

declare -A target_batch_ranges=(
    ["xjtu"]="1"          # Only batch1 for xjtu
    ["tongji"]="3"        # Only batch3 for tongji
    ["mit"]="1 2 3"       # Batches 1-3 for mit
    ["hust"]="1" # Batches 1-10 for hust
)

for bat_number in 2; do
    for source_dataset in "${datasets[@]}"; do
    
    source_batch=${source_batches[$source_dataset]}
    if [ $source_dataset == "hust" ]; then
        source_batch="1"
        ckpt_path="/mnt/wenjt5/project1/experiments_logs/hust/cnn_hust_batch${source_batch}/"
    elif [ $source_dataset == "xjtu" ]; then
        source_batch="1"
        ckpt_path="/mnt/wenjt5/project1/experiments_logs/xjtu/cnn_batch${source_batch}_random_select_0_bat/"
    elif [ $source_dataset == "mit" ]; then
        source_batch="1"
        ckpt_path="/mnt/wenjt5/project1/experiments_logs/mit/cnn_batch${source_batch}_random_select_0_bat/"
    elif [ $source_dataset == "tongji" ]; then
        source_batch="3"
        ckpt_path="/mnt/wenjt5/project1/experiments_logs/tongji/cnn_${tongji_subsets[$((source_batch))-1]}/"
    fi

        for target_dataset in "${datasets[@]}"; do
            if [ "$target_dataset" == "$source_dataset" ]; then
                continue  # Skip when source and target are the same
            fi
        # Function to get batch range based on source_batch
            target_batches=${target_batch_ranges[$target_dataset]}

            for target_batch in $target_batches; do
                # Determine the selected_subset based on target_dataset
                case $source_dataset in
                    "xjtu")
                        selected_subset="cell_XJTU_batch${source_batch}_data"
                        ;;
                    "mit")
                        selected_subset="cell_mit_batch${source_batch}_data"
                        ;;
                    "hust")
                        selected_subset="cell_HUST_batch${source_batch}_data"
                        ;;
                    "tongji")
                        selected_subset="cell_Tongji_${tongji_subsets[source_batch-1]}_data"  # Arrays are 0-indexed
                        ;;
                esac

                case $target_dataset in
                    "xjtu")
                        target_subset="cell_XJTU_batch${target_batch}_data"
                        ;;
                    "mit")
                        target_subset="cell_mit_batch${target_batch}_data"
                        ;;
                    "hust")
                        target_subset="cell_HUST_batch${target_batch}_data"
                        ;;
                    "tongji")
                        target_subset="cell_Tongji_${tongji_subsets[target_batch-1]}_data"
                        ;;
                    *)
                        echo "Unknown target dataset: $target_dataset"
                        exit 1
                        ;;
                esac

                if [[ "$target_dataset" == "tongji" ]]; then
                    run_description="${base_model}_${source_dataset}_batch${i}_to_${target_dataset}_${tongji_subsets[$((target_batch))-1]}_transfer"    
                elif [[ "$source_dataset" == "tongji" ]]; then
                    run_description="${base_model}_${source_dataset}_${tongji_subsets[$((source_batch))-1]}_to_${target_dataset}_batch${target_batch}_transfer"
                else
                    run_description="${base_model}_${source_dataset}_batch${source_batch}_to_${target_dataset}_batch${target_batch}_transfer"
                fi
                # Tips1：确保ckpt全都存在
                # Tips2: 确保random pick的sample都存在（1 bat and 2 bat)
                # Tips3: 确保mixture dataset都存在
                echo "Loading checkpoint from: ${ckpt_path}"
                echo "Using target_dataset from: ${target_dataset}-${target_batch}"
                echo "Saving to: ${run_description}"
                echo "USing ${bat_number} bat for transfer learning！"
                python -m transfer \
                    --target_dataset $target_dataset \
                    --source_dataset $source_dataset \
                    --experiment_description $source_dataset \
                    --selected_subset $selected_subset \
                    --target_batch $target_batch \
                    --run_description $run_description \
                    --base_model $base_model \
                    --training_mode $training_mode \
                    --target_data_path "${data_paths[$target_dataset]}${target_subset}" \
                    --source_data_path ${data_paths[$source_dataset]} \
                    --selected_ckpt $ckpt_path \
                    --bat_num $bat_number
            done
        done
    done
done
