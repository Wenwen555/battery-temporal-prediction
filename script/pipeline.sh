# 定义数据集路径和模型名称数组
DATASETS=("mit" "xjtu" "hust","tongji")
selected_subset_mit=(
'cell_mit_batch1_prepocess_data_Configs.py',
'cell_mit_batch2_prepocess_data_Configs.py',
'cell_mit_batch3_prepocess_data_Configs.py
')
selected_subset_xjtu=(
"cell_XJTU_batch1_prepocess_data_Configs.py",
"cell_XJTU_batch2_prepocess_data_Configs.py",
"cell_XJTU_batch3_prepocess_data_Configs.py",
"cell_XJTU_batch4_prepocess_data_Configs.py",
"cell_XJTU_batch5_prepocess_data_Configs.py",
"cell_XJTU_batch6_prepocess_data_Configs.py"
)
selected_subset_hust=(
"cell_HUST_batch1_data_Configs.py",
"cell_HUST_batch2_data_Configs.py",
"cell_HUST_batch3_data_Configs.py",
"cell_HUST_batch4_data_Configs.py",
"cell_HUST_batch5_data_Configs.py",
"cell_HUST_batch6_data_Configs.py",
"cell_HUST_batch7_data_Configs.py",
"cell_HUST_batch8_data_Configs.py",
"cell_HUST_batch9_data_Configs.py",
"cell_HUST_batch10_data_Configs.py"
)
selected_subset_tongji=(
"cell_Tongji_NCA_data_Configs.py",
"cell_Tongji_NCM_NCA_data_Configs.py",
"cell_Tongji_NCM_data_Configs.py"
)


# 循环遍历数据集和模型，生成并运行train.sh
for DATASET in "${DATASETS[@]}"; do
  subset_var="selected_subset_${DATASET}"
  subset_files=("${!subset_var}")
  
  for config in "${subset_files[@]}"; do
    # 生成 train.sh 文件
    cp train.sh train_temp.sh
    if [[ $config =~ (batch[0-9]+) ]]; then
	batch_value="${BASH_REMATCH[1]}"
    fi
    sed -i "s|__dataset_name__|$DATASET|g" train.sh
    sed -i "s|__selected_subset__|$config|g" train.sh
    sed -i "s|__run_description__|$batch_value|g" train.sh

    echo "Running training with dataset: $DATASET : $batch_value$"
    # 运行train.sh
    bash train.sh
  done
done


