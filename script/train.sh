export CUDA_VISIBLE_DEVICES=1
python -m main.py \
--dataset tongji \
--experiment_description NCM_NCA_test \
--selected_subset cell_Tongji_NCM_NCA_data \
--run_description test \
--base_model mlp \
--training_mode supervised \
--data_path data/Tongji/ \
