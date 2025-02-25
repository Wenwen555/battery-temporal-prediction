export CUDA_VISIBLE_DEVICES=1
python -m main.py \
--dataset tongji \
--experiment_description problem_check \
--selected_subset cell_Tongji_NCA_data \
--run_description test \
--base_model cnn \
--training_mode supervised \
--data_path data/Tongji/ \
