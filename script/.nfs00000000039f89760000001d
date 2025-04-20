export CUDA_VISIBLE_DEVICES=0
for training_mode in {'supervised','supervised_with_contrast'}; do
    python -m main.py \
    --dataset hust \
    --experiment_description hust \
    --selected_subset cell_HUST_batch9_data \
    --run_description cnn_hust_batch2 \
    --base_model cnn \
    --training_mode $training_mode \
    --data_path data/HUST/
done