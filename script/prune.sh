#!/bin/bash

# 1. 准备工作
RANKING_FILE="feature_ranking.npy"

# 步骤 A: 计算特征重要性 
# (假设我们用 MLP 模型的特征梯度作为“黄金标准”，或者你可以分别计算)
echo "Calculating Feature Importance..."
python calculate_importance.py \
  --model_type cnn_mlp \
  --checkpoint_path ckpt/model_mlp.pth \
  --save_path $RANKING_FILE

# # 步骤 B: 评估 Linear Predictor 的曲线
# echo "Running Linear Predictor Curve..."
# python eval_dimension_curve.py \
#   --predictor_type linear \
#   --checkpoint_path ckpt/model_linear.pth \
#   --ranking_path $RANKING_FILE \
#   --device cuda:0

# # 步骤 C: 评估 MLP Predictor 的曲线
# echo "Running MLP Predictor Curve..."
# python eval_dimension_curve.py \
#   --predictor_type predictor_mlp \
#   --checkpoint_path ckpt/model_mlp.pth \
#   --ranking_path $RANKING_FILE \
#   --device cuda:0

# 步骤 D: (可选) 调用画图脚本
# python plot_curves.py
