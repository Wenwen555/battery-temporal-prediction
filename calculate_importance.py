import torch
import torch.nn as nn
import numpy as np
import argparse
import os
import sys
import importlib
import random

torch.backends.cudnn.enabled = False

from dataloader.dataloader import Load_Dataset

def dynamic_import(module_path, class_name):
    """动态导入模块中的类"""
    try:
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except ImportError as e:
        print(f"Error importing {class_name} from {module_path}: {e}")
        sys.exit(1)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

# 假设 Load_Dataset 在 dataloader.py 中，如果不是请修改此处


# ==========================================
# 1. 核心逻辑：计算特征重要性
# ==========================================

def get_feature_importance(model, dataloader, device, criterion):
    model.eval()
    # 此时不能使用 no_grad，因为我们需要计算 Input/Feature 的梯度
    
    total_gradients = None
    batch_count = 0

    print("Start calculating gradient-based importance...")
    
    for i, (batch_x, batch_y) in enumerate(dataloader):
        batch_x = batch_x.to(device).float()
        batch_y = batch_y.to(device).float()
        
        batch_size, cycles, channels, seq_len = batch_x.size()
        
        # --- Step 1: CNN Forward (手动拆解 base_Model.forward) ---
        # 为了获取 conv_block 输出的梯度，我们需要手动运行这一步
        x_reshaped = batch_x.view(batch_size * cycles, channels, seq_len)
        
        # 运行 conv_block
        features_raw = model.conv_block(x_reshaped) 
        # features_raw shape: [batch*cycles, 256, 1] (假设 output_size=1)
        
        # *** 关键 ***：注册 hook 或者 retain_grad 来获取中间层梯度
        features_raw.retain_grad()
        
        # --- Step 2: Predictor Forward ---
        # 模拟 base_Model 后续的处理逻辑
        x = features_raw.squeeze(-1)          # [batch*cycles, 256]
        x = x.view(batch_size, cycles, -1)    # [batch, cycles, 256]
        
        predictions = model.predictor(x).squeeze(-1)
        
        # --- Step 3: Backward ---
        loss = criterion(predictions, batch_y)
        
        model.zero_grad()
        loss.backward()
        
        # --- Step 4: 提取梯度并聚合 ---
        # features_raw.grad shape: [batch*cycles, 256, 1]
        if features_raw.grad is not None:
            # 对 batch, cycles, 和最后的维度取平均，只保留 channel (256) 维度
            grads = features_raw.grad.abs().mean(dim=(0, 2)) 
            # 注意：因为 features_raw 是 (B*C, Dim, 1)，mean(dim=0) 会把 B*C 维度平均掉
            
            if total_gradients is None:
                total_gradients = grads
            else:
                total_gradients += grads
            batch_count += 1
        else:
            print(f"Warning: Batch {i} yielded no gradients.")

    if total_gradients is None:
        raise ValueError("No gradients were computed. Check if the model graph is broken.")

    # 计算全局平均梯度
    avg_gradients = total_gradients / batch_count
    
    # 排序：从大到小，返回索引
    # sorted_indices[0] 是最重要的特征索引
    sorted_indices = torch.argsort(avg_gradients, descending=True)
    
    return sorted_indices.cpu().numpy(), avg_gradients.cpu().numpy()

# ==========================================
# 2. 主流程
# ==========================================

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    # --------------------------------------
    # A. 准备 Configs
    # --------------------------------------
    
    # 我们直接传入xjtu-batch1的config，因为当前其为我们的测试实验数据集
    Configs = dynamic_import('config_files.xjtu.cell_XJTU_batch1_data_Configs', 'Config')
    configs = Configs()

    # --------------------------------------
    # B. 加载数据
    # --------------------------------------
    print(f"Loading dataset from {args.data_path} ...")
    dataset_content = torch.load(args.data_path, map_location='cpu', weights_only=False) # weights_only=False 默认
    
    # 使用 training_mode='train' (或者 'test'，建议用训练集或验证集计算重要性)
    train_dataset = Load_Dataset(dataset_content, configs, training_mode='analysis', dataset_name='xjtu')
    
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=configs.batch_size,
        drop_last=True, # Drop last 避免 shape 不匹配问题
        shuffle=True,
        num_workers=0
    )

    # --------------------------------------
    # C. 初始化模型
    # --------------------------------------
    model_module_path = f'models.{args.base_model}'
    print(f"Importing base_Model from {model_module_path}...")
    base_Model_Class = dynamic_import(model_module_path, 'base_Model')
    
    model = base_Model_Class(configs).to(device)
    
    # --------------------------------------
    # D. 加载预训练权重
    # --------------------------------------
    # 构建路径逻辑 (参考你的描述)
    base_path = "/mnt/wenjt5/TC-SOH/experiments_logs/xjtu"
    if args.output_module == "linear":
        load_from = os.path.join(base_path, 'cnn_batch1_random_select_0_bat', f"supervised_with_contrast_seed_{args.seed}", str(args.fold), "saved_models")
        ckpt_file = os.path.join(load_from, "ckp_last.pt")
    else:
        load_from = os.path.join(base_path, f"Predictor_{args.output_module}", 'cnn_batch1_random_select_0_bat', f"supervised_with_contrast_seed_{args.seed}", str(args.fold), "saved_models")
        ckpt_file = os.path.join(load_from, "ckp_last.pt")
    
    print(f"Loading checkpoint from: {ckpt_file}")
    if not os.path.exists(ckpt_file):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_file}")

    chkpoint = torch.load(ckpt_file, map_location=device)
    pretrained_dict = chkpoint["model_state_dict"]
    
    # 加载权重 (strict=True 确保结构匹配)
    if args.output_module == 'linear':
        new_state_dict = {}
        for k, v in pretrained_dict.items():
            name = k.replace("module.", "")
            # === 重命名 fc -> predictor ===
            # 如果 checkpoint 里叫 'fc.weight'，改名为 'predictor.weight'
            if name.startswith("fc."):
                name = name.replace("fc.", "predictor.")
                print(f"Remapping key: {k} -> {name}") # 打印出来让你放心
                
            new_state_dict[name] = v
        msg = model.load_state_dict(new_state_dict, strict=True)
    else:
        msg = model.load_state_dict(pretrained_dict, strict=True)
    print(f"Model loaded. {msg}")

    # --------------------------------------
    # E. 计算并保存重要性
    # --------------------------------------
    criterion = nn.MSELoss() # 你的损失函数
    
    print(f"Starting calculating for {args.output_module}...")
    ranked_indices, raw_scores = get_feature_importance(model, train_loader, device, criterion)
    ranked_scores = raw_scores[ranked_indices]
    print("Top 10 Important Feature Indices:", ranked_indices[:10])
    print("Their relative avg_gradients:", ranked_scores[:10])
    
    # 保存结果
    save_path = os.path.join(args.save_dir, f"xjtu_batch1_importance_ranking_{args.output_module}.npy")
    os.makedirs(args.save_dir, exist_ok=True)
    
    np.save(save_path, ranked_indices)
    print(f"Importance ranking saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate Feature Importance via Gradients")

    # Model Configs (这些参数必须与训练时一致，用于初始化模型结构)
    parser.add_argument('--base_model', type=str, default='cnn', help='Name of the python file in models/ folder')
    parser.add_argument('--output_module', type=str, default='lstm', choices=['predictor_mlp', 'predictor_cnn', 'predictor_lstm', 'predictor_transformer', 'linear'], help='Which predictor to use')

    # Paths
    parser.add_argument('--data_path', type=str, default='/mnt/wenjt5/TC-SOH/data/XJTU/cell_XJTU_batch1_data/train_val.pt', help='Path to .pt dataset file')
    parser.add_argument('--save_dir', type=str, default='/mnt/wenjt5/TC-SOH/experiments_logs/importance', help='Where to save the .npy file')

    # Experiment Settings
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--fold', type=str, default='2', help='Fold number string')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', type=str, default='cuda')
    
    # 这里的 args 将被作为 configs 传入模型
    args = parser.parse_args()
    
    main(args)
