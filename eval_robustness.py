import torch
import numpy as np
import argparse
import pandas as pd
import importlib
import os
from dataloader.dataloader import Load_Dataset


def dynamic_import(module_path, class_name):
    """动态导入类"""
    module = importlib.import_module(module_path)
    return getattr(module, class_name)

import torch
import math

def get_sigma_from_snr_loader(dataloader, target_snr_db, device='cpu'):
    """
    根据目标信噪比(dB)和整个Dataloader的数据，计算需要的高斯噪声标准差 sigma。
    原理:
    Global_Signal_Power = sum(x^2) / total_elements
    P_noise = P_signal / 10^(SNR_db / 10)
    sigma = sqrt(P_noise)
    """
    # 1. 如果目标是无穷大信噪比（无噪），直接返回0
    if target_snr_db == float('inf'):
        return 0.0

    total_square_sum = 0.0
    total_elements = 0
    
    # 2. 遍历 Loader 计算全局信号功率 (P_signal)
    # 使用 no_grad 避免显存占用
    with torch.no_grad():
        for batch in dataloader:
            # 假设 dataloader 返回的是 (inputs, labels) 或者 [inputs, labels]
            # 我们只关心 inputs
            if isinstance(batch, (list, tuple)):
                inputs = batch[0]
            else:
                inputs = batch # 或者是直接返回的数据
            
            # 将数据放到指定设备（建议用CPU计算统计量，防止显存溢出，除非数据很小）
            inputs = inputs.to(device)
            
            # 累加平方和 (x^2)
            total_square_sum += torch.sum(inputs ** 2).item()
            # 累加元素总数
            total_elements += inputs.numel()
    
    # 计算全局平均信号功率
    global_signal_power = total_square_sum / total_elements
    
    # 3. 计算目标噪声功率
    # SNR = 10 * log10(P_signal / P_noise)  =>  P_noise = P_signal / 10^(SNR/10)
    noise_power = global_signal_power / (10 ** (target_snr_db / 10))
    
    # 4. 得到 sigma
    sigma = math.sqrt(noise_power)
    
    return sigma


def check_noise_level(sample, sigma):
    noise = torch.randn_like(sample) * sigma
    
    # 计算信噪比
    signal_power = (sample ** 2).mean()
    noise_power = (noise ** 2).mean()
    snr = 10 * torch.log10(signal_power / noise_power)
    
    print(f"信噪比: {snr:.2f} dB")
    print(f"噪声标准差: {sigma}")
    print(f"数据标准差: {sample.std():.4f}")
    print(f"相对噪声强度: {sigma/sample.std():.4f}")
    
    return snr

def add_noise(sample, sigma):
    """
    向输入样本添加高斯噪声
    sample: tensor [Batch, Seq_Len, ...]
    sigma: 噪声标准差
    """
    if sigma <= 0:
        return sample
    
    # 生成与输入相同形状的噪声
    noise = torch.randn_like(sample) * sigma
    noisy_sample = sample + noise
    return noisy_sample

def analyze_sensitivity(model, loader, device, noise_sigma=0.0):
    """
    执行 SOH 灵敏度分段分析。
    将 SOH 分为不同阶段，分别计算 MSE，用于证明模型在早期/健康阶段的高感知力。
    """
    print(f"\nRunning Sensitivity Analysis with Noise Sigma: {noise_sigma} ...")
    
    all_preds = []
    all_targets = []
    
    model.eval()
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            
            # 注入噪声 (通常此实验在 sigma=0 下做，展示最大精度)
            x_input = add_noise(x, noise_sigma)
            
            # 前向传播
            preds, _ = model(x_input)
            
            # 收集数据
            all_preds.append(preds.view(-1).cpu().numpy())
            all_targets.append(y.view(-1).cpu().numpy())
    # 拼接数据
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    
    # 计算平方误差
    squared_errors = (all_preds - all_targets) ** 2
    
    # 创建 DataFrame
    df_results = pd.DataFrame({
        'SOH_True': all_targets,
        'SOH_Pred': all_preds,
        'Squared_Error': squared_errors
    })
    
    # === 定义分段区间 (针对 0.7 - 1.0 的范围) ===
    # 逻辑：
    # 1. Healthy (>0.95): 极早期，特征极不明显
    # 2. Early Decay (0.90 - 0.95): 早期衰退
    # 3. Middle Decay (0.80 - 0.90): 中期
    # 4. Late/Failure (<0.80): 晚期，特征明显
    # 注意：bins 的左边放宽到 0.0 以防有异常值或略低于 0.7 的数据
    bins = [-0.1, 0.8, 0.9, 0.95, 2.0] 
    labels = ['Late/Failure (<0.80)', 'Middle Decay (0.80-0.90)', 'Early Decay (0.90-0.95)', 'Healthy (>0.95)']
    
    # 分组
    df_results['Stage'] = pd.cut(df_results['SOH_True'], bins=bins, labels=labels, right=False)
    
    # 计算各阶段统计量
    stage_mse = df_results.groupby('Stage', observed=False)['Squared_Error'].mean()
    stage_count = df_results['Stage'].value_counts()
    # 打印精美的报表
    print(f"\n{'='*20} EXPERIMENT 1: SENSITIVITY REPORT {'='*20}")
    print(f"{'SOH Stage':<30} | {'MSE Loss':<15} | {'Samples'}")
    print("-" * 65)
    
    # 倒序打印 (从健康 -> 故障)
    for stage in labels[::-1]:
        mse = stage_mse.get(stage, float('nan'))
        count = stage_count.get(stage, 0)
        print(f"{stage:<30} | {mse:.8f}        | {count}")
    print(f"{'='*68}\n")
    
    return df_results # 返回原始数据以便后续画图或保存

def validate_robustness(model, loader, noise_level, device):
    """
    在特定噪声水平下评估模型
    """
    model.eval()
    total_loss = 0
    criterion = torch.nn.MSELoss()
    
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            
            # # 只在第一次迭代时检查噪声水平
            # if i == 0:
            #     check_noise_level(x, noise_level)
            
            # === 核心步骤：注入噪声 ===
            x_noisy = add_noise(x, noise_level)
            
            # End-to-End 模型前向传播 (注意元组解包 preds, _)
            preds, _ = model(x_noisy)
            
            loss = criterion(preds, y)
            total_loss += loss.item()
            
    return total_loss / len(loader)

def main(args):
    # === 2. 加载配置 ===
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # 动态加载 Config
    # 注意：根据您的文件结构，这里可能需要调整路径
    Configs = dynamic_import('config_files.xjtu.cell_XJTU_batch1_data_Configs', 'Config')
    configs = Configs()
    
    # 覆盖 output_module 配置
    configs.output_module = f"predictor_{args.output_module}"
    
    # === 3. 初始化模型 ===
    model_module_path = f'models.{args.base_model}'
    base_Model = dynamic_import(model_module_path, 'base_Model')
    model = base_Model(configs).to(args.device)
    
    # === 4. 加载 Checkpoint (完全复用您的逻辑) ===
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
    
    # 权重加载与 Key 映射处理
    if args.output_module == 'linear':
        new_state_dict = {}
        for k, v in pretrained_dict.items():
            name = k.replace("module.", "")
            # 重命名 fc -> predictor
            if name.startswith("fc."):
                name = name.replace("fc.", "predictor.")
                # print(f"Remapping key: {k} -> {name}") 
            new_state_dict[name] = v
        msg = model.load_state_dict(new_state_dict, strict=True)
    else:
        msg = model.load_state_dict(pretrained_dict, strict=True)
    print(f"Model loaded status: {msg}")
    

    # ===  准备测试数据 ===
    print(f"Loading dataset from {args.data_path} ...")
    # weights_only=False 是为了兼容旧版 pytorch 保存的文件
    dataset_content = torch.load(args.data_path, map_location='cpu', weights_only=False) 
    
    # 使用 analysis 模式或 test 模式加载数据
    test_dataset = Load_Dataset(dataset_content, configs, training_mode='analysis', dataset_name='xjtu')
    
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=configs.batch_size,
        drop_last=False, # False用于sensitivity分析，True用于Robustness分析
        shuffle=False,
        num_workers=0
    )

    analyze_sensitivity(model, test_loader, device, noise_sigma=0.0)
    # === 6. 鲁棒性评估循环 ===
    # noise_levels = [0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
    target_snrs = [float('inf'), 40, 30, 25, 20, 15, 10, 5]
    results = []
    
    print(f"Starting Robustness Evaluation for {args.output_module}...")
    print(f"{'Target SNR(dB)': <15} | {'Noise Level':<15} | {'MSE Loss':<15}")
    print("-" * 50)
    
    for snr in target_snrs:
    # 1. 为 Auto-embedding 模型计算 sigma
    # 假设 auto_test_loader 是原始数据的 loader
        sigma_auto = get_sigma_from_snr_loader(test_loader, snr, device='cpu')
        val_loss = validate_robustness(model, test_loader, sigma_auto, args.device)
        
        print(f"{snr:<15} | {sigma_auto:<15.3f} | {val_loss:.6f}")
        results.append({'snr':snr, 'noise_sigma': sigma_auto, 'loss': val_loss})
        
    # === 7. 保存结果 ===
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        
    save_name = f'robustness_{args.output_module}_seed{args.seed}_fold{args.fold}.npy'
    save_path = os.path.join(args.save_dir, save_name)
    np.save(save_path, results)
    print(f"\nResults saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate E2E Model Robustness")

    # Model Configs
    parser.add_argument('--base_model', type=str, default='cnn', help='Name of the python file in models/ folder')
    parser.add_argument('--output_module', type=str, default='mlp', choices=['predictor_mlp', 'predictor_cnn', 'predictor_lstm', 'predictor_transformer', 'linear'], help='Which predictor to use')

    # Paths
    parser.add_argument('--data_path', type=str, default='/mnt/wenjt5/TC-SOH/data/XJTU/cell_XJTU_batch1_data/test.pt', help='Path to .pt dataset file')
    parser.add_argument('--save_dir', type=str, default='/mnt/wenjt5/TC-SOH/experiments_logs/robustness', help='Where to save the results')

    # Experiment Settings (用于定位 checkpoint)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--fold', type=str, default='2', help='Fold number string')
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    main(args)
