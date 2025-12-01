import torch
import numpy as np
import argparse
import importlib
from dataloader.dataloader import Load_Dataset
import os

def dynamic_import(module_path, class_name):
    """动态导入类"""
    module = importlib.import_module(module_path)
    return getattr(module, class_name)

def validate(model, loader, device):
    model.eval()
    total_loss = 0
    criterion = torch.nn.MSELoss()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds, _ = model(x)
            loss = criterion(preds, y)
            total_loss += loss.item()
    return total_loss / len(loader)


def main(args):
    # 1. 加载配置和模型
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    Configs = dynamic_import('config_files.xjtu.cell_XJTU_batch1_data_Configs', 'Config')
    configs = Configs()
    
    configs.output_module = f"predictor_{args.output_module}"
    
    model_module_path = f'models.{args.base_model}'
    base_Model = dynamic_import(model_module_path, 'base_Model')
    model = base_Model(configs).to(args.device)
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
    
    # import ipdb; ipdb.set_trace()
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
    
    # 2. 加载特征排序
    ranking = np.load(args.ranking_path)
    model.feature_ranking = ranking # 赋值给模型
    
    # 3. 准备测试集
    print(f"Loading dataset from {args.data_path} ...")
    dataset_content = torch.load(args.data_path, map_location='cpu', weights_only=False) # weights_only=False 默认
    
    # 使用 training_mode='train' (或者 'test'，建议用训练集或验证集计算重要性)
    test_dataset = Load_Dataset(dataset_content, configs, training_mode='analysis', dataset_name='xjtu')
    
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=configs.batch_size,
        drop_last=True, # Drop last 避免 shape 不匹配问题
        shuffle=False,
        num_workers=0
    )
 
    # 4. 定义要测试的维度列表
    dims_to_test = [10, 30, 50, 80, 100, 130, 150, 200, 256]
    
    results = []
    
    print(f"Starting evaluation for {args.output_module}...")
    
    for k in dims_to_test:
        model.active_feature_count = k
        # 运行验证/测试循环
        val_loss = validate(model, test_loader, args.device) # 你的验证函数
        
        print(f"Dim: {k}, Loss: {val_loss}")
        results.append({'dim': k, 'loss': val_loss})
        
    # 5. 保存结果用于画图
    np.save(f'{args.save_dir}/results_{args.output_module}.npy', results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate Feature Importance via Gradients")

    # Model Configs (这些参数必须与训练时一致，用于初始化模型结构)
    parser.add_argument('--base_model', type=str, default='cnn', help='Name of the python file in models/ folder')
    parser.add_argument('--output_module', type=str, default='lstm', choices=['predictor_mlp', 'predictor_cnn', 'predictor_lstm', 'predictor_transformer', 'linear'], help='Which predictor to use')

    # Paths
    parser.add_argument('--data_path', type=str, default='/mnt/wenjt5/TC-SOH/data/XJTU/cell_XJTU_batch1_data/test.pt', help='Path to .pt dataset file')
    parser.add_argument('--save_dir', type=str, default='/mnt/wenjt5/TC-SOH/experiments_logs/importance', help='Where to save the .npy file')
    parser.add_argument('--ranking_path', type=str, help='path to importance ranking')

    # Experiment Settings
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--fold', type=str, default='2', help='Fold number string')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', type=str, default='cuda')
    
    # 这里的 args 将被作为 configs 传入模型
    args = parser.parse_args()
    if args.ranking_path is None:
        args.ranking_path = f'/mnt/wenjt5/TC-SOH/experiments_logs/importance/xjtu_batch1_importance_ranking_{args.output_module}.npy'
    
    main(args)
