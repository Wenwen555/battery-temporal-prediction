import argparse
import os
from datetime import datetime
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dataloader.dataloader_analyze import Load_Dataset
from models.TC import TC
from trainer.fine_tuned_trainer import Trainer_f
from utils import _logger, set_requires_grad
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, Subset
import importlib
from scipy.signal import savgol_filter
from tqdm import tqdm
from npeet_plus import mi
from sklearn.feature_selection import mutual_info_regression

def dynamic_import(module_path, class_name):
    """动态导入类"""
    module = importlib.import_module(module_path)
    return getattr(module, class_name)

def parse_arguments():
    """解析命令行参数"""
    home_dir = os.getcwd()
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='mit', type=str, help='Experiment Dataset')
    parser.add_argument('--experiment_description', default='', type=str, help='Experiment Description')
    parser.add_argument('--selected_subset', default='cell_mit_batch1', help='subset of a dataset')
    parser.add_argument('--run_description', default='test1', type=str, help='Experiment Description')
    parser.add_argument('--base_model', default='cnn', type=str, help='cnn, lstm, mlp, imv_lstm, transformer')
    parser.add_argument('--seed', default=123, type=int, help='seed value')
    parser.add_argument('--training_mode', default='supervised_with_contrast', type=str,
                      help='Modes of choice: supervised, supervised_with_contrast, predict_module')
    parser.add_argument('--small_sample_num', default=None, type=int, help='The number of small sample in experiment')
    parser.add_argument('--random_select', default=home_dir, type=str, help='Project home directory')
    parser.add_argument('--data_path', default=r'data/', type=str, help='Path containing dataset')
    parser.add_argument('--logs_save_dir', default='experiments_logs', type=str, help='saving directory')
    parser.add_argument('--device', default='cuda:0', type=str, help='cpu or cuda')
    parser.add_argument('--home_path', default=home_dir, type=str, help='Project home directory')
    parser.add_argument('--model_path', default=None, type=str, help='Path to checkpoint')
    parser.add_argument('--manual_data_path', default=None, type=str, help='Path to manual_data')
    return parser.parse_args()

def setup_environment(args):
    """设置实验环境和随机种子"""
    # 创建日志目录
    os.makedirs(args.logs_save_dir, exist_ok=True)
    
    # 设置随机种子
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)

def load_from_ckpt(model, temporal_contr_model, model_path, device):
     # 从检查点加载模型
    chkpoint = torch.load(os.path.join(model_path, "ckp_last.pt"), map_location=device)
    pretrained_model_dict = chkpoint["model_state_dict"]
    pretrained_temporal_model_dict = chkpoint["temporal_contr_model_state_dict"]
    
    model.load_state_dict(pretrained_model_dict, strict=False)
    set_requires_grad(model, pretrained_model_dict, requires_grad=False)
    
    return model, temporal_contr_model

def load_data(data_path, configs, training_mode, dataname):
    """加载数据集"""
    test_dataset = torch.load(os.path.join(data_path, "test.pt"), weights_only=False)
    test_dataset = Load_Dataset(test_dataset, configs, training_mode, dataname)
    test_loader = DataLoader(dataset=test_dataset, batch_size=16,
                           shuffle=False, drop_last=True, num_workers=0)
    return test_loader

def extract_features(model, data_loader, device):
    """提取特征"""
    all_features = []
    all_labels = []
    for batch_idx, (data, labels) in enumerate(data_loader):
        data = data.to(device)
        labels = labels.to(device)
        _, features = model(data)
        all_features.append(features)
        all_labels.append(labels)
    
    all_features = torch.concatenate(all_features, axis=0)
    all_labels = torch.concatenate(all_labels, axis=0)
    return all_features, all_labels

def preprocess_features(features, use_pca):
    """特征预处理"""
    if use_pca:
        features_np = features.cpu().numpy()
        features_np[np.abs(features_np) < 1e-10] = 0
        pca = PCA(n_components=50)
        features_pca = pca.fit_transform(features_np)
        features_pca = np.sign(features_pca) * np.log1p(np.abs(features_pca)) * 10
        return features_pca
    else:
        return features.cpu()

def plot_t_sne_features(tsne_results, dataname, data_type, manual=False):
    """可视化特征"""
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        tsne_results[:, 0],
        tsne_results[:, 1],
        c=np.arange(len(tsne_results)),
        cmap='viridis',
        alpha=0.4,
        s=30,
    )
    cbar = plt.colorbar(scatter)
    cbar.set_label('Cycle Index', fontsize=22)
    if not manual:
        plt.title("Automatic Extraction", fontsize=24)
    else:
        plt.title("Manual Features", fontsize=24)
    plt.grid(alpha=0.2)
    
    plt.tight_layout()
    plt.tick_params(axis='both', labelsize=16)
    prefix = 'manual' if manual else 'test'
    plt.savefig(f'plot/{prefix}-{dataname}-{data_type}-tsne.pdf')
    plt.show()

def analyze_mutual_information(features_manual, is_manual, dataset, subset):
    """分析互信息"""
    print("Analyzing Mutual Infro...")
    N = len(features_manual)
    mi_matrix = np.zeros((N, N))
    # 初始化进度条
    with tqdm(total=(N * (N - 1)) // 2) as pbar:
        for i in range(1, N):
            for j in range(i + 1, N):
                mutual_info = mi(features_manual[i], features_manual[j])
                mi_matrix[i, j] = mutual_info
                mi_matrix[j, i] = mutual_info
                pbar.update(1)
    
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(mi_matrix, cmap='viridis', cbar=True)
    plt.title("Mutual Information Between All Cycle Pairs")
    if is_manual:
        plt.savefig(f"plot/manual_mutual_info-{dataset}-{subset}.svg")
        np.save(f'manual_mi_matrix-{dataset}-{subset}.npy',mi_matrix)
    else: 
        plt.savefig(f"plot/mutual_info-{dataset}-{subset}.svg")
        np.save(f'mi_matrix-{dataset}-{subset}.npy',mi_matrix)

def comparison_mutual_information(dataset, subset):
    print("Comparison begin...")
    mi_matrix = np.load(f'mi_matrix-{dataset}-{subset}.npy')
    mi_matrix_manual = np.load(f'manual_mi_matrix-{dataset}-{subset}.npy')
    mi_auto_smooth = savgol_filter(mi_matrix[2][3:], window_length=11, polyorder=3)
    mi_manual_smooth = savgol_filter(mi_matrix_manual[2][3:], window_length=11, polyorder=3)

    plt.plot(mi_auto_smooth, 'r', alpha=0.5, linewidth=2, label='Auto Features (smooth)')
    plt.plot(mi_manual_smooth, 'b', alpha=0.5, linewidth=2, label='Manual Features (smooth)')
    plt.legend()
    # plt.plot(mi_matrix[0][1:], 'r', alpha=0.7)
    # plt.plot(mi_matrix_manual[0][1:], 'b', alpha=0.7)
    plt.savefig(f'plot/comparison_mi-{dataset}-{subset}.svg')

def mutual_infomation_with_label(features, labels, is_manual):
    # 计算每个特征与标签的NMI
    mi = mutual_info_regression(features, labels)
    # print("mi scores per feature:", mi)
    if is_manual:
        print("Max mi in manual is: ", max(mi))
    else:
        print("Max mi is: ", max(mi))
    return mi

def main():
    # 1. 解析参数
    args = parse_arguments()
    # 2. 设置环境
    setup_environment(args)
    
    # 3. 加载配置
    config_module_path = f'config_files.{args.dataset}.{args.selected_subset}_Configs'
    Configs = dynamic_import(config_module_path, 'Config')
    configs = Configs()

    # 4. 加载模型
    model_module_path = f'models.{args.base_model}'
    base_Model = dynamic_import(model_module_path, 'base_Model')
    model = base_Model(configs).to(args.device)
    temporal_contr_model = TC(configs, args.device).to(args.device)
    model, temporal_contr_model = load_from_ckpt(model,temporal_contr_model,args.model_path, args.device)
    
    # 5. 加载数据
    data_path = os.path.join(args.data_path, args.selected_subset)
    test_loader = load_data(data_path, configs, args.training_mode, args.dataset)

    # 6. 提取特征
    all_features, all_labels = extract_features(model, test_loader, args.device)
    manual_data = pd.read_csv(args.manual_data_path, encoding="utf-8")
    features_manual = manual_data.iloc[:, :-1].values  # 转为 numpy array
    labels_manual = manual_data.iloc[:, -1].values
    if np.isnan(features_manual).any() or np.isinf(features_manual).any():
        print("Warning: features_manual contains infinity values. Replacing with finite values.")
        features_manual = np.nan_to_num(features_manual,nan=0.0, posinf=1e5, neginf=-1e5)
    
    # 7. 特征预处理
    features = preprocess_features(all_features, use_pca=False)
    
    # 分析表征和labels之间的关联
    # mi_scores = mutual_infomation_with_label(features, all_labels.cpu(), is_manual=False)
    # mi_scores_manual = mutual_infomation_with_label(features_manual[:, :-1], features_manual[:, -1], is_manual=True)

    # 8. 可视化分析
    tsne = TSNE(n_components=2, perplexity=23, n_iter=1000, random_state=42)
    # # plot
    tsne_results = tsne.fit_transform(features)
    plot_t_sne_features(tsne_results, args.dataset, args.selected_subset)
    tsne_results_manual = tsne.fit_transform(features_manual)
    plot_t_sne_features(tsne_results_manual, args.dataset, args.selected_subset, manual=True)
    
    # 对比两种方法得到的表征
    # comparison_mutual_information(args.dataset, 'batch1')

    # 9. 加载手动特征并分析
    # analyze_mutual_information(features, False, args.dataset, 'batch1')
    # analyze_mutual_information(features_manual, True, args.dataset, 'batch1')

if __name__ == "__main__":
    main()