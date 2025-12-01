import argparse
import os
from datetime import datetime
import numpy as np
import torch
from dataloader.dataloader import Load_Dataset
from models.TC import TC
from trainer.trainer import Trainer, Trainer_S
from trainer.small_sample_trainer import train_small_sample
from trainer.fine_tuned_trainer import Trainer_f
from utils import _logger, set_requires_grad
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset
import importlib

def dynamic_import(module_path, class_name):
    """动态导入类"""
    module = importlib.import_module(module_path)
    return getattr(module, class_name)

def run_training(args=None, train_dataloader=None, val_dataloader=None, test_dataloader=None):
    # 如果从外部调用时没有传入args，则从命令行解析
    if args is None:
        start_time = datetime.now()
        parser = argparse.ArgumentParser()
        ######################## Model parameters ########################
        home_dir = os.getcwd()
        parser.add_argument('--dataset', default='mit', type=str, help='Experiment Dataset')
        parser.add_argument('--experiment_description', default='', type=str, help='Experiment Description')
        parser.add_argument('--selected_subset', default='cell_mit_batch1',help='subset of a dataset')
        parser.add_argument('--run_description', default='test1', type=str, help='Experiment Description')
        parser.add_argument('--base_model',default='cnn',type=str, help='cnn, lstm, mlp, imv_lstm, transformer')
        parser.add_argument('--seed', default=123, type=int, help='seed value')
        parser.add_argument('--training_mode', default='supervised_with_contrast', type=str,
                            help='Modes of choice: supervised, supervised_with_contrast, transfer')
        parser.add_argument('--small_sample_num', default=None, type=int, help='The number of small sample in experiment')
        parser.add_argument('--data_path', default=r'data/', type=str, help='Path containing dataset')
        parser.add_argument('--logs_save_dir', default='experiments_logs', type=str, help='saving directory')
        parser.add_argument('--device', default='cuda:0', type=str, help='cpu or cuda')
        parser.add_argument('--home_path', default=home_dir, type=str, help='Project home directory')
        args = parser.parse_args()
    else:
        start_time = datetime.now()
    
    device = torch.device(args.device)
    experiment_description = args.experiment_description
    data_type = args.selected_subset
    dataname = args.dataset
    training_mode = args.training_mode
    small_sample_num = args.small_sample_num
    run_description = args.run_description

    logs_save_dir = args.logs_save_dir
    os.makedirs(logs_save_dir, exist_ok=True)

    # exec(f'from config_files.{dataname}.{data_type}_Configs import Config as Configs')
    # exec (f'from models.{base_model} import base_Model')
    # load config
    config_module_path = f'config_files.{args.dataset}.{args.selected_subset}_Configs'
    Configs = dynamic_import(config_module_path, 'Config')
    configs = Configs()

    # load model
    model_module_path = f'models.{args.base_model}'
    base_Model = dynamic_import(model_module_path, 'base_Model')

    # ##### fix random seeds for reproducibility ########
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)
    #####################################################

    # Load Model
    temporal_contr_model = TC(configs,device).to(device)

    # Load datasets
    data_path = os.path.join(args.data_path, data_type)
    batch_size = configs.batch_size

    dataset = torch.load(os.path.join(data_path, 'train_val.pt'),weights_only=False)
    test_dataset = torch.load(os.path.join(data_path, "test.pt"),weights_only=False)
    test_dataset = Load_Dataset(test_dataset, configs, training_mode, dataname) 
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size,
                                              shuffle=False, drop_last=True, num_workers=4)

    if small_sample_num != 0:
        data_path = os.path.join(data_path, f"random_select_{small_sample_num}_bat")
        experiment_log_dir = os.path.join(logs_save_dir, experiment_description, run_description, training_mode + f"_seed_{SEED}")
        os.makedirs(experiment_log_dir, exist_ok=True)
        train_small_sample(data_path, experiment_log_dir, SEED, training_mode, small_sample_num, configs, device, test_loader)
    elif train_dataloader != None:
        assert val_dataloader != None and test_dataloader != None
        model_optimizer = torch.optim.Adam(list(model.parameters()), lr=configs.lr, 
            betas=(configs.beta1, configs.beta2), weight_decay=3e-4)
        training_mode = "supervised_with_contrast"
        temporal_contr_optimizer = torch.optim.Adam(list(temporal_contr_model.parameters()), lr=configs.lr, betas=(configs.beta1, configs.beta2), weight_decay=3e-4)
        Trainer(model,temporal_contr_model,model_optimizer, temporal_contr_optimizer, train_dataloader, val_dataloader, test_dataloader, device, logger, configs, experiment_log_dir, training_mode)
    else:
        k_folds = configs.k_fold
        kf = KFold(n_splits=k_folds, shuffle=True)
        dataset_size = len(dataset)
        indices = np.arange(dataset_size)
        # 使用k-fold交叉验证来训练!
        for fold, (train_idx, valid_idx) in enumerate(kf.split(indices)):
            model = base_Model(configs).to(device)
            print(f'Fold {fold + 1}/{k_folds}')
            if configs.output_module != 'linear':
                experiment_log_dir = os.path.join(logs_save_dir, 
                experiment_description, configs.output_module, run_description, training_mode + f"_seed_{SEED}", str(fold + 1))
            else:
                experiment_log_dir = os.path.join(logs_save_dir, 
                experiment_description, run_description, training_mode + f"_seed_{SEED}", str(fold + 1))
            os.makedirs(experiment_log_dir, exist_ok=True)

            # Logging
            log_file_name = os.path.join(experiment_log_dir,
                                            f"logs_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log")
            logger = _logger(log_file_name)
            logger.debug("=" * 45)
            logger.debug(f'Dataset: {data_type}')
            logger.debug(f'Mode:    {training_mode}')
            logger.debug("=" * 45)

            train_subset = Subset(dataset, train_idx)
            valid_subset = Subset(dataset, valid_idx)

            train_dataset = Load_Dataset(train_subset, configs, training_mode, dataname)
            valid_dataset = Load_Dataset(valid_subset, configs, training_mode, dataname)
            train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, drop_last=True, shuffle=True, num_workers=4)
            valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=64, shuffle=False, drop_last=configs.drop_last, num_workers=4)

            # 使用 Subset 创建训练集和验证集
            model_optimizer = torch.optim.Adam(list(model.parameters()), lr=configs.lr,
            betas=(configs.beta1, configs.beta2), weight_decay=3e-4)
            temporal_contr_optimizer = torch.optim.Adam(list(temporal_contr_model.parameters()), lr=configs.lr, betas=(configs.beta1, configs.beta2), weight_decay=3e-4)
            Trainer(model,temporal_contr_model,model_optimizer, temporal_contr_optimizer, train_loader, valid_loader, test_loader, device, logger, configs, experiment_log_dir, training_mode)

    print("Finish training!")
    # logger.debug(f"Training time is : {datetime.now() - start_time}")

    return {
        'model': model,
        'temporal_contr_model': temporal_contr_model,
        'configs': configs,
        'device': device
    }

# 当作为脚本运行时执行
if __name__ == '__main__':
    run_training()