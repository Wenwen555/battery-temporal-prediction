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
import importlib

def dynamic_import(module_path, class_name):
    """动态导入类"""
    module = importlib.import_module(module_path)
    return getattr(module, class_name)

def run_training(args=None, train_dataloader=None, val_dataloader=None, test_dataloader=None, logger=None, experiment_log_dir=None):
    # 如果从外部调用时没有传入args，则从命令行解析
    assert args is not None
    start_time = datetime.now()
    
    device = torch.device(args.device)
    experiment_description = args.experiment_description
    data_type = args.selected_subset
    dataname = args.source_dataset
    run_description = args.run_description
    training_mode = args.training_mode
    run_description = args.run_description
    base_model = args.base_model

    logs_save_dir = args.logs_save_dir
    os.makedirs(logs_save_dir, exist_ok=True)
    
    # load config
    config_module_path = f'config_files.{args.source_dataset}.{args.selected_subset}_Configs'
    Configs = dynamic_import(config_module_path, 'Config')
    configs = Configs()


    # ##### fix random seeds for reproducibility ########
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)
    #####################################################

    # Load Model
    temporal_contr_model = TC(configs,device).to(device)
    model_module_path = f'models.{args.base_model}'
    base_Model = dynamic_import(model_module_path, 'base_Model')
    model = base_Model(configs).to(device)
    # Load datasets
    batch_size = configs.batch_size

    model_optimizer = torch.optim.Adam(list(model.parameters()), lr=configs.lr, 
        betas=(configs.beta1, configs.beta2), weight_decay=3e-4)
    temporal_contr_optimizer = torch.optim.Adam(list(temporal_contr_model.parameters()), lr=configs.lr, betas=(configs.beta1, configs.beta2), weight_decay=3e-4)
    Trainer(model,temporal_contr_model,model_optimizer, temporal_contr_optimizer, train_dataloader, val_dataloader, test_dataloader, device, logger, configs, experiment_log_dir, training_mode)

    print("Finish training!")
    logger.debug(f"Training time is : {datetime.now() - start_time}")

    return {
        'model': model,
        'temporal_contr_model': temporal_contr_model,
        'configs': configs,
        'device': device
    }

# 当作为脚本运行时执行
if __name__ == '__main__':
    run_training()