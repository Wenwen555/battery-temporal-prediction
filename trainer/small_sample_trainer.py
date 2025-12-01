# small_sample_trainer.py

import os
import torch
from datetime import datetime
from models.cnn import base_Model
from trainer.trainer import Trainer_S
from dataloader.dataloader import Load_Dataset
from utils import _logger
from models.TC import TC


def train_small_sample(data_path, experiment_log_dir, SEED, training_mode, small_sample_num, configs, device, test_loader):
    """
    小样本训练函数
    :param data_path: 数据路径
    :param experiment_log_dir: 日志保存目录
    :param SEED: 随机种子
    :param training_mode: 训练模式（supervised_with_contrast等）
    :param small_sample_num: 小样本数量
    :param configs: 配置参数对象
    :param device: 设备（cpu/cuda）
    :param test_loader: 测试集DataLoader
    """
    model = base_Model(configs).to(device)
    ## log_save_dir 需要修改

    # Logging
    log_file_name = os.path.join(
        experiment_log_dir,
        f"logs_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log"
    )
    logger = _logger(log_file_name)
    logger.debug("=" * 45)
    logger.debug(f'Random selection training')
    logger.debug("=" * 45)
    logger.debug(f'Dataset: cell_mit_batch1')
    logger.debug(f'Mode:    {training_mode}')
    logger.debug(f'Sample Nums: {small_sample_num}')
    logger.debug("=" * 45)

    # 加载小样本数据
    dataset_name = ['one_bat_1.pt','two_bat_1-2.pt','three_bat_1-2-3.pt','four_bat_1-2-3-5.pt']
    print("Loading data from: ", data_path)
    dataset = torch.load(os.path.join(data_path, dataset_name[small_sample_num-1]),weights_only=False)
    train_dataset = Load_Dataset(dataset, configs, training_mode, 'mixture')
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=16,
        drop_last=True,
        shuffle=True,
        num_workers=0
    )

    # 初始化优化器
    model_optimizer = torch.optim.Adam(
        list(model.parameters()),
        lr=configs.lr,
        betas=(configs.beta1, configs.beta2),
        weight_decay=3e-4
    )
    temporal_contr_model = TC(configs, device).to(device)
    temporal_contr_optimizer = torch.optim.Adam(
        list(temporal_contr_model.parameters()),
        lr=configs.lr,
        betas=(configs.beta1, configs.beta2),
        weight_decay=3e-4
    )

    # 开始训练
    Trainer_S(
        model=model,
        temporal_contr_model=temporal_contr_model,
        model_optimizer=model_optimizer,
        temp_cont_optimizer=temporal_contr_optimizer,
        train_dl=train_loader,
        test_dl=test_loader,
        device=device,
        logger=logger,
        config=configs,
        experiment_log_dir=experiment_log_dir,
        training_mode=training_mode
    )
    print("Finish small sample training!")