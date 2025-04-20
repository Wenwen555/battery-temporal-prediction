import argparse
import os

from datetime import datetime

import numpy as np
import torch

from dataloader.dataloader import Load_Dataset
from models.TC import TC
# from dataloader.original_dataloader import custom_collate_fn_valid
# from models.TC_batch import TC  # testing
from trainer.trainer import Trainer, Trainer_S
from trainer.fine_tuned_trainer import Trainer_f
from utils import _logger, set_requires_grad

from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset

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
                    help='Modes of choice: supervised, supervised_with_contrast, predict_module')
parser.add_argument('--small_sample_num', default=None, type=int, help='The number of small sample in experiment')
parser.add_argument('--random_select', default=home_dir, type=str, help='Project home directory')
parser.add_argument('--data_path', default=r'data/', type=str, help='Path containing dataset')
parser.add_argument('--logs_save_dir', default='experiments_logs', type=str, help='saving directory')
parser.add_argument('--device', default='cuda:0', type=str, help='cpu or cuda')
parser.add_argument('--home_path', default=home_dir, type=str, help='Project home directory')
args = parser.parse_args()

device = torch.device(args.device)
experiment_description = args.experiment_description
data_type = args.selected_subset
dataname = args.dataset
run_description = args.run_description
training_mode = args.training_mode
small_sample_num = args.small_sample_num
run_description = args.run_description
base_model = args.base_model
random_select = args.random_select

logs_save_dir = args.logs_save_dir
os.makedirs(logs_save_dir, exist_ok=True)

exec(f'from config_files.{dataname}.{data_type}_Configs import Config as Configs')
exec (f'from models.{base_model} import base_Model')
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

# Load datasets
data_path = os.path.join(args.data_path, data_type)

batch_size = configs.batch_size
if training_mode == "fine_tuned":
    dataset = torch.load(os.path.join(data_path, 'one_bat.pt'),weights_only=False)
else:
    if small_sample_num != None and random_select == None:
        dataset_name = ['one_bat.pt','two_bat.pt','three_bat.pt','four_bat.pt']
        dataset = torch.load(os.path.join(data_path, dataset_name[small_sample_num-1]),weights_only=False)
    else:
        dataset = torch.load(os.path.join(data_path, 'train_val.pt'),weights_only=False)
test_dataset = torch.load(os.path.join(data_path, "test.pt"),weights_only=False)
test_dataset = Load_Dataset(test_dataset, configs, training_mode,dataname) 
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size,
                                          shuffle=False, drop_last=True, num_workers=4)


# 思路：首先在一个数据集上训练，得到TC模型的参数，然后在另一个数据集的某一个batch上，把TC模块的参数freeze
# 考虑用XJTU得到的model在TJU的1st Battery上fine_tuned
if training_mode == "fine_tuned":
    model = base_Model(configs).to(device)
    path_xjtu = '/mnt/wenjt5/project1/experiments_logs/hust/cnn_hust/'
    load_from = os.path.join(
        os.path.join(path_xjtu, f"supervised_with_contrast_seed_{SEED}", '1/'
                     "saved_models"))
    chkpoint = torch.load(os.path.join(load_from, "ckp_last.pt"), map_location=device)
    pretrained_model_dict = chkpoint["model_state_dict"]
    pretrained_temporal_model_dict = chkpoint["temporal_contr_model_state_dict"]
    
    model_dict = model.state_dict()
    temporal_contr_model_dict = temporal_contr_model.state_dict()
    
    # 1. filter out unnecessary keys
    # pretrained_dict = {k: v for k, v in pretrained_model_dict.items() if k in model_dict}
   # 1. 修改预训练模型的权重（从HUST的[64,3,20]扩展到MIT的[64,4,20]）
    if 'conv_block.0.weight' in pretrained_model_dict:
        pretrained_conv_weight = pretrained_model_dict['conv_block.0.weight']  # shape [64,3,20]
        # 初始化一个新的权重张量 [64,4,20]
        expanded_conv_weight = torch.zeros(64, 4, 20)  

        # 将预训练权重填入前3个通道（假设第4个通道是新增的temperature，初始化为0或随机值）
        expanded_conv_weight[:, :3, :] = pretrained_conv_weight  # 复制前3个通道

        # 第4个通道（temperature相关）可以保持为0或随机初始化
        # expanded_conv_weight[:, 3:, :] = torch.randn(64, 1, 20) * 0.01  # 小随机数初始化
        
        # 更新预训练字典中的权重
        pretrained_model_dict['conv_block.0.weight'] = expanded_conv_weight
    # 更新模型参数
    model_dict.update(pretrained_model_dict)
    temporal_contr_model_dict.update(pretrained_temporal_model_dict)
    
    # 加载模型参数
    model.load_state_dict(pretrained_model_dict,strict=False)
    temporal_contr_model.load_state_dict(pretrained_temporal_model_dict)
    set_requires_grad(temporal_contr_model, pretrained_temporal_model_dict, requires_grad=False) # Freeze TC module.
    # 检查冻结模块的梯度
    # for param in temporal_contr_model.parameters():
    #     if param.requires_grad:
    #         print(f'Gradients for param after backward: {param.grad}')
    #     else:
    #         print(f'No gradients for param (frozen)')
    
    ### fine-tuning on 1st battery in TJU
    experiment_log_dir = os.path.join(logs_save_dir, experiment_description, run_description,
                                      training_mode + f"_seed_{SEED}")
    os.makedirs(experiment_log_dir, exist_ok=True)
    # Logging
    log_file_name = os.path.join(experiment_log_dir,
                                 f"logs_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log")
    logger = _logger(log_file_name)
    logger.debug("=" * 45)
    logger.debug(f'Dataset: {data_type}')
    logger.debug(f'Mode:    {training_mode}')
    logger.debug("=" * 45)
    
    train_dataset = Load_Dataset(dataset, configs, training_mode, dataname)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=32, drop_last=True, shuffle=True, num_workers=4)
    model_optimizer = torch.optim.Adam(list(model.parameters()), lr=configs.lr_f, betas=(configs.beta1, configs.beta2), weight_decay=3e-4)
    # Trainer
    Trainer_f(model, temporal_contr_model, model_optimizer, train_loader, test_loader, device, logger, configs, experiment_log_dir, training_mode)
    print("Finish fine-tuning!")
    
else:
    if small_sample_num != None:
        if random_select:
            random_selection_folder_path = os.path.join(data_path,f"random_select_{small_sample_num}_bat")
            pt_files = [f for f in os.listdir(random_selection_folder_path) if f.endswith(".pt")]
            for file in pt_files:
                dataset = torch.load(os.path.join(random_selection_folder_path, file),weights_only=False)
                model = base_Model(configs).to(device)
                file_save = file.split(".")[0]
                experiment_log_dir = os.path.join(logs_save_dir, experiment_description, run_description, file_save,
                                                      training_mode + f"_seed_{SEED}")
                os.makedirs(experiment_log_dir, exist_ok=True)

                # Logging
                log_file_name = os.path.join(experiment_log_dir,
                                             f"logs_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log")
                logger = _logger(log_file_name)
                logger.debug("=" * 45)
                logger.debug(f'Random selection training')
                logger.debug("=" * 45)
                logger.debug(f'Dataset: {data_type}')
                logger.debug(f'Mode:    {training_mode}')
                logger.debug(f'Sample Nums: {small_sample_num}')
                logger.debug("=" * 45)

                train_dataset = Load_Dataset(dataset, configs, training_mode, dataname)
                train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=16, drop_last=True, shuffle=True, num_workers=0)

                model_optimizer = torch.optim.Adam(list(model.parameters()), lr=configs.lr, betas=(configs.beta1, configs.beta2), weight_decay=3e-4)
                temporal_contr_optimizer = torch.optim.Adam(list(temporal_contr_model.parameters()), lr=configs.lr, betas=(configs.beta1, configs.beta2), weight_decay=3e-4)
                # Trainer S
                Trainer_S(model, temporal_contr_model, model_optimizer,temporal_contr_optimizer, train_loader, test_loader, device, logger, configs, experiment_log_dir, training_mode)
        else:
            model = base_Model(configs).to(device)
            experiment_log_dir = os.path.join(logs_save_dir, experiment_description, run_description,
                                                  training_mode + f"_seed_{SEED}")
            os.makedirs(experiment_log_dir, exist_ok=True)

            # Logging
            log_file_name = os.path.join(experiment_log_dir,
                                         f"logs_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log")
            logger = _logger(log_file_name)
            logger.debug("=" * 45)
            logger.debug(f'Dataset: {data_type}')
            logger.debug(f'Mode:    {training_mode}')
            logger.debug(f'Sample Nums: {small_sample_num}')
            logger.debug("=" * 45)

            train_dataset = Load_Dataset(dataset, configs, training_mode, dataname)
            train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=16, drop_last=True, shuffle=True, num_workers=0)

            model_optimizer = torch.optim.Adam(list(model.parameters()), lr=configs.lr, betas=(configs.beta1, configs.beta2), weight_decay=3e-4)
            temporal_contr_optimizer = torch.optim.Adam(list(temporal_contr_model.parameters()), lr=configs.lr, betas=(configs.beta1, configs.beta2), weight_decay=3e-4)
            # Trainer S
            Trainer_S(model, temporal_contr_model, model_optimizer,temporal_contr_optimizer, train_loader, test_loader, device, logger, configs, experiment_log_dir, training_mode)
    else:
        k_folds = configs.k_fold
        kf = KFold(n_splits=k_folds, shuffle=True)
        dataset_size = len(dataset)
        indices = np.arange(dataset_size)
        # 使用k-fold交叉验证来训练!
        for fold, (train_idx, valid_idx) in enumerate(kf.split(indices)):
            model = base_Model(configs).to(device)
            print(f'Fold {fold + 1}/{k_folds}')
            #
            experiment_log_dir = os.path.join(logs_save_dir, experiment_description, run_description,
                                              training_mode + f"_seed_{SEED}", str(fold + 1))
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
            train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=32, drop_last=True, shuffle=True, num_workers=4)
            valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=32,                        shuffle=False, drop_last=configs.drop_last, num_workers=4)

            # 使用 Subset 创建训练集和验证集
            model_optimizer = torch.optim.Adam(list(model.parameters()), lr=configs.lr,
    betas=(configs.beta1, configs.beta2), weight_decay=3e-4)
            temporal_contr_optimizer = torch.optim.Adam(list(temporal_contr_model.parameters()), lr=configs.lr, betas=(configs.beta1, configs.beta2), weight_decay=3e-4)

            Trainer(model,temporal_contr_model,model_optimizer, temporal_contr_optimizer, train_loader, valid_loader, test_loader, device, logger, configs, experiment_log_dir, training_mode)
    print("Finish training!")
    logger.debug(f"Training time is : {datetime.now() - start_time}")


