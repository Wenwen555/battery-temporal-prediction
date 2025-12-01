import os
import argparse
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold

import glob
import random
# 自定义模块导入
from models.TC import TC
from main4transfer import run_training

from dataloader.dataloader import Load_Dataset
from trainer.fine_tuned_trainer import Trainer_f
from utils import _logger, set_requires_grad

import importlib
def dynamic_import(module_path, class_name):
    """动态导入类"""
    module = importlib.import_module(module_path)
    return getattr(module, class_name)

class FineTunerRunner:
    '''
    类功能：实现三种迁移学习
    1. Source only: 直接在目标域(target domain)进行评估
    2. Fine_tuning: 用目标域的1~2个bat进行微调，冻结TC模块，只修改encoder
    3. Mixture Training: 将数据集混合，调用main函数进行训练，但问题在于：
        （1）数据不对齐
        （2）channel不对齐
        （3）如何合并两个数据集？ testloader是target_set的 
    因此需要回到My_Battery中重新处理
    '''
    def __init__(self):
        self.args = self._parse_args()
        self.device = torch.device(self.args.device)
        self.SEED = self.args.seed
        self._set_seeds()

        # 初始化变量
        self.configs = None
        self.model = None
        self.temporal_contr_model = None
        self.test_loader = None
        self.experiment_log_dir = None
        self.logger = None

    def _parse_args(self):
        """解析命令行参数"""
        parser = argparse.ArgumentParser(description="Fine-tuning script with contrastive learning")
        home_dir = os.getcwd()
        parser.add_argument('--target_dataset', default='tongji', type=str, help='Target Dataset')
        parser.add_argument('--source_dataset', default='hust', type=str, help='Source Dataset')
        parser.add_argument('--experiment_description', default='', type=str, help='Experiment Description')
        parser.add_argument('--selected_subset', default='cell_mit_batch1', help='subset of source dataset')
        parser.add_argument('--target_batch', default=None, help='subset of target_batch')
        parser.add_argument('--run_description', default='test1', type=str, help='Experiment Description')
        parser.add_argument('--base_model', default='cnn', type=str, help='Model architecture: cnn, lstm, mlp, etc.')
        parser.add_argument('--seed', default=123, type=int, help='Random seed for reproducibility')
        parser.add_argument('--training_mode', default='supervised_with_contrast', type=str, help='Training mode')
        parser.add_argument('--source_data_path', default=r'data/', type=str, help='Path containing source dataset')
        parser.add_argument('--target_data_path', default=r'data/', type=str, help='Path containing target dataset')
        parser.add_argument('--logs_save_dir', default='experiments_logs', type=str, help='Directory to save logs')
        parser.add_argument('--device', default='cuda:0', type=str, help='Device: cpu or cuda')
        parser.add_argument('--home_path', default=home_dir, type=str, help='Project root directory')
        parser.add_argument('--selected_ckpt', default=None, type=str, help='Path to pre-trained model checkpoint')
        parser.add_argument('--bat_num', default=1, type=int, help='Number of bat for finetuning.')
        return parser.parse_args()

    def _set_seeds(self):
        """设置随机种子"""
        torch.manual_seed(self.SEED)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = False
        np.random.seed(self.SEED)

    def _setup_directories_and_configs(self):
        """加载配置文件 & 创建实验目录"""
        config_module_path = f'config_files.{self.args.source_dataset}.{self.args.selected_subset}_Configs'
        Configs = dynamic_import(config_module_path, 'Config')
        self.configs = Configs()

        # 构建实验日志目录
        self.experiment_log_dir = os.path.join(
            self.args.logs_save_dir,
            self.args.experiment_description,
            self.args.run_description,
            self.args.training_mode + f"_seed_{self.SEED}"
        )
        os.makedirs(self.experiment_log_dir, exist_ok=True)

        # 初始化日志
        log_file_name = os.path.join(
            self.experiment_log_dir,
            f"logs_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log"
        )
        self.logger = _logger(log_file_name)
        self.logger.debug("=" * 45)
        self.logger.debug(f'Dataset: {self.args.selected_subset}')
        self.logger.debug(f'Mode:    {self.args.training_mode}')
        self.logger.debug("=" * 45)

    def _load_datasets(self):
        """加载数据集"""
        source_data_path = os.path.join(self.args.source_data_path, self.args.selected_subset)
        target_data_path = self.args.target_data_path
        #设置随机数来确保复现
        seed_value = 42
        random.seed(seed_value)
        mixture_bat_data_path = os.path.join(source_data_path, 'mixture', f"mixed_{self.args.target_dataset}", f'mixed_{self.args.source_dataset}_with_{self.args.target_dataset}_batch{self.args.target_batch}_{self.args.bat_num}_bat')
        target_bat_data_path = os.path.join(target_data_path, f'random_select_{self.args.bat_num}_bat')
        
        if self.args.bat_num == 2:
            file_pattern = os.path.join(target_bat_data_path, 'two_bat_*.pt')
        else: 
            file_pattern = os.path.join(target_bat_data_path, 'one_bat_*.pt')
        files = glob.glob(file_pattern)
        selected_files = random.sample(files, k=1)

        # 这里的target_train指的是finetuning过程中微调的电池
        target_train_data = torch.load(selected_files[0], weights_only=False)
        # 此处test_data取到的cycles有问题，只有6个cycles，还是随机的)
        target_test_data = torch.load(os.path.join(target_data_path, "test.pt"), weights_only=False)

        mixture_train_data = torch.load(os.path.join(mixture_bat_data_path, "mixture_train.pt"), weights_only=False)
        val_data = torch.load(os.path.join(mixture_bat_data_path, "val.pt"), weights_only=False)

         #This dataset is for finetuning.
        self.dataset =  Load_Dataset(target_train_data, self.configs, self.args.training_mode, "mixture")
        self.mixture_train_dataset = Load_Dataset(mixture_train_data, self.configs, self.args.training_mode, "mixture")
        self.val_dataset = Load_Dataset(val_data, self.configs, self.args.training_mode, "mixture")
        self.test_dataset = Load_Dataset(target_test_data, self.configs, self.args.training_mode, "mixture")
        # DataLoader
        # print("Length of dataset is: ", len(self.dataset))
        self.train_dataloader = DataLoader(
            dataset=self.dataset,
            batch_size=self.configs.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=0
        )
        self.mixture_train_dataloader = DataLoader(
            dataset=self.mixture_train_dataset,
            batch_size=self.configs.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=0
        )
        self.val_dataloader = DataLoader(
            dataset=self.val_dataset,
            batch_size=self.configs.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0
        )
        self.test_dataloader = DataLoader(
            dataset=self.test_dataset,
            batch_size=self.configs.batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=0
        )

    def _build_model(self):
        """构建模型 & TC模块 & 加载预训练权重"""
        model_module_path = f'models.{self.args.base_model}'
        base_Model = dynamic_import(model_module_path, 'base_Model')
        self.model = base_Model(self.configs).to(self.device)
        self.temporal_contr_model = TC(self.configs, self.device).to(self.device)

        # 加载预训练模型
        ckpt_path = self.args.selected_ckpt
        load_from = os.path.join(ckpt_path, f"supervised_with_contrast_seed_{self.SEED}", '4', "saved_models")
        print("Loading from: ", load_from)
        chkpoint = torch.load(os.path.join(load_from, "ckp_last.pt"), map_location=self.device)

        pretrained_model_dict = chkpoint["model_state_dict"]
        pretrained_temporal_model_dict = chkpoint["temporal_contr_model_state_dict"]

        # 扩展卷积层（适配通道数）
        if 'conv_block.0.weight' in pretrained_model_dict:
            pretrained_weight = pretrained_model_dict['conv_block.0.weight']
            expanded_weight = torch.zeros_like(self.model.conv_block[0].weight)
            expanded_weight[:, :pretrained_weight.shape[1], :] = pretrained_weight
            pretrained_model_dict['conv_block.0.weight'] = expanded_weight

        self.model.load_state_dict(pretrained_model_dict, strict=False)
        self.temporal_contr_model.load_state_dict(pretrained_temporal_model_dict, strict=False)

        # 冻结 TC 模块
        set_requires_grad(self.temporal_contr_model, pretrained_temporal_model_dict, requires_grad=False)

    # def _run_kfold_training(self):
    #     """K-Fold 交叉验证训练流程"""
    #     k_folds = self.configs.k_fold
    #     kf = KFold(n_splits=k_folds, shuffle=True)
    #     indices = np.arange(len(self.dataset))

    #     for fold, (train_idx, valid_idx) in enumerate(kf.split(indices)):
    #         self.logger.info(f'Fold {fold + 1}/{k_folds}')
    #         # 构造子集
    #         train_subset = Subset(self.dataset, train_idx)
    #         valid_subset = Subset(self.dataset, valid_idx)
    #         train_dataset = Load_Dataset(train_subset, self.configs, self.args.training_mode, self.args.dataset)
    #         valid_dataset = Load_Dataset(valid_subset, self.configs, self.args.training_mode, self.args.dataset)
    #         train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=self.configs.batch_size, drop_last=True, shuffle=True, num_workers=4)
    #         valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=self.configs.batch_size, shuffle=False, drop_last=self.configs.drop_last, num_workers=4)

    #         # 每个 Fold 都重新初始化模型
    #         self._build_model()
    #         model = self.model

    #         # 优化器
    #         model_optimizer = torch.optim.Adam(
    #             list(model.parameters()),
    #             lr=self.configs.lr_f,
    #             betas=(self.configs.beta1, self.configs.beta2),
    #             weight_decay=3e-4
    #         )

    #         # 开始训练
    #         Trainer_f(
    #             model,
    #             self.temporal_contr_model,
    #             model_optimizer,
    #             train_loader,
    #             self.test_loader,
    #             self.device,
    #             self.logger,
    #             self.configs,
    #             self.experiment_log_dir,
    #             self.args.training_mode
    #         )

    def _source_only(self):
        # 此函数用于直接加载模型后在target的测试集进行评估
        # 为tranfer下的source_only单独记录一份实验结果
        self.args.training_mode = "source_only"
        self._setup_directories_and_configs()

        self._build_model()
        model = self.model
        model_optimizer = torch.optim.Adam(
                list(model.parameters()),
                lr=self.configs.lr_f,
                betas=(self.configs.beta1, self.configs.beta2),
                weight_decay=3e-4
            )
        #结果是：(mape, rmse)
        test_results = Trainer_f(
            model,
            self.temporal_contr_model,
            model_optimizer,
            self.train_dataloader,
            self.test_dataloader,
            self.device,
            self.logger,
            self.configs,
            self.experiment_log_dir,
            "source_only"
        )

    def _finetuning(self):
        if self.args.bat_num == 1:
            print("111")
            self.args.training_mode = "fine_tuning_1bat"
        else:
            print("222")
            self.args.training_mode = "fine_tuning_2bat"
        self._setup_directories_and_configs()
            
        self._build_model()
        model = self.model

        model_optimizer = torch.optim.Adam(
                list(model.parameters()),
                lr=self.configs.lr_f,
                betas=(self.configs.beta1, self.configs.beta2),
                weight_decay=3e-4
            )

        # 开始训练
        Trainer_f(
            model,
            self.temporal_contr_model,
            model_optimizer,
            self.train_dataloader,
            self.test_dataloader,
            self.device,
            self.logger,
            self.configs,
            self.experiment_log_dir,
            self.args.training_mode
        )

    def _train_with_target(self):
        # 调用main函数来训练混合数据 
        # Question: Channel不一样怎么办？
        # 目前可以配对的是： (tongji, hust), (xjtu, mit)
        # 之后再考虑删除temperature这一项
        if self.args.bat_num == 1:
            print("111")
            self.args.training_mode = "mixture_training_1bat"
        else:
            print("222")
            self.args.training_mode = "mixture_training_2bat"
        self._setup_directories_and_configs()
        args = self.args
        training_result = run_training(args, self.mixture_train_dataloader, self.val_dataloader, self.test_dataloader, self.logger,
                    self.experiment_log_dir)
    
    def run(self):
        """主流程入口"""
        self._setup_directories_and_configs()
        self._load_datasets()
        # self._source_only()
        self._finetuning()
        # self._train_with_target()

        self.logger.info("✅ Fine-tuning completed successfully.")


if __name__ == "__main__":
    runner = FineTunerRunner()
    runner.run()