import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict, Any
from torch.utils.data import DataLoader, Subset

import seaborn as sns
import pandas as pd
from matplotlib.lines import Line2D

from dataloader.dataloader import Load_Dataset
from models.TC import TC
from trainer.fine_tuned_trainer import Trainer_f
from utils import _logger, set_requires_grad

import importlib
def dynamic_import(module_path, class_name):
    """动态导入类"""
    module = importlib.import_module(module_path)
    return getattr(module, class_name)

class Plot:
    def __init__(self):
        self.args = self._parse_args()
        self.device = 'cuda'

        config_module_path = f'config_files.{self.args.dataset}.{self.args.batch}_Configs'
        Configs = dynamic_import(config_module_path, 'Config')
        self.configs = Configs()

        self.model = None
        self.temporal_contr_model = None
    
    def _parse_args(self):
        """解析命令行参数"""
        parser = argparse.ArgumentParser(description="Plot script.")
        home_dir = os.getcwd()
        parser.add_argument('--dataset', default='hust', type=str, help='Dataset')
        parser.add_argument('--batch', default='cell_mit_batch1', help='subset of dataset')
        parser.add_argument('--data_path', default=r'data/', type=str, help='Path containing dataset')
        parser.add_argument('--home_path', default=home_dir, type=str, help='Project root directory')
        parser.add_argument('--selected_ckpt', default=None, type=str, help='Path to pre-trained model checkpoint')
        return parser.parse_args()
        
    def load_weights_from_checkpoint(self, ckpt_path: str, model_name: str = 'cnn', seed: int = 123, fold: str = '5') -> None:
        """
        Load model weights from checkpoint (similar to your reference code)
        """
        if model_name == 'mlp':
            if self.args.dataset == 'tongji':
                ckpt_path = f"/mnt/wenjt5/project1/experiments_logs/{self.args.dataset}/mlp_batch2_random_select_0_bat/"
            else: 
                ckpt_path = f"/mnt/wenjt5/project1/experiments_logs/{self.args.dataset}/mlp_batch1_random_select_0_bat/"
        model_module_path = f'models.{model_name}'
        base_Model = dynamic_import(model_module_path, 'base_Model')
        self.model = base_Model(self.configs).to(self.device)
        self.temporal_contr_model = TC(self.configs, self.device).to(self.device)
        
        load_from = os.path.join(ckpt_path, f"supervised_with_contrast_seed_{seed}", fold, "saved_models")
        print(f"Loading from: {load_from}")
        
        checkpoint = torch.load(
            os.path.join(load_from, "ckp_last.pt"),
            map_location=self.device
        )
        
        # Load model state dicts
        pretrained_model_dict = checkpoint["model_state_dict"]
        pretrained_temporal_dict = checkpoint["temporal_contr_model_state_dict"]
        
        # Handle channel dimension expansion if needed
        if model_name == 'cnn':
            if 'conv_block.0.weight' in pretrained_model_dict:
                pretrained_weight = pretrained_model_dict['conv_block.0.weight']
                expanded_weight = torch.zeros_like(self.model.conv_block[0].weight)
                expanded_weight[:, :pretrained_weight.shape[1], :] = pretrained_weight
                pretrained_model_dict['conv_block.0.weight'] = expanded_weight
            
        # Load weights
        self.model.load_state_dict(pretrained_model_dict, strict=False)
        if self.temporal_contr_model is not None:
            self.temporal_contr_model.load_state_dict(pretrained_temporal_dict, strict=False)
            # Freeze temporal contrast module
            self._set_requires_grad(self.temporal_contr_model, requires_grad=False)

    def _load_datasets(self):
        """加载数据集"""        
        data_path = os.path.join(self.args.data_path, self.args.batch)
        train_data = torch.load(os.path.join(data_path, "train_val.pt"), weights_only=False)
        test_data = torch.load(os.path.join(data_path, "test.pt"), weights_only=False)

         #This dataset is for finetuning.
        self.dataset =  Load_Dataset(train_data, self.configs, 'plot', "mixture")
        self.test_dataset = Load_Dataset(test_data, self.configs, 'plot', "mixture")
        # DataLoader
        # print("Length of dataset is: ", len(self.dataset))
        self.train_dataloader = DataLoader(
            dataset=self.dataset,
            batch_size=self.configs.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=0
        )
        self.test_dataloader = DataLoader(
            dataset=self.test_dataset,
            batch_size=1,
            shuffle=False,
            drop_last=False,
            num_workers=0
        )

    def _set_requires_grad(self, model: torch.nn.Module, requires_grad: bool = False) -> None:
        """Helper to freeze/unfreeze model parameters"""
        for param in model.parameters():
            param.requires_grad = requires_grad
    
    def predicion(self):
        training_mode = "plot"
        model = self.model
        model_optimizer = torch.optim.Adam(
                list(model.parameters()),
                lr=self.configs.lr_f,
                betas=(self.configs.beta1, self.configs.beta2),
                weight_decay=3e-4
            )
        #结果是：(mape, rmse)
        predicions, labels = Trainer_f(
            model,
            self.temporal_contr_model,
            model_optimizer,
            self.train_dataloader,
            self.test_dataloader,
            self.device,
            None,
            self.configs,
            None,
            "plot"
        )
        return predicions, labels

    def get_metric(self):
        error_dict = {}
        models = ['cnn', 'mlp']
        for model_name in models:
            if model_name != 'cnn':
                self.load_weights_from_checkpoint(ckpt_path=self.args.selected_ckpt, model_name=model_name)
            error_dict[model_name] = {}
            model_optimizer = torch.optim.Adam(
                    list(self.model.parameters()),
                    lr=self.configs.lr_f,
                    betas=(self.configs.beta1, self.configs.beta2),
                    weight_decay=3e-4
                )
            #结果是：(mape, rmse)
            mape, rmse = Trainer_f(
                self.model,
                self.temporal_contr_model,
                model_optimizer,
                self.train_dataloader,
                self.test_dataloader,
                self.device,
                None,
                self.configs,
                None,
                "plot_metric"
            )
            error_dict[model_name]['mape'] = mape
            error_dict[model_name]['rmse'] = [x * 100 for x in rmse]
        return error_dict

    def plot_prediction_scatter(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        # title: str = "Predictions vs True SOH",
        xlabel: str = "True SOH",
        ylabel: str = "Predictions",
        figsize: tuple = (8, 6),
        alpha: float = 0.9,
        color: str = 'blue',
        diagonal_color: str = 'red',
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create scatter plot comparing predictions vs true labels with perfect prediction diagonal.
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)

        distance = np.abs(y_true - y_pred)
        norm = plt.Normalize(vmin=distance.min(), vmax=distance.max())

        # Scatter plot
        scatter = ax.scatter(y_pred, y_true, c=distance, cmap='Blues_r', alpha=alpha, norm=norm, 
                  label=f'Predictions (n={len(y_true)})')

        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Absolute error', fontsize=18)
        # Perfect prediction line (y=x)
        min_val = min(min(y_true), min(y_pred))
        max_val = max(max(y_true), max(y_pred))
        ax.plot([min_val, max_val], [min_val, max_val], 
               linestyle='--', color=diagonal_color, 
               label='Perfect Prediction')
        
        # Formatting
        ax.set_xlabel(xlabel, fontsize=16)
        ax.set_ylabel(ylabel, fontsize=16)
        ax.grid(True, linestyle='--', alpha=0.7)

        ax.legend(
            loc='upper left',
            fontsize=18
        )

        plt.xlim(min_val, 1.0)
        plt.ylim(min_val, 1.0)
        plt.tick_params(axis='both', labelsize=14)  # 'x'或'y'单独设置
        plt.tight_layout()
        if save_path:
            plt.savefig(f'{save_path}/scatter.pdf', dpi=300, bbox_inches='tight')
            
        return fig

    # def plot_error_violin_comparison(
    #     self,
    #     error_dict: Dict,
    #     save_path: Optional[str] = None,
    #     figsize: tuple = (12, 8)
    #     ) -> plt.Figure:
    #         """
    #         Plot violin plots comparing error metrics (MAE/MAPE/RMSE) across multiple models and datasets.
    #         Args:
    #             error_data: Nested dictionary containing error values.
    #                     Structure: {dataset: {model: {metric: array_of_values}}}
    #                     Example:
    #                     'XJTU':
    #                     {
    #                         'CNN': {'MAE': [...], 'MAPE': [...], 'RMSE': [...]},
    #                         'MLP': {'MAE': [...], 'MAPE': [...], 'RMSE': [...]},
    #                         'PINN': {'MAE': [...], 'MAPE': [...], 'RMSE': [...]}
    #                     }
    #             save_path: Path to save the figure (optional).
    #             figsize: Figure size (width, height).
    #         Returns:
    #             matplotlib Figure object.
    #         """
            
    #         # Prepare DataFrame for plotting
    #         df_list = []
    #         for model, metrics in error_dict.items():
    #             for metric, values in metrics.items():
    #                 for value in values:
    #                     df_list.append({
    #                         'model': model,
    #                         'metric': metric,
    #                         'error': value
    #                     })
            
    #         df = pd.DataFrame(df_list)

    #         # Create figure
    #         fig, axs = plt.subplots(3, 1, figsize=figsize, dpi=200, sharex=True)
    #         metrics = ['mape', 'rmse']

    #         # Blue, Green, Red for different models?
    #         colors = ['#c44e52', '#4c72b0', '#55a868']
            
    #         # Plot each metric in a separate row
    #         for i, metric in enumerate(metrics):
    #             ax = axs[i]
    #             metric_df = df[df['metric'] == metric]

    #             # Draw violin plots
    #             sns.violinplot(
    #                 x='metric',
    #                 y='error',
    #                 hue='model',
    #                 data=metric_df,
    #                 palette=colors,
    #                 inner='quartile',
    #                 cut=0,
    #                 bw_method=0.2,
    #                 linewidth=0.5,
    #                 ax=ax
    #             )
                
    #             # Add mean markers
    #             # for j, dataset in enumerate(error_dict.keys()):
    #             #     for k, model in enumerate(['cnn', 'mlp']):
    #             #         if model in error_dict:
    #             #             mean_val = metric_df[
    #             #                 (metric_df['Dataset'] == dataset) & 
    #             #                 (metric_df['Model'] == model)
    #             #             ]['Error'].mean()
                            
    #             #             ax.scatter(
    #             #                 (k-1)*0.2,  # Offset positions for different models
    #             #                 mean_val,
    #             #                 color='white',
    #             #                 edgecolor='black',
    #             #                 s=60,
    #             #                 zorder=10
    #             #             )
                
    #             ax.set_title(metric, fontsize=10, pad=4)
    #             ax.set_ylabel("Error Value", fontsize=9)
    #             ax.set_xlabel("")
                
    #             # Format y-axis for MAPE
    #             if metric == 'mape':
    #                 ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}%'))
                
    #             ax.grid(axis='y', linestyle='--', alpha=0.3)
                
    #             # Remove legend except for last subplot
    #             if i != 2:
    #                 ax.get_legend().remove()
            
    #         # Create custom legend
    #         legend_elements = [
    #             # Line2D([0], [0], color=colors[0], lw=4, label='Ours'),
    #             Line2D([0], [0], color=colors[0], lw=4, label='cnn'),
    #             Line2D([0], [0], color=colors[1], lw=4, label='mlp'),
    #             Line2D([0], [0], marker='o', color='w', markeredgecolor='k',
    #                 markersize=8, label='Mean', linestyle='None')
    #         ]
            
    #         axs[2].legend(
    #             handles=legend_elements,
    #             loc='upper center',
    #             bbox_to_anchor=(0.5, -0.2),
    #             ncol=4,
    #             fontsize=9
    #         )
            
    #         # plt.suptitle(title, y=1.02, fontsize=12)
    #         plt.tight_layout()
            
    #         if save_path:
    #             print("Saving to the path: ", save_path)
    #             plt.savefig(f'{save_path}/violin.svg', bbox_inches='tight', dpi=300)
            
    #         return fig

    def plot_error_violin_comparison(
        self,
        error_dict: Dict,
        save_path: Optional[str] = None,
        dataset_name: str = None,
        figsize: tuple = (12, 6)  # 调整画布大小
    ) -> plt.Figure:
        """
        Plot violin plots comparing MAPE and RMSE for CNN and MLP on a single figure.
        Args:
            error_dict: Nested dictionary containing error values.
                Structure: {model: {metric: array_of_values}}
                Example:
                {
                    'CNN': {'mape': [...], 'rmse': [...]},
                    'MLP': {'mape': [...], 'rmse': [...]}
                }
            save_path: Path to save the figure (optional).
            figsize: Figure size (width, height).
        Returns:
            matplotlib Figure object.
        """
        df_list = []
        # for batch in batches:
            ############################
        if self.args.dataset == 'xjtu':
            df1 = pd.read_excel(f'data/PINN4plot/PINN-XJTU-results.xlsx',
                    engine='openpyxl',
                    sheet_name=f'battery_mean_1')
        elif self.args.dataset == 'tongji':
            df1 = pd.read_excel(f'data/PINN4plot/PINN-TJU-results.xlsx',
                    engine='openpyxl',
                    sheet_name=f'battery_mean_0')
        else:
            df1 = pd.read_excel(f'data/PINN4plot/PINN-{(self.args.dataset).upper()}-results.xlsx',
                    engine='openpyxl',
                    sheet_name=f'battery_mean_0')

        df1['model'] = ['PINN'] * df1.shape[0]
        melted_df1 = pd.melt(df1, id_vars=['model'],
                            value_vars=['MAPE','RMSE'],
                            var_name='metric', value_name='error')
        melted_df1["metric"] = melted_df1["metric"].str.lower()
        melted_df1["error"] = melted_df1["error"] * 100

        df_list_ours = []
        for model, metrics in error_dict.items():
            # if model != 'cnn':
            #     continue
            for metric, values in metrics.items():
                for value in values:
                    df_list_ours.append({
                        'model': model,
                        'metric': metric,
                        'error': value
                    })        
        df_ours = pd.DataFrame(df_list_ours)
        df = pd.concat([df_ours, melted_df1], ignore_index=True)
        # Filter only MAPE and RMSE
        df = df[df['metric'].isin(['mape', 'rmse'])]
        # 复制 DataFrame 避免修改原数据（可选）
        df_plot = df.copy()
        df_plot['metric'] = df_plot['metric'].str.upper()  # 转为大写
        # import ipdb
        # ipdb.set_trace()

        # Create a single figure (not subplots)
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=400)

        # Colors for CNN and MLP
        colors = ['#092147', '#1A488E','#97B2DE' ]  # Blue for CNN, Red for MLP
        
        # Draw violin plots on the same axis
        sns.violinplot(
            x='metric',      # x-axis: 'mape' and 'rmse'
            y='error',
            hue='model',     # Different models (CNN/MLP)
            data=df_plot,
            palette=colors,
            density_norm='count',
            inner='point',
            dodge=True,
            saturation=1,
            cut=0,
            bw_method=0.5,
            linewidth=0,
            ax=ax,
            width=0.8,      # 控制整个violin组的宽度（默认0.8）
        )

        if dataset_name == 'xjtu':
            plt.ylim(0,5)
        elif dataset_name == 'tongji':
            plt.ylim(0,10)

        # Add mean markers (optional)
        for i, metric in enumerate(['mape', 'rmse']):
            for j, model in enumerate(['CNN', 'MLP', 'PINN']):
                if model in error_dict:
                    mean_val = df[
                        (df['metric'] == metric) & 
                        (df['model'] == model.lower())  # Ensure case matches
                    ]['error'].mean()
                    
                    ax.scatter(
                        i + j*0.1 - 0.05,  # Adjust position to avoid overlap
                        mean_val,
                        color='white',
                        edgecolor='black',
                        s=60,
                        zorder=10
                    )
        
        ax.set_ylabel("Error", fontsize=20)
        ax.tick_params(axis='x', labelsize=20)  # 调整字体大小（默认是 10）
        ax.set_xlabel("")
        plt.tick_params(axis='both', labelsize=18)  # 'x'或'y'单独设置
         
        # Format y-axis for MAPE (percentage)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(
            lambda x, _: f'{x:.0f}%' if ax.get_xlim()[0] == 0 else f'{x:.2f}'
        ))
        
        # ax.grid(axis='y', linestyle='--', alpha=0.1)

        # Custom legend
        legend_elements = [
            Line2D([0], [0], color=colors[0], lw=6, label='Auto-CNN'),
            Line2D([0], [0], color=colors[1], lw=6, label='Auto-MLP'),
            Line2D([0], [0], color=colors[2], lw=6, label='PINN'),
            # Line2D([0], [0], marker='o', color='w', markeredgecolor='k',
            #     markersize=8, label='Mean', linestyle='None')
        ]
        
        ax.legend(
            handles=legend_elements,
            loc='upper left',
            fontsize=22
        )
        plt.tight_layout()
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            print("Saving to path: ", save_path)
            plt.savefig(f'{save_path}/violin.pdf', bbox_inches='tight', dpi=300)
        
        return fig
        
    def run(self):
        """主流程入口"""
        self.load_weights_from_checkpoint(ckpt_path=self.args.selected_ckpt)
        self._load_datasets()
        # prediction, label= self.predicion()
        # scatter_save_path = f'/mnt/wenjt5/project1/plot/scatter/{self.args.dataset}-{self.args.batch}/'
        # os.makedirs(scatter_save_path, exist_ok=True)
        # self.plot_prediction_scatter(label.view(-1).cpu(), prediction.view(-1).cpu(), save_path=scatter_save_path)
        error_dict = self.get_metric()
        violin_save_path = f'/mnt/wenjt5/project1/plot/violin/main/{self.args.dataset}-{self.args.batch}/'
        os.makedirs(violin_save_path, exist_ok=True)
        # 需要获取model的error_data
        self.plot_error_violin_comparison(error_dict=error_dict, save_path=violin_save_path, dataset_name=self.args.dataset)
        print("✅ Plotting completed successfully.")

if __name__ == "__main__":
    runner = Plot()
    runner.run()