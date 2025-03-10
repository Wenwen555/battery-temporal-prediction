import os

import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset




# 数据加载过程如下:
# 1. 提供数据集然后进入不同的key name
# 2. 将data_split和dataloader结合,利用k折交叉验证保证模型的稳定性
import torch

class Load_Dataset(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, dataset, config, training_mode, dataset_name):
        super(Load_Dataset, self).__init__()
        self.training_mode = training_mode

        labelset = [d['summary'] for d in dataset]
        labels = []
        data = [d['cycle'] for d in dataset]
        records = []

        if dataset_name == "tongji":
            # ks = ['Ecell/V','<I>/mA', 'Q discharge/mA.h', 'Q charge/mA.h']
            ks = ['Ecell/V', '<I>/mA', 'Q charge/mA.h']
        elif dataset_name == "xjtu":
            ks = ['current_A', 'voltage_V', 'capacity_Ah', 'temperature_C']
        elif dataset_name == "mit":
            ks = ['current (A)', 'voltage (V)', 'charge Q (Ah)', 'Temperature']
        elif dataset_name == "hust":
            # the data maker of hust doesn't provide temperature data.
            ks = ['Current (mA)', 'Voltage (V)', 'Capacity (mAh)']

        for i in range(len(dataset)):
            cell_data = data[i]
            cell_label = labelset[i]
            cycles = sorted(cell_data.keys(), key=lambda x: int(x))[:]
            labels.append(
                # todo: 此处HUST的数据集出现了一个偏差,后续需要修正
                np.asarray(cell_label[:], dtype=np.float32)
            )
            
            if dataset_name != "hust":
                records.append(
                    np.asarray([[cell_data[c][k][:] for k in ks] for c in cycles],
                               dtype=np.float32)
                )
            else:
                #先前的数据预处理部分没设计好
                records.append(
                    np.asarray([[cell_data[c]['Current (mA)'][k][:] for k in ks] for c in cycles],
                               dtype=np.float32)
                )
        del dataset, labelset

        # using cumsum to calculate the index of the idx of pair:(bat,cyc)
        num_samples = [len(d) for d in records]
        self._cum_sum = np.cumsum(num_samples)
        
        self.indexes = {}
        start = 0
        for i, s in enumerate(self._cum_sum):
            for idx in range(start, s):
                curr_idx = idx - start
                self.indexes[idx] = (i, curr_idx)
            start = s

        self.x_data = records
        self.y_data = labels
        
        for idx in range(len(self.x_data)):
            if isinstance(self.x_data[idx], np.ndarray):
                self.x_data[idx] = torch.from_numpy(self.x_data[idx])
                self.y_data[idx] = torch.from_numpy(self.y_data[idx])
        
        
        #首先sequence prediction过程暂时不考虑augmentation
        self.samples = []
        self.samples_labels = []
        window_size = 10
        step = 1
        for idx, bat in enumerate(self.x_data):
            for start in range(0,len(bat)-window_size, step):
                temp_sample = torch.stack([cyc for cyc in bat[start:start+window_size]])
                temp_labels = self.y_data[idx][start:start+window_size]
                self.samples.append(temp_sample)
                self.samples_labels.append(temp_labels)
        
        self.len = len(self.samples)

    def __getitem__(self, index):
        i, si = self.indexes[index]
        return self.samples[index], self.samples_labels[index]

    def __len__(self):
        return self.len
