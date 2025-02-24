import os

import numpy as np
# from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset

from .augmentations import DataTransform


# 数据加载过程如下:
# 1. 提供数据集然后进入不同的key name
# 2. 将data_split和dataloader结合,利用k折交叉验证保证模型的稳定性

class Load_Dataset(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, dataset, config, training_mode, dataset_name):
        super(Load_Dataset, self).__init__()
        self.training_mode = training_mode

        labelset = [d['summary'] for d in dataset]
        labels = []
        data = [d['cycle'] for d in dataset]
        records = []
        # 初始化 MinMaxScaler
        scaler = StandardScaler()

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
                normalized_results = []
                for c in cycles:
                    normalized_cycle = []
                    for k in ks:
                        reshaped_data = cell_data[c][k][:].values.reshape(-1,1)
                        normalized_value = scaler.fit_transform(reshaped_data)
                        normalized_cycle.append(normalized_value.flatten())
                    normalized_results.append(normalized_cycle)
                records.append(np.asarray(normalized_results,dtype=np.float32))
                # records.append(
                #     np.asarray([[cell_data[c][k][:] for k in ks] for c in cycles],
                #                dtype=np.float32)
                # )
            else:
                normalized_results = []
                for c in cycles:
                    normalized_cycle = []
                    for k in ks:
                        temp_min = cell_data[c][k][:].min()
                        temp_max = cell_data[c][k][:].max()
                        normalized_value = (cell_data[c][k][:] - temp_min) / (temp_max - temp_min)
                        normalized_cycle.append(normalized_value)
                    normalized_results.append(normalized_cycle)
                records.append(np.asarray(normalized_results,dtype=np.float32))
                # records.append(
                #     np.asarray([[cell_data[c]['Current (mA)'][k][:] for k in ks] for c in cycles],
                #                dtype=np.float32)
                # )
        del dataset

        # using cumsum to calculate the index of the idx of pair:(bat,cyc)
        num_samples = [len(d) for d in records]
        self._cum_sum = np.cumsum(num_samples)
        self.len = sum(num_samples)
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

    # def __getitem__(self, index):
    #     i, si = self.indexes[index]
    #     return self.x_data[i][si], self.y_data[i][si], self.x_data[i][si], self.x_data[i][si]

    def __getitem__(self, index):
        i, si = self.indexes[index]
        return self.x_data[i][si], self.y_data[i][si]

    def __len__(self):
        return self.len

def custom_collate_fn_valid(batch):
    """
    自定义 collate_fn，用于从电池数据中按滑动窗口选取 batch
    :param batch: 包含多个电池的所有 cycle 数据
    """
    custom_batch_size = 80
    step = 4  # 滑动步长
    batch_x = []
    batch_y = []

    for x,y in batch:
        num_cycles = len(x)
        # 如果 cycle 数少于 128，则直接返回所有 cycle
        if num_cycles <= custom_batch_size:
            continue
            # batch_x.append(x)
            # batch_y.append(y)
        else:
            # 从最后一个 cycle 开始滑动选取 128 个 cycle
            # print("original shape of x is: ", x.shape)
            # print("original shape of y is: ", y.shape)
            for start in range(0, num_cycles - custom_batch_size, step):
                if start + custom_batch_size > num_cycles:
                    continue
                batch_x.append(x[start:start + custom_batch_size])
                batch_y.append(y[start:start + custom_batch_size])
    batch_x = torch.stack(batch_x) if isinstance(batch_x[0], torch.Tensor) else batch_x
    batch_y = torch.stack(batch_y) if isinstance(batch_y[0], torch.Tensor) else batch_y
    return batch_x, batch_y

