import pickle
import os
import torch
import matplotlib.pyplot as plt
from itertools import combinations
import random
import glob

def get_save_and_file_path(target_dataset):
    n_value_dict = {
        'xjtu':6,
        'mit':3,
        'hust':10,
        'tongji':3,
    }
    batch_dict = {}
    for i in range(1, n_value_dict[target_dataset] + 1):
        key = f"batch{i}"
        if target_dataset == 'xjtu':
            value = f"/mnt/wenjt5/project1/data/XJTU/cell_XJTU_batch{i}_data/"
        elif target_dataset == 'mit':
            value = f"/mnt/wenjt5/project1/data/MIT/cell_mit_batch{i}_data/"
        elif target_dataset == 'hust':
            value = f"/mnt/wenjt5/project1/data/HUST/cell_HUST_batch{i}_data/"
        elif target_dataset == 'tongji':
            subset = ['NCA','NCM','NCM_NCA']
            value = f"/mnt/wenjt5/project1/data/Tongji/cell_Tongji_{subset[i-1]}_data/"
        batch_dict[key] = value

    file_dict = {}
    for i in range(1, n_value_dict[target_dataset] + 1):
        key = f"batch{i}"
        if target_dataset == 'xjtu':
            value = f"/mnt/wenjt5/project1/data/XJTU/XJTU_batch{i}_prepocess.pkl"
        elif target_dataset == 'mit':
            value = f"/mnt/wenjt5/project1/data/MIT/batch{i}_prepocess.pkl"
        elif target_dataset == 'hust':
            value = f"/mnt/wenjt5/project1/data/HUST/HUST_batch{i}_prepocess.pkl"
        elif target_dataset == 'tongji':
            subset = ['NCA','NCM','NCM_NCA']
            value = f"/mnt/wenjt5/project1/data/Tongji/Tongji_{subset[i-1]}_prepocess.pkl"
        file_dict[key] = value
    
    mix_file_dict = {}
    for i in range(1, n_value_dict[target_dataset] + 1):
        key = f"batch{i}"
        if target_dataset == 'xjtu':
            value = f"/mnt/wenjt5/project1/data/XJTU/mixture/XJTU_batch{i}_prepocess_for_mix.pkl"
        elif target_dataset == 'mit':
            value = f"/mnt/wenjt5/project1/data/MIT/mixture/MIT_{i}_prepocess_for_mix.pkl"
        elif target_dataset == 'hust':
            value = f"/mnt/wenjt5/project1/data/HUST/mixture/HUST_{i}_prepocess_for_mix.pkl"
        elif target_dataset == 'tongji':
            subset = ['NCA','NCM','NCM_NCA']
            value = f"/mnt/wenjt5/project1/data/Tongji/mixture/Tongji_{subset[i-1]}_prepocess_for_mix.pkl"
        mix_file_dict[key] = value
    
    return batch_dict, file_dict, mix_file_dict


class Load_Dataset():
    def __init__(self, save_path, target_dataset):
        super(Load_Dataset).__init__()
        self.save_path = save_path
        self.dataset_name = target_dataset

    def read_pkl_file(self, filename):
        with open(filename, 'rb') as file:
            data = pickle.load(file)
        return data

    @staticmethod
    def train_val_test_split(data):
        test_bat_idx = []
        # test_name = ['R3_battery-4', 'R3_battery-8', 'RW_battery-4', 'RW_battery-8']
        for bat_name in data.keys():
            if bat_name % 4 == 0:
            # if "4" in bat_name or "8" in bat_name:
                test_bat_idx.append(bat_name)
        train_val = []
        test_bat = []
        
        for idx in data.keys():
            if idx not in test_bat_idx:
                train_val.append(data[idx])
        # 此处不需要split labels，因为labels是存在data内部的
        for idx in test_bat_idx:
            test_bat.append(data[idx])
        print("Length of train_bats is: " ,len(train_val))
        print("Length of test_bats is: " ,len(test_bat_idx))
        return train_val, test_bat
    
    def small_sample_random_select(self, data, sample_num, save_path):
        # for xjtu
        if self.dataset_name == 'xjtu':
            batteries = [bat_name for bat_name in data.keys() if '4' not in bat_name and '8' not in bat_name and '12' not in bat_name]
        elif self.dataset_name == 'tongji':
            batteries = [ bat_name for bat_name in data.keys() if (bat_name % 4 != 0)]
        elif self.dataset_name == 'hust':
            batteries = [ bat_name for bat_name in data.keys() if (bat_name % 4 != 0)]
        elif self.dataset_name == 'mit':
            batteries = [ bat_name for bat_name in data.keys() if (bat_name % 4 != 0)]
        all_combinations = list(combinations(batteries, sample_num))
        number2text = ['one','two','three','four']
        for combo in all_combinations:
            train_set = []
            # 提取数字部分，如 "2C_battery-1" -> "1"
            num_strs = []
            for item in combo:
                if isinstance(item, str):
                    # 比如 "bat-1" -> "1"
                    parts = item.split("-")
                    if len(parts) > 1:
                        num_strs.append(parts[1])
                    else:
                        num_strs.append(parts[0])  # 如果没有 -, 就整个作为标识
                else:
                    # 非字符串转为字符串加入
                    num_strs.append(str(item))

            # 拼接数字部分
            if sample_num == 1:
                combined_num = ''.join(num_strs)
            else:
                combined_num = '-'.join(num_strs)
            # 构造文件名
            filename = f"{number2text[sample_num - 1]}_bat_{combined_num}.pt"
            train_file_path = os.path.join(save_path, filename)
            # 存入数据
            for bat_name in data.keys():
                if bat_name in combo:
                    train_set.append(data[bat_name])
            # save the file
            torch.save(train_set, train_file_path, _use_new_zipfile_serialization=False)
            print(f"Saved: {filename} -> {combo}")
            break
    
    def mix_source_target_bats(self, source_dataset, target_file_path, source_dataset_name, target_dataset_name, target_subsets, sample_num=2):
        """
        将 target dataset 的 1~2 个 bat 混合进 source dataset 的 small sample 数据中。
        参数:
            source_dataset: source 数据集（原始数据）
            target_file_path: str, target 数据集路径（包含 one_bat_*.pt or two_bat_*.pt 数据）
            sample_num: int, target 数据集中原本是几个 bat 的组合（比如 two_bat）
        """
        SEED = 42
        random.seed(SEED)
        # mixture path在source dataset文件夹下
        os.makedirs(self.save_path, exist_ok=True)
        num2str = ['one', 'two', 'three', 'four']
        target_pattern = os.path.join(target_file_path, f"random_select_{sample_num}_bat", f'{num2str[sample_num-1]}_bat_*.pt')
        target_files_all = glob.glob(target_pattern)
        target_files = random.choice(target_files_all)
        target_data = torch.load(target_files, weights_only=False)
        
        
        source_train_val_data, _ = self.train_val_test_split(source_dataset)
        k = len(source_train_val_data)
        val_len = int(0.2 * k)
        indices = random.sample(range(k), k)
        source_val_set = []
        for i in range(val_len):
            source_val_set.append(source_train_val_data[indices[i]])
        mixture_train_data = []
        for i in range(len(source_train_val_data)):
            if i not in indices[:val_len]:
                mixture_train_data.append(source_train_val_data[i])
        
        # 下面做一个check mechanism，用于检验数据形状是否一致，从而找到错误（不符合要求的random_pick)

        format_check(mixture_train_data, target_data)
        for bat in target_data:
            mixture_train_data.append(bat)
        
        # Save
        save_folder_path = os.path.join(mixture_save_path, f"mixed_{target_dataset_name}", f"mixed_{source_dataset_name}_with_{target_dataset_name}_{target_subsets}_{sample_num}_bat")
        os.makedirs(save_folder_path, exist_ok=True)
        save_train_path = os.path.join(save_folder_path, 'mixture_train.pt')
        save_val_path = os.path.join(save_folder_path, 'val.pt')
        torch.save(mixture_train_data, save_train_path, _use_new_zipfile_serialization=False)
        torch.save(source_val_set, save_val_path, _use_new_zipfile_serialization=False)
        print(f"已保存混合数据至: {save_folder_path}")

    def save_dict(self, train_dict, test_dict=None, sample_num=None):
        save_path = self.save_path
        os.makedirs(save_path, exist_ok=True)
        dataset_save_name = ['one_bat.pt','two_bat.pt','three_bat.pt','four_bat.pt']
        # note _use_new_zipfile_serialization will use zip_file format for storing data.
        if sample_num == None:
            assert test_dict != None
            train_val_file_path = os.path.join(save_path, 'train_val.pt')
            test_file_path = os.path.join(save_path, 'test.pt')
            torch.save(train_dict, train_val_file_path, _use_new_zipfile_serialization=False)
            torch.save(test_dict, test_file_path, _use_new_zipfile_serialization=False)
        else:
            assert test_dict == None
            train_file_path = os.path.join(save_path, dataset_save_name[sample_num-1])
            torch.save(train_dict, train_file_path, _use_new_zipfile_serialization=False)

def format_check(origin_data:list, target_data:list):
    # 首先check自身
    constant_channels = len(origin_data[0]['cycle'][0])
    constant_lengths = len(origin_data[0]['cycle'][0]['current'])
    target_lengths = len(target_data[0]['cycle'][0]['current'])
    print("Constant length is: ", constant_lengths)
    print("Constant channel is: ", constant_channels)
    print("Constant target length is: ", target_lengths)
    for idx in range(len(origin_data)):
        for cyc in origin_data[idx]['cycle'].keys():
            if len(origin_data[idx]['cycle'][cyc]) != constant_channels:
                raise Warning("Channels数无法对齐! 当前的channels数为：", len(origin_data[idx]['cycle'][cyc]), "当前bat和cycle分别为：", idx, cyc)
            if len(origin_data[idx]['cycle'][cyc]['current']) != constant_lengths:
                raise Warning("长度不对齐！")
    for idx in range(len(target_data)):
        for cyc in target_data[idx]['cycle'].keys():
            if len(target_data[idx]['cycle'][cyc]['current']) != constant_lengths:
                raise Warning("源域和目标域长度不对齐！")

current_path = os.getcwd()
parent_path = os.path.dirname(current_path)

# There are parameters that can control the dataset_generation from pkl file.
source_dataset_name = 'hust'
#, 'batch4', 'batch5', 'batch6', 'batch7', 'batch8', 'batch9', 'batch10'
source_subsets = ['batch1']

samll_sample_flag = True
for_mixture = False #False True

target_dataset_name = "mit"
# 'batch1', 'batch2', 
target_subsets = ['batch1', 'batch2', 'batch3']
target_sample_nums = 1

batch_dict, file_dict, mix_file_dict = get_save_and_file_path(source_dataset_name)
target_batch_dict, target_file_dict, _ = get_save_and_file_path(target_dataset_name)

# for target_sample_num in target_sample_nums:
for subset in source_subsets:
    save_path = os.path.join(parent_path, batch_dict[subset])
    filepath = os.path.join(parent_path, mix_file_dict[subset])
    ld = Load_Dataset(save_path, source_dataset_name)
    source_dataset = ld.read_pkl_file(filepath)
    if samll_sample_flag:
        # This is data selection for small sample experiment
        # for sample in [1,2,3,4]:
        filepath = os.path.join(parent_path, mix_file_dict[subset])
        ld = Load_Dataset(save_path, source_dataset_name)
        source_dataset_for_mix = ld.read_pkl_file(filepath)
        print("Loading from: ", filepath)
        for sample in [1, 2, 3, 4]:
            random_select_save_path = os.path.join(save_path, f"random_select_{sample}_bat")
            # 确保目标文件夹存在，如果不存在则创建
            os.makedirs(random_select_save_path, exist_ok=True)
            ld.small_sample_random_select(source_dataset_for_mix, sample, random_select_save_path)
            print("Saving to: ", random_select_save_path)
    elif for_mixture:
        # mixture的source dataset和 target dataset需要多经过一层mixture文件夹
        mixture_save_path = os.path.join(parent_path, batch_dict[subset], "mixture")
        filepath = os.path.join(parent_path, mix_file_dict[subset])
        mix_ld = Load_Dataset(mixture_save_path, source_dataset_name)
        source_mixture_dataset = mix_ld.read_pkl_file(filepath)
        print("Loading from: ", filepath)
        for target_subset in target_subsets:
            target_save_path = os.path.join(parent_path, target_batch_dict[target_subset])
            mix_ld.mix_source_target_bats(source_mixture_dataset, target_save_path, source_dataset_name, target_dataset_name, target_subset, sample_num=target_sample_num)
    else:
        train_val_bat, test_bat = ld.train_val_test_split(source_dataset)
        ld.save_dict(train_val_bat, test_bat)
        print("Saving to: ", save_path)
print("pkl file has been split!")



