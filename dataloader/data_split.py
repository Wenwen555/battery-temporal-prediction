import pickle
import os
import torch
import matplotlib.pyplot as plt


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
            value = f"data/XJTU/cell_XJTU_batch{i}_prepocess_data"
        elif target_dataset == 'mit':
            value = f"data/MIT/cell_mit_batch{i}_data"
        elif target_dataset == 'hust':
            value = f"data/HUST/cell_HUST_batch{i}_data"
        elif target_dataset == 'tongji':
            subset = ['NCA','NCM','NCM_NCA']
            value = f"data/Tongji/cell_Tongji_{subset[i]}_data'"
        batch_dict[key] = value

    file_dict = {}
    for i in range(1, n_value_dict[target_dataset] + 1):
        key = f"batch{i}"
        if target_dataset == 'xjtu':
            value = f"data/XJTU/XJTU_batch{i}_prepocess.pkl"
        elif target_dataset == 'mit':
            value = f"data/MIT/batch{i}_prepocess.pkl"
        elif target_dataset == 'hust':
            value = f"data/HUST/HUST_batch{i}_prepocess.pkl"
        elif target_dataset == 'tongji':
            subset = ['NCA','NCM','NCM_NCA']
            value = f"data/Tongji/Tongji_{subset[i]}_prepocess.pkl"
        file_dict[key] = value
    return batch_dict, file_dict


class Load_Dataset():
    def __init__(self, save_path):
        super(Load_Dataset).__init__()
        self.save_path = save_path

    def read_pkl_file(self, filename):
        with open(filename, 'rb') as file:
            data = pickle.load(file)
        return data

    @staticmethod
    def train_val_test_split(data):
        test_bat_idx = []
        test_name = ['2C_battery-4','2C_battery-8']
        for bat_name in data.keys():
            # if bat_name % 4 == 0:
            if bat_name in test_name:
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
    
    @staticmethod
    def small_sample(data, sample_num):
        cnt = 0
        test_bat = ['2C_battery-4','2C_battery-8']
        # test_bat = [4,8]
        train_set = []
        for bat_name in data.keys():
            if cnt < sample_num and bat_name not in test_bat:
                train_set.append(data[bat_name])
                cnt += 1
        return train_set
    
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

current_path = os.getcwd()
parent_path = os.path.dirname(current_path)

# There are parameters that can control the dataset_generation from pkl file.
target_dataset = 'xjtu'
target_subset = 'batch1'
samll_sample_flag = True
batch_dict, file_dict = get_save_and_file_path(target_dataset)

save_path = os.path.join(parent_path, batch_dict[target_subset])
filepath = os.path.join(parent_path, file_dict[target_subset])

ld = Load_Dataset(save_path)
dataset = ld.read_pkl_file(filepath)
if samll_sample_flag:
    # This is data selection for small sample experiment
    for sample in [1,2,3,4]:
        train_dataset = ld.small_sample(dataset, sample)
        ld.save_dict(train_dict=train_dataset, sample_num=sample)
else:
    train_val_bat, test_bat = ld.train_val_test_split(dataset)
    
print("pkl file has been split!")


