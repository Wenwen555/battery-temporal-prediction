import pickle
import os
import torch
import matplotlib.pyplot as plt

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
        # test_bat_idx = [3, 6, 7, 13, 43] #This is a mit example.
        # for hust dataset, we usually take 4,8 as test sample.
        test_bat_idx = []
        for bat_name in data.keys():
            if bat_name % 4 == 0:
                test_bat_idx.append(bat_name)
        # test_bat_idx = [4,9,14,19,24,29,34,39,44]
        # test_bat_idx = ['R2.5_battery-4','R2.5_battery-8']
        # It is easy to check the length of data and make desicion.
        train_val = []
        test_bat = []

        for idx in data.keys():
            if idx not in test_bat_idx:
                train_val.append(data[idx])
        # 此处不需要split labels，因为labels是存在data内部的
        for idx in test_bat_idx:
            test_bat.append(data[idx])
        return train_val, test_bat


    def save_dict(self, train_dict, test_dict):
        save_path = self.save_path
        os.makedirs(save_path, exist_ok=True)
        train_val_file_path = os.path.join(save_path, 'train_val.pt')
        test_file_path = os.path.join(save_path, 'test.pt')
        # note _use_new_zipfile_serialization will use zip_file format for storing data.
        torch.save(train_dict, train_val_file_path, _use_new_zipfile_serialization=False)
        torch.save(test_dict, test_file_path, _use_new_zipfile_serialization=False)
        print("pkl file has been split!")

current_path = os.getcwd()
parent_path = os.path.dirname(current_path)
save_path = os.path.join(parent_path, 'data/Tongji/cell_Tongji_NCM_NCA_data')
filepath = os.path.join(parent_path, 'data/Tongji/Tongji_NCM_NCA_prepocess.pkl')

ld = Load_Dataset(save_path)
dataset = ld.read_pkl_file(filepath)
train_val_bat, test_bat = ld.train_val_test_split(dataset)
ld.save_dict(train_val_bat, test_bat)

