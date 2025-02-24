import numpy as np
import torch
import torch.nn as nn

from .attention import Seq_Transformer

#
# class TC(nn.Module):
#     def __init__(self, configs, device):
#         super(TC, self).__init__()
#         self.num_channels = configs.final_out_channels
#         self.timestep = configs.TC.timesteps
#         self.Wk = nn.ModuleList([nn.Linear(configs.TC.hidden_dim, self.num_channels) for i in range(self.timestep)])
#         self.lsoftmax = nn.LogSoftmax()
#         self.device = device
#
#         self.projection_head = nn.Sequential(
#             nn.Linear(configs.TC.hidden_dim, configs.final_out_channels // 2),
#             nn.BatchNorm1d(configs.final_out_channels // 2),
#             nn.ReLU(inplace=True),
#             nn.Linear(configs.final_out_channels // 2, configs.final_out_channels // 4),
#         )
#
#         self.seq_transformer = Seq_Transformer(patch_size=self.num_channels, dim=configs.TC.hidden_dim, depth=4,
#                                                heads=4, mlp_dim=64)
#
#     def forward(self, z_aug1, z_aug2):
#         seq_len = z_aug1.shape[0]
#         batch = z_aug1.shape[0]
#
#
#         t_samples = torch.randint(seq_len - self.timestep, size=(1,)).long().to(
#             self.device)  # randomly pick time stamps
#
#         nce = 0  # average over timestep and batch
#         encode_samples = torch.empty((self.timestep, 1, self.num_channels)).float().to(self.device)
#
#         for i in np.arange(1, self.timestep + 1):
#             encode_samples[i - 1] = z_aug2[t_samples + i, 1, :].view(batch, self.num_channels)
#         forward_seq = z_aug1[:t_samples + 1, : ,1] #(batch, features, seq_len)
#
#         c_t = self.seq_transformer(forward_seq)
#
#         pred = torch.empty((self.timestep, batch, self.num_channels)).float().to(self.device)
#         for i in np.arange(0, self.timestep):
#             linear = self.Wk[i]
#             pred[i] = linear(c_t)
#         for i in np.arange(0, self.timestep):
#             total = torch.mm(encode_samples[i], torch.transpose(pred[i], 0, 1))
#             nce += torch.sum(torch.diag(self.lsoftmax(total)))
#         nce /= -1. * batch * self.timestep
#         return nce, self.projection_head(c_t)


import torch
import torch.nn as nn

from .attention import Seq_Transformer


class TC(nn.Module):
    def __init__(self, configs, device):
        super(TC, self).__init__()
        self.num_channels = configs.final_out_channels  # 特征维度
        self.timestep = configs.TC.timesteps  # 时间步长
        self.Wk = nn.ModuleList([nn.Linear(configs.TC.hidden_dim, self.num_channels) for _ in range(self.timestep)])
        self.W = nn.Linear(configs.TC.hidden_dim, self.num_channels)
        self.lsoftmax = nn.LogSoftmax(dim=-1)
        self.device = device

        self.projection_head = nn.Sequential(
            nn.Linear(configs.TC.hidden_dim, configs.final_out_channels // 2),
            # nn.BatchNorm1d(configs.final_out_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(configs.final_out_channels // 2, configs.final_out_channels // 4),
        )


        self.seq_transformer = Seq_Transformer(
            patch_size=256,  # 每个周期表示一个序列元素
            dim=configs.TC.hidden_dim,
            depth=2,
            heads=4,
            mlp_dim=64,
        )

    def forward(self, z_aug1, z_aug2):
        # 输入形状 (cycles_number, features, 1)
        cycles_number, features, _ = z_aug1.shape

        # 去掉最后一维，保持输入形状为 (cycles_number, features)
        z_aug1 = z_aug1.squeeze(-1)  # (cycles_number, features)
        z_aug2 = z_aug2.squeeze(-1)  # (cycles_number, features)

        # 设定序列长度: 此处序列理解为cycles拼接起来的序列
        seq_len = cycles_number
        if seq_len < self.timestep:
            self.timestep = seq_len // 4 #保证取各个分位点时不会越界
        # 随机采样时间戳，保证不越界
        t_samples = torch.randint(seq_len - self.timestep, size=(1,)).long().to(self.device)
        # 下面提供一个特别的t_samples设定，它会循环计算多遍，此处暂时考虑几个分位数点来作为t_sample划分点
        # 那么在此处写一个循环来从不同的t_sample进行计算，并观察各个loss的特点，判别是否要用loss的平均

        quantile = [0.25,0.5,0.75]
        total_nce = []
        for split in quantile:
            t_samples = int(len(z_aug1) * split) - 1
            nce = 0  # 平均计算 nce 损失
            encode_samples = torch.empty((self.timestep, features)).float().to(self.device)

            # 从 z_aug2 中提取样本
            for i in range(self.timestep):
                encode_samples[i] = z_aug2[t_samples + i, :]  # (features,)

            # 截取 z_aug1 的序列
            forward_seq = z_aug1[:t_samples + 1, :]  # (t_samples + 1, features)

            # 使用 seq_transformer 获取上下文表示
            c_t = self.seq_transformer(forward_seq.unsqueeze(0))  # (1, t_samples + 1, features)
            c_t = c_t.squeeze(0)  # 去掉第一个维度，变为 (t_samples + 1, features)

            # 使用 Wk 线性层进行预测
            pred = torch.empty((self.timestep, features)).float().to(self.device)
            for i in range(self.timestep):
                # linear = self.Wk[i]
                linear = self.W
                pred[i] = linear(c_t)  # (features,)

            # 计算 nce 损失
            # for i in range(self.timestep):
            total = torch.mm(encode_samples, torch.transpose(pred,0,1))  # (t_samples, t_samples)
            nce += torch.sum(torch.diag(self.lsoftmax(total)))


            nce = nce / (-1.0 * self.timestep)
            total_nce.append(nce)

        return sum(total_nce)/len(total_nce), self.projection_head(c_t.unsqueeze(0))