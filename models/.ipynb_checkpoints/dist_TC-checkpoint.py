import torch
import torch.nn as nn


class TC(nn.Module):
    def __init__(self, configs, device):
        super(TC, self).__init__()
        self.lsoftmax = nn.LogSoftmax(dim=-1)
        self.device = device


    def forward(self, z_aug1, z_aug2):
        # 输入形状 (cycles_number, features, 1)
        batch, features, _ = z_aug1.shape
        
        total = torch.mm(z_aug1.squeeze(-1),  torch.transpose(z_aug2.squeeze(-1),0,1))
        nce = torch.sum(torch.diag(self.lsoftmax(total)))
        
        return nce