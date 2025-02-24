import torch
import torch.nn as nn
import os
from torch.autograd import grad
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Sin(nn.Module):
    def __init__(self):
        super(Sin, self).__init__()

    def forward(self, x):
        return torch.sin(x)

class MLP(nn.Module):
    def __init__(self,input_dim=17,output_dim=1,layers_num=4,hidden_dim=50,droupout=0.2):
        super(MLP, self).__init__()

        assert layers_num >= 2, "layers must be greater than 2"
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layers_num = layers_num
        self.hidden_dim = hidden_dim

        self.layers = []
        for i in range(layers_num):
            if i == 0:
                self.layers.append(nn.Linear(input_dim,hidden_dim))
                self.layers.append(Sin())
            elif i == layers_num-1:
                self.layers.append(nn.Linear(hidden_dim,output_dim))
            else:
                self.layers.append(nn.Linear(hidden_dim,hidden_dim))
                self.layers.append(Sin())
                self.layers.append(nn.Dropout(p=droupout))
        self.net = nn.Sequential(*self.layers)
        self._init()

    def _init(self):
        for layer in self.net:
            if isinstance(layer,nn.Linear):
                nn.init.xavier_normal_(layer.weight)

    def forward(self,x):
        x = self.net(x)
        return x


class Predictor(nn.Module):
    def __init__(self,input_dim=40):
        super(Predictor, self).__init__()
        self.net = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(input_dim,32),
            Sin(),
            nn.Linear(32,1)
        )
        self.input_dim = input_dim
    def forward(self,x):
        return self.net(x)


class Solution_u(nn.Module):
    def __init__(self):
        super(Solution_u, self).__init__()
        self.encoder = MLP(input_dim=17,output_dim=32,layers_num=3,hidden_dim=60,droupout=0.2)
        self.predictor = Predictor(input_dim=32)
        self._init_()

    def get_embedding(self,x):
        return self.encoder(x)

    def forward(self,x):
        x = self.encoder(x)
        x = self.predictor(x)
        return x

    def _init_(self):
        for layer in self.modules():
            if isinstance(layer,nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.constant_(layer.bias,0)
            elif isinstance(layer,nn.Conv1d):
                nn.init.xavier_normal_(layer.weight)
                nn.init.constant_(layer.bias,0)


class PINN(nn.Module):
    def __init__(self,configs):
        super(PINN, self).__init__()

        self.solution_u = Solution_u().to(device)
        # todo: 此处的dynamical_F input_dim可能需要调整
        self.dynamical_F = MLP(input_dim=35,output_dim=1,
                               layers_num=configs.F_layers_num,
                               hidden_dim=configs.F_hidden_dim,
                               droupout=0.2).to(device)

        # self.optimizer = torch.optim.Adam(self.parameters(), lr=configs.warmup_lr)
        self.optimizer1 = torch.optim.Adam(self.solution_u.parameters(), lr=configs.warmup_lr)
        self.optimizer2 = torch.optim.Adam(self.dynamical_F.parameters(), lr=configs.lr_F)

        self.scheduler = LR_Scheduler(optimizer=self.optimizer1,
                                      warmup_epochs=configs.warmup_epochs,
                                      warmup_lr=configs.warmup_lr,
                                      num_epochs=configs.epochs,
                                      base_lr=configs.lr,
                                      final_lr=configs.final_lr)
        self.relu = nn.ReLU()


        # loss = loss1 + alpha*loss2 + beta*loss3
        self.alpha = self.configs.alpha
        self.beta = self.configs.beta

    def predict(self,xt):
        return self.solution_u(xt)

    def forward(self,xt):
        # todo: 由于此处的PINN需要传入连续的两个cycle的数据,因此需要修改到dataloader,此处暂时先考虑接别的模型来作为输出.
        xt.requires_grad = True
        x = xt[:,0:-1]
        t = xt[:,-1:]

        u = self.solution_u(torch.cat((x,t),dim=1))

        u_t = grad(u.sum(),t,
                   create_graph=True,
                   only_inputs=True,
                   allow_unused=True)[0]
        u_x = grad(u.sum(),x,
                   create_graph=True,
                   only_inputs=True,
                   allow_unused=True)[0]

        # todo: check dimension below if i change the dim of original input.
        F = self.dynamical_F(torch.cat([xt,u,u_x,u_t],dim=1))
        f = u_t - F

        for iter,(x1,x2,y1,y2) in enumerate(dataloader):
            x1,x2,y1,y2 = x1.to(device),x2.to(device),y1.to(device),y2.to(device)
            u1,f1 = self.forward(x1)
            u2,f2 = self.forward(x2)

            # data loss
            loss1 = 0.5*self.loss_func(u1,y1) + 0.5*self.loss_func(u2,y2)

            # PDE loss
            f_target = torch.zeros_like(f1)
            loss2 = 0.5*self.loss_func(f1,f_target) + 0.5*self.loss_func(f2,f_target)

            # physics loss  u2-u1<0
            loss3 = self.relu(u2-u1).sum()

            # total loss
            loss = loss1 + self.alpha*loss2 + self.beta*loss3

        return predictions, loss
