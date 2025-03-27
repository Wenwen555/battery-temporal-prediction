import torch
import torch.nn as nn

class base_Model(nn.Module):
    def __init__(self, config):
        super(base_Model, self).__init__()
        self.hidden_size = config.hidden_size
        # LSTM 层
        self.lstm = nn.LSTM(config.input_size, config.hidden_size, batch_first=False)
        
        # 全连接层：用于生成预测结果
        self.fc_pred = nn.Linear(256, 1)
        
        # 全连接层：用于生成 256 维输出表征
        self.fc_repr = nn.Linear(config.hidden_size, 256)

    def forward(self, x):
        
        batch, cycles, features, seq_len = x.shape
        x = x.view(batch*cycles, features, seq_len)
        x = x.permute(2, 0, 1) # [seq_len, batch * cycles, features]
        
        lstm_out, (h_n, c_n) = self.lstm(x) # LSTM 输出形状: [seq_len, batch * cycles, hidden_size]
        # 取最后一个时间步的输出作为表征
        # h_n 形状: [1, batch * cycles, hidden_size]
        last_hidden = h_n[-1]  # [batch * cycles, hidden_size]
        # 生成预测结果
        # pred = self.fc_pred(last_hidden)  # [batch * cycles, 1]
        
        # 生成 256 维输出表征
        features = self.fc_repr(last_hidden)  # [batch * cycles, 256]
        pred = self.fc_pred(features)
        pred = pred.squeeze(-1)  # [batch * cycles]
        pred = pred.view(batch, cycles)
        features = features.view(batch, cycles ,256)
        return pred, repr
