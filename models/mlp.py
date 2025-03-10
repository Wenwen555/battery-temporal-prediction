from torch import nn

class base_Model(nn.Module):
    def __init__(self, configs):
        super(base_Model, self).__init__()
        self.fc1 = nn.Linear(configs.mlp_input_channels * configs.original_seq_len,configs.hidden_channels)
        self.fc2 = nn.Linear(configs.hidden_channels,configs.hidden_channels)
        self.fc3 = nn.Linear(configs.hidden_channels, configs.mlp_final_output)
        
        self.fc4 = nn.Linear(configs.mlp_final_output, 1)
        self.dropout = nn.Dropout(configs.dropout)

    def forward(self, x):
        # 输入形状: [batch_size, cycles, features, seq_len]
        batch_size, cycles, features, seq_len = x.shape
        x = x.view(batch_size, cycles, -1)  # 形状变为 [batch_size, cycles, features * seq_len]
        
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        features = self.fc3(x)
        logits = self.fc4(features)
        logits = logits.squeeze(-1)
        return logits, features
