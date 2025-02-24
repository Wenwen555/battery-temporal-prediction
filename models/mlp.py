from torch import nn

class base_Model(nn.Module):
    def __init__(self, configs):
        super(base_Model, self).__init__()
        self.fc1 = nn.Linear(configs.original_input_channels * configs.original_seq_len,configs.hidden_channels)
        self.fc2 = nn.Linear(configs.hidden_channels,configs.hidden_channels)
        self.fc3 = nn.Linear(configs.hidden_channels, configs.num_classes)
        self.dropout = nn.Dropout(configs.dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        logits = self.fc3(x)

        return logits, x
