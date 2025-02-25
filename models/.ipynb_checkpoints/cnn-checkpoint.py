import torch
import torch.nn as nn

class base_Model(nn.Module):
    def __init__(self, configs):
        super(base_Model, self).__init__()
        
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels=configs.cnn_input_channels_1, out_channels=configs.cnn_output_channels_1, kernel_size=25, stride=3, padding=2),
            nn.ReLU(),
            # nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=configs.cnn_input_channels_2, out_channels=configs.cnn_output_channels_2, kernel_size=25, stride=3, padding=2), 
            nn.ReLU(),
            # nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=configs.cnn_input_channels_3, out_channels=configs.cnn_output_channels_3, kernel_size=25, stride=3, padding=2), 
            nn.ReLU(),
            # nn.MaxPool1d(kernel_size=2, stride=2),
            nn.AdaptiveAvgPool1d(output_size=1)
        )
        
        # 全连接层：生成预测结果 (32, 30)
        self.fc = nn.Linear(configs.cnn_output_channels_3, 1)

    def forward(self, x):
        batch_size, cycles, channels, seq_len = x.size()
        x = x.view(batch_size * cycles, channels, seq_len)
        x = self.conv_block(x)
        x = x.squeeze(-1)
        x = x.view(batch_size, cycles, -1)
        predictions = self.fc(x).squeeze(-1)
        return predictions, x