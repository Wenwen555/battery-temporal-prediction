# import torch.nn as nn
# import torch
# import matplotlib.pyplot as plt
#
# class base_Model(nn.Module):
#     def __init__(self,configs):
#         super(base_Model, self).__init__()
#
#         self.conv_block = nn.Sequential(
#             nn.Conv1d(in_channels=configs.cnn_input_channels_1, out_channels=configs.cnn_output_channels_1, kernel_size=3, stride=2, padding=1),
#             nn.ReLU(),
#             # nn.MaxPool1d(kernel_size=2, stride=2),
#             nn.Conv1d(in_channels=configs.cnn_input_channels_2, out_channels=configs.cnn_output_channels_2, kernel_size=3, stride=2, padding=1),
#             nn.ReLU(),
#             # nn.MaxPool1d(kernel_size=2, stride=2),
#             nn.Conv1d(in_channels=configs.cnn_input_channels_3, out_channels=configs.cnn_output_channels_3, kernel_size=3, stride=2, padding=1),  # Output: (128, 256, 68)
#         )
#         self.avgpool = nn.AdaptiveAvgPool1d(1)
#
#         self.fc = nn.Linear(configs.cnn_output_channels_3, 1)  # 256 channels -> 1 output
#
#
#     def forward(self, x):
#         input = x
#         logits_x = self.conv_block(x)
#         x = self.avgpool(logits_x)
#         flatten = torch.flatten(x, 1)
#         logits = self.fc(flatten)
#         return logits, x


import torch.nn as nn
import torch
import matplotlib.pyplot as plt

class base_Model(nn.Module):
    def __init__(self,configs):
        super(base_Model, self).__init__()

        # self.conv = nn.Conv2d(in_channels=3, out_channels=256, kernel_size=(1,537))
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels=configs.cnn_input_channels_1, out_channels=configs.cnn_output_channels_1, kernel_size=25, stride=3, padding=1),
            nn.ReLU(),
            # nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=configs.cnn_input_channels_2, out_channels=configs.cnn_output_channels_2, kernel_size=25, stride=3, padding=1),
            nn.ReLU(),
            # nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=configs.cnn_input_channels_3, out_channels=configs.cnn_output_channels_3, kernel_size=25, stride=3, padding=1),  # Output: (128, 256, 68)
        )
        # self.avgpool = nn.AdaptiveAvgPool2d((configs.cnn_output_channels_3,1))
        self.avgpool = nn.AdaptiveAvgPool1d(1) #池化会削减差别
        self.fc = nn.Linear(configs.cnn_output_channels_3, 1)  # 256 *  channels -> 1 output


    def forward(self, x):
        input = x
        x = self.conv_block(x)
        x = self.avgpool(x)
        flatten = torch.flatten(x, 1)
        logits = self.fc(flatten)
        return logits, x

