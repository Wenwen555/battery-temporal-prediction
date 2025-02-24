import torchvision.models

from models.resnet import resnet34
import torch.nn as nn


class base_Model(nn.Module):
    def __init__(self, configs, strides=1):
        super().__init__()
        self.resnet34 = resnet34()


    def forward(self, x):
        # The result of returned values is a pair: (logics, x)
        # In which logics is the predicted value of model.
        # The batchsize is carefully calculated and the result is 256.
        return self.resnet34(x)


# Below code is for counting the parameters of model.
# def count_parameters(model):
#     count = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     print('The model has {} trainable parameters'.format(count))
#
#
# if __name__ == "__main__":
#     model = base_Model(None)
#     count_parameters(model)
#
