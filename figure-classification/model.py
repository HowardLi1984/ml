import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

class mlp_net(nn.Module):
    def __init__(self, config):
        super(mlp_net, self).__init__()
        self.layer1 = nn.Linear(config.mlp_input_size, config.mlp_hidden_size)
        self.layer2 = nn.Linear(config.mlp_hidden_size, config.mlp_output_size)
        self.layer3 = nn.Linear(config.mlp_output_size, config.mlp_output_size)
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.layer1(x)
        x = F.relu(x)
        x = self.layer2(x)
        x = F.relu(x)
        output = self.layer3(x)
        return output


class cnn_net(nn.Module):
    def __init__(self, config):
        super(cnn_net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=config.in_channel, out_channels=config.hidden_channel_1,\
                                 kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(config.hidden_channel_1, affine=True)
        self.conv2 = nn.Conv2d(in_channels=config.hidden_channel_1, out_channels=config.hidden_channel_2,\
                                 kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(config.hidden_channel_2, affine=True)
        self.dropout = nn.Dropout(p=0.5)                        
        self.out = nn.Linear(config.hidden_channel_2 * 7 * 7, 10)
    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = F.relu(x1)
        x2 = F.max_pool2d(x1, (2,2))

        x3 = self.conv2(x2)
        x3 = self.bn2(x3)
        x3 = self.dropout(x3)

        x3 = F.relu(x3)
        x4 = F.max_pool2d(x3, (2,2))
        x5 = x4.view(x4.size(0), -1)
        output = self.out(x5)
        return output