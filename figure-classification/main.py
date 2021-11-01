import json
import torch
import torch.nn as nn
import torch.utils.data as Data
# import torch.utils.data.DataLoader as DataLoader
import torchvision
import torch.nn.functional as F
import numpy as np

from model import mlp_net, cnn_net

class Config():
    def __init__(self):
        self.cuda = True
        self.train_batch = 64
        self.test_batch = 64
        self.epoch = 4
        self.lr = 5e-4
        self.download_mnist = True
        self.mlp_input_size = 28*28
        self.mlp_hidden_size = 2048
        self.mlp_output_size = 10
        
        self.image_size = 28
        # cnn
        self.in_channel = 1
        self.hidden_channel_1 = 16
        self.hidden_channel_2 = 32

def build_dataloader(config):
    train_data = torchvision.datasets.MNIST(root='./mnist/', train=True, transform=torchvision.transforms.ToTensor(),
                                            download=config.download_mnist, )
    test_data = torchvision.datasets.MNIST(root='./mnist/', train=False, transform=torchvision.transforms.ToTensor())
    train_dataloader = Data.DataLoader(
        train_data, batch_size=config.train_batch, shuffle=True
    )
    test_dataloader = Data.DataLoader(
        test_data, batch_size=config.test_batch, shuffle=True
    )
    return train_dataloader, test_dataloader


def main():
    config = Config()
    train_dataloader, test_dataloader = build_dataloader(config)

    # net = mlp_net(config)
    net = cnn_net(config)
    if config.cuda: net.to("cuda:0")
    optimizer = torch.optim.Adam(net.parameters(), lr=config.lr)
    loss_function = nn.CrossEntropyLoss()
    test_result_list = []

    # training
    for epoch in range(config.epoch):
        print("epoch = ", epoch)
        for idx, (train_x, train_y) in enumerate(train_dataloader):
            if config.cuda:
                train_x, train_y = train_x.cuda(), train_y.cuda()
            pred_y = net(train_x)
            loss = loss_function(pred_y, train_y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if idx % 100 == 0: # testing
                total_correct, total_ans = 0, 0
                for idx, (test_x, test_y) in enumerate(test_dataloader):
                    if config.cuda:
                        test_x, test_y = test_x.cuda(), test_y.cuda()
                    test_output = net(test_x)
                    pred_y = torch.max(test_output, 1)[1].data.squeeze()
                    total_correct += torch.sum(pred_y == test_y).type(torch.FloatTensor)
                    total_ans += test_y.size(0)
                print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.cpu().numpy(),\
                                        '| test accuracy: %.3f' % (total_correct/total_ans))
                test_result_list.append(float(total_correct/total_ans))
                # print(type(test_result_list[-1]))
    
    result_dict = json.load(open("result.json"))
    result_dict['cnn_total'] = test_result_list
    json.dump(result_dict, open("result.json", "w"))
    return
main()