import torch

import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential

import numpy as np

from collections import OrderedDict

from .layers import GlobalPooling
from .layers import GraphConvolution
from .utils import build_graph
from .myutils import my_build_graph


class MultiLayerGCN(nn.Module):
    def __init__(self, dropout=0.5, num_classes=40):
        super(MultiLayerGCN, self).__init__()
        self.conv0 = GraphConvolution(3, 64, bias=False)
        self.conv1 = GraphConvolution(64, 64, bias=False)
        self.conv2 = GraphConvolution(64, 128, bias=False)
        self.conv3 = GraphConvolution(128, 256, bias=False)
        self.conv4 = GraphConvolution(512, 1024, bias=False)
        self.bn0 = nn.BatchNorm1d(64)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(1024)
        self.pool = GlobalPooling()
        self.classifier = nn.Sequential(OrderedDict([
            ('fc0', nn.Linear(2048, 512, bias=False)),
            ('relu0', nn.LeakyReLU(negative_slope=0.2)),
            ('bn0', nn.BatchNorm1d(512)),
            ('drop0', nn.Dropout(p=dropout)),
            ('fc1', nn.Linear(512, 256, bias=False)),
            ('relu1', nn.LeakyReLU(negative_slope=0.2)),
            ('bn1', nn.BatchNorm1d(256)),
            ('drop1', nn.Dropout(p=dropout)),
            ('fc2', nn.Linear(256, num_classes)),
        ]))

    def forward(self, adj, x):
        x0 = F.leaky_relu(self.bn0(self.conv0(adj, x)), negative_slope=0.2)
        x1 = F.leaky_relu(self.bn1(self.conv1(adj, x0)), negative_slope=0.2)
        x2 = F.leaky_relu(self.bn2(self.conv2(adj, x1)), negative_slope=0.2)
        x3 = F.leaky_relu(self.bn3(self.conv3(adj, x2)), negative_slope=0.2)
        x = torch.cat((x0, x1, x2, x3), dim=1)
        x = F.leaky_relu(self.bn4(self.conv4(adj, x)), negative_slope=0.2)
        x = self.pool(x)
        x = self.classifier(x)
        return x

class GCNFeatureExtractor(nn.Module):
    def __init__(self, input_dim=2, output_dim=1024):
        super(GCNFeatureExtractor, self).__init__()
        self.conv0 = GraphConvolution(input_dim, 64, bias=False)
        self.conv1 = GraphConvolution(64, 128, bias=False)
        self.conv2 = GraphConvolution(128, output_dim, bias=False)
        # self.conv3 = GraphConvolution(128, output_dim, bias=False)
        # self.conv4 = GraphConvolution(512, output_dim, bias=False)
        self.bn0 = nn.BatchNorm1d(64)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(output_dim)
        # self.bn3 = nn.BatchNorm1d(output_dim)
        # self.bn4 = nn.BatchNorm1d(output_dim)
        # self.pool = GlobalPooling()
        # self.classifier = nn.Sequential(OrderedDict([
        #     ('fc0', nn.Linear(2048, 512, bias=False)),
        #     ('relu0', nn.LeakyReLU(negative_slope=0.2)),
        #     ('bn0', nn.BatchNorm1d(512)),
        #     ('drop0', nn.Dropout(p=dropout)),
        #     ('fc1', nn.Linear(512, 256, bias=False)),
        #     ('relu1', nn.LeakyReLU(negative_slope=0.2)),
        #     ('bn1', nn.BatchNorm1d(256)),
        #     ('drop1', nn.Dropout(p=dropout)),
        #     ('fc2', nn.Linear(256, num_classes)),
        # ]))

    def forward(self, x):
        # x = x.permute(0, 2, 1)
        # print('x: ', x.shape)
        # adjs = []
        # for i in range(x.shape[0]):
        #     adj = build_graph(x[i], k=4).todense()
        #     adjs.append(adj)
        # adjs = torch.tensor(np.array(adjs)).to(torch.float32).to(x.device)
        # print('adjs: ', adjs[-1])
        # x = x.permute(0, 2, 1)
        adjs0 = my_build_graph(x, x, 8).to(torch.float32).to(x.device)#1200*20*20
        # adjs1 = my_build_graph(x, x, 8).to(torch.float32).to(x.device)
        # adjs2 = my_build_graph(x, x, 8).to(torch.float32).to(x.device)
        # print('adjs: ', adjs.requires_grad)
        # print('adjs: ', adjs)
        # assert False
        x = x.permute(0, 2, 1)

        x0 = F.leaky_relu(self.bn0(self.conv0(adjs0, x)), negative_slope=0.2)#1200*64*20
        x1 = F.leaky_relu(self.bn1(self.conv1(adjs0, x0)), negative_slope=0.2)#1200*128*20

        # x_temp = x1.permute(0, 2, 1)
        # adjs1 = my_build_graph(x_temp, x_temp, 8).to(torch.float32).to(x.device)#1200*128*128
        # x_temp = x_temp.permute(0, 2, 1)

        x2 = F.leaky_relu(self.bn2(self.conv2(adjs0, x1)), negative_slope=0.2)
        # x3 = F.leaky_relu(self.bn3(self.conv3(adjs1, x2)), negative_slope=0.2)
        # x = torch.cat((x0, x1, x2, x3), dim=1)
        # x = F.leaky_relu(self.bn4(self.conv4(adjs, x)), negative_slope=0.2)
        # x = self.pool(x)
        # x = self.classifier(x)
        
        # x = torch.cat((x0, x1), dim=-2)
        # x = torch.cat((x, x2), dim=-2)
        x = x2.permute(0, 2, 1)
        return x

class Downsample(nn.Module):
    def __init__(self, feature_dim, activation='relu'):
        super().__init__()
        # self.mlp = Sequential(
        #     FullyConnected(feature_dim, feature_dim // 2, activation=activation),
        #     FullyConnected(feature_dim // 2, feature_dim // 4, activation=activation),
        #     FullyConnected(feature_dim // 4, 2, activation=None)
        # )
        self.mlp = Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, feature_dim // 4),
            nn.ReLU(),
            nn.Linear(feature_dim // 4, 2),
            nn.ReLU(),
        )

    def forward(self, pos, x):
        """
        :param  pos:    (B, N, 3)
        :param  x:      (B, N, d)
        :return (B, rN, d)
        """
        # idx, pos, x = self.pool(pos, x)
        # idx = None
        # if self.pre_filter:
        #     pos = pos + self.mlp(x)
        # return idx, pos, x
        pos = pos + self.mlp(x)
        return pos

def main():
    features = torch.rand(4, 3, 1024)
    adj = torch.rand(4, 1024, 1024)
    model = MultiLayerGCN()
    print('Model:', utils.get_total_parameters(model))
    score = model(adj, features)
    print('Classification:', score.size())


if __name__ == '__main__':
    main()
