import torch
from torch.nn import Module, Linear, ModuleList
import torch.nn.functional as F

from .utils import *


class DenseEdgeConv(Module):

    def __init__(self, in_channels, num_layers, layer_out_dim, knn=16, aggr='max', activation='relu'):
        super().__init__()
        self.in_channels = in_channels
        self.knn = knn
        assert num_layers > 2
        self.num_layers = num_layers
        self.layer_out_dim = layer_out_dim
        
        # Densely Connected Layers
        self.layer_first = FullyConnected(2*in_channels, layer_out_dim, bias=True, activation=activation)
        self.layer_last = FullyConnected(in_channels + (num_layers - 1) * layer_out_dim, layer_out_dim, bias=True, activation=None)
        self.layers = ModuleList()
        for i in range(1, num_layers-1):
            self.layers.append(FullyConnected(in_channels + i * layer_out_dim, layer_out_dim, bias=True, activation=activation))

        self.aggr = Aggregator(aggr)

    @property
    def out_channels(self):
        return self.in_channels + self.num_layers * self.layer_out_dim

    def get_edge_feature(self, x, knn_idx):
        """
        :param  x:          (B, N, d)
        :param  knn_idx:    (B, N, K)
        :return (B, N, K, 2*d)
        """
        knn_feat = group(x, knn_idx)   # B * N * K * d
        x_tiled = x.unsqueeze(-2).expand_as(knn_feat)
        # edge_feat = torch.cat([x_tiled, knn_feat, knn_feat - x_tiled], dim=3)
        edge_feat = torch.cat([x_tiled, knn_feat], dim=3)
        return edge_feat

    def forward(self, x, pos):
        """
        :param  x:  (B, N, d)
        :return (B, N, d+L*c)
        """
        knn_idx = get_knn_idx(pos, pos, k=self.knn, offset=1)

        # First Layer
        edge_feat = self.get_edge_feature(x, knn_idx)
        y = torch.cat([
            self.layer_first(edge_feat),              # (B, N, K, c)
            x.unsqueeze(-2).repeat(1, 1, self.knn, 1) # (B, N, K, d)
        ], dim=-1)  # (B, N, K, d+c)

        # Intermediate Layers
        for layer in self.layers:
            y = torch.cat([
                layer(y),           # (B, N, K, c)
                y,                  # (B, N, K, c+d)
            ], dim=-1)  # (B, N, K, d+c+...)
        
        # Last Layer
        y = torch.cat([
            self.layer_last(y), # (B, N, K, c)
            y                   # (B, N, K, d+(L-1)*c)
        ], dim=-1)  # (B, N, K, d+L*c)

        # Pooling
        y = self.aggr(y, dim=-2)
        
        return y

class OrderEdgeConv(Module):

    def __init__(self, in_channels, num_layers, layer_out_dim, knn=16, aggr='max', activation='relu'):
        super().__init__()
        self.in_channels = in_channels
        assert knn%2 == 0, 'Currently, only supports knn as an even number'
        self.knn = knn
        assert num_layers > 2
        self.num_layers = num_layers
        self.layer_out_dim = layer_out_dim
        
        # Densely Connected Layers
        self.layer_first = FullyConnected(3*in_channels, layer_out_dim, bias=True, activation=activation)
        self.layer_last = FullyConnected(in_channels + (num_layers - 1) * layer_out_dim, layer_out_dim, bias=True, activation=None)
        self.layers = ModuleList()
        for i in range(1, num_layers-1):
            self.layers.append(FullyConnected(in_channels + i * layer_out_dim, layer_out_dim, bias=True, activation=activation))

        self.aggr = Aggregator(aggr)

    @property
    def out_channels(self):
        return self.in_channels + self.num_layers * self.layer_out_dim

    def get_edge_feature(self, x, knn_idx):
        """
        :param  x:          (B, N, d)
        :param  knn_idx:    (B, N, K)
        :return (B, N, K, 2*d)
        """
        bs, pts, k = knn_idx.shape
        k_half = int(k/2)
        knn_feat = group(x, knn_idx)   # B * N * K * d
        # print('knn feat: ', knn_feat[:, :, 0, :].shape)
        # print(knn_feat.shape)
        x_tiled = x.unsqueeze(-2).expand_as(knn_feat)
        ordered_feat = torch.zeros_like(knn_feat)
        ordered_feat[:, :, 0:k_half, :] = x.unsqueeze(2) - knn_feat[:, :, 0:k_half, :]
        ordered_feat[:, :, k_half:, :] = knn_feat[:, :, k_half:, :] - x.unsqueeze(2)
        # ordered_feat = knn_feat - x_tiled
        # print(ordered_feat.shape)
        edge_feat = torch.cat([x_tiled, knn_feat, ordered_feat], dim=3)
        # edge_feat = torch.cat([x_tiled, knn_feat], dim=3)
        return edge_feat

    def get_ordered_knn_idx(self, pos, k):
        bs, pts, dim = pos.shape
        k_half = int(k/2)
        temp_k = torch.tensor(range(k+1)) - k/2
        temp_k = torch.cat((temp_k[:k_half], temp_k[k_half+1:])).view(k, -1)

        temp_idx = torch.tensor(range(pts)).repeat(k, 1)
        single_idx = torch.clamp(temp_idx - temp_k, 0, pts-1).long().permute(1, 0)
        idx = single_idx.repeat(bs, 1, 1)

        return idx

    def forward(self, x, pos):
        """
        :param  x:  (B, N, d)
        :return (B, N, d+L*c)
        """
        knn_idx = self.get_ordered_knn_idx(pos, k=self.knn).to(pos.device)
        # knn_idx = get_knn_idx(pos, pos, k=self.knn, offset=1)

        # First Layer
        edge_feat = self.get_edge_feature(x, knn_idx)
        y = torch.cat([
            self.layer_first(edge_feat),              # (B, N, K, c)
            x.unsqueeze(-2).repeat(1, 1, self.knn, 1) # (B, N, K, d)
        ], dim=-1)  # (B, N, K, d+c)

        # Intermediate Layers
        for layer in self.layers:
            y = torch.cat([
                layer(y),           # (B, N, K, c)
                y,                  # (B, N, K, c+d)
            ], dim=-1)  # (B, N, K, d+c+...)
        
        # Last Layer
        y = torch.cat([
            self.layer_last(y), # (B, N, K, c)
            y                   # (B, N, K, d+(L-1)*c)
        ], dim=-1)  # (B, N, K, d+L*c)

        # Pooling
        y = self.aggr(y, dim=-2)
        
        return y
