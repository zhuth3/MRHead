from torch.nn.modules.activation import ReLU
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch import tensor
from copy import deepcopy
from typing import List, Tuple
from mmdet.core.bbox.match_costs.builder import MATCH_COST
from mmdet.models import HEADS
from mmdet3d.models import ASSIGNER, MATCH

def log_sinkhorn_iterations(Z: torch.Tensor, log_mu: torch.Tensor, log_nu: torch.Tensor, iters: int) -> torch.Tensor:
    """ Perform Sinkhorn Normalization in Log-space for stability"""
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)


def log_optimal_transport(scores: torch.Tensor, alpha: torch.Tensor, iters: int) -> torch.Tensor:
    """ Perform Differentiable Optimal Transport in Log-space for stability"""
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns = (m*one).to(scores), (n*one).to(scores)

    bins0 = alpha.expand(b, m, 1)
    bins1 = alpha.expand(b, 1, n)
    alpha = alpha.expand(b, 1, 1)

    couplings = torch.cat([torch.cat([scores, bins0], -1),
                           torch.cat([bins1, alpha], -1)], 1)

    norm = - (ms + ns).log()
    log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
    log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = Z - norm  # multiply probabilities by M+N
    return Z

def MLP(channels: List[int], do_bn: bool = True) -> nn.Module:
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n-1):
            if do_bn:
                layers.append(nn.BatchNorm1d(channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)

def attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor]:
    dim = query.shape[1]
    scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim**.5
    prob = torch.nn.functional.softmax(scores, dim=-1)
    return torch.einsum('bhnm,bdhm->bdhn', prob, value), prob

class MultiHeadedAttention(nn.Module):
    """ Multi-head attention to increase model expressivitiy """
    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        batch_dim = query.size(0)
        query, key, value = [l(x).view(batch_dim, self.dim, self.num_heads, -1)
                             for l, x in zip(self.proj, (query, key, value))]
        x, _ = attention(query, key, value)
        return self.merge(x.contiguous().view(batch_dim, self.dim*self.num_heads, -1))

class AttentionalPropagation(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int):
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, feature_dim)
        self.mlp = MLP([feature_dim*2, feature_dim*2, feature_dim])
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        message = self.attn(x, source, source)
        return self.mlp(torch.cat([x, message], dim=1))

def arange_like(x, dim: int):
    return x.new_ones(x.shape[dim]).cumsum(0) - 1  # traceable in 1.1

def unique_max(scores):
    scores = scores.softmax(-2)
    max0 = scores[:, :, :].max(2)
    hash_table = torch.zeros(scores.shape[1])
    max1 = torch.zeros(scores.shape[2])
    idx = 0
    while(idx < scores.shape[2]):
      max_temp = scores[:, :, idx].max(1)
      if (hash_table[max_temp.indices] == 0):
        hash_table[max_temp.indices] = idx+1
        max1[idx] = max_temp.indices
        idx += 1
        continue
      else:
        scores[:, max_temp.indices, idx] = 0.0
        # print('Duplicate Error')

    return max0.indices, max1.unsqueeze(0).long()

class PositionalEncoding(nn.Module):
    """
    compute sinusoid encoding.
    """
    def __init__(self, d_model, max_len):
        """
        constructor of sinusoid encoding class

        :param d_model: dimension of model
        :param max_len: max sequence length
        :param device: hardware device setting
        """
        super(PositionalEncoding, self).__init__()

        # same size with input matrix (for adding with input matrix)
        self.encoding = torch.zeros(max_len, d_model)
        self.encoding.requires_grad = False  # we don't need to compute gradient

        pos = torch.arange(0, max_len)
        pos = pos.float().unsqueeze(dim=1)
        # 1D => 2D unsqueeze to represent word's position

        _2i = torch.arange(0, d_model, step=2).float()
        # 'i' means index of d_model (e.g. embedding size = 50, 'i' = [0,50])
        # "step=2" means 'i' multiplied with two (same with 2 * i)

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
        # compute positional encoding to consider positional information of words

    def forward(self, x):
        # self.encoding
        # [max_len = 512, d_model = 512]

        # batch_size, seq_len = x.size()
        # [batch_size = 128, seq_len = 30]
        encoding = self.encoding.to(x.device)
        return x+encoding[:x.size(1), :].repeat(x.size(0), 1, 1)
        # [seq_len = 30, d_model = 512]
        # it will add with tok_emb : [128, 30, 512] 

@MATCH.register_module()
class myMatch(nn.Module):
    def __init__ (self, feature_dim, layer_names):
        super().__init__()
        # self.points = nn.Sequential(
        #     nn.Conv1d(5, 32, kernel_size=1, bias=True),
        #     nn.BatchNorm1d(32),
        #     nn.ReLU(),
        #     nn.Conv1d(32, 64, kernel_size=1, bias=True),
        #     nn.BatchNorm1d(64),
        #     nn.ReLU(),
        #     nn.Conv1d(64, 256, kernel_size=1, bias=True),
        #     nn.BatchNorm1d(256),
        #     nn.ReLU(),
        #     # nn.Linear(20, 1),
        #     # nn.softmax()
        #     )
        # self.points = nn.Transformer(d_model=5, nhead=5, num_encoder_layers=6, num_decoder_layers=6)
        
        self._init_layers(feature_dim, layer_names)

    def _init_layers(self, feature_dim, layer_names):
        self.embding = nn.Sequential(
            nn.Conv1d(5, 256, kernel_size=1, bias=True),
            nn.BatchNorm1d(256),
            nn.Softmax(dim=-1),
            )
        self.position_encoder = PositionalEncoding(256, 20)
        encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=4)
        self.points = nn.Sequential(
            nn.TransformerEncoder(encoder_layer, num_layers=1),
            nn.Linear(256, 1),
            nn.BatchNorm1d(20),
            nn.Softmax(dim=-2),
            )

        self.layers = nn.ModuleList([
                AttentionalPropagation(feature_dim, 4)
                for _ in range(len(layer_names))])
        self.names = layer_names
        # self.layers = AttentionalPropagation(256, 4)

        bin_score = torch.nn.Parameter(torch.tensor(1.))
        self.register_parameter('bin_score', bin_score)

    def forward(self, pred_position, gt_position, pred_class, gt_class):
        #conbine position and class
        pred_encoding = torch.cat([pred_position, pred_class.unsqueeze(1).repeat(1, 20, 1)], 2)#[50, 20, 2+3]
        gt_class_onehot = F.one_hot(gt_class)
        gt_encoding = torch.cat([gt_position, gt_class_onehot.unsqueeze(1).repeat(1, 20, 1)], 2)

        pred_encoding = self.embding(pred_encoding.permute(0, 2, 1))
        pred_encoding = self.position_encoder(pred_encoding.permute(0, 2, 1))
        pred_encoding = self.points(pred_encoding)#[50, 20, 1]
        gt_encoding = self.embding(gt_encoding.permute(0, 2, 1))
        gt_encoding = self.position_encoder(gt_encoding.permute(0, 2, 1))
        gt_encoding = self.points(gt_encoding)
        # print('pred_encoding: ', pred_encoding.shape)
        # assert False, 'test'

        #attention superglue
        #print(outputs.shape)
        gt_encoding = gt_encoding.permute(2, 1, 0)
        pred_encoding = pred_encoding.permute(2, 1, 0)
        for layer, name in zip(self.layers, self.names):
            if name == 'cross':
                pred_temp, gt_temp = gt_encoding, pred_encoding
            else:  # if name == 'self':
                pred_temp, gt_temp = pred_encoding, gt_encoding

        pred_temp, gt_temp = layer(pred_encoding, pred_temp), layer(gt_encoding, gt_temp)
        pred_encoding, gt_encoding = (pred_encoding + pred_temp), (gt_encoding + gt_temp)

        #compute score matrix
        scores_compute = torch.einsum('bdn,bdm->bnm', pred_encoding, gt_encoding)#[1, 50, 11]
        scores = log_optimal_transport(scores_compute, self.bin_score, iters=50)
        # print(scores.shape)

        # max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
        # print(max1.indices)

        max0, max1 = unique_max(scores[:, :-1, :-1])
        #max0, max1 = unique_max(scores[:, :, :])
        # print(max1.shape)
        assigned_gt_inds = torch.zeros(max0.shape[1]).long().to(pred_position.device)
        # scores_matrix = torch.zeros(max0.shape[1]).long().to(pred_position.device)
        # print(max0.shape)
        idx = torch.from_numpy(np.array([i+1 for i in range(max1.shape[1])])).to(pred_position.device)
        assigned_gt_inds[max1] = idx
        # scores_matrix[max1] = 1
        # print(max0)

        # loss = torch.norm(scores_matrix - scores_compute.squeeze(0))

        assigned_labels = torch.zeros(max0.shape[1]).long().to(pred_position.device)
        assigned_labels[:] = -1
        # print(assigned_labels.shape)
        # print(gt_class.shape)
        assigned_labels[max1] = gt_class.squeeze(-1).long().to(pred_position.device)

        return assigned_gt_inds, assigned_labels


def main():
    gt_position = [tensor([71,  1, 75, 11, 75, 20, 72, 29, 66, 35, 63, 37, 51, 41,  1, 56,  0]),
    tensor([1,  81,  87,  55,  96,  54, 102,  54, 106,  57, 121,  68, 138,  93, 0]),
    tensor([192,  81, 190,  79, 145,  11, 144,   6, 145,   1,   0]),
    tensor([161,  93, 134,  54, 134,  52, 136,  50, 138,  51, 166,  93,   0]),
    tensor([78,  1, 73,  4, 71,  1,  0]),
    tensor([155,  26, 161,  34, 114,  64, 105,  57, 155,  26,   0]),
    tensor([102,  54,  91,  55,  73,  27,  76,  14, 102,  54,   0]),
    tensor([177,  93, 146,  47,   0]),
    tensor([150,  93, 126,  59,   0]),
    tensor([186,  89, 154,  42,   0]),
    tensor([84, 43,  1, 69,  0])]
    #gt_class = tensor(np.random.rand(11, 1)).float()
    gt_class = tensor(np.array([[0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1]]).reshape(11, -1)).long()
    pred_position = tensor(np.random.rand(120, 2, 256)).float()
    pred_class = tensor(np.random.rand(120, 3)).float()
    # print(pred.shape)

    # print(input.shape)
    # n, d = input.shape
    # input = input[:, :-1].reshape(n, -1, 2)
    # input = torch.tensor(input, dtype=torch.float32).permute(0, 2, 1)
    # # input = input.unsqueeze(1)
    # #print(input)
    # model = transScale()
    # out = model(input)
    # print(out.shape)

    feature_dim = 256
    layer_names = ['self', 'coss'] * 3

    model = myMatch(feature_dim, layer_names)
    outx, outpred = model(pred_position, gt_position, pred_class, gt_class)

    # print(outx.shape)
    # print(outpred.shape)