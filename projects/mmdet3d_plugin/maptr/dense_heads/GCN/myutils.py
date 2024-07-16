import argparse
import h5py

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

def _get_weights(dist, indices):
    b, num_nodes, k = dist.shape
    # print(indices.shape)
    # assert b, num_nodes, k == indices.shape
    # print(dist.shape)
    dist = torch.nan_to_num(dist, nan=1000.0)
    assert dist.min() >= 0
    # weight matrix
    # print('dist: ', dist.shape)

    temp = dist[:, :, -1]
    sigma2 = torch.mean(temp, dim=-1)**2
    sigma2 = sigma2.unsqueeze(-1).repeat(1, num_nodes*k).reshape(b, num_nodes, k)
    # print('dist: ', dist.shape)
    # print('sigma2: ', sigma2.shape)
    dist = torch.exp(torch.div(-dist**2, sigma2))
    # n = torch.tensor(np.arange(0, b)).repeat(num_nodes * k, 1).transpose(1, 0)
    i = torch.tensor(np.arange(0, num_nodes)).repeat(k, 1).transpose(1, 0).reshape(1, num_nodes * k).repeat(b, 1).to(dist.device)
    j = indices.reshape(b, num_nodes * k)
    v = dist.reshape(b, num_nodes * k)
    # weights = torch.sparse_csr_tensor(i, j, v, (b, num_nodes, num_nodes)).to_dense()
    weights = torch.zeros([b, num_nodes*num_nodes]).to(dist.device)
    # print('i: ', i.device)
    # print('j: ', j.device)
    idx = i*num_nodes + j
    # print('idx: ', idx.device)
    # print('v: ', v.device)
    # print('weights: ', weights.device)
    weights = weights.scatter_(1, idx, v).reshape(b, num_nodes, num_nodes)
    temp = torch.diagonal(weights, dim1=-1, dim2=-2)
    #   print(temp.shape)
    temp = torch.diag_embed(temp)
    #   print(temp.shape)
    weights = weights-temp
    # undirected graph
    bigger = weights.permute(0, 2, 1) > weights
    weights = weights - weights.multiply(bigger) + weights.permute(0, 2, 1).multiply(bigger)
    return weights

def _get_normalize_adj(dist, indices):
    # print('get norm: ', indices[0][0])
    adj = _get_weights(dist, indices)
    # adj = adj.to_sparse_csr
    row_sum = torch.sum(adj, dim=2)
    d_inv = torch.pow(row_sum, -0.5)
    d_inv[torch.isinf(d_inv)] = 0.0
    d_mat_inv_sqrt = torch.diag_embed(d_inv)
    out = torch.bmm(torch.bmm(adj, d_mat_inv_sqrt).transpose(2, 1), d_mat_inv_sqrt).detach()
    return out

def get_ordered_knn_idx(pos, k):
    # print('pos.shape', pos.shape)
    bs, pts, dim = pos.shape
    k_half = int(k/2)
    temp_k = torch.tensor(range(k+1)) - k/2
    temp_k = torch.cat((temp_k[:k_half], temp_k[k_half+1:])).view(k, -1)

    temp_idx = torch.tensor(range(pts)).repeat(k, 1)
    single_idx = torch.clamp(temp_idx - temp_k, 0, pts-1).long().permute(1, 0)
    idx = single_idx.repeat(bs, 1, 1)

    return idx

def my_build_graph(pos, query, k=4, offset=0):
    """
    :param coordinates: positions for 3D point cloud (N * 3)
    :param k: number of nearest neighbors
    :return: adjacency matrix for 3D point cloud
    """
    B, N, F = tuple(pos.size())
    M = query.size(1)
    x = pos

    pos = pos.unsqueeze(1).expand(B, M, N, F)
    query  = query.unsqueeze(2).expand(B, M, N, F)   # B * M * N * F
    dist = torch.sum((pos - query) ** 2, dim=3, keepdim=False)   # B * M * N
    dist = torch.sqrt(dist)
    knn_idx = torch.argsort(dist, dim=2)[:, :, offset:k+offset]   # B * M * k
    # knn_idx = get_ordered_knn_idx(x, k).to(dist.device)
    knn_dist = torch.gather(dist, dim=2, index=knn_idx)           # B * M * k
    return _get_normalize_adj(knn_dist, knn_idx)