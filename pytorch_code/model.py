#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
Created on July, 2018

@author: Tangrizzly
"""

import datetime
import math
import numpy as np
import torch
from torch import nn
from torch.nn import Module, Parameter
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GatedGraphConv, SAGEConv
from torch_geometric.utils import to_networkx
from torch_cluster import random_walk
import networkx as nx


class GNN(Module):
    def __init__(self, hidden_size, step=1):
        super(GNN, self).__init__()
        self.step = step
        self.hidden_size = hidden_size
        self.input_size = hidden_size * 2
        self.gate_size = 3 * hidden_size
        self.w_ih = Parameter(torch.Tensor(self.gate_size, self.input_size))
        self.w_hh = Parameter(torch.Tensor(self.gate_size, self.hidden_size))
        self.b_ih = Parameter(torch.Tensor(self.gate_size))
        self.b_hh = Parameter(torch.Tensor(self.gate_size))
        self.b_iah = Parameter(torch.Tensor(self.hidden_size))
        self.b_oah = Parameter(torch.Tensor(self.hidden_size))

        self.linear_edge_in = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_out = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_f = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

        self.conv1 = GATConv(self.hidden_size, self.hidden_size, heads=8, dropout=0.6)
        # On the Pubmed dataset, use heads=8 in conv2.
        self.conv2 = GATConv(8 * 100, self.hidden_size, heads=1, concat=False, dropout=0.6)
        self.ggnn = GatedGraphConv(self.hidden_size, step+1)

    def GNNCell(self, A, hidden, edge_index):
        # todo 加上公式 8, 9的 GAT
        hy = self.conv1(hidden, edge_index)

        input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
        input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
        inputs = torch.cat([input_in, input_out], 2)
        gi = F.linear(inputs, self.w_ih, self.b_ih)
        gh = F.linear(hidden, self.w_hh, self.b_hh)
        i_r, i_i, i_n = gi.chunk(3, 2)
        h_r, h_i, h_n = gh.chunk(3, 2)
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + resetgate * h_n)
        hy = newgate + inputgate * (hidden - newgate)
        return hy

    def forward(self, A, hidden, edge_index=None):
        # 原始 paper用 GAT取代 GGNN
        # hidden = F.relu(self.conv1(hidden, edge_index))
        # hidden = self.conv2(hidden, edge_index)

        # todo 用 GGNN layer
        hidden = self.ggnn(hidden, edge_index)

        # for i in range(self.step):
        #     hidden = self.GNNCell(A, hidden, edge_index)
        return hidden


class SessionGraph(Module):
    def __init__(self, opt, n_node):
        super(SessionGraph, self).__init__()
        self.hidden_size = opt.hiddenSize
        self.n_node = n_node
        self.batch_size = opt.batchSize
        self.nonhybrid = opt.nonhybrid
        self.embedding = nn.Embedding(self.n_node, self.hidden_size)
        self.gnn = GNN(self.hidden_size, step=opt.step)
        self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_three = nn.Linear(self.hidden_size, 1, bias=False)
        self.linear_transform = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def compute_scores(self, hidden, mask):
        ht = hidden[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]  # batch_size x latent_size
        q1 = self.linear_one(ht).view(ht.shape[0], 1, ht.shape[1])  # batch_size x 1 x latent_size
        q2 = self.linear_two(hidden)  # batch_size x seq_length x latent_size
        alpha = self.linear_three(torch.sigmoid(q1 + q2))
        a = torch.sum(alpha * hidden * mask.view(mask.shape[0], -1, 1).float(), 1)
        if not self.nonhybrid:
            a = self.linear_transform(torch.cat([a, ht], 1))
        b = self.embedding.weight[1:]  # n_nodes x latent_size
        scores = torch.matmul(a, b.transpose(1, 0))
        return scores

    def forward(self, inputs, A, edge_index=None):
        hidden = self.embedding(inputs).squeeze()
        hidden = self.gnn(A, hidden, edge_index)
        pad = self.embedding(torch.Tensor([0]).to(torch.int64).cuda())
        return hidden, pad


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


def get(padding_emb, i, hidden, alias_inputs, length):
    # 手動padding
    h_ = hidden[i][alias_inputs[i]]
    if h_.shape[0] == length:
        return h_
    else:
        r_ = padding_emb.repeat(length - h_.shape[0], 1)
        return torch.cat([h_, r_])


def forward(model, i, data):
    # 改成用 geometric的Data格式
    items, targets, mask, batch, seq = data.x, data.y, data.sequence_mask, data.batch, data.sequence
    seq = seq.view(targets.shape[0], -1).cpu().numpy()
    seq_len = data.sequence_len.view(-1).cpu().numpy()
    mask = mask.view(targets.shape[0], -1)

    A = []
    # datas = data.to_data_list()
    # graphs = [to_networkx(d) for d in datas]
    # A = [nx.convert_matrix.to_pandas_adjacency(g).values for g in graphs]  # 無向圖adj = in + out
    # # todo 用原始seq建 in& out adj
    # A_out = [g for g in graphs]  # 有向圖的adj就是A_out
    # # todo 從 adj, in_edge建 in adj

    hidden, pad = model(items, A, data.edge_index)
    # 推回原始序列
    sections = torch.bincount(batch).cpu().numpy()
    # split whole x back into graphs G_i
    hidden = torch.split(hidden, tuple(sections))
    x = torch.split(items, tuple(sections))
    # x是unique nodes, sequence是原始序列
    # todo 測試不padding直接算attention的版本

    alias_inputs = []
    for i in range(targets.shape[0]):
        x_ = x[i].cpu().numpy().reshape(-1)
        len_ = seq_len[i]
        seq_ = seq[i, :len_]
        alias_inputs.append([np.where(x_ == j)[0][0] for j in seq_])

    leng = seq.shape[1]
    seq_hidden = torch.stack([get(pad, i, hidden, alias_inputs, leng) for i in torch.arange(len(alias_inputs)).long()])
    return targets, model.compute_scores(seq_hidden, mask)


def train_test(model, train, test):
    model.scheduler.step()
    print('start training: ', datetime.datetime.now())
    model.train()
    total_loss = 0.0
    for i, batch in enumerate(train):
        model.optimizer.zero_grad()
        targets, scores = forward(model, i, batch.to('cuda'))
        # targets = trans_to_cuda(torch.Tensor(targets).long())
        loss = model.loss_function(scores, targets - 1)
        loss.backward()
        model.optimizer.step()
        total_loss += loss
        if i % int(len(train) / 5 + 1) == 0:
            print('[%d/%d] Loss: %.4f' % (i, len(train), loss.item()))
    print('\tLoss:\t%.3f' % total_loss)

    print('start predicting: ', datetime.datetime.now())
    model.eval()
    hit, mrr = [], []
    for i, batch in enumerate(test):
        targets, scores = forward(model, i, batch.to('cuda'))
        targets -= 1
        sub_scores = scores.topk(20)[1]
        sub_scores = trans_to_cpu(sub_scores).detach().numpy()
        targets = targets.cpu().numpy()  # target & score must both be numpy arrays
        for score, target in zip(sub_scores, targets):
            hit.append(np.isin(target, score))
            if not np.isin(target, score):
            # if len(np.where(score == target)[0]) == 0:
                mrr.append(0)
            else:
                # at_where = np.where(score == target)
                mrr.append(1 / (np.where(score == target)[0][0] + 1))
    hit = np.mean(hit) * 100
    mrr = np.mean(mrr) * 100
    return hit, mrr
