# -*- coding: utf-8 -*-

import pickle
import torch
import collections
from torch_geometric.data import InMemoryDataset, Data


class MultiSessionsGraph(InMemoryDataset):
    """Every session is a graph."""

    def __init__(self, root, phrase, transform=None, pre_transform=None):
        assert phrase in ['train', 'test']
        self.phrase = phrase
        super(MultiSessionsGraph, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [self.phrase + '.txt']

    @property
    def processed_file_names(self):
        return [self.phrase + '.pt']

    def download(self):
        pass

    def process(self):
        # data = pickle.load(open(self.raw_dir + '/' + self.raw_file_names[0], 'rb'))
        data = pickle.load(open(self.root + '/' + self.raw_file_names[0], 'rb'))
        data_list = []
        # 找最長的 sequence
        us_lens = [len(upois) for upois in data[0]]
        len_max = max(us_lens)
        padding_item = [0]

        for sequence, y in zip(data[0], data[1]):
            # sequence = [1, 3, 2, 2, 1, 3, 4]
            i = 0
            nodes = {}  # dict{15: 0, 16: 1, 18: 2, ...}
            senders = []
            x = []
            for node in sequence:
                if node not in nodes:
                    nodes[node] = i  # node第一次出現的位置
                    x.append([node])  # 同一sequence下的unique node
                    i += 1
                senders.append(nodes[node])  # 為了建 out A
            receivers = senders[:]

            if len(senders) != 1:
                del senders[-1]  # the last item is a receiver
                del receivers[0]  # the first item is a sender

            # undirected
            # senders, receivers = senders + receivers, receivers + senders

            # padding sequence
            sequence_len = torch.tensor([len(sequence)], dtype=torch.long)  # 原始sequence長度
            mask = [1] * len(sequence) + padding_item * (len_max - len(sequence))
            sequence = sequence + padding_item * (len_max - len(sequence))

            pair = {}  # 紀錄item間的edge
            sur_senders = senders[:]
            sur_receivers = receivers[:]
            i = 0
            for sender, receiver in zip(sur_senders, sur_receivers):
                if str(sender) + '-' + str(receiver) in pair:
                    pair[str(sender) + '-' + str(receiver)] += 1  # 有考慮一個pattern出現幾次
                    del senders[i]  # 確保是unique
                    del receivers[i]
                else:
                    pair[str(sender) + '-' + str(receiver)] = 1
                    i += 1

            count = collections.Counter(senders)  # list內element的數量
            # 算out的edge數並標準化
            out_degree_inv = torch.tensor([1 / count[i] for i in senders], dtype=torch.float)

            count = collections.Counter(receivers)
            in_degree_inv = torch.tensor([1 / count[i] for i in receivers], dtype=torch.float)
            # 計算node間的pattern次數, edge的初始weight
            edge_attr = torch.tensor([pair[str(senders[i]) + '-' + str(receivers[i])] for i in range(len(senders))],
                                     dtype=torch.float)

            edge_index = torch.tensor([senders, receivers], dtype=torch.long)
            x = torch.tensor(x, dtype=torch.long)  # 保留原始的node id
            y = torch.tensor([y], dtype=torch.long)
            # custom args
            sequence = torch.tensor(sequence, dtype=torch.long)
            mask = torch.tensor(mask, dtype=torch.long)
            # 相當於networkx的graph
            session_graph = Data(x=x, y=y,
                                 edge_index=edge_index, edge_attr=edge_attr,
                                 sequence=sequence, sequence_len=sequence_len,
                                 sequence_mask=mask,
                                 out_degree_inv=out_degree_inv, in_degree_inv=in_degree_inv)
            data_list.append(session_graph)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
