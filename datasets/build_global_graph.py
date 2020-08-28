from torch_geometric.data import Data
from torch_geometric.utils import from_networkx, to_networkx
import torch
import numpy as np
import pickle
from tqdm import tqdm
import networkx as nx
import torch

# Global graph
with open('datasets/sample/train.txt', 'rb') as f:
    train = pickle.load(f)
# unique items in train
unique_nodes = []
for i, seq in tqdm(enumerate(train[0])):
    for j, node in enumerate(seq):
        if node not in unique_nodes:
            unique_nodes.append(node)
# 所有unique item在所有session中曾出現的前後2個item建graph
# 從所有session中先記錄每個item前後2個item有誰
graph_node = {k: [] for k in unique_nodes}
epsilon = 2
for i, seq in tqdm(enumerate(train[0])):
    if len(seq) > 0:
        for j, node in enumerate(seq):
            # 有考慮self-loop
            if j+epsilon < len(seq)-1:
                graph_node[node]+=seq[j:j+epsilon]
            else:
                graph_node[node]+=seq[j:len(seq)]

edge_lists = []
for i, k in tqdm(enumerate(graph_node.keys())):
    # 對graph node每個key取set
    graph_node[k] = list(set(graph_node[k]))
    # 根據graph node建出所有edge list
    if len(graph_node[k]) > 0:
        edge_lists+=[[k, j] for j in graph_node[k]]
edge_lists = [list(x) for x in set(tuple(x) for x in edge_lists)]  # uniques lists in list
# edge_lists += [x[::-1] for x in edge_lists]  # 為了無向圖, 所有edge list都要是雙向
# edge_lists = [list(x) for x in set(tuple(x) for x in edge_lists)]
# edge_index = torch.tensor(edge_lists, dtype=torch.long)
# data2 = Data(edge_index=edge_index.t().contiguous())

# edge_lists = [x for x in edge_lists if not x[0]==x[1]]  # remove self loop
l = np.array(edge_lists).transpose()
r = np.array([l[1, :], l[0, :]])  # 建無向圖
# x = [[l[0]] for l in edge_lists]  # get node id from edge index
# x = torch.tensor(x, dtype=torch.float)  # fixme 應該只有304個
edge_index = torch.from_numpy(r).long()

# todo 全部有 309nodes, 但只有304個有edge, 要reindex?
# 建無向圖
x = torch.tensor(unique_nodes).long()  # edge index內就是node id
# x = torch.arange(l[0].max()+1).long()
data3 = Data(x, edge_index)  # 假設x是從node id 0 開始遞增

# 建nx的無向圖
G = nx.Graph()
G.add_edges_from(edge_lists)
# networkx無向圖轉Data
data = from_networkx(G)
data.x = torch.tensor(list(G.nodes)).unsqueeze(-1)  # 不一定對的上, 因為node id可能早被重新編碼

# 很費時, 建好graph就直接存檔(networkx(用Data直接轉) + Data)
torch.save(data, 'global_graph.pt')

# 用link prediction train

# 如果是dynamic版的話:
# 用Data()建T張獨立的graph, 不要把這graph放進dataloader => 各自train GAT到T emb
# 根據session graph的時間取第T張graph的item emb
