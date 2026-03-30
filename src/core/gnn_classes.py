## gnn_classes.py
# imports 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GNNPipeline:

    def __init__(self, preprocess, model_fn, config):
        self.preprocess = preprocess
        self.model_fn = model_fn
        self.config = config

    def fit(self, train_df):
        data = self.preprocess(train_df)
        self.result = self.model_fn(data, self.config)
        return self

    def predict(self, test_df):
        data = self.preprocess(test_df)
        return self.model_fn.predict(data)
    

class SimpleGNN(torch.nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, 32)
        self.conv2 = GCNConv(32, 1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x.squeeze()


class SimpleBinaryGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        return self.head(x).squeeze(-1)