import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
import torch_geometric.nn.conv as conv_g
import torch_geometric.nn.models as model
from torch_geometric.nn import global_add_pool
from torch.nn import ModuleList, Sequential, Linear, BatchNorm1d, ReLU
from utils.rna_lib import global_softmax


def MLP(in_channels: int, out_channels: int) -> torch.nn.Module:
    return Sequential(
        Linear(in_channels, out_channels),
        BatchNorm1d(out_channels),
        ReLU(inplace=True),
        Linear(out_channels, out_channels),
    )


class GINE(nn.Module):
    def __init__(self, in_size, out_size, edge_dim):
        super(GINE, self).__init__()
        self.in_size = in_size
        self.out_size = out_size

        mlp = MLP(in_size, out_size)
        self.GIN = conv_g.GINEConv(mlp, edge_dim=edge_dim)

    def forward(self, x, edge_index, edge_attr=None):
        y = self.GIN(x=x, edge_index=edge_index, edge_attr=edge_attr)
        return y


class BackboneNet(nn.Module):
    def __init__(self, in_size, out_size, edge_size, hide_size_list, n_layers):
        super(BackboneNet, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.hide_size_list = hide_size_list
        self.n_layers = n_layers
        self.size_layer_list = [in_size] + hide_size_list + [out_size]

        self.layers_gin = []

        self.edge_dim = edge_size
        for i in range(n_layers):
            self.layers_gin.append(
                GINE(self.size_layer_list[i], self.size_layer_list[i+1], self.edge_dim)
            )
            self.add_module('gin_block_{}'.format(i), self.layers_gin[i])
            # edge_dim = self.size_layer_list[i]

    def forward(self, x, edge_index, edge_attr=None):
        for layer in self.layers_gin:
            x = layer(x=x, edge_index=edge_index, edge_attr=edge_attr)
            x = F.relu(x)
        return x


class ActorNet(nn.Module):
    def __init__(self, in_size, out_size, hide_size_list, hide_size_fc, n_layers, edge_dim):
        super(ActorNet, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.hide_size_list = hide_size_list
        self.hide_size_fc = hide_size_fc
        self.n_layers = n_layers
        self.size_layer_list = [in_size] + hide_size_list

        self.layers_gin = []

        # edge_dim = 1
        for i in range(n_layers):
            self.layers_gin.append(
                GINE(self.size_layer_list[i], self.size_layer_list[i + 1], edge_dim=edge_dim)
            )
            self.add_module('gin_block_{}'.format(i), self.layers_gin[i])

        self.layers_gin = nn.ModuleList(self.layers_gin)

        self.fc1 = nn.Linear(self.size_layer_list[-1], self.hide_size_fc, bias=False)
        self.fc2 = nn.Linear(self.hide_size_fc, self.out_size, bias=False)

    def forward(self, x, edge_index, batch, edge_attr=None):
        for layer in self.layers_gin:
            x = layer(x=x, edge_index=edge_index, edge_attr=edge_attr)
            x = F.relu(x)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # x = x.view(x.size(0)//max_size, max_size, -1)
        x = torch.flatten(x, 1, 2)
        batch_flatten = batch.unsqueeze(1).repeat(1, 4).view(-1,)
        # action_prob = F.softmax(x, dim=1)
        action_prob = global_softmax(x, batch_flatten)

        return action_prob


class CriticNet(nn.Module):
    def __init__(self, in_size, out_size, hide_size_list, hide_size_fc, n_layers, edge_dim):
        super(CriticNet, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.hide_size_list = hide_size_list
        self.hide_size_fc = hide_size_fc
        self.n_layers = n_layers
        self.size_layer_list = [in_size] + hide_size_list

        self.layers_gin = []

        # edge_dim = 1
        for i in range(n_layers):
            self.layers_gin.append(
                GINE(self.size_layer_list[i], self.size_layer_list[i + 1], edge_dim=edge_dim)
            )
            self.add_module('gin_block_{}'.format(i), self.layers_gin[i])

        self.layers_gin = nn.ModuleList(self.layers_gin)

        self.fc1 = nn.Linear(self.size_layer_list[-1], self.hide_size_fc, bias=False)
        self.fc2 = nn.Linear(self.hide_size_fc, self.out_size, bias=False)

    def forward(self, x, edge_index, batch, edge_attr=None):
        for layer in self.layers_gin:
            x = layer(x=x, edge_index=edge_index, edge_attr=edge_attr)
            x = F.relu(x)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # x = x.view(x.size(0)//max_size, max_size, -1)
        x = global_add_pool(x, batch)
        value = torch.sum(x, dim=1)

        return value


class Position_Actor(nn.Module):
    def __init__(self, in_size, out_size, hide_size_list, hide_size_fc, n_layers, edge_dim):
        super(Position_Actor, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.hide_size_list = hide_size_list
        self.hide_size_fc = hide_size_fc
        self.n_layers = n_layers
        self.size_layer_list = [in_size] + hide_size_list

        self.layers_gin = []

        # edge_dim = 1
        for i in range(n_layers):
            self.layers_gin.append(
                GINE(self.size_layer_list[i], self.size_layer_list[i + 1], edge_dim=edge_dim)
            )
            self.add_module('gin_block_{}'.format(i), self.layers_gin[i])

        self.layers_gin = nn.ModuleList(self.layers_gin)

        self.fc1 = nn.Linear(self.size_layer_list[-1], self.hide_size_fc, bias=False)
        self.fc2 = nn.Linear(self.hide_size_fc, self.out_size, bias=False)

    def forward(self, x, edge_index, batch, edge_attr=None):
        for layer in self.layers_gin:
            x = layer(x=x, edge_index=edge_index, edge_attr=edge_attr)
            x = F.relu(x)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # x = x.view(x.size(0)//max_size, max_size, -1)
        # x = torch.flatten(x, 1, 2)
        # batch_flatten = batch.unsqueeze(1).repeat(1, 4).view(-1,)
        # action_prob = F.softmax(x, dim=1)
        # action_prob = global_softmax(x, batch_flatten)
        action_prob = global_softmax(x, batch)

        return action_prob


class Position_Critic(nn.Module):
    def __int__(self, in_size, out_size, hide_size_list, hide_size_fc, n_layers, edge_dim):
        super(Position_Critic, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.hide_size_list = hide_size_list
        self.hide_size_fc = hide_size_fc
        self.n_layers = n_layers
        self.size_layer_list = [in_size] + hide_size_list

        self.layers_gin = []

        # edge_dim = 1
        for i in range(n_layers):
            self.layers_gin.append(
                GINE(self.size_layer_list[i], self.size_layer_list[i + 1], edge_dim=edge_dim)
            )
            self.add_module('gin_block_{}'.format(i), self.layers_gin[i])

        self.layers_gin = nn.ModuleList(self.layers_gin)

        self.fc1 = nn.Linear(self.size_layer_list[-1], self.hide_size_fc, bias=False)
        self.fc2 = nn.Linear(self.hide_size_fc, self.out_size, bias=False)

    def forward(self, x, edge_index, batch, edge_attr=None):
        for layer in self.layers_gin:
            x = layer(x=x, edge_index=edge_index, edge_attr=edge_attr)
            x = F.relu(x)

        x = global_add_pool(x, batch)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # x = x.view(x.size(0)//max_size, max_size, -1)

        value = torch.sum(x, dim=1)

        return value


class Base_Actor(nn.Module):
    def __int__(self, in_size, out_size, hide_size_fc):
        super(Base_Actor, self).__int__()
        self.in_size = in_size
        self.out_size = out_size
        self.hide_size_fc = hide_size_fc

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        action_prob = F.softmax(x)

        return action_prob


class Base_Critic(nn.Module):
    def __init__(self, in_size, out_size, hide_size_fc):
        super(Base_Critic, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.hide_size_fc = hide_size_fc

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


