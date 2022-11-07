import math
import random
from itertools import product
# from distance import hamming
import RNA
import gym
import torch
from tqdm import tqdm
from torch import nn
from torch.nn import functional as F
import numpy as np
import os
from torch_geometric.data import DataLoader as DataLoader_g

from embedding.embedding import emb_full
from utils.rna_lib import random_init_sequence, random_init_sequence_pair, graph_padding, forbidden_actions_pair, \
    get_distance_from_graph_norm, get_edge_h, get_topology_distance, rna_act_pair, get_energy_from_graph, \
    get_distance_from_graph, get_topology_distance_norm, structure_dotB2Edge, get_graph, get_dotB_from_graph, \
    get_edge_attr, simple_init_sequence, simple_init_sequence_pair, seq_base2Onehot, freeze_actions_from_graph, \
    freeze_actions_from_graph_2, adjacent_distance, base_pair_list_4, base_pair_list_6, base_list, edge2Adj, \
    adjacent_distance_sim, adjacent_distance_only, get_bp, get_edge_attr_from_bp, edge_embedding
from collections import namedtuple
import torch_geometric
from utils.config_ppo import device
import multiprocessing as mp
from functools import partial
import pathos.multiprocessing as pathos_mp

from utils.rna_lib_edge_emb import get_graph_with_edge_emb, forbidden_actions_pair_with_edge_emb, \
    rna_act_pair_with_edge_emb
from utils.rna_split import RNATree


Transition = namedtuple('Transition', ['state', 'action', 'a_log_prob', 'reward', 'next_state'])


def sigmoid(input):
    return 1 / (1 + math.exp(-input))


class RNA_Env_Edge_Emb(gym.Env):
    def __init__(self, dotB, id=None, init_base_order=None, init_pair_order=None, action_space=4, h_weight=2,
                 dist_norm=False, mutation_threshold=5, sigmoid_limit=True, max_size=None, base_threshold=2,
                 reward_type="bp", edge_threshold=0.1):
        super(RNA_Env_Edge_Emb, self).__init__()
        self.dotB = dotB
        self.tree = RNATree(dotB)
        self.tree.external_loop_create()
        self.tree_level = len(self.tree.branch_log.items()) - 1
        self.branchs = [[]] * self.tree_level
        self.length = len(dotB)
        self.init_base_order = init_base_order
        self.init_pair_order = init_pair_order
        self.action_space = action_space
        self.dist_norm = dist_norm
        self.sigmoid_limit = sigmoid_limit
        self.h_weight = h_weight
        self.id = id
        self.edge_threshold = edge_threshold
        if max_size is None:
            self.max_size = self.length
        else:
            self.max_size = max_size

        self.dist_show = self.length

        self.base_threshold = base_threshold

        self.mutation_threshold = mutation_threshold

        self.edge_index = structure_dotB2Edge(self.dotB)
        if self.init_base_order is not None:
            self.init_seq_base, self.init_seq_onehot = \
                simple_init_sequence_pair(self.dotB, self.edge_index, self.init_base_order, self.init_pair_order,
                                          self.max_size, self.action_space)
        else:
            self.init_seq_base, self.init_seq_onehot = simple_init_sequence_pair(self.dotB, self.edge_index, 0, 0,
                                          self.max_size, self.action_space)

        # self.level = self.tree_level - 1

        self.aim_scope = None
        self.aim_graph = None
        self.aim_dotB = None
        self.aim_level = -1
        self.settle_nodes = []
        self.aim_branchs = []
        self.aim_branch = None
        self.settled_place = []
        self.max_level = self.tree_level - 1

        self.last_energy = 1e6
        self.last_distance = self.length
        self.last_score = 0
        self.forbidden_actions = []
        # self.aim_edge_h = []
        self.freeze_actions = []
        # self.limit_actions = []
        self.base_pair_list = base_pair_list_4 if action_space == 4 else base_pair_list_6
        self.settle_seq = list(self.init_seq_base)
        self.reward_type = reward_type

        self.done = 0

    class settle_branch:
        def __init__(self, scope, seq_base, seq_onehot):
            self.scope = scope
            self.seq_base = seq_base
            self.seq_onehot = seq_onehot

    def merge_sub_branch_to_loop(self, sub_branch_seqs, branch):
        loop = branch.get_loop()
        if type(loop[0]) == int:
            start = loop[0]
        else:
            start = loop[0].get_start()
        if type(loop[-1]) == int:
            end = loop[-1]
        else:
            end = loop[-1].get_end()

        self.aim_dotB = self.dotB[start:end+1] if end < self.length else self.dotB[start:]
        self.aim_graph = get_graph(self.aim_dotB, self.max_size)
        self.aim_level = branch.get_level()

    def reset(self):
        self.log_seq = []
        self.log_forbid = []
        if self.init_base_order is None:
            self.init_seq_base, self.init_seq_onehot = \
                random_init_sequence_pair(self.dotB, self.edge_index, self.max_size, self.action_space)
            self.settle_seq = list(self.init_seq_base)
        self.tree.reset()
        self.settled_place = []
        self.aim_level = self.tree_level - 1
        self.aim_branchs = self.tree.branch_log[str(self.aim_level)].copy()
        self.last_distance = 0
        self.done = 0

        self.switch_aim_branch()

        slack_skip = ((self.aim_level > 0 and len(self.tree.external_loop) == 1) or
                      (self.aim_level > -1 and len(self.tree.external_loop) > 1)) \
                     and self.last_distance <= self.base_threshold

        # while self.last_distance == 0 and not self.done:
        while (slack_skip or self.last_distance == 0) and not self.done:
            self.switch_aim_branch()
            slack_skip = ((self.aim_level > 0 and len(self.tree.external_loop) == 1) or
                          (self.aim_level > -1 and len(self.tree.external_loop) > 1)) \
                         and self.last_distance <= self.base_threshold

        # action limit
        if self.last_distance > 0:
            self.freeze_actions = self._get_freeze_actions(self.last_distance)
            self.forbidden_actions = forbidden_actions_pair_with_edge_emb(self.aim_graph, self.action_space)

        self.log_seq.append(self.aim_graph.y['seq_base'])
        self.log_forbid.append(self.forbidden_actions)

        # bp = get_bp(self.aim_graph.y['seq_base'], self.max_size)
        bp = self.aim_graph.y['bp']

        # aim_attr = get_edge_attr_from_bp(self.aim_graph.edge_index, bp)
        # real_attr = get_edge_attr_from_bp(self.aim_graph.y['real_edge_index'], bp)
        # self.aim_graph.edge_attr = aim_attr
        # self.aim_graph.y['real_attr'] = real_attr

        if self.reward_type == "bp":
            score = np.dot(self.aim_graph.y['adj_aim'], bp)
            self.last_score = score.sum().sum()

        return self.aim_graph.clone()

    def check_settle(self, branch):
        settle_place = []
        branch_scope = branch.scope()
        # seq_base = self.init_seq_base[branch_scope[0]: branch_scope[1]+1]
        seq_base = self.settle_seq[branch_scope[0]: branch_scope[1]+1]
        seq_base = list(seq_base)
        for node in branch.get_loop():
            if type(node) != int:
                scope = node.scope()
                settle_place += list(range(scope[0], scope[1]+1))
                # seq_base[scope[0]: scope[1]+1] = list(node.settled_seq)
        seq_base = ''.join(seq_base)
        return settle_place, seq_base

    def switch_aim_branch(self):
        # switch branch
        # self.settle_list = []
        if len(self.aim_branchs) > 0:
            # self.aim_branchs = self.tree.branch_log[str(self)]
            self.aim_branch = self.aim_branchs.pop()

        else:
            if self.aim_level == -1:
                if len(self.tree.external_loop) > 1:
                    self.aim_branchs = self.tree.branch_log[str(self.aim_level)].copy()
                    random.shuffle(self.aim_branchs)
                    self.aim_branch = self.aim_branchs.pop()
                    self.aim_level -= 1
                else:
                    self.done = True
                    self.aim_branch = None
                    return None
            elif self.aim_level >= 0 and self.tree.external_loop:
                self.aim_level -= 1
                self.aim_branchs = self.tree.branch_log[str(self.aim_level)].copy()
                self.aim_branch = self.aim_branchs.pop()
            else:
                self.done = 1
                self.aim_branch = None
                return None

        self.aim_scope = self.aim_branch.scope()
        self.aim_dotB = self.dotB[self.aim_scope[0]:] if self.aim_scope[1] >= self.length - 1 \
            else self.dotB[self.aim_scope[0]:self.aim_scope[1] + 1]

        self.settled_place, seq_base = self.check_settle(self.aim_branch)
        seq_onehot = seq_base2Onehot(seq_base, max_size=self.max_size)
        self.aim_graph = get_graph_with_edge_emb(self.aim_dotB, self.max_size, seq_base, seq_onehot,
                                                 self.edge_threshold)

        # self.aim_edge_h = get_edge_h(self.aim_dotB)

        # sequence
        # self.aim_graph.x = seq_onehot
        # self.aim_graph.y['seq_base'] = seq_base
        # edge
        # self.aim_graph.y['real_dotB'] = get_dotB_from_graph(self.aim_graph)
        # self.aim_graph.y['real_edge_index'] = structure_dotB2Edge(self.aim_graph.y['real_dotB'])

        # self.aim_graph.y['real_attr'] = get_edge_attr(self.aim_graph.y['real_edge_index'], self.h_weight)
        # adjacent matrix
        # self.aim_graph.y['adj_real'] = edge2Adj(self.aim_graph.y['real_edge_index'], self.max_size)

        # distance
        self.last_distance = self._adjacent_distance(self.aim_graph, self.dist_norm)
        # self.dist_show = self.last_distance + self.length - self.aim_scope[1] + self.aim_scope[0]
        self.dist_show = self.last_distance
        if self.last_distance == 0:
            # self.aim_branch.settle(self.aim_graph.y['seq_base'])
            self.settle()

        # action limit
        # else:
        self.freeze_actions = self._get_freeze_actions(self.last_distance)
        self.forbidden_actions = forbidden_actions_pair_with_edge_emb(self.aim_graph, self.action_space)

        self.aim_l_seq = len(self.aim_graph.y['seq_base'])
        self.aim_l_dotB = len(self.aim_graph.y['dotB'])

        return self.aim_graph.clone()

    def settle(self):
        seq = list(self.aim_graph.y['seq_base'])
        for i, j in zip(range(self.aim_scope[0], self.aim_scope[1]+1), range(len(seq))):
            self.settle_seq[i] = seq[j]
            if i not in self.settled_place:
                self.settled_place.append(i)

    def _get_freeze_actions(self, distance, freeze_actions=None):
        ratio = random.random() / 2 + 0.5
        # ratio = 1
        if self.dist_norm:
            sig_threshold = sigmoid(distance)
        else:
            sig_threshold = sigmoid(distance / len(self.aim_dotB))

        # if self.sigmoid_limit and ratio > sig_threshold:
        #     if freeze_actions is None:
        #         freeze_actions = self.freeze_actions_from_graph_split()
        # else:
        #     freeze_actions = []

        if self.sigmoid_limit :
            single_free = (ratio > sig_threshold)
            if freeze_actions is None:
                freeze_actions = self.freeze_actions_from_graph_split(single_free)
        else:
            freeze_actions = []
        return freeze_actions

    def freeze_actions_from_graph_split(self, single_free=False):
        l = len(self.aim_graph.y['dotB'])
        # loop_nodes = self.aim_branch.get_loop()
        # sub_branchs = [node for node in loop_nodes if type(node) != int]
        # settle_scope_list = [branch.scope() for branch in sub_branchs]
        settle_nodes = [place - self.aim_scope[0] for place in self.settled_place]
        # for scope in settle_scope_list:
        #     for i in range(scope[0], scope[1]):
        #         settle_nodes.append(i - self.aim_branch.get_start())

        diff_adj = self.aim_graph.y['adj_aim'] - self.aim_graph.y['adj_real']
        same_index = [i for i in range(l) if np.all(diff_adj[1] == 0)]

        if single_free:
            single_index = [i for i in range(l) if self.aim_graph.y['dotB'][i] == '.']
                            # and i not in settle_nodes]
        else:
            single_index = []

        freeze_actions = []
        for index in same_index:
            if index not in single_index:
                for j in range(4):
                    freeze_actions.append(4 * index + j)

        return freeze_actions

    def step(self, action):
        self.aim_graph, self.forbidden_actions = rna_act_pair_with_edge_emb(action, self.aim_graph,
                                                              self.forbidden_actions, self.action_space)
        real_dotB = get_dotB_from_graph(self.aim_graph)
        real_edge_index = structure_dotB2Edge(real_dotB)
        # real_attr = get_edge_attr(real_edge_index, self.h_weight)

        self.aim_graph.y['real_dotB'] = real_dotB
        self.aim_graph.y['real_edge_index'] = real_edge_index
        # self.aim_graph.y['real_attr'] = real_attr

        self.aim_graph.y['adj_real'] = edge2Adj(real_edge_index, self.max_size)

        # local improvement
        self.aim_graph, self.forbidden_actions = self._local_improve_for_graph(self.aim_graph, self.forbidden_actions)

        distance = self._adjacent_distance(self.aim_graph, self.dist_norm)

        self.freeze_actions = self.freeze_actions_from_graph_split()

        self.last_distance = distance

        self.aim_graph.y['bp'] = get_bp(self.aim_graph.y['seq_base'], self.max_size)

        if self.reward_type == "bp":
            # bp = get_bp(self.aim_graph.y['seq_base'], self.max_size)
            score = np.dot(self.aim_graph.y['adj_aim'], self.aim_graph.y['bp'])
            score = score.sum().sum()
            reward = score - self.last_score

        else:
            reward = self.last_distance - distance

        # self.dist_show = self.last_distance + self.length - len(self.settled_place)
        self.dist_show = self.last_distance

        is_terminal = 0

        slack_skip = ((self.aim_level > 0 and len(self.tree.external_loop) == 1) or
                      (self.aim_level > -1 and len(self.tree.external_loop) > 1)) \
                     and self.last_distance <= self.base_threshold

        if distance == 0 or slack_skip:
            self.settle()
            if not self.done:
                while (self.last_distance == 0 or slack_skip) and not self.done:
                    self.switch_aim_branch()
                    slack_skip = ((self.aim_level > 0 and len(self.tree.external_loop) == 1) or
                                  (self.aim_level > -1 and len(self.tree.external_loop) > 1)) \
                                 and self.last_distance <= self.base_threshold
                if self.done:
                    is_terminal = 1
                    reward += 1000
            else:
                is_terminal = 1

        self.log_seq.append(self.aim_graph.y['seq_base'])
        self.log_forbid.append(self.forbidden_actions)

        # bp = get_bp(self.aim_graph.y['seq_base'], self.max_size)
        # aim_attr = get_edge_attr_from_bp(self.aim_graph.edge_index, bp)
        # real_attr = get_edge_attr_from_bp(self.aim_graph.y['real_edge_index'], bp)
        # self.aim_graph.edge_attr = aim_attr
        # self.aim_graph.y['real_attr'] = real_attr

        edge_index, edge_attr = edge_embedding(self.aim_graph.y['aim_edge_index'], self.aim_graph.y['bp'],
                                               self.edge_threshold)

        self.aim_graph.edge_index = edge_index
        self.aim_graph.edge_attr = edge_attr

        if self.reward_type == "bp":
            score = np.dot(self.aim_graph.y['adj_aim'], self.aim_graph.y['bp'])
            score = score.sum().sum()
            self.last_score = score

        return self.aim_graph, reward, is_terminal

    def _adjacent_distance(self, graph, normalize=False):
        l = self.max_size
        adj = graph.y['adj_aim'] - graph.y['adj_real']
        same_index_list = [i for i in range(self.max_size) if np.all(adj[i] == 0)]
        distance = l - len(same_index_list)
        if normalize:
            distance /= l
        return distance

    def _local_improve_for_graph(self, graph, forbid_actions):
        dist = adjacent_distance_sim(graph, normalize=False)
        if dist <= self.mutation_threshold and dist > 0:
            dist, graph, forbid_actions = self._local_improve(graph, dist, forbid_actions)
            # forbid_actions = forbidden_actions_pair(graph, self.action_space)
        return graph, forbid_actions

    def _get_mutatuion(self, seq_list, mutations_s, mutations_p, index_s, index_p):
        for site_s, mutation_s in zip(index_s, mutations_s):
            seq_list[site_s] = base_list[int(mutation_s)]
        for site_p, mutation_p in zip(index_p, mutations_p):
            seq_list[site_p[0]] = self.base_pair_list[int(mutation_p)][0]
            seq_list[site_p[1]] = self.base_pair_list[int(mutation_p)][1]
        return seq_list

    def _local_improve(self, graph, dist, forbidden_actions):
        # print('\nimprove')
        max_size = graph.x.shape[0]
        dotB = graph.y['dotB']
        real_dotB = graph.y['real_dotB']
        l = len(dotB)

        adj = graph.y['adj_aim'] - graph.y['adj_real']
        diff_index = [i for i in range(len(graph.y['dotB'])) if not np.all(adj[i] == 0)]
        diff_single_index = [i for i in diff_index if dotB[i] == '.']
        diff_pair_index_s = [i for i in diff_index if dotB[i] == '(']
        diff_pair_index = []
        for i in diff_pair_index_s:
            indexs = [j for j in range(len(graph.y['aim_edge_index'][0])) if graph.y['aim_edge_index'][0][j] == i]
            for index in indexs:
                if i < graph.y['aim_edge_index'][1][index] - 1:
                    diff_pair_index.append([i, int(graph.y['aim_edge_index'][1][index])])

        # dotB_diff_index = [i for i in range(l) if dotB[i] != real_dotB[i]]
        # improved_dist = [dist]
        improved_dist = []
        old_seq_base = graph.y['seq_base']
        seq_base_list = list(old_seq_base)
        # improved_seq_list = [seq_base_list.copy()]
        # improved_dotB_list = [real_dotB]
        improved_seq_list = []
        improved_dotB_list = []
        # for index in dotB_diff_index:
        #    change_bases = [base for base in base_list if base != self.seq_base_list[index]]
        #    for change_base in change_bases:

        bases_order_str = [str(i) for i in range(self.action_space)]
        bases_order_str = ''.join(bases_order_str)

        for mutations_s in product("0123", repeat=len(diff_single_index)):
            for mutations_p in product(bases_order_str, repeat=len(diff_pair_index)):
                tmp_seq_list = seq_base_list.copy()
                tmp_seq_list = self._get_mutatuion(tmp_seq_list, mutations_s, mutations_p, diff_single_index,
                                                      diff_pair_index)
                improved_seq_list.append(tmp_seq_list)
                tmp_seq = ''.join(tmp_seq_list)
                changed_dotB = RNA.fold(tmp_seq)[0]
                improved_dotB_list.append(changed_dotB)
                changed_edge = structure_dotB2Edge(changed_dotB)
                changed_adj = edge2Adj(changed_edge, self.max_size)
                changed_dist = adjacent_distance_only(graph, changed_adj)
                improved_dist.append(changed_dist)
                if changed_dist == 0:
                    graph.x = seq_base2Onehot(tmp_seq, max_size)
                    graph.y['seq_base'] = tmp_seq
                    graph.y['real_dotB'] = changed_dotB
                    graph.y['real_edge_index'] = structure_dotB2Edge(changed_dotB)
                    # graph.y['real_attr'] = get_edge_attr(graph.y['real_edge_index'], self.h_weight)
                    # graph.y['adj_real'] = edge2Adj(edge_index=graph.y['real_edge_index'], max_size=self.max_size)
                    graph.y['adj_real'] = changed_adj
                    return 0, graph, forbidden_actions

        min_dist = min(improved_dist)
        min_places = [i for i in range(len(improved_dist)) if min_dist == improved_dist[i]]
        min_index = random.choice(min_places)
        if min_index > 0:
            min_seq_list = improved_seq_list[min_index]
            min_dotB = improved_dotB_list[min_index]

            min_seq = ''.join(min_seq_list)
            graph.x = seq_base2Onehot(min_seq, max_size)
            graph.y['seq_base'] = min_seq
            graph.y['real_dotB'] = min_dotB
            graph.y['real_edge_index'] = structure_dotB2Edge(min_dotB)
            # graph.y['real_attr'] = get_edge_attr(graph.y['real_edge_index'], self.h_weight)
            graph.y['adj_real'] = edge2Adj(edge_index=graph.y['real_edge_index'], max_size=self.max_size)

            improved_place = [i for i in range(len(min_seq)) if min_seq[i] != old_seq_base[i]]

            for place in improved_place:
                pair_index = (graph.y['aim_edge_index'][0, :] == place).nonzero()
                pair_places = graph.y['aim_edge_index'][1, pair_index]
                h_pair_place = [pair_place for pair_place in pair_places
                                if (pair_place > place + 1 or pair_place < place - 1)]

                if len(h_pair_place) == 0:
                    forbid_changes_old = [j for j in range(len(self.base_pair_list))
                                          if base_list[j][0] == old_seq_base[place]]
                    forbid_changes_new = [j for j in range(len(self.base_pair_list))
                                          if base_list[j][0] == min_seq[place]]
                    for change_old in forbid_changes_old:
                        forbidden_action_old = (place * self.action_space + change_old)
                        forbidden_actions.remove(forbidden_action_old)
                        # print("remove: {}".format(forbidden_action_old))
                    # 加入新禁止动作
                    for change in forbid_changes_new:
                        forbidden_action = (place * self.action_space + change)
                        forbidden_actions.append(forbidden_action)
                        # print("add: {}".format(forbidden_action))

                else:
                    # if place < h_pair_place[0]:
                    base_old = old_seq_base[place]
                    pair_base_old = old_seq_base[h_pair_place[0]]
                    base_pair_new = [min_seq[place], min_seq[h_pair_place[0]]]
                    if [base_old, pair_base_old] in self.base_pair_list:
                        # 找到以前的位置及禁止动作
                        forbidden_change_old = self.base_pair_list.index([base_old, pair_base_old])
                        forbidden_change_old_pair = self.base_pair_list.index([pair_base_old, base_old])
                        # 获取旧禁止动作
                        forbidden_action_old = (place * self.action_space + forbidden_change_old)
                        forbidden_action_old_pair = (
                                    h_pair_place[0] * self.action_space + forbidden_change_old_pair).item()
                        # 删去旧动作
                        # print("f_o: {}, p_f_o: {}".format(forbidden_action_old, forbidden_action_old_pair))
                        try:
                            forbidden_actions.remove(forbidden_action_old)
                            forbidden_actions.remove(forbidden_action_old_pair)
                        except:
                            pass
                        # print("remove: {}, {}".format(forbidden_action_old, forbidden_action_old_pair))
                    # 加入新的禁止动作
                    # if base_pair in base_pair_list:
                    if base_pair_new in self.base_pair_list:
                        forbidden_change = self.base_pair_list.index(base_pair_new)
                        base_pair_r = base_pair_new.copy()
                        base_pair_r.reverse()
                        forbidden_change_pair = self.base_pair_list.index(base_pair_r)
                        forbidden_action = (place * self.action_space + forbidden_change)
                        forbidden_action_pair = (h_pair_place[0] * self.action_space + forbidden_change_pair).item()
                        # 加入新动作
                        # print("f: {}, p_f: {}\n".format(forbidden_action, forbidden_action_pair))
                        if forbidden_action not in forbidden_actions:
                            forbidden_actions.append(forbidden_action)
                        if forbidden_action_pair not in forbidden_actions:
                            forbidden_actions.append(forbidden_action_pair)
                        # print("add: {}, {}".format(forbidden_action, forbidden_action_pair))

        return min_dist, graph, forbidden_actions

    def get_emb(self, emb_dict, mer):
        seq_emb = emb_full(self.aim_graph.y['seq_base'], emb_dict, mer, max_size=self.max_size)
        graph = self.aim_graph.clone()
        graph.x = seq_emb
        return graph


























