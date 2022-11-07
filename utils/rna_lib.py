import math

import RNA
import random
import matplotlib.pyplot as plt
import numpy as np
import torch
# import Levenshtein
import torch_geometric
from distance import hamming
from tqdm import tqdm
from random import choice
import multiprocessing as mp

#############################################################
# 全局常量
#############################################################
base_color_dict = {'A': 'y', 'U': 'b', 'G': 'r', 'C': 'g'}
base_list = ['A', 'U', 'C', 'G']
onehot_list = [np.array([1, 0, 0, 0]), np.array([0, 1, 0, 0]), np.array([0, 0, 1, 0]), np.array([0, 0, 0, 1])]
base_pair_dict_6 = {'A': ['U'], 'U': ['A', 'G'], 'C': ['G'], 'G': ['U', 'C']}
base_pair_dict_4 = {'A': ['U'], 'U': ['A'], 'C': ['G'], 'G': ['C']}
base_pair_list_6 = [['A', 'U'], ['U', 'A'], ['U', 'G'], ['G', 'U'], ['G', 'C'], ['C', 'G']]
base_pair_list_4 = [['A', 'U'], ['U', 'A'], ['C', 'G'], ['G', 'C']]

#############################################################
# 数据结构和工具
#############################################################

class Stack(object):
    """栈"""
    def __init__(self):
         self.items = []

    def is_empty(self):
        """判断是否为空"""
        return self.items == []

    def push(self, item):
        """加入元素"""
        self.items.append(item)

    def pop(self):
        """弹出元素"""
        return self.items.pop()

    def peek(self):
        """返回栈顶元素"""
        return self.items[len(self.items)-1]

    def size(self):
        """返回栈的大小"""
        return len(self.items)


def dice(max, min=0):
    """
    骰子工具
    :param max: 最大数+1
    :param min: 最小数，默认为0
    :return: 随机采样数
    """
    return random.randrange(min, max, 1)


def delete_zero_row(t):
    """
    删除矩阵里的全零行
    :param t: 原矩阵张量
    :return: 去零后的矩阵
    """
    idx = torch.where(torch.all(t[..., :] == 0, axis=1))
    idx = idx[0]
    all = torch.arange(t.shape[0])
    for i in range(len(idx)):
        all = all[torch.arange(all.size(0)) != idx[i] - i]
    t = torch.index_select(t, 0, all)
    return t


#############################################################
# 结构转换
#############################################################

def  structure_dotB2Edge(dotB):
    """
    将点括号结构转化为边集
    :param dotB: 点括号结构
    :return: 边集，有向图
    """
    l = len(dotB)
    # 初始化
    u = []
    v = []
    for i in range(l - 1):
        u += [i, i + 1]
        v += [i + 1, i]
    str_list = list(dotB)
    stack = Stack()
    for i in range(l):
        if (str_list[i] == '('):
            stack.push(i)
        elif (str_list[i] == ')'):
            last_place = stack.pop()
            u += [i, last_place]
            v += [last_place, i]
    edge_index = torch.tensor(np.array([u, v]))
    return edge_index


def structure_edge2DotB(edge_index):
    """
    将边集转换为点括号结构
    :param edge_index: 边集，tensor
    :return:
    """
    u = edge_index[0, :]
    v = edge_index[1, :]


#############################################################
# 序列转换
#############################################################

def base2Onehot(base):
    """
    碱基字符转onehot编码
    :param base:
    :return:
    """
    onehot = torch.tensor(np.zeros((4,)), dtype=torch.long)
    i = 0
    if (base == "A"):
        i = 0
    elif (base == "U"):
        i = 1
    elif (base == "C"):
        i = 2
    else:
        i = 3
    onehot[i] = 1

    return onehot


def seq_base2Onehot(seq_base, max_size=None):
    """
    将碱基对序列编码为onehot形式
    :param: seq_base: 碱基序列
    :param: max_size: 序列最大长度
    :return: 碱基序列的onehot编码，tensor形式
    """
    # pool = mp.Pool()
    l = len(seq_base)
    # seq_onehot = pool.map(base2Onehot, list(seq_base))
    # pool.close()
    # pool.join()
    seq_onehot = map(base2Onehot, list(seq_base))
    seq_onehot = list(seq_onehot)
    if max_size is not None:
        seq_onehot += [torch.tensor([0, 0, 0, 0])] * (max_size - l)
    seq_onehot = torch.stack(seq_onehot, dim=0)
    return seq_onehot


def onehot2Base(onehot):
    """
    onehot编码转碱基字符
    :param onehot:  onehot编码，tensor
    :return: 碱基字符
    """
    base = ['A', 'U', 'C', 'G']
    onehot = onehot.numpy()[0]
    if np.all(onehot == 0):
        return ''
    i = np.where(onehot == 1)
    i = i[0].item()
    return base[i]


def seq_onehot2Base(seq_onehot):
    """
    onehot编码序列转为碱基字符串
    :param seq_onehot: onehot编码序列，tensor
    :return: 碱基字符串
    """
    # seq_onehot = delete_zero_row(seq_onehot)
    # pool = mp.Pool()
    seq = list(torch.split(seq_onehot, 1, dim=0))
    # seq_base = pool.map(onehot2Base, seq)
    # pool.close()
    # pool.join()
    seq_base = map(onehot2Base, seq)
    seq_base = ''.join(list(seq_base))
    return seq_base


#############################################################
# 能量计算
#############################################################

def get_energy_from_onehot(seq_onehot, dotB):
    """
    由onehot序列计算能量
    :param seq_onehot: onehot序列
    :return: 能量
    """
    seq_base = seq_onehot2Base(seq_onehot)
    energy = RNA.energy_of_struct(seq_base, dotB)
    return energy


def get_energy_from_base(seq_base, dotB):
    """
    由碱基序列序列计算能量
    :param seq_base: 碱基序列
    :return: 能量
    """
    energy = RNA.energy_of_struct(seq_base, dotB)
    return energy


def get_energy_from_graph(graph):
    """
    由graph计算能量
    :param graph: 图，pyg.Data
    :return: 能量
    """
    # energy = get_energy_onehot(graph.x, graph.y['dotB'])
    energy = get_energy_from_base(graph.y['seq_base'], graph.y['dotB'])
    return energy


#############################################################
# 计算距离
#############################################################

def get_distance_from_onehot(seq_onehot ,dotB_Aim):
    """
    由onehot编码序列计算与目标结构的距离
    :param seq_onehot: onehot序列
    :param dotB_Aim: 目标结构的dotB结构
    :return: 距离
    """
    seq_base = seq_onehot2Base(seq_onehot)
    dotB_Real = RNA.fold(seq_base)[0]
    # distance = Levenshtein.distance(dotB_Real, dotB_Aim)
    distance = hamming(dotB_Real, dotB_Aim)
    return distance


def get_distance_from_base(seq_base ,dotB_Aim):
    """
    由onehot编码序列计算与目标结构的距离
    :param seq_onehot: onehot序列
    :param dotB_Aim: 目标结构的dotB结构
    :return: 距离
    """
    dotB_Real = RNA.fold(seq_base)[0]
    # distance = Levenshtein.distance(dotB_Real, dotB_Aim)
    distance = hamming(dotB_Real, dotB_Aim)
    return distance


def get_distance_from_base_norm(seq_base ,dotB_Aim):
    """
    由onehot编码序列计算与目标结构的距离
    :param seq_onehot: onehot序列
    :param dotB_Aim: 目标结构的dotB结构
    :return: 距离
    """
    dotB_Real = RNA.fold(seq_base)[0]
    #distance = Levenshtein.distance(dotB_Real, dotB_Aim) / len(dotB_Aim)
    distance = hamming(dotB_Real, dotB_Aim) / len(dotB_Aim)
    return distance


def get_distance_from_graph(graph):
    """
    由graph计算与目标结构的距离
    :param graph: 图
    :return: 距离
    """
    # distance = get_distance_Levenshtein(graph.x, graph.y['dotB'])
    seq_base = graph.y['seq_base']
    dotB_aim = graph.y['dotB']
    # dotB_real = RNA.fold(seq_base)[0]
    dotB_real = graph.y['real_dotB']
    # distance = Levenshtein.distance(dotB_real, dotB_aim)
    distance = hamming(dotB_real, dotB_aim)
    return distance


def get_distance_from_graph_norm(graph):
    """
    由graph计算与目标结构的距离
    :param graph: 图
    :return: 距离
    """
    # distance = get_distance_Levenshtein(graph.x, graph.y['dotB'])
    seq_base = graph.y['seq_base']
    dotB_aim = graph.y['dotB']
    # d otB_real = RNA.fold(seq_base)[0]
    dotB_real = graph.y['real_dotB']
    # distance = Levenshtein.distance(dotB_real, dotB_aim) / len(dotB_aim)
    distance = hamming(dotB_real, dotB_aim) / len(dotB_aim)
    return distance


#############################################################
# 序列初始化
#############################################################

def random_base():
    """
    随机获得一个碱基
    :return: 碱基字符，onehot编码
    """
    onehot = [0, 0, 0, 0]  # A, U, C, G
    seed = random.randrange(0, 4, 1)
    base = base_list[seed]
    onehot[seed] = 1
    return base, onehot


def random_init_sequence(dotB, max_size=None):
    """
    随机初始化碱基序列
    :param dotB: 点括号结构
    :param max_size: 序列最大长度
    :return: 随机碱基序列，随机onehot序列
    """
    l = len(dotB)
    seq_base = ''
    seq_onehot = []
    for i in range(l):
        base_tmp, onehot_tmp = random_base()
        seq_base += base_tmp
        seq_onehot.append(onehot_tmp)
    seq_onehot += [[0, 0, 0, 0]] * (max_size - l)
    seq_onehot = torch.tensor(seq_onehot)
    return seq_base, seq_onehot


def random_base_pair(action_space):
    """
    随机获得一个碱基对
    :return: 碱基字符对，onehot编码对
    """
    if action_space == 4:
        base_pair_dict = base_pair_dict_4
    elif action_space == 6:
        base_pair_dict = base_pair_dict_6
    onehot = [0, 0, 0, 0]  # A, U, C, G
    seed = random.randrange(0, 4, 1)
    base = base_list[seed]
    onehot[seed] = 1
    base_pair = base_pair_dict[base][dice(len(base_pair_dict[base]), 0)]
    onehot_pair = base2Onehot(base_pair)
    return [base, onehot], [base_pair, onehot_pair]


def random_init_sequence_pair(dotB, edge_index, max_size, action_space):
    """
    配对初始化序列
    """
    if action_space == 4:
        base_pair_dict = base_pair_dict_4
    elif action_space == 6:
        base_pair_dict = base_pair_dict_6

    l = len(dotB)
    seq_base = ''
    for i in range(l):
        base_tmp, _ = random_base()
        seq_base += base_tmp
    seq_onehot = [np.array([0., 0., 0., 0.])] * max_size
    seq_base = list(seq_base)

    for i in range(edge_index.shape[1]):
        place = edge_index[0][i]
        pair_place = edge_index[1][i]
        # if place > pair_place:
        #     continue
        #if np.all(seq_onehot[place] == 0.):
        base_tmp = seq_base[place]
        seq_onehot[place] = base2Onehot(base_tmp).numpy()
        if place < pair_place-1:
            seq_base[pair_place] = base_pair_dict[base_tmp][dice(len(base_pair_dict[base_tmp]), 0)]
            seq_onehot[pair_place] = base2Onehot(seq_base[pair_place]).numpy()
    seq_onehot = torch.tensor(np.array(seq_onehot), dtype=torch.int64)
    seq_base = ''.join(seq_base)
    return seq_base, seq_onehot


def simple_init_sequence(dotB, base_order, max_size=None):
    l = len(dotB)
    seq_base = base_list[base_order] * l
    seq_onehot = [onehot_list[base_order]] * l
    seq_onehot += [[0, 0, 0, 0]] * (max_size - l)
    seq_onehot = torch.tensor(seq_onehot)
    return seq_base, seq_onehot

def simple_init_sequence_pair(dotB, edge_index, base_order, pair_order, max_size, action_space):
    if action_space == 4:
        base_pair_dict = base_pair_dict_4
        base_pair_list = base_pair_list_4
    elif action_space == 6:
        base_pair_dict = base_pair_dict_6
        base_pair_list = base_pair_list_6

    base = base_list[base_order]
    onehot = onehot_list[base_order]
    pair_base = base_pair_list[pair_order]

    l = len(dotB)
    seq_base = ''
    # for i in range(l):
    #     base_tmp, _ = random_base()
    #     seq_base += base_tmp
    seq_base = base * l
    seq_onehot = [np.array(onehot)] * max_size
    seq_base = list(seq_base)

    for i in range(edge_index.shape[1]):
        place = edge_index[0][i]
        pair_place = edge_index[1][i]
        # if place > pair_place:
        #     continue
        #if np.all(seq_onehot[place] == 0.):
        # base_tmp = seq_base[place]
        # seq_onehot[place] = base2Onehot().numpy()
        if place < pair_place-1:
            seq_base[place] = pair_base[0]
            seq_onehot[place] = base2Onehot(pair_base[0]).numpy()
            seq_base[pair_place] = pair_base[1]
            seq_onehot[pair_place] = base2Onehot(pair_base[1]).numpy()
    seq_onehot = torch.tensor(np.array(seq_onehot))
    seq_base = ''.join(seq_base)
    return seq_base, seq_onehot


#############################################################
# 建图
#############################################################

def get_graph(dotB=None, max_size=None, seq_base=None, seq_onehot=None, h_weight=2):
    """
    由初始序列和，加入边的权重
    :param max_size: 最大长度，未设定即默认为本身长度
    :param seq_base: 碱基序列，未设定则默认随机初始化
    :param dotB: 点括号结构，未设定则默认为序列结构折叠
    :param h_weight: 氢键边权重
    :return:
    """
    if max_size is None:
        max_size = len(seq_base if seq_base is not None else dotB)
    if seq_base == None:
        seq_base, seq_onehot = random_init_sequence(dotB, max_size)
    else:
        if seq_onehot is None:
            seq_onehot = seq_base2Onehot(seq_base)

    edge_index = structure_dotB2Edge(dotB)

    edge_attr = get_edge_attr(edge_index, h_weight)

    # adj_aim = torch_geometric.utils.to_dense_adj(edge_index).squeeze(0).numpy()
    adj_aim = edge2Adj(edge_index, max_size)

    y = {'dotB': dotB, 'seq_base': seq_base, 'real_dotB': dotB, 'real_edge_index': edge_index, 'real_attr': edge_attr,
         'adj_aim': adj_aim, 'adj_real': adj_aim}

    graph = torch_geometric.data.Data(x=seq_onehot, y=y, edge_index=edge_index, edge_attr=edge_attr)

    return graph


def get_edge_attr(edge_index, h_weight=2):
    edge_attr = []

    for i in range(edge_index.shape[1]):
        place = edge_index[0][i]
        pair_place = edge_index[1][i]
        if place + 1 < pair_place or place > pair_place + 1:
            edge_attr.append(h_weight)
        else:
            edge_attr.append(1)

    edge_attr = torch.tensor(edge_attr).view(-1, 1).float()

    return edge_attr


def get_edge_attr_from_bp(edge_index, bp, h_weight=2):
    edge_attr = []

    for i in range(edge_index.shape[1]):
        place = edge_index[0][i]
        pair_place = edge_index[1][i]
        if place + 1 < pair_place or place > pair_place + 1:
            edge_attr.append(h_weight)
        else:
            edge_attr.append(1)

    edge_attr = torch.tensor(edge_attr).view(-1, 1).float()

    for u, v , i in zip(edge_index[0], edge_index[1], range(len(edge_index[0]))):
        if u > v + 1 or u < v - 1:
            edge_attr[i][0] = bp[u][v]

    return edge_attr

def get_real_graph(graph):
    x = graph.x.clone()
    edge_index = graph.y['real_edge_index']
    edge_attr = graph.y['real_attr']
    new_graph = torch_geometric.data.Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return new_graph


def graph_padding(graph, max_size=None):
    """
    将图补到固定浐长度
    :param graph: 原图
    :param max_size: 补到长度
    :return: 新图
    """
    l = graph.x.size()[0]
    padding = torch.zeros((max_size-l, graph.x.size()[1]))
    graph.x = torch.cat([graph.x,padding], dim=0)
    return graph


#############################################################
# 检验碱基配对率
#############################################################

def get_pair_ratio(graph, action_space):
    if action_space == 4:
        base_pair_dict = base_pair_dict_4
    elif action_space == 6:
        base_pair_dict = base_pair_dict_6
    pair_cnt = 0
    paired_cnt = 0
    # seq_base = list(seq_onehot2Base(graph.x))
    seq_base = graph.y['seq_base']
    for i in range(graph.edge_index.shape[1]):
        place = graph.edge_index[0][i]
        pair_place = graph.edge_index[1][i]
        if place + 1 < pair_place:
            pair_cnt += 1
            base = seq_base[place]
            pair_base = seq_base[pair_place]
            if pair_base in base_pair_dict[base]:
                paired_cnt += 1
    ratio = float(paired_cnt) / (pair_cnt + 1e-10)
    return ratio


#############################################################
# 动作
#############################################################

def rna_act(action, graph):
    place = action // 4
    base = base_list[action % 4]
    onehot = base2Onehot(base)
    graph.x[place] = onehot
    return graph


def rna_act_pair(action_, graph, forbidden_actions, action_space):
    """
    动作空间 action_space
    :param action: 动作编号
    :param graph: 受动作的图
    :param forbidden_actions: 禁止动作表
    :return: 动作后的图
    """
    # graph = graph_.clone()
    # 获取动作位置和改写的碱基对
    # print('\nact')
    if action_space == 4:
        base_pair_dict = base_pair_dict_4
        base_pair_list = base_pair_list_4
    elif action_space == 6:
        base_pair_dict = base_pair_dict_6
        base_pair_list = base_pair_list_6
    action = action_.cpu()
    # place = action // action_space
    place = torch.div(action, action_space, rounding_mode='trunc')
    base_pair = base_pair_list[action % action_space]
    onehot = base2Onehot(base_pair[0])
    pair_onehot = base2Onehot(base_pair[1])

    seq_base_list = list(graph.y['seq_base'])

    # 改变本位
    graph.x[place] = onehot
    base_old = seq_base_list[place]
    seq_base_list[place] = base_pair[0]
    # 获取邻居
    index = (graph.edge_index[0, :] == place).nonzero()
    pair_places = graph.edge_index[1, index]
    # 寻找氢键并改碱基
    place_h = -1
    pair_base_old = ''
    for pair_place in pair_places:
        if place < pair_place - 1 or place > pair_place + 1:
            place_h = pair_place
            pair_base_old = seq_base_list[pair_place]
            graph.x[pair_place] = pair_onehot
            seq_base_list[pair_place] = base_pair[1]

    # 如果有氢键
    # 需要考虑本位和对位
    if place_h >= 0:
        # print("action: {}".format(action.item()))
        # 删去旧的禁止动作
        if [base_old, pair_base_old] in base_pair_list:
            # 找到以前的位置及禁止动作
            forbidden_change_old = base_pair_list.index([base_old, pair_base_old])
            forbidden_change_old_pair = base_pair_list.index([pair_base_old, base_old])
            # 获取旧禁止动作
            forbidden_action_old = (place * action_space + forbidden_change_old).item()
            forbidden_action_old_pair = (place_h * action_space + forbidden_change_old_pair).item()
            # 删去旧动作
            # print("f_o: {}, p_f_o: {}".format(forbidden_action_old, forbidden_action_old_pair))
            forbidden_actions.remove(forbidden_action_old)
            forbidden_actions.remove(forbidden_action_old_pair)
            # print("remove: {}, {}".format(forbidden_action_old, forbidden_action_old_pair))
        # 加入新的禁止动作
        # if base_pair in base_pair_list:
        forbidden_change = base_pair_list.index(base_pair)
        base_pair_r = base_pair.copy()
        base_pair_r.reverse()
        forbidden_change_pair = base_pair_list.index(base_pair_r)
        forbidden_action = (place * action_space + forbidden_change).item()
        forbidden_action_pair = (place_h * action_space + forbidden_change_pair).item()
        # print("add: {}, {}".format(forbidden_action, forbidden_action_pair))
        # 加入新动作
        # print("f: {}, p_f: {}\n".format(forbidden_action, forbidden_action_pair))
        forbidden_actions.append(forbidden_action)
        forbidden_actions.append(forbidden_action_pair)
    # 如果没有氢键
    # 只考虑本位
    elif place_h < 0:
        forbid_changes_old = [j for j in range(len(base_pair_list)) if base_pair_list[j][0] == base_old]
        forbidden_changes = [j for j in range(len(base_pair_list)) if base_pair_list[j][0] == base_pair[0]]
        # 删除旧禁止动作
        for change_old in forbid_changes_old:
            forbidden_action_old = (place * action_space + change_old).item()
            forbidden_actions.remove(forbidden_action_old)
            # print("remove: {}".format(forbidden_action_old))
        # 加入新禁止动作
        for change in forbidden_changes:
            forbidden_action = (place * action_space + change).item()
            forbidden_actions.append(forbidden_action)
            # print("add: {}".format(forbidden_action))

    graph.y['seq_base'] = ''.join(seq_base_list)
    # forbidden_actions.sort()

    return graph, forbidden_actions

def forbidden_actions_pair(graph, action_space):
    if action_space == 4:
        base_pair_dict = base_pair_dict_4
        base_pair_list = base_pair_list_4
    elif action_space == 6:
        base_pair_dict = base_pair_dict_6
        base_pair_list = base_pair_list_6
    # seq_base = seq_onehot2Base(graph.x)
    seq_base = graph.y['seq_base']
    if seq_base != graph.y['seq_base']:
        print('error')

    forbidden_actions = []
    # 检查每一位
    for i in range(len(graph.y['dotB'])):
        base = seq_base[i]
        # 获取氢键
        place_h = -1
        index = (graph.edge_index[0, :] == i).nonzero()
        pair_places = graph.edge_index[1, index]
        for pair_place in pair_places:
            if i < pair_place - 1 or i > pair_place + 1:
                place_h = pair_place

        # 如果有氢键
        if place_h >= 0:
            base_pair = seq_base[place_h]
            if [base, base_pair] in base_pair_list:
                forbidden_changes = base_pair_list.index([base, base_pair])
                forbidden_actions.append(i * action_space + forbidden_changes)
        # 如果没有氢键
        elif place_h < 0:
            forbidden_changes = [j for j in range(len(base_pair_list)) if base_pair_list[j][0] == base]
            for change in forbidden_changes:
                forbidden_actions.append(i * action_space + change)

    forbidden_actions.sort()

    return forbidden_actions


#############################################################
# 数据校验
#############################################################

def check_data(dataloader):
    with tqdm(total=len(dataloader), desc='Check') as pbar:
        for data in dataloader:
            max_size = int(data.batch.shape[0] / len(data.y))
            dotB_list = [y['dotB'] for y in data.y]
            label_energy_list = [y['energy'] for y in data.y]
            seq_onehot_list = torch.split(data.x, max_size, dim=0)
            seq_base_list = list(map(seq_onehot2Base, seq_onehot_list))
            energy_list = list(map(RNA.energy_of_struct, seq_base_list, dotB_list))
            delta_list = [label_energy - energy for label_energy, energy in zip(label_energy_list, energy_list)]
            num_e = 0
            for delta in delta_list:
                if delta != 0.:
                    print('Data error!')
                    num_e += 1
            pbar.set_postfix({'Error data': num_e})
            pbar.update(1)


#############################################################
# 获取拓扑信息
#############################################################

def get_edge_h(dotB):
    """
    获取氢键边的边集
    :param dotB:
    :return:
    """
    str_list = list(dotB)
    l = len(str_list)
    u = []
    v = []

    # for i in range(l - 1):
    #     u += [i]
    #     v += [i + 1]

    stack = Stack()
    for i in range(l):
        if (str_list[i] == '('):
            stack.push(i)
        elif (str_list[i] == ')'):
            last_place = stack.pop()
            u += [last_place]
            v += [i]
    edge_index = torch.tensor(np.array([u, v]))

    return edge_index


def edge_distance(edge_real, edge_aim):
    """
    拓扑结构距离
    :param edge_real:
    :param edge_aim:
    :param normlize:
    :return:
    """
    real_h = edge_real.t().float()
    aim_h = edge_aim.t().float()

    padding = torch.tensor([0, 0]).view(1,-1).float()

    if real_h.size(0) == 0:
        real_h = torch.cat([padding, real_h], dim=0)

    if aim_h.size(0) == 0:
        aim_h = torch.cat([padding, aim_h], dim=0)
    distance_matrix = torch.cdist(real_h, aim_h, p=2.0)
    distance_r_a = torch.min(distance_matrix, 1)[0]
    distance_a_r = torch.min(distance_matrix, 0)[0]
    distance_tensor = torch.cat([distance_a_r, distance_r_a], dim=0)
    distance = torch.norm(distance_tensor)
    # distance_r_a = torch.sum(distance_r_a)
    # distance_a_r = torch.sum(distance_a_r)
    # distance = distance_r_a.item() + distance_a_r.item()
    return distance


def edge_distance_norm(edge_real, edge_aim, l_real, l_aim):
    real_h = edge_real.t().float()
    aim_h = edge_aim.t().float()

    n_real = real_h.size(0)
    n_aim = aim_h.size(0)

    padding = torch.tensor([0, 0]).view(1, -1).float()

    if n_real == 0:
        real_h = torch.cat([padding, real_h], dim=0)

    if n_aim == 0:
        aim_h = torch.cat([padding, aim_h], dim=0)

    norm_base = math.sqrt(
        n_real * ((l_aim//2)**2 + (l_aim//2-1)**2) + n_aim * ((l_real//2)**2 + (l_real//2-1)**2)
    )

    distance_matrix = torch.cdist(real_h, aim_h, p=2.0)
    distance_r_a = torch.min(distance_matrix, 1)[0]
    distance_a_r = torch.min(distance_matrix, 0)[0]
    distance_tensor = torch.cat([distance_a_r, distance_r_a], dim=0)
    distance = torch.norm(distance_tensor)
    distance = distance / norm_base
    return distance


def get_topology_distance(graph, aim_edge_h):
    """
    获取拓扑结构距离
    :param graph:
    :param aim_edge_h:
    :return:
    """
    # real_dotB = RNA.fold(graph.y['seq_base'])[0]
    real_dotB = graph.y['real_dotB']
    real_edge_h = get_edge_h(real_dotB)
    distance = edge_distance(real_edge_h, aim_edge_h)
    return distance


def get_topology_distance_norm(graph, aim_edge_h):
    """
    获取标准化拓扑距离
    """
    # real_dotB = RNA.fold(graph.y['seq_base'])[0]
    real_dotB = graph.y['real_dotB']
    real_edge_h = get_edge_h(real_dotB)
    l_real = len(real_dotB)
    l_aim = len(graph.y['dotB'])
    distance = edge_distance_norm(real_edge_h, aim_edge_h, l_real, l_aim)
    return distance

def get_dotB_from_graph(graph):
    return RNA.fold(graph.y['seq_base'])[0]

def get_dotB(x):
    pass

def freeze_actions(aim_dotB, real_dotB):
    dotB_same_index = [i for i in range(len(aim_dotB)) if aim_dotB[i] == real_dotB[i]]
    freeze_actions = []
    for index in dotB_same_index:
        for j in range(4):
            freeze_actions.append(4 * index + j)
    return freeze_actions


def freeze_actions_from_graph(graph):
    return freeze_actions(graph.y['dotB'], graph.y['real_dotB'])


def freeze_actions_from_graph_2(graph):
    adj = graph.y['adj_aim'] - graph.y['adj_real']
    freeze_index = [i for i in range(len(graph.y['dotB'])) if np.all(adj[i] == 0)]
    single_index = [i for i in freeze_index if graph.y['dotB'][i] == '.']
    freeze_actions = []
    for index in freeze_index:
        if index not in single_index:
            for j in range(4):
                freeze_actions.append(4 * index + j)
    return freeze_actions


def adjacent_distance(graph, normalize=False):
    l = len(graph.y['dotB'])
    adj = graph.y['adj_aim'] - graph.y['adj_real']
    same_index_list = [i for i in range(len(graph.y['dotB'])) if np.all(adj[i] == 0)]
    distance = l - len(same_index_list)
    if normalize:
        distance /= l
    freeze_actions = []
    for index in same_index_list:
        for j in range(4):
            freeze_actions.append(4 * index + j)
    return distance, freeze_actions


def adjacent_distance_2(graph, normalize=False):
    l = len(graph.y['dotB'])
    adj = graph.y['adj_aim'] - graph.y['adj_real']
    same_index_list = [i for i in range(len(graph.y['dotB'])) if np.all(adj[i] == 0)]
    distance = l - len(same_index_list)
    if normalize:
        distance /= l
    single_index = [i for i in same_index_list if graph.y['dotB'][i] == '.']
    freeze_actions = []
    for index in same_index_list:
        if index not in single_index:
            for j in range(4):
                freeze_actions.append(4 * index + j)
    return distance, freeze_actions


def adjacent_distance_sim(graph, normalize=False):
    l = len(graph.y['dotB'])
    adj = graph.y['adj_aim'] - graph.y['adj_real']
    same_index_list = [i for i in range(len(graph.y['dotB'])) if np.all(adj[i] == 0)]
    distance = l - len(same_index_list)
    if normalize:
        distance /= l
    return distance

def adjacent_distance_only(graph, adj_2):
    l = len(graph.y['dotB'])
    adj = graph.y['adj_aim'] - adj_2
    diff_index_list = [i for i in range(l) if np.all(adj[i] == 0)]
    distance = l - len(diff_index_list)
    return distance


def edge2Adj(edge_index, max_size):
    adj = np.zeros((max_size, max_size))
    for i, j in zip(edge_index[0], edge_index[1]):
        adj[i][j] = 1

    return adj


def simplify_graph(graph):
    out_dict = {}

    out_dict['x'] = graph.x
    out_dict['y'] = graph.y
    out_dict['edge_index'] = graph.edge_index
    out_dict['edge_attr'] = graph.edge_attr

    return out_dict


def recover_graph(graph_dict):
    graph = torch_geometric.data.Data(x=graph_dict['x'], y=graph_dict['y'], edge_index=graph_dict['edge_index'],
                                      edge_attr=graph_dict['edge_attr'])
    return graph


def get_bp(seq, max_size=None):
    fc = RNA.fold_compound(seq)
    (propensity, ensemble_energy) = fc.pf()
    basepair_probs = fc.bpp()
    bp = np.array(basepair_probs)[1:,1:]

    l = len(seq)

    if max_size is not None and max_size > l:
        bp_tmp = np.zeros((max_size, max_size), dtype=float)
        bp_tmp[:l,:l] = bp
        bp = bp_tmp
    return bp


def edge_embedding(edge_index, bp, threshold=0.):

    edge_attr = []

    edge_index_new = edge_index.clone().tolist()

    edge_buffer = []
    for i in range(np.size(bp[0])):
        index = [_index for _index in range(len(edge_index_new[0])) if edge_index_new[0][_index] == i]
        pair_index = []
        for k in index:
            pair_index.append(edge_index_new[1][k])

        # for j in range(i+2, np.size(bp[0])):
        next_list = np.where(bp[i] > threshold)[0]
        for j in next_list:
            if j > i + 1:
                if j not in pair_index:
                    edge_buffer.append([i, j])

    for u, v, i in zip(edge_index_new[0], edge_index_new[1], range(len(edge_index[0]))):
        if u == v + 1 or u == v - 1:
            edge_attr.append([1., 1, 0, 0])
        else:
            edge_attr.append([bp[u][v], 0, 1, 0])

    for erro_edge in edge_buffer:
        i = erro_edge[0]
        j = erro_edge[1]
        edge_index_new[0].append(i)
        edge_index_new[1].append(j)
        edge_attr.append([bp[i][j], 0, 0, 1])
        edge_index_new[0].append(j)
        edge_index_new[1].append(i)
        edge_attr.append([bp[j][i], 0, 0, 1])

    edge_index_new = torch.tensor(np.array(edge_index_new))
    edge_attr = torch.tensor(np.array(edge_attr), dtype=torch.float32)

    return edge_index_new, edge_attr


def global_softmax(x, batch):
    device = x.device
    l_list = []
    for i in range(max(batch) + 1):
        index = torch.where(batch == i)
        l = len(index[0])
        l_list.append(l)
    n = len(l_list)
    l_max = max(l_list)
    end_list = [0]
    for i in range(n):
        end_list.append(end_list[i] + l_list[i])

    out_m = torch.zeros([n, l_max]).to(device)

    for i in range(n):
        tmp_tensor = x[end_list[i]:end_list[i + 1]]
        out_m[i, :l_list[i]] = torch.softmax(tmp_tensor, dim=0)

    return out_m



