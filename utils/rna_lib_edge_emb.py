import RNA
import torch
import torch_geometric

from utils.rna_lib import random_init_sequence, seq_base2Onehot, structure_dotB2Edge, get_bp, edge_embedding, edge2Adj, \
    base_pair_dict_4, base_pair_list_4, base_pair_dict_6, base_pair_list_6, base2Onehot


def get_graph_with_edge_emb(dotB=None, max_size=None, seq_base=None, seq_onehot=None, edge_threshold=0.):
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

    aim_dotB = dotB
    aim_edge_index = structure_dotB2Edge(aim_dotB)

    bp = get_bp(seq_base)

    real_dotB = RNA.fold(seq_base)[0]
    real_edge_index = structure_dotB2Edge(real_dotB)

    edge_index, edge_attr = edge_embedding(aim_edge_index, bp, threshold=edge_threshold)

    # adj_aim = torch_geometric.utils.to_dense_adj(edge_index).squeeze(0).numpy()
    adj_aim = edge2Adj(aim_edge_index, max_size)
    adj_real = edge2Adj(real_edge_index, max_size)

    y = {'dotB': dotB, 'seq_base': seq_base, 'real_dotB': real_dotB, 'aim_edge_index': aim_edge_index,
         'real_edge_index': real_edge_index, 'real_attr': edge_attr,
         'adj_aim': adj_aim, 'adj_real': adj_real, 'bp': bp}

    graph = torch_geometric.data.Data(x=seq_onehot, y=y, edge_index=edge_index, edge_attr=edge_attr)

    return graph


def rna_act_pair_with_edge_emb(action_, graph, forbidden_actions, action_space):
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
    index = (graph.y['aim_edge_index'][0, :] == place).nonzero()
    pair_places = graph.y['aim_edge_index'][1, index]
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


def forbidden_actions_pair_with_edge_emb(graph, action_space):
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
        index = (graph.y['aim_edge_index'][0, :] == i).nonzero()
        pair_places = graph.y['aim_edge_index'][1, index]
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