import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.data
from tqdm import tqdm
from utils.rna_lib import get_distance_from_graph, get_energy_from_graph, forbidden_actions_pair, get_real_graph, \
    recover_graph
from collections import namedtuple
from utils.config_gin import device
from networks.RD_GIN import BackboneNet, Position_Actor, Position_Critic, Base_Actor, Base_Critic
from torch.autograd import Variable
from torch import no_grad, clamp
import os
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from itertools import chain
import multiprocessing as mp
# import pathos.multiprocessing as pathos_mp
from torch.distributions import Categorical
from functools import partial

Transition = namedtuple('Transition', ['state', 'action', 'a_log_prob', 'reward', 'next_state', 'done'])


def get_element_index(ob_list, index):
    """
    从list中选择多个项
    :param ob_list: 整个list
    :param word:
    :return:
    """
    return [ob_list[i] for i in index]


def get_action_sample_forbid(pro_list_, length, forbidden_actions, freeze_actions, num_change):
    """
    sample with probability
    :param pro_list:
    :param length:
    :return:
    """
    pro_list = pro_list_.clone().view(-1, )[0:length * num_change]
    for action in forbidden_actions:
        pro_list[action] = 0
    for action in freeze_actions:
        pro_list[action] = 0
    try:
        action = torch.multinomial(pro_list, 1).item()
    except: # 若概率全0
        pro_list = torch.ones(pro_list.shape, dtype=pro_list.dtype).scatter_(0, torch.tensor(forbidden_actions), 0) / pro_list.shape[0]
        action = torch.multinomial(pro_list, 1).item()

    return action


def get_action_max_forbid(pro_list_, length, forbidden_actions, freeze_actions, num_change):
    """
    choose the action with the largest probability
    :param pro_list:
    :param length:
    :return:
    """
    pro_list = pro_list_.clone().view(-1, )[0:length * num_change]
    for action in forbidden_actions:
        pro_list[action] = 0
    for action in freeze_actions:
        pro_list[action] = 0
    action = torch.argmax(pro_list)
    return action.item()


def get_place_sample(pro_list_, freeze_place_list):
    pro_list = pro_list_.clone().view(-1, )
    # for action in forbidden_actions:
    #     pro_list[action] = 0
    for forbidden_place in freeze_place_list:
        pro_list[forbidden_place] = 0
    try:
        place = torch.multinomial(pro_list, 1).item()
    except:  # 若概率全0
        pro_list = torch.ones(pro_list.shape, dtype=pro_list.dtype) / \
                   pro_list.shape[0]
        place = torch.multinomial(pro_list, 1).item()

    return place


def get_place_max(pro_list_, freeze_place_list):
    pro_list = pro_list_.clone().view(-1, )
    # for action in forbidden_actions:
    #     pro_list[action] = 0
    for action in freeze_place_list:
        pro_list[action] = 0
    action = torch.argmax(pro_list)
    return action.item()


def get_base_sample(pro_list_, forbidden_base_list):
    pro_list = pro_list_.clone().view(-1, )
    for forbidden_base in forbidden_base_list:
        pro_list[forbidden_base] = 0
    try:
        base = torch.multinomial(pro_list, 1).item()
    except:
        pro_list = torch.ones(pro_list.shape, dtype=pro_list.dtype) / \
                   pro_list.shape[0]
        base = torch.multinomial(pro_list, 1).item()
    return base


def get_base_max(pro_list_, forbidden_base_list):
    pro_list = pro_list_.clone().view(-1, )
    # for action in forbidden_actions:
    #     pro_list[action] = 0
    for action in forbidden_base_list:
        pro_list[action] = 0
    action = torch.argmax(pro_list)
    return action.item()


def action_convert(place, base_order, action_space):
    action = place * action_space + base_order
    return action


class PPO_Log(nn.Module):
    def __init__(self,
                 BackBone=BackboneNet,
                 Position_Actor=Position_Actor, Position_Critic=Position_Critic,
                 Base_Actor=Base_Actor, Base_Critic=Base_Critic,
                 backboneParam=None,
                 Position_ActorParam=None, Position_CriticParam=None,
                 Base_ActorParam=None, Base_CriticParam=None,
                 lr_backbone=0.001, lr_critic=0.001, lr_actor=0.001,
                 K_epoch=5, train_batch_size=100, actor_freeze_ep=10, eps_clips=0.2, gamma=0.9, num_graph=100, max_grad_norm=1,
                 pool=None, action_space=4, max_loop=1, use_crtic=False, backbone_freeze_ep=0):
        """
        PPO类初始函数
        :param backboneParam: backbone网络参数，被封装为ArgumentParser
        :param criticParam: critic网络参数，被封装为ArgumentParser
        :param actorParam: actor网络参数，被封装为ArgumentParser
        :param lr_backbone: backbone学习率
        :param lr_critic: critic学习率
        :param lr_actor: actor学习率
        :param K_epoch: 每次训练轮次
        :param train_batch_size: 训练的batch size
        :param actor_freeze_ep: actor冻结的游戏局
        :param eps_clips: 训练限制
        :param gamma: reward衰减率
        :param num_graph: 环境中的场景个数
        :param max_grad_norm: 最大更新梯度
        """
        super(PPO_Log, self).__init__()
        if pool is None:
            # self.pool = pathos_mp.ProcessingPool()
            self.map = map
        else:
            self.pool = pool
            self.map = self.pool.map
        self.actor_freeze_ep = actor_freeze_ep
        self.backbone_freeze_ep = backbone_freeze_ep
        self.eps_clips = eps_clips
        self.use_cuda = (True if torch.cuda.is_available() else False)
        self.K_epoch = K_epoch
        self.gamma = gamma
        self.num_chain = num_graph
        self.max_grad_norm = max_grad_norm
        self.use_crtic = use_crtic
        # self.feature_concat = feature_concat

        # 构建网络
        self.backbone = BackBone(backboneParam.in_size, backboneParam.out_size, backboneParam.edge_size,
                                 backboneParam.hide_size_list, backboneParam.n_layers).to(device)
        self.critic = None
        if use_crtic:
            self.critic = Critic(criticParam.in_size, criticParam.out_size, criticParam.hide_size_list, criticParam.hide_size_fc,
                                      criticParam.n_layers, criticParam.edge_dim).to(device)
        self.actor = Actor(actorParam.in_size, actorParam.out_size, actorParam.hide_size_list, actorParam.hide_size_fc,
                                  actorParam.n_layers, criticParam.edge_dim).to(device)
        self.batch_size = train_batch_size
        self.lr_b = lr_backbone
        self.lr_c = lr_critic
        self.lr_a = lr_actor
        # self.optimizer_b = torch.optim.Adam(filter(lambda p: p.requires_grad, self.backbone.parameters()), lr=self.lr_b)
        # self.optimizer_c = torch.optim.Adam(filter(lambda p: p.requires_grad, self.critic.parameters()), lr=self.lr_c)
        # self.optimizer_a = torch.optim.Adam(filter(lambda p: p.requires_grad, self.actor.parameters()), lr=self.lr_a)
        self.optimizer_b = torch.optim.Adam(self.backbone.parameters(), lr=self.lr_b)
        self.optimizer_c = None
        if use_crtic:
            self.optimizer_c = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c)
        self.optimizer_a = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a)
        # self.optimizer = torch.optim.Adam([
        #     {'params': self.backbone.parameters(), 'lr': self.lr_b},
        #     {'params': self.actor.parameters(), 'lr': self.lr_a},
        #     {'params': self.critic.parameters(), 'lr': self.lr_c}
        # ])
        # 设置网络参数是否训练
        for param in self.backbone.parameters():
            param.requires_grad = True
        if use_crtic:
            for param in self.critic.parameters():
                param.requires_grad = True
        for param in self.actor.parameters():
            # param.requires_grad = False
            param.requires_grad = True
        self.backbone.train()
        if use_crtic:
            self.critic.train()
        # self.actor.eval()
        self.actor.train()
        # 设置数据池
        # 试试多条链合并训练的结果
        # self.buffer = []
        self.buffer = [[]] * self.num_chain
        # for j in range(len(self.buffer)):
        #     del self.buffer[j][:]
        self.loss_a_list = []
        self.loss_c_list = []
        self.action_space = action_space
        self.buffer_cnt = 0
        self.buffer_loop = max_loop

    def forward(self, data_batch, max_size, actions):
        data_batch_ = data_batch.clone().to(device)
        x = Variable(data_batch_.x.float().to(device))
        edge_index = Variable(data_batch_.edge_index.to(device))
        edge_attr = Variable(data_batch_.edge_attr.to(device))
        # edge_weight = edge_attr.view(-1, ).float()

        # real_data_batch_ = real_data_batch.clone()
        # real_edge_index = Variable(real_data_batch_.edge_index.to(device))
        # real_edge_attr = Variable(real_data_batch_.edge_attr.to(device))
        # real_edge_weight = real_edge_attr.view(-1, ).float()
        # feature extract
        # feature_aim = self.backbone(x, edge_index, edge_attr)
        feature = self.backbone(x, edge_index, edge_attr)
        # feature_real = self.backbone(x, real_edge_index, real_edge_attr)

        # feature = torch.cat([feature_aim, feature_real], dim=1)
        # if self.feature_concat:
        #     feature = torch.cat([feature_aim, feature_real], dim=1)
        # else:
        #     feature = feature_aim - feature_real
        # value
        values = None
        if self.use_crtic:
            values = self.critic(feature, edge_index, max_size, edge_attr)
        # action
        action_probs = self.actor(feature, edge_index, max_size, edge_attr)

        dist = Categorical(action_probs)
        action_log_probs = dist.log_prob(actions)
        dist_entropy = dist.entropy()

        return action_log_probs, values, dist_entropy

    def work(self, data_batch, len_list, max_size, forbidden_actions_list, type_='selectAction',
             freeze_actions_list=None):
        """
        产生动作
        :param data_batch: 输入图batch
        :param len_list: RNA
        :param type_: 产生动作的方式，selectAction为按概率随机采样，selectActionMax为选择概率最大的动作
        :return: 产生的动作以及动作的概率
        """
        # 由于actor会对图进行修改，所以要事先克隆
        data_batch_ = data_batch.clone().to(device)
        # real_data_batch_ = real_data_batch.clone().to(device)

        x = Variable(data_batch_.x.float().to(device))

        edge_index = Variable(data_batch_.edge_index.to(device))

        edge_attr = Variable(data_batch_.edge_attr.to(device))

        batch = data_batch_.batch.to(device)
        # edge_weight = edge_attr.view(-1, ).float()

        # real_edge_index = Variable(real_data_batch_.edge_index.to(device))
        # real_edge_attr = Variable(real_data_batch_.edge_attr.to(device))
        # real_edge_weight = real_edge_attr.view(-1, ).float()
        # 产生动作，不是产生训练数据，梯度阶段
        with no_grad():
            # 特征提取
            # feature_aim = self.backbone(x, edge_index, edge_attr)
            feature = self.backbone(x, edge_index, edge_attr)
            # feature_aim_numpy = feature_aim.cpu().numpy()
            # feature_real = self.backbone(x, real_edge_index, real_edge_attr)
            # feature_real_numpy = feature_real.cpu().numpy()
            # feature_numpy = feature_aim_numpy - feature_real_numpy

            # if self.feature_concat:
            #     feature = torch.cat([feature_aim, feature_real], dim=1)
            # else:
            #     feature = feature_aim - feature_real

            # 计算动作概率
            action_prob = self.actor(feature, edge_index, batch, edge_attr).cpu()
            action_prob_list = torch.split(action_prob, 1, dim=0)

        # 产生动作batch
        if type_ == 'selectAction':
            action_work = partial(get_action_sample_forbid, num_change=self.action_space)
            actions = list(self.map(action_work, action_prob_list, len_list, forbidden_actions_list,
                                    freeze_actions_list))
            actions = torch.tensor(list(actions), dtype=torch.long).view(-1,)
            dist = Categorical(action_prob)
            action_log_probs = dist.log_prob(actions)
        else:
            action_work = partial(get_action_max_forbid, num_change=self.action_space)
            actions = list(self.map(action_work, action_prob_list, len_list, forbidden_actions_list,
                                    freeze_actions_list))
            # actions = self.map(get_action_max_forbid, action_prob_list, len_list, forbidden_actions_list)
            actions = torch.tensor(list(actions), dtype=torch.long).view(-1, )
            dist = Categorical(action_prob)
            action_log_probs = dist.log_prob(actions)

        return actions.detach(), action_log_probs.detach()

    def storeTransition(self, transition, id_chain):
        """
        保存四元组
        :param transition:四元组
        :param id_chain: 当前四元组所属的链的id
        :return:
        """
        buffer_tmp = self.buffer[id_chain] + [transition]
        self.buffer[id_chain] = buffer_tmp

    def trainStep(self, ep, max_size, batchSize=None):
        """
        模型训练
        :param ep: 当前的轮次
        :param batchSize: 训练的batch
        :return:
        """
        # 解除actor的冻结
        # if ep == self.actor_freeze_ep + 1:
        #     for param in self.actor.parameters():
        #         param.requires_grad = True
        #     self.actor.train()

        # 等待buffer收集一个batch的数据
        if batchSize is None:
            # if len(self.buffer) < self.batch_size:
            #     return
            batchSize = self.batch_size

        graphs = []
        actions = []
        rewards = []
        old_action_log_prob = []
        Gt = []

        # 每一条完整四元组链，分开处理，计算期望的估计
        for id_chain in range(self.num_chain):
            # 加载数据
            # graphs_tmp = [t.state.to(torch.device("cpu")) for t in self.buffer[id_chain]]
            graphs_tmp = [t.state for t in self.buffer[id_chain]]
            actions_tmp = [t.action for t in self.buffer[id_chain]]
            reward_tmp = [t.reward for t in self.buffer[id_chain]]
            old_action_log_prob_tmp = [t.a_log_prob for t in self.buffer[id_chain]]
            done_tmp = [t.done for t in self.buffer[id_chain]]

            # 计算奖励期望的蒙特卡洛统计
            R = 0
            Gt_tmp = []
            for r, done in zip(reward_tmp[::-1], done_tmp[::-1]):
                if done:
                    R = 0
                R = r + self.gamma * R
                Gt_tmp.insert(0, R)

            graphs += graphs_tmp
            Gt += Gt_tmp
            actions += actions_tmp
            old_action_log_prob += old_action_log_prob_tmp

        Gt = torch.tensor(Gt, dtype=torch.float).to(device)
        actions = torch.tensor(actions).view(-1,).to(device)
        old_action_log_prob = torch.tensor(old_action_log_prob, dtype=torch.float).view(-1,).to(device)
        # real_graph_list = list(self.map(get_real_graph, graphs))

        loss_a_all = 0
        loss_c_all = 0

        for i in range(1, self.K_epoch + 1):
            loss_a = 0
            loss_c = 0
            loss_a_log = 0
            loss_c_log = 0
            n_log = 0
            # 打乱顺序，进行训练
            # with tqdm(total=math.ceil(len(graphs)/self.batch_size), desc=f'Train: Epoch {i}/{self.K_epoch}') as pbar:
            for index in BatchSampler(SubsetRandomSampler(range(len(graphs))), batchSize, False):
                Gt_index = Gt[index]
                # 抽取图
                graphs_index = get_element_index(graphs, index)
                graphs_index = list(self.map(recover_graph, graphs_index))

                real_graphs_index = list(self.map(get_real_graph,graphs_index))

                graphs_index = torch_geometric.data.Batch.from_data_list(graphs_index).to(device)

                # real_graphs_index = get_element_index(real_graph_list, index)
                real_graphs_index = torch_geometric.data.Batch.from_data_list(real_graphs_index).to(device)

                actions_index = actions[index]
                # x, edge_index = Variable(graphs_index.x.float().to(device)), Variable(graphs_index.edge_index.to(device))
                # edge_attr = Variable(graphs_index.edge_attr.to(device))
                # 计算优势
                action_log_probs, V, dist_entropy = self.forward(graphs_index, real_graphs_index, max_size, actions_index)

                if self.use_crtic:
                    delta = Gt_index.view(-1,) - V.detach().view(-1,)
                else:
                    delta = Gt_index.view(-1, )
                advantage = delta.view(-1,)

                # 获取动作概率
                # actions_log_probs = probs.gather(1, actions[index])

                ratio = torch.exp(action_log_probs - old_action_log_prob[index].detach())
                surr1 = ratio * advantage
                surr2 = clamp(ratio, 1 - self.eps_clips, 1 + self.eps_clips) * advantage

                loss_a = -torch.min(surr1, surr2).mean()
                loss_c = F.mse_loss(Gt_index.view(-1,), V.view(-1,)) if self.use_crtic else torch.tensor(0)
                loss_all = loss_a + 0.5 * loss_c - 0.01 * dist_entropy.mean()

                l = len(index)
                n_log += l
                loss_a_log += loss_a.item() * l
                loss_c_log += loss_c.item() * l

                self.optimizer_b.zero_grad()
                self.optimizer_a.zero_grad()
                if self.use_crtic:
                    self.optimizer_c.zero_grad()
                # self.optimizer.zero_grad()

                loss_all.backward()
                nn.utils.clip_grad_norm_(self.backbone.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                if self.use_crtic:
                    nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)

                # self.optimizer_b.step()
                # if self.actor_freeze_ep < ep:
                #     self.optimizer_a.step()
                self.optimizer_b.step()
                self.optimizer_a.step()
                if self.use_crtic:
                    self.optimizer_c.step()
                # self.optimizer.step()
                    # tqdm更新显示
                    # pbar.set_postfix({'LC': loss_c.item(), 'LA': loss_a.item()})
                    # pbar.update(1)
            loss_a_log = loss_a_log / n_log
            loss_c_log = loss_c_log / n_log
            print("Loss_A: {}, Loss_c: {}".format(loss_a_log, loss_c_log))

            loss_a_all += loss_a_log
            loss_c_all += loss_c_log

        self.loss_a_list.append(loss_a.item())
        self.loss_c_list.append(loss_c.item())

        return loss_a_log, loss_c_log


    def clean_buffer(self):
        self.buffer_cnt += 1
        if self.buffer_cnt % self.buffer_loop == 0:
            self.buffer_cnt = 0
            for j in range(len(self.buffer)):
                del self.buffer[j][:]

    def save(self, model_dir, episode):
        """
        保存模型
        :param model_dir: 模型保存地址
        :param i_episode: 当前的游戏轮次
        :return:
        """
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        torch.save(self.backbone.state_dict(), model_dir + 'backbone_' + str(episode) + '.pth')
        torch.save(self.actor.state_dict(), model_dir + 'actor_' + str(episode) + '.pth')
        if self.use_crtic:
            torch.save(self.critic.state_dict(), model_dir + 'critic_' + str(episode) + '.pth')

    def load(self, model_dir, episode):
        """
        保存模型
        :param model_dir: 模型保存地址
        :param i_episode: 当前的游戏轮次
        :return:
        """
        if not os.path.exists(model_dir):
            raise ValueError('File not exist!')

        self.backbone.load_state_dict(torch.load(model_dir + 'backbone_' + str(episode) + '.pth', map_location=lambda storage, loc: storage))
        self.actor.load_state_dict(torch.load(model_dir + 'actor_' + str(episode) + '.pth', map_location=lambda storage, loc: storage))
        if self.use_crtic:
            self.critic.load_state_dict(torch.load(model_dir + 'critic_' + str(episode) + '.pth', map_location=lambda storage, loc: storage))
        print('Load weights from {} with episode {}'.format(model_dir, episode))


    def load_backbone(self, model_dir, episode):
        if not os.path.exists(model_dir):
            raise ValueError('File not exist!')

        self.backbone.load_state_dict(torch.load(model_dir + 'backbone_' + str(episode) + '.pth', map_location=lambda storage, loc: storage))

        print('Load weights from {} with episode {}'.format(model_dir, episode))

    def trainStep_with_freeze(self, ep, max_size, batchSize=None):
        """
        模型训练
        :param ep: 当前的轮次
        :param batchSize: 训练的batch
        :return:
        """
        # 解除actor的冻结
        # if ep == self.actor_freeze_ep + 1:
        #     for param in self.actor.parameters():
        #         param.requires_grad = True
        #     self.actor.train()

        # 等待buffer收集一个batch的数据
        if batchSize is None:
            # if len(self.buffer) < self.batch_size:
            #     return
            batchSize = self.batch_size

        if ep <= self.backbone_freeze_ep:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()
        else:
            for param in self.backbone.parameters():
                param.requires_grad = True
            self.backbone.train()

        graphs = []
        actions = []
        rewards = []
        old_action_log_prob = []
        Gt = []

        # 每一条完整四元组链，分开处理，计算期望的估计
        for id_chain in range(self.num_chain):
            # 加载数据
            # graphs_tmp = [t.state.to(torch.device("cpu")) for t in self.buffer[id_chain]]
            graphs_tmp = [t.state for t in self.buffer[id_chain]]
            actions_tmp = [t.action for t in self.buffer[id_chain]]
            reward_tmp = [t.reward for t in self.buffer[id_chain]]
            old_action_log_prob_tmp = [t.a_log_prob for t in self.buffer[id_chain]]
            done_tmp = [t.done for t in self.buffer[id_chain]]

            # 计算奖励期望的蒙特卡洛统计
            R = 0
            Gt_tmp = []
            for r, done in zip(reward_tmp[::-1], done_tmp[::-1]):
                if done:
                    R = 0
                R = r + self.gamma * R
                Gt_tmp.insert(0, R)

            graphs += graphs_tmp
            Gt += Gt_tmp
            actions += actions_tmp
            old_action_log_prob += old_action_log_prob_tmp

        Gt = torch.tensor(Gt, dtype=torch.float).to(device)
        actions = torch.tensor(actions).view(-1,).to(device)
        old_action_log_prob = torch.tensor(old_action_log_prob, dtype=torch.float).view(-1,).to(device)
        # real_graph_list = list(self.map(get_real_graph, graphs))

        loss_a_all = 0
        loss_c_all = 0

        for i in range(1, self.K_epoch + 1):
            loss_a = 0
            loss_c = 0
            loss_a_log = 0
            loss_c_log = 0
            n_log = 0
            # 打乱顺序，进行训练
            # with tqdm(total=math.ceil(len(graphs)/self.batch_size), desc=f'Train: Epoch {i}/{self.K_epoch}') as pbar:
            for index in BatchSampler(SubsetRandomSampler(range(len(graphs))), batchSize, False):
                Gt_index = Gt[index]
                # 抽取图
                graphs_index = get_element_index(graphs, index)
                graphs_index = list(self.map(recover_graph, graphs_index))

                # real_graphs_index = list(self.map(get_real_graph,graphs_index))

                graphs_index = torch_geometric.data.Batch.from_data_list(graphs_index).to(device)

                # real_graphs_index = get_element_index(real_graph_list, index)
                # real_graphs_index = torch_geometric.data.Batch.from_data_list(real_graphs_index).to(device)

                actions_index = actions[index]
                # x, edge_index = Variable(graphs_index.x.float().to(device)), Variable(graphs_index.edge_index.to(device))
                # edge_attr = Variable(graphs_index.edge_attr.to(device))
                # 计算优势
                action_log_probs, V, dist_entropy = self.forward(graphs_index, max_size, actions_index)

                if self.use_crtic:
                    delta = Gt_index.view(-1,) - V.detach().view(-1,)
                else:
                    delta = Gt_index.view(-1, )
                advantage = delta.view(-1,)

                # 获取动作概率
                # actions_log_probs = probs.gather(1, actions[index])

                ratio = torch.exp(action_log_probs - old_action_log_prob[index].detach())
                surr1 = ratio * advantage
                surr2 = clamp(ratio, 1 - self.eps_clips, 1 + self.eps_clips) * advantage

                loss_a = -torch.min(surr1, surr2).mean()
                loss_c = F.mse_loss(Gt_index.view(-1,), V.view(-1,)) if self.use_crtic else torch.tensor(0)
                loss_all = loss_a + 0.5 * loss_c - 0.01 * dist_entropy.mean()

                l = len(index)
                n_log += l
                loss_a_log += loss_a.item() * l
                loss_c_log += loss_c.item() * l

                if ep > self.backbone_freeze_ep:
                    self.optimizer_b.zero_grad()

                self.optimizer_a.zero_grad()

                if self.use_crtic:
                    self.optimizer_c.zero_grad()
                # self.optimizer.zero_grad()

                loss_all.backward()
                nn.utils.clip_grad_norm_(self.backbone.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                if self.use_crtic:
                    nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)

                # self.optimizer_b.step()
                # if self.actor_freeze_ep < ep:
                #     self.optimizer_a.step()
                if ep > self.backbone_freeze_ep:
                    self.optimizer_b.step()

                self.optimizer_a.step()

                if self.use_crtic:
                    self.optimizer_c.step()
                # self.optimizer.step()
                    # tqdm更新显示
                    # pbar.set_postfix({'LC': loss_c.item(), 'LA': loss_a.item()})
                    # pbar.update(1)
            loss_a_log = loss_a_log / n_log
            loss_c_log = loss_c_log / n_log
            print("Loss_A: {}, Loss_c: {}".format(loss_a_log, loss_c_log))

            loss_a_all += loss_a_log
            loss_c_all += loss_c_log

        self.loss_a_list.append(loss_a.item())
        self.loss_c_list.append(loss_c.item())

        return loss_a_log, loss_c_log







