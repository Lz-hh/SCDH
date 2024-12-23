import torch
from torch import nn
from .alias_multinomial import AliasMethod
import math
import torch.nn.functional as F


class NCEAverage(nn.Module):

    def __init__(self, inputSize, outputSize, K, T=0.07, momentum=0.5, classes=24, use_softmax=True):
        super(NCEAverage, self).__init__()
        self.nLem = outputSize  # 传进来的是 ndata：整个数据集的长度，也就是 memory bank 的长度
        self.unigrams = torch.ones(self.nLem)
        self.multinomial = AliasMethod(self.unigrams)  # AliasMethod 在这里是用于：随机采样负样本
        self.multinomial.cuda()
        self.K = K  # 随机采样负样本的数量
        self.use_softmax = use_softmax
        self.register_buffer('params', torch.tensor(
            [K, T * math.sqrt(inputSize), -1, -1, momentum]))  # 用params 保存参数 K, T, momentum 的值，用于后面计算 NCE loss
        stdv = 1. / math.sqrt(inputSize / 3)
        rnd = torch.randn(outputSize, inputSize).mul_(2 * stdv).add_(-stdv)  # 随机初始化memory bank
        self.register_buffer('memory', F.normalize(rnd.sign(), dim=1))
        self.label_memory = torch.zeros(outputSize, classes).cuda()

    def update_memory(self, data):
        memory = 0
        for i in range(len(data)):
            memory += data[i]
        memory /= memory.norm(dim=1, keepdim=True)
        self.memory.mul_(0).add_(memory)

    def forward(self, labels, l, ab, y, classes, pos_num, neg_num, idx=None, epoch=None):
        K = int(self.params[0].item())
        T = self.params[1].item()
        Z_l = self.params[2].item()
        Z_ab = self.params[3].item()

        momentum = self.params[4].item() if (epoch is None) else (0 if epoch < 0 else self.params[4].item())
        batchSize = l.size(0)
        outputSize = self.memory.size(0)
        inputSize = self.memory.size(1)
        pos_indices = None
        neg_indices = None
        y.cuda()
        # 初始化保存正负样本索引和weight的列表
        pos_indices_list = []
        neg_indices_list = []
        weight_list = []

        # score computation
        if idx is None:
            # idx = self.multinomial.draw(self.nLem)
            # # idx.select(1, 0).copy_(y.data)  # 将真实标签索引放到idx的第一列，y表示真实标签索引

            # 随机生成1w个索引，它们将在整个self.memory中进行采样
            idx_total = torch.randperm(self.memory.size(0))
        # sample
        if momentum <= 0:
            weight = (l + ab) / 2.
            inx = torch.stack([torch.arange(batchSize)] * batchSize)
            inx = torch.cat(
                [torch.arange(batchSize).view([-1, 1]), inx[torch.eye(batchSize) == 0].view([batchSize, -1])],
                dim=1).to(weight.device).view([-1])
            # print(inx)
            weight = weight[inx].view([batchSize, batchSize, -1])
        else:
            for i in range(batchSize):
                device = self.memory.device  # 获取 self.memory 所在的设备
                # 获取当前样本的标签
                current_label = self.label_memory[y[i]]
                idx_total = idx_total.to(device)
                # 根据当前样本标签，判断idx_total中的样本是正样本还是负样本
                label_similarity = current_label @ self.label_memory[idx_total].T

                # 找到正负样本的索引
                pos_indices = idx_total[label_similarity > 0]
                neg_indices = idx_total[label_similarity == 0]
                pos_indices = pos_indices.to(device)  # 将 pos_indices 转移到该设备
                neg_indices = neg_indices.to(device)  # 将 neg_indices 转移到该设备

                if len(pos_indices) == 0:  # 如果没有正样本，找标签相似度最高的样本作为正样本
                    most_similar_idx = torch.argmax(label_similarity)
                    pos_indices = idx_total[most_similar_idx].unsqueeze(0)  # 将其增加一个维度，使其成为一维张量
                    neg_indices = idx_total[torch.arange(len(idx_total), device=device) != most_similar_idx].to(device)

                # 如果不足，通过重复来填充
                if len(pos_indices) < pos_num:
                    repeat_times = (pos_num + len(pos_indices) - 1) // len(pos_indices)
                    pos_indices = pos_indices.repeat(repeat_times)

                if len(neg_indices) < neg_num:
                    repeat_times = (neg_num + len(neg_indices) - 1) // len(neg_indices)
                    neg_indices = neg_indices.repeat(repeat_times)

                pos_indices = pos_indices[:pos_num]
                neg_indices = neg_indices[:neg_num]

                # 将索引添加到对应的列表中
                pos_indices_list.append(pos_indices)
                neg_indices_list.append(neg_indices)

                # 获取对应的weight，并添加到weight_list中
                weight = torch.index_select(self.memory, 0, torch.cat([pos_indices, neg_indices])).detach()
                weight_list.append(weight)

                # 将weight_list转换为张量，形状为[batchSize, 4096, inputSize]
            weight_tensor = torch.stack(weight_list)
            weight = weight_tensor

        weight = weight.sign_()
        # out_ab为ab与正负样本点积计算得到的概率，也可用其进一步得到归一化常数Z
        out_ab = torch.bmm(weight, ab.view(batchSize, inputSize, 1))
        # sample
        out_l = torch.bmm(weight, l.view(batchSize, inputSize, 1))
        if self.use_softmax:  # 归一化
            out_ab = torch.div(out_ab, T)
            out_l = torch.div(out_l, T)
            out_l = out_l.contiguous()
            out_ab = out_ab.contiguous()
        else:
            out_ab = torch.exp(torch.div(out_ab, T))
            out_l = torch.exp(torch.div(out_l, T))
            # set Z_0 if haven't been set yet,
            # Z_0 is used as a constant approximation of Z, to scale the probs
            if Z_l < 0:
                self.params[2] = out_l.mean() * outputSize
                Z_l = self.params[2].clone().detach().item()
                print("normalization constant Z_l is set to {:.1f}".format(Z_l))
            if Z_ab < 0:
                self.params[3] = out_ab.mean() * outputSize
                Z_ab = self.params[3].clone().detach().item()
                print("normalization constant Z_ab is set to {:.1f}".format(Z_ab))
            # compute out_l, out_ab
            out_l = torch.div(out_l, Z_l).contiguous()
            out_ab = torch.div(out_ab, Z_ab).contiguous()

        # # update memory
        with torch.no_grad():
            l = (l + ab) / 2.  # shape:256,32 连续值
            l.div_(l.norm(dim=1, keepdim=True))
            l_pos = torch.index_select(self.memory, 0, y.view(-1))  # self.memory: 18015,32 连续值 y:256
            l_pos.mul_(momentum)  # l_pos:256，32
            l_pos.add_(torch.mul(l, 1 - momentum))  # 动量更新公式
            l_pos = l_pos.div_(l_pos.norm(dim=1, keepdim=True))
            self.memory.index_copy_(0, y, l_pos)
            self.label_memory.index_copy_(0, y, labels.float())  # labels 为一个batch的张量

        return out_l, out_ab, momentum
