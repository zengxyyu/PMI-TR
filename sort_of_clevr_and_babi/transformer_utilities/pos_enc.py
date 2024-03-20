

import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import torch.nn.functional as F
import random

# 对输入进行位置编码
class PositionEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len = 300):
        super().__init__()
        self.d_model = d_model
        # create constant 'pe' matrix with values dependant on
        # pos and i
        # 初始化一个位置编码矩阵, 它是一个0阵，矩阵的大小是max_seq_len x d_model.
        # max_seq_len为句子的单词个数  d_model为词嵌入维度
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = \
                math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))

        # pe现在还只是一个二维矩阵，要想和embedding的输出（一个三维张量）相加，就必须拓展一个维度
        pe = pe.unsqueeze(0)
        # 最后把pe位置编码矩阵注册成模型的buffer，什么是buffer呢，
        # 我们把它认为是对模型效果有帮助的，但是却不是模型结构中超参数或者参数，不需要随着优化步骤进行更新的增益对象.
        # 注册之后我们就可以在模型保存后重加载时和模型结构与参数一同被加载.
        self.register_buffer('pe', pe)
        # 与pe维度一样的全1的矩阵
        self.pos_emb_weight = nn.Parameter(torch.ones_like(pe))

    def forward(self, x):
        """forward函数的参数是x, 表示文本序列的词嵌入表示"""
        # make embeddings relatively larger  交换块和行
        x = x.permute(1,0,2)

        #x = x * math.sqrt(self.d_model)
        #add constant to embedding

        seq_len = x.size(1)  # seq_len为x每一块中的行数

        #width x channel
        #pe_use = F.interpolate(self.pe.permute(0,2,1), size=seq_len).permute(0,2,1)
        # 在相加之前我们对pe做一些适配工作， 将这个三维张量的第二维也就是句子最大长度的那一维将切片到与输入的x的第二维相同即x.size(1)，
        # 因为我们默认max_len为300一般来讲实在太大了，很难有一条句子包含300个词汇，所以要进行与输入张量的适配.
        # 最后使用Variable进行封装，使其与x的样式相同，但是它是不需要进行梯度求解的，因此把requires_grad设置成false.
        # zxy cuda
        # pe_use = Variable(self.pe[:,:seq_len] * F.sigmoid(self.pos_emb_weight[:,:seq_len]), requires_grad=False).cuda()
        pe_use = Variable(self.pe[:, :seq_len] * torch.sigmoid(self.pos_emb_weight[:, :seq_len]),
                          requires_grad=False)

        #bs x pos x nhid --> bs x nhid x pos --> bs x pos x nhid
        x = x + pe_use
        x = x.permute(1,0,2)

        return x
