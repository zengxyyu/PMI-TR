import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from transformer_utilities.sw_ltm import SW_LTM
from transformer_utilities.relational_memory_volatile import RelationalMemory
from transformers import TransformerEncoder
from einops import rearrange, repeat
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ConvInputModel(nn.Module):
    def __init__(self):
        super(ConvInputModel, self).__init__()

        self.conv1 = nn.Conv2d(3, 24, 3, stride=2, padding=1)
        self.batchNorm1 = nn.BatchNorm2d(24)
        self.conv2 = nn.Conv2d(24, 24, 3, stride=2, padding=1)
        self.batchNorm2 = nn.BatchNorm2d(24)
        self.conv3 = nn.Conv2d(24, 24, 3, stride=2, padding=1)
        self.batchNorm3 = nn.BatchNorm2d(24)
        self.conv4 = nn.Conv2d(24, 24, 3, stride=2, padding=1)
        self.batchNorm4 = nn.BatchNorm2d(24)

    def forward(self, img):
        """convolution"""
        x = self.conv1(img)
        x = F.relu(x)
        x = self.batchNorm1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.batchNorm2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.batchNorm3(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.batchNorm4(x)
        return x

# 走基础模型
class BasicModel(nn.Module):
    def __init__(self, args, name):
        super(BasicModel, self).__init__()
        self.name = name

    def train_(self, input_img, input_qst, label):
        self.optimizer.zero_grad()
        output = self(input_img, input_qst)  # (64,10) label(64)
        loss = F.nll_loss(output, label)
        loss.backward()
        self.optimizer.step()
        # max(1)返回数组每一行最大值组成的一维数组  max(1)[1]返回最大值的所在行的索引
        pred = output.data.max(1)[1]  # 64,
        correct = pred.eq(label.data).cpu().sum()
        accuracy = correct * 100. / len(label)
        return accuracy, loss

    def test_(self, input_img, input_qst, label):
        output = self(input_img, input_qst)
        loss = F.nll_loss(output, label)
        pred = output.data.max(1)[1]
        correct = pred.eq(label.data).cpu().sum()
        accuracy = correct * 100. / len(label)
        return accuracy, loss

    def save_model(self, epoch, save_dir):
        import os
        name = 'epoch_{}_{:02d}.pth'.format(self.name, epoch)
        path = os.path.join(save_dir, name)
        # 仅保存和加载模型参数
        torch.save(self.state_dict(), path)

class FCOutputModel(nn.Module):
    def __init__(self):
        super(FCOutputModel, self).__init__()

        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.fc2(x)
        x = F.relu(x)
        x = F.dropout(x)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

class CNN_MLP(BasicModel):

    def __init__(self, args):
        super(CNN_MLP, self).__init__(args, 'CNNMLP')

        self.conv = ConvInputModel()
        self.fc1 = nn.Linear(96, 256)  # question concatenated to all
        self.fcout = FCOutputModel()

        # self.optimizer = optim.Adam(self.parameters(), lr=args.lr)
        self.optimizer = optim.AdamW(self.parameters(), lr=args.lr, weight_decay=1e-4)
        # print([ a for a in self.parameters() ] )

    def forward(self, img):
        x = self.conv(img)  ## x = (64 x 24 x 5 x 5)

        """fully connected layers"""
        x = x.view(x.size(0), -1)  # 64,600  64,96
        x_ = self.fc1(x)
        x_ = F.relu(x_)

        return self.fcout(x_)

class CNN_PMI(nn.Module):
    def __init__(self, args):
        # super(CNN_MI, self).__init__(args, 'CNNPMI')
        super().__init__()
        self.use_topk = args.use_topk
        self.topk = args.topk
        self.use_long_men = args.use_long_men
        self.long_mem_segs = args.long_mem_segs
        self.long_mem_aggre = args.long_mem_aggre
        self.use_wm_inference = args.use_wm_inference

        self.conv = ConvInputModel()
        self.f_fc1 = nn.Linear(256, 256)
        self.fcout = FCOutputModel()

        h_dim = args.h_dim
        self.patch_to_embedding = nn.Linear(24, h_dim)  # 最初為24
        self.num_heads = 8
        self.head_dim = args.h_dim // self.num_heads

        self.relational_memory = SW_LTM(
            mem_slots=args.mem_slots,
            head_size=self.head_dim,
            input_size=args.h_dim,
            output_size=args.h_dim,
            num_heads=self.num_heads,
            num_blocks=1,
            forget_bias=1,
            input_bias=0,
            gate_style="unit",
            attention_mlp_layers=4,
            key_size=32,
            return_all_outputs=False,
            use_topk=self.use_topk,
            topk=self.topk,
            use_long_men=self.use_long_men,
            long_mem_segs=self.long_mem_segs,
            long_mem_aggre=self.long_mem_aggre,
            use_wm_inference=self.use_wm_inference,
            num_steps=int(4),
            null_attention=args.null_attention
        )
        self.memory = None
        self.Mr = None
        # self.optimizer = optim.Adam(self.parameters(), lr=args.lr)
        # self.optimizer = optim.AdamW(self.parameters(), lr=args.lr, weight_decay=0.05)  #zxy
        # self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=20, T_mult=1, eta_min=0.0001)

        # 训练循环
        self.mlp_head = nn.Linear(h_dim, 10)

    def forward(self, img):
        x = self.conv(img)  # x = (64 x 24 x 2 x 2) 3L:64,24,4,4
        bs = x.size()[0]
        n_channels = x.size()[1]
        d = x.size()[2]
        x_flat = x.view(bs, n_channels, d * d)  # x_flat = (64 x 24 x 4)
        x_flat = x_flat.permute(0,2,1)  # (64 x 4 x 24)
        x = self.patch_to_embedding(x_flat.to(device)) # (64 x 24 x 256)

        if self.memory is None and self.Mr is None:
            self.memory, self.Mr = self.relational_memory.initial_state(x.size(0),
                                                                        x.size(1))  # query(64,4,24)
            print("In multi-head self-attention, the initial size of working memory is:", self.memory.size())  # (64,8,256)
            print("In multi-head self-attention, the initial size of long-term memory is:", self.Mr.size())  # (64,8,256)

        _, _, self.memory, self.Mr, hx = self.relational_memory(inputs=x.to(device), memory=self.memory.detach().to(device),
                                                                            Mr=self.Mr.detach().to(device))
        # x_g = hx.sum(1).squeeze()
        x_g = hx[:, 0]
        # x_g = F.log_softmax(self.mlp_head(x[:, 0]), dim=1)  # (64,10)

        """f"""
        x_ = self.f_fc1(x_g)   # x_g 64, 256
        x_ = F.relu(x_)

        # return x_g
        return self.fcout(x_)

class CNN_TRHSW(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.use_topk = args.use_topk
        self.topk = args.topk

        self.conv = ConvInputModel()
        self.f_fc1 = nn.Linear(256, 256)
        self.fcout = FCOutputModel()

        h_dim = args.h_dim
        self.patch_to_embedding = nn.Linear(24, h_dim)  # 最初為24
        self.num_heads = 8
        self.head_dim = args.h_dim // self.num_heads

        self.relational_memory = RelationalMemory(
            mem_slots=args.mem_slots,
            head_size=self.head_dim,
            input_size=args.h_dim,
            output_size=args.h_dim,
            num_heads=self.num_heads,
            num_blocks=1,
            forget_bias=1,
            input_bias=0,
            gate_style="unit",
            attention_mlp_layers=4,
            key_size=32,
            return_all_outputs=False,
            use_topk=self.use_topk,
            topk=self.topk,
            num_steps=int(4),
            null_attention=args.null_attention
        )
        self.memory = None
        # self.optimizer = optim.Adam(self.parameters(), lr=args.lr)
        # self.optimizer = optim.AdamW(self.parameters(), lr=args.lr, weight_decay=0.05)  #zxy
        # self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=20, T_mult=1, eta_min=0.0001)

        # 训练循环
        self.mlp_head = nn.Linear(h_dim, 10)

    def forward(self, img):
        x = self.conv(img)  # x = (64 x 24 x 2 x 2) 3L:64,24,4,4
        bs = x.size()[0]
        n_channels = x.size()[1]
        d = x.size()[2]
        x_flat = x.view(bs, n_channels, d * d)  # x_flat = (64 x 24 x 4)
        x_flat = x_flat.permute(0,2,1)  # (64 x 4 x 24)
        x = self.patch_to_embedding(x_flat.to(device)) # (64 x 24 x 256)

        if self.memory is None:
            self.memory = self.relational_memory.initial_state(x.size(0), x.size(1))  # query(27,64,256)
            print("The initial workspace size is:", self.memory.size())  # (64,8,256)

        _, _, self.memory, hx = self.relational_memory(inputs=x.to(device), memory=self.memory.detach().to(device))
        # x_g = hx.sum(1).squeeze()
        x_g = hx[:, 0]
        # x_g = F.log_softmax(self.mlp_head(x[:, 0]), dim=1)  # (64,10)

        """f"""
        x_ = self.f_fc1(x_g)   # x_g 64, 256
        x_ = F.relu(x_)

        # return x_g
        return self.fcout(x_)