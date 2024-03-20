from __future__ import print_function
import argparse
import json
import os
# import cPickle as pickle
import pickle
import torch.autograd as autograd
autograd.set_detect_anomaly(True)
import random
import numpy as np
import csv
import torch
# from torch.utils.tensorboard import SummaryWriter
from torch import nn
from torch.autograd import Variable
from sort_of_clevr.data.prioritysort import PrioritySortDataset
from sort_of_clevr.data.nfar import NFarDataset
from model import RN, CNN_MLP
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tensorboardX as tbX
from torch.autograd import Variable
from transformers import TransformerEncoder
from einops import rearrange, repeat
from transformer_utilities.set_transformer import SetTransformer

# from torch.cuda import memory
# memory._set_max_split_size(max_split_size_mb=512)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 走基础模型
class BasicModel(nn.Module):
    def __init__(self, args, name):
        super(BasicModel, self).__init__()
        self.name = name

    def train_(self, input_data, label, losses, errors):
        self.optimizer.zero_grad()
        # output = self(input_data.detach())  # (64,8) label(1,64,8)
        if "nfar" not in args.task_name:
            # 若任务名称中不包含"nfar"，则将in_data初始化为与输入相同形状的全零张量
            in_data = torch.zeros(input_data.size()).to(device)
            in_data = in_data.permute(1,0,2)[-(label.size()[0]):].permute(1,0,2)
            output= self(in_data.detach())
            out = torch.sigmoid(output).repeat(label.size()[0],1,1)

        criterion = nn.BCELoss()
        # loss = F.nll_loss(out2, torch.argmax(label2, -1))  # (64,10) (64,)
        loss = criterion(out, label)
        losses.append(loss.item())
        loss.backward()
        if args.clip_grad > 0:
            nn.utils.clip_grad_value_(model.parameters(), args.clip_grad)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        self.optimizer.step()
        # 5.计算错误率error
        binary_output = out.clone().to('cpu').detach().apply_(lambda x: 0 if x < 0.5 else 1).to(device)
        error = torch.sum(torch.abs(binary_output - label)) / args.batch_size
        errors.append(error.item())
        return losses, errors

    def test_(self, input_data, label):
        output = self(input_data)
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

# 此处开始为新增内容
class Transformer(BasicModel):
    def __init__(self,  args):
        super(Transformer, self).__init__(args, 'Transformer')

        h_dim = args.embed_dim
        num_classes = 32

        if args.functional:
            # 置换不变性 FFFT
            print('USING SET TRANSFORMER')
            self.net = SetTransformer(h_dim, dim_hidden = 512, num_inds = args.mem_slots)   # 原来都为512
        else:
            self.net = TransformerEncoder(
                            h_dim,
                            512,
                            num_layers = args.num_layers,
                            num_heads = 4,
                            dropout = 0.1,
                            share_parameters = args.share_vanilla_parameters,
                            shared_memory_attention = args.shared_memory_attention,
                            use_topk = args.use_topk,
                            topk = args.topk,
                            mem_slots = args.mem_slots,
                            null_attention = args.null_attention )

        self.input_to_embedding = nn.Linear(34, h_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, h_dim))

        if args.functional:
            self.mlp_head = nn.Linear(512, num_classes)
        else:
            self.mlp_head = nn.Linear(h_dim, num_classes)
        self.optimizer = optim.Adam(self.parameters(), lr=args.lr)

    def forward(self, x):
        x = self.input_to_embedding(x).to(device)  # 将40变为256维
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)  # (64,8+1,256)
        # 计算x 要么为transformer的输出，要么为transformer+GRU的输出
        x = self.net(x.to(device))   # 输出的x为(64,27,256)
        x = self.mlp_head(x[:, 0].detach().clone())
        x = F.log_softmax(x, dim=1)
        # x = F.log_softmax(self.mlp_head(x[:,0]), dim = 1)  # (64,8)
        return x



def str2bool(v):
    """Method to map string to bool for argument parser"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


# Training settings   1.设置参数对象parser
parser = argparse.ArgumentParser(description='PyTorch Relational-Network sort-of-CLVR Example')
# 2.调用 add_argument() 方法添加参数
parser.add_argument('--model', type=str, choices=['RN', 'CNN_MLP', 'Transformer'], default='Transformer',
                    help='resume from model stored')
parser.add_argument('--epochs', type=int, default=50000, metavar='N',
                    help='number of epochs to train (default: 20)')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                    help='learning rate (default: 0.0001)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
# parser.add_argument('--seed', type=int, default=1, metavar='S',
#                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--resume', type=str,
                    help='resume from model stored')
parser.add_argument('--relation-type', type=str, default='binary',
                    help='what kind of relations to learn. options: binary, ternary (default: binary)')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')

# 1 ~9
# TR + HSW 256 4 True True 5 True 8 1 False
parser.add_argument('--embed_dim', type=int, default=128)  # default=256  h_dim
parser.add_argument('--num_layers', type=int, default=4)
# 层之间参数是否共享，TR+HC和ISAB 不会共享，其他都是共享
parser.add_argument('--share_vanilla_parameters', type=str2bool, default=True)  # default=False
parser.add_argument('--use_topk', type=str2bool, default=True)  # default=False
parser.add_argument('--topk', type=int, default=8)  # default=3
parser.add_argument('--shared_memory_attention', type=str2bool, default=True) # default=False
parser.add_argument('--mem_slots', type=int, default=8)  # default=4
parser.add_argument('--seed', type=int, default=1)  # default=0
parser.add_argument('--functional', type=str2bool, default=False,
                    help='ues set_transformer or not') # default=False

parser.add_argument('--save_dir', type=str, default='model_nfar')
parser.add_argument('--null_attention', type=str2bool, default=False)
# zxy add
parser.add_argument('-clip_grad', type=int, default=10,
                        help='clip gradient')

# 3.使用 parse_args() 解析添加的参数
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
np.random.seed(args.seed)


args.image_size = 75
args.patch_size = 15

# 调用模型
if args.model == 'CNN_MLP':
    model = CNN_MLP(args)
elif args.model == 'Transformer':
    model = Transformer(args)
    # print(model)
else:
    model = RN(args)

model_dirs = args.save_dir
bs = args.batch_size
model.to(device)

# def tensor_data(data, i):
#     # img为tensor类型， from_numpy()方法把数组转换成张量，且二者共享内存
#     # np.asarray()将数据结构转化为ndarray的形式  每次取0:64 64:128 128:192
#     img = torch.from_numpy(np.asarray(data[0][bs * i:bs * (i + 1)])) # 64,3,75,75
#     qst = torch.from_numpy(np.asarray(data[1][bs * i:bs * (i + 1)])) # 64,18
#
#     input_data.data.resize_(img.size()).copy_(img)
#     label.data.resize_(qst.size()).copy_(qst)


def cvt_data_axis(data):
    input = [e[0] for e in data]
    target = [e[1] for e in data]
    return (input, target)

# 模型训练
# def train(epoch, input_data, label):
#     # data 为dict 含input(8,128,40) target(1,128,8)
#     model.train()
#     # random.shuffle(data)
#     # data为二元组形式：输入、答案
#     # data_tensor = cvt_data_axis(data)
#     acc_total = []
#     loss_total = []
#     input_data = input_data.permute(1,0,2)  # (8,128,40)-->(128,8,40)
#     label = label
#
#     # rel[0]为98000个（3,75,75）的图片
#     for batch_idx in range(bs):
#         loss, error = model.train_(input_data, label)
#         losses.append(loss.item())
#         errors.append(error.item())
#
#         # if batch_idx % args.log_interval == 0:
#         #     print('Train Epoch: {} [{}/{} ({:.0f}%)] '
#         #           'Accuracy: {:.0f}%'.format(
#         #         epoch,
#         #         batch_idx * bs * 2,
#         #         len(rel[0]) * 2,
#         #         100. * batch_idx * bs / len(rel[0]),
#         #         acc_total), flush=True)
#
#     # avg_acc = sum(acc_total) / len(acc_total)
#     #
#     # avg_loss = sum(loss_total) / len(loss_total)
#
#     return losses, errors



# 模型测试
# def test(epoch, ternary, rel, norel):
#     model.eval()
#     if not len(rel[0]) == len(norel[0]):
#         print('Not equal length for relation dataset and non-relation dataset.', flush=True)
#         return
#
#     ternary = cvt_data_axis(ternary)
#     rel = cvt_data_axis(rel)
#     norel = cvt_data_axis(norel)
#
#     accuracy_ternary = []
#     accuracy_rels = []
#     accuracy_norels = []
#
#     loss_ternary = []
#     loss_binary = []
#     loss_unary = []
#
#     for batch_idx in range(len(rel[0]) // bs):
#         tensor_data(ternary, batch_idx)
#         acc_ter, l_ter = model.test_(input_img, input_qst, label)
#         accuracy_ternary.append(acc_ter.item())
#         loss_ternary.append(l_ter.item())
#
#         tensor_data(rel, batch_idx)
#         acc_bin, l_bin = model.test_(input_img, input_qst, label)
#         accuracy_rels.append(acc_bin.item())
#         loss_binary.append(l_bin.item())
#
#         tensor_data(norel, batch_idx)
#         acc_un, l_un = model.test_(input_img, input_qst, label)
#         accuracy_norels.append(acc_un.item())
#         loss_unary.append(l_un.item())
#
#     accuracy_ternary = sum(accuracy_ternary) / len(accuracy_ternary)
#     accuracy_rel = sum(accuracy_rels) / len(accuracy_rels)
#     accuracy_norel = sum(accuracy_norels) / len(accuracy_norels)
#     print('\n Test set: Ternary accuracy: {:.0f}% Binary accuracy: {:.0f}% | Unary accuracy: {:.0f}%\n'.format(
#         accuracy_ternary, accuracy_rel, accuracy_norel), flush=True)
#
#     # summary_writer.add_scalars('Accuracy/test', {
#     #    'ternary': accuracy_ternary,
#     #    'binary': accuracy_rel,
#     #    'unary': accuracy_norel
#     # }, epoch)
#
#     loss_ternary = sum(loss_ternary) / len(loss_ternary)
#     loss_binary = sum(loss_binary) / len(loss_binary)
#     loss_unary = sum(loss_unary) / len(loss_unary)
#
#     # summary_writer.add_scalars('Loss/test', {
#     #    'ternary': loss_ternary,
#     #    'binary': loss_binary,
#     #    'unary': loss_unary
#     # }, epoch)
#
#     return accuracy_ternary, accuracy_rel, accuracy_norel


# 数据加载
# def load_data():
#     print('loading data...')
#     dirs = './data'
#     filename = os.path.join(dirs, 'sort-of-clevr.pickle')
#     with open(filename, 'rb') as f:
#         train_datasets, test_datasets = pickle.load(f)
#     ternary_train = []
#     ternary_test = []
#     rel_train = []
#     rel_test = []
#     norel_train = []
#     norel_test = []
#     print('processing data...', flush=True)
#
#     for img, ternary, relations, norelations in train_datasets:
#         img = np.swapaxes(img, 0, 2)
#         for qst, ans in zip(ternary[0], ternary[1]):
#             ternary_train.append((img, qst, ans))
#         for qst, ans in zip(relations[0], relations[1]):
#             rel_train.append((img, qst, ans))
#         for qst, ans in zip(norelations[0], norelations[1]):
#             norel_train.append((img, qst, ans))
#
#     for img, ternary, relations, norelations in test_datasets:
#         img = np.swapaxes(img, 0, 2)
#         for qst, ans in zip(ternary[0], ternary[1]):
#             ternary_test.append((img, qst, ans))
#         for qst, ans in zip(relations[0], relations[1]):
#             rel_test.append((img, qst, ans))
#         for qst, ans in zip(norelations[0], norelations[1]):
#             norel_test.append((img, qst, ans))
#
#     return (ternary_train, ternary_test, rel_train, rel_test, norel_train, norel_test)

if __name__ == "__main__":

    # 1.处理数据集
    task_params = json.load(open("./task/prioritysort.json"))
    dataset = PrioritySortDataset(task_params)
    args.task_name = task_params['task']
    log_dir = f'./logs_{args.task_name}_share_stm'
    summary_writer = tbX.SummaryWriter(log_dir)
    # data 为dict 含input(8,128,40) target(1,128,8)
    # data = dataset.get_sample_wlen(bs=bs)
    # input, target = data['input'].to(device), data['target'].to(device)  # input(8,128,40) target(1,128,8)
    # out = torch.zeros(target.size()).to(device)  # out(1,128,8)

    try:
        os.makedirs(model_dirs)
    except:
        print('directory {} already exists'.format(model_dirs), flush=True)

    if args.resume:
        filename = os.path.join(model_dirs, args.resume)
        if os.path.isfile(filename):
            print('==> loading checkpoint {}'.format(filename))
            # 仅保存和加载模型参数的形式load_state_dict
            checkpoint = torch.load(filename)
            model.load_state_dict(checkpoint)
            print('==> loaded checkpoint {}'.format(filename), flush=True)

    with open(f'{args.save_dir}/{args.model}_{args.seed}_log.csv', 'w') as log_file:
        csv_writer = csv.writer(log_file, delimiter=',')
        csv_writer.writerow(['epoch', 'train_acc_ternary', 'train_acc_rel',
                             'train_acc_norel', 'test_acc_ternary', 'test_acc_rel', 'test_acc_norel'])

        print(f"Training {args.model} {f'({args.relation_type})' if args.model == 'RN' else ''} model...", flush=True)

        losses = []
        errors = []
        for epoch in range(1, args.epochs + 1):
            # 开始训练
            # loss, error = train(epoch, input, target)
            data = dataset.get_sample_wlen(bs=args.batch_size)
            input, target = data['input'].to(device), data['target'].to(device)  # input(8,128,40) target(1,128,8)
            out = torch.zeros(target.size()).to(device)  # out(1,128,8)
            model.train()
            input_data = input.permute(1, 0, 2)  # (8,128,40)-->(128,8,40)
            losses, errors = model.train_(input_data, target, losses, errors)
            losses_out = np.mean(losses)
            errors_out = np.mean(errors)
            if epoch % 10 == 0:
                print('Iteration: %d\tLoss: %.2f\tError in bits per sequence: %.2f' %
                      (epoch, losses_out, errors_out))
            summary_writer.add_scalars('Errors/train', {
                'losses': losses_out,
                'errors': errors_out
            }, epoch)

            # 开始测试模型
            # test_acc = test(epoch, data)
            # 写入excel表格
            csv_writer.writerow([epoch, errors_out])
            # 每次epoch都保存模型
            # model.save_model(epoch, args.save_dir)
            torch.cuda.empty_cache()
        print("epoch end!")
