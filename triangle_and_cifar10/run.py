import random
import argparse
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
import tensorboardX as tbX
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from model import CNN_MLP, CNN_PMI, CNN_TRHSW
from transformer_utilities.set_transformer import SetTransformer
from transformers import TransformerEncoder  # FunctionalVisionTransformer, ViT
from einops import rearrange, repeat
from dataset import TriangleDataset, CountingMNISTDataset
import os
from utils.utils import WarmupScheduler
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
log_dir = './24_3.19_OUTRE_cifar10_VIT_256_256_0.0002_zhuji_N4H8_0.55_Sche_T5_wd0.09'
summary_writer = tbX.SummaryWriter(log_dir)

def str2bool(v):
    """Method to map string to bool for argument parser"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='Image Classification Tasks')
parser.add_argument('--model', default="default", type=str, choices=('default', 'functional', 'CNN_MLP', 'CNN_PMI','CNN_TRHSW'),
                    help='type of models to use')  # default="functional" yuanwei default
parser.add_argument('--data', default="Triangle", type=str,
                    choices=('cifar10', 'cifar100', 'pathfinder', 'MNIST', 'Triangle'), help='data to train on')
parser.add_argument('--version', default=0, type=int, help='version for shared transformer-- 0 or 1')
parser.add_argument('--num_templates', default=12, type=int, help='num of templates for shared transformer')
parser.add_argument('--num_heads', default=4, type=int, help='num of heads in Multi Head attention layer')
# Triangle is 32   cifar10 is 4 or 8
# zxy zxy test test2
parser.add_argument('--patch_size', default=32, type=int, help='patch_size for transformer')
parser.add_argument('--epochs', default=1000, type=int, help='num of epochs to train default=200')
parser.add_argument('--lr', default=0.0002, type=float, help='learning rate')
parser.add_argument('--dropout', default=0.1, type=float, help='dropout')
parser.add_argument('--name', default="model", type=str, help='Model name for logs and checkpoint')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')

# parameters setting
parser.add_argument('--batch_size', default=64, type=int, help='batch_size to use')
parser.add_argument('--num_layers', default=4, type=int, help='num of layers')  # default=12
parser.add_argument('--h_dim', type=int, default=256)
parser.add_argument('--ffn_dim', type=int, default=256)  # default=512
parser.add_argument('--share_vanilla_parameters', type=str2bool, default=True) # default=False
parser.add_argument('--use_topk', type=str2bool, default=True)  # default=False
parser.add_argument('--topk', type=int, default=5) # default=3
parser.add_argument('--shared_memory_attention', type=str2bool, default=True) # default=False
parser.add_argument('--seed', type=int, default=1)  # default=0
parser.add_argument('--mem_slots', type=int, default=8) # default=4
parser.add_argument('--use_long_men', type=str2bool, default=True,
                    help='ues long-term memory or not')
parser.add_argument('--long_mem_segs', type=int, default=5)
parser.add_argument('--long_mem_aggre', type=str2bool, default=True,
                    help='uses cross-attention between WM and LTM or not')
parser.add_argument('--use_wm_inference', type=str2bool, default=True,
                    help='WM involvement during inference or not')

parser.add_argument('--num_gru_schemas', type=int, default=1)
parser.add_argument('--num_attention_schemas', type=int, default=1)
parser.add_argument('--schema_specific', type=str2bool, default=False)
parser.add_argument('--num_eval_layers', type=int, default=1)
parser.add_argument('--num_digits_for_mnist', type=int, default=3)
parser.add_argument('--null_attention', type=str2bool, default=False)
args = parser.parse_args()

MIN_NUM_PATCHES = 0   # 16

# logging config

# if not os.path.isdir('logs'):
#    os.mkdir('logs')

# logging.basicConfig(filename='./logs/%s.log' % args.name,
#                        level=logging.DEBUG, format='%(asctime)s %(levelname)-10s %(message)s')

# logging.info("Using args: {}".format(args))

def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def mnist_acc(outputs, targets):
    outputs[outputs >= 0.5] = 1.
    outputs[outputs < 0.5] = 0.

    # print(outputs)
    # print(targets)

    equality = torch.eq(outputs, targets)

    equality = equality.int()

    # print(equality)
    print('-----')
    equality = equality.sum(1)
    equality[equality < num_classes] = 0
    equality[equality == num_classes] = 1

    correct = equality.sum().item()
    print(correct)

    return correct

def train(epoch):
    print('\nEpoch: %d' % epoch)
    # logging.info('Epoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        # print(inputs.shape)  # (64,3,32,32)
        # print(targets)  # (64,)
        optimizer.zero_grad()
        outputs = net(inputs)
        outputs = pre_loss_fn(outputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        scheduler.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        if args.data == "MNIST":
            correct += mnist_acc(outputs, targets)
        else:
            correct += predicted.eq(targets).sum().item()

        if batch_idx % 100 == 99:  # print every 100 mini-batches
            # net.net.enc.self_attn.relational_memory.print_log()
            print('[%d, %5d] loss: %.3f accuracy:%.3f' %
                  (epoch, batch_idx + 1, train_loss / (batch_idx + 1), 100. * correct / total))
            # logging.info('[%d, %5d] loss: %.3f accuracy:%.3f' %
            #     (epoch + 1, batch_idx + 1, train_loss / (batch_idx+1), 100.*correct/total))
    # summary_writer.add_scalars('Accuracy/train', {
    #     'train_acc': correct,
    #     'train_loss': train_loss
    # }, epoch)
    train_acc = 100. * correct / total
    return train_acc, train_loss


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            outputs = pre_loss_fn(outputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            if args.data == "MNIST":
                correct += mnist_acc(outputs, targets)
            else:
                correct += predicted.eq(targets).sum().item()
    # summary_writer.add_scalars('Accuracy/test', {
    #     'test_acc': correct,
    #     'test_loss': test_loss
    # }, epoch)

    # Save checkpoint.
    acc = 100. * correct / total
    print("test_accuracy is %.3f after epochs %d" % (acc, epoch))
    # logging.info("test_accuracy is %.3f after epochs %d"%(acc,epoch))
    if acc > best_acc:
        print('Saving..')
        # logging.info("==> Saving...")
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/' + args.name + '_ckpt.pth')
        best_acc = acc
    return acc

if __name__ == "__main__":
    seed_everything(seed=args.seed)

    image_size = 0
    num_classes = 0

    # logging.info("Loading data: {}".format(args.data))
    # 1.Processing the dataset
    if args.data == "cifar10":
        # settings from https://github.com/kuangliu/pytorch-cifar/blob/master/main.py

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                                  shuffle=True, num_workers=2, drop_last = True)

        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                               download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                                 shuffle=False, num_workers=2, drop_last = True)

        classes = ('plane', 'car', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        num_classes = 10
        image_size = 32
        channels = 3
    elif args.data == "cifar100":
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),  # 随机裁剪图像并调整大小为 224x224
            transforms.RandomHorizontalFlip(),  # 随机进行水平翻转
            transforms.ToTensor(),  # 转换为张量
            transforms.Normalize((0.5070, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761)),  # 归一化
        ])

        transform_test = transforms.Compose([
            transforms.Resize(256),  # 调整图像大小为 256x256
            transforms.CenterCrop(224),  # 中心裁剪图像为 224x224
            transforms.ToTensor(),  # 转换为张量
            transforms.Normalize((0.5070, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761)),  # 归一化
        ])

        trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                                 download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                                  shuffle=True, num_workers=2, drop_last = True)

        testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                                download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                                 shuffle=False, num_workers=2, drop_last = True)

        num_classes = 100
        image_size = 224
        channels = 3

        # 原来的大小
        # settings from https://github.com/weiaicunzai/pytorch-cifar100/blob/master/conf/global_settings.py

        # transform_train = transforms.Compose([
        #     transforms.RandomCrop(32, padding=4),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.RandomRotation(15),
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.5070, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761)),
        # ])
        #
        # transform_test = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.5070, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761)),
        # ])
        #
        # trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
        #                                          download=True, transform=transform_train)
        # trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
        #                                           shuffle=True, num_workers=2)
        #
        # testset = torchvision.datasets.CIFAR100(root='./data', train=False,
        #                                         download=True, transform=transform_test)
        # testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
        #                                          shuffle=False, num_workers=2)
        #
        # num_classes = 100
        # image_size = 32
        # channels = 3
    elif args.data == "pathfinder":
        trainset = np.load('./data/train.npz')
        trainset = torch.utils.data.TensorDataset(torch.Tensor(trainset['x']).reshape(-1, 1, 32, 32),
                                                  torch.LongTensor(trainset['y']))
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                                  shuffle=True, num_workers=2)
        testset = np.load('./data/test.npz')
        testset = torch.utils.data.TensorDataset(torch.Tensor(testset['x']).reshape(-1, 1, 32, 32),
                                                 torch.LongTensor(testset['y']))

        testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                                 shuffle=False, num_workers=2)
        num_classes = 2
        image_size = 32
        channels = 1
    elif args.data == 'MNIST':
        train_dataset = CountingMNISTDataset(split="train", path="MNIST", dig_range=[1, 3], num_examples=10000)
        test_dataset = CountingMNISTDataset(split="test", path="MNIST", dig_range=[4, 5], num_examples=2000)

        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=2, shuffle=False)
        testloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=2, shuffle=False)
        num_classes = 10
        image_size = 100
        channels = 1
    elif args.data == 'Triangle':
        train_dataset = TriangleDataset(num_examples=50000)
        test_dataset = TriangleDataset(num_examples=10000)
        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=2, shuffle=False)
        testloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=2, shuffle=False)
        num_classes = 4
        image_size = 64
        channels = 1
    print("Task is :{}".format(args.data))
    best_acc = 0  # best test accuracy
    start_epoch = 1  # start from epoch 0 or last checkpoint epoch

    # 2.Model configuration
    if args.model == "functional":
        print("using SetTransformer!!!!!!!!!!!!!")
        transformer = SetTransformer(args.h_dim, dim_hidden=args.h_dim, num_inds=args.mem_slots)
        # net = FunctionalVisionTransformer(
        #    image_size = image_size,
        #    patch_size = args.patch_size,
        #    num_classes = num_classes,
        #    dim = 1024,
        #    depth = args.num_layers,
        #    heads = args.num_heads,
        #    mlp_dim = 2048,
        #    dropout = args.dropout,
        #    emb_dropout = 0.1,
        #    num_templates = args.num_templates,
        #    version = args.version,
        #    channels=channels

        #    )
    elif args.model == "default":
        transformer = TransformerEncoder(
            args.h_dim,
            args.ffn_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            dropout=args.dropout,
            share_parameters=args.share_vanilla_parameters,
            shared_memory_attention=args.shared_memory_attention,
            use_topk=args.use_topk,
            topk=args.topk,
            mem_slots=args.mem_slots,
            use_long_men=args.use_long_men,
            long_mem_segs=args.long_mem_segs,
            long_mem_aggre=args.long_mem_aggre,
            use_wm_inference=args.use_wm_inference,
            null_attention=args.null_attention,
            num_steps=int((image_size * image_size) / (args.patch_size * args.patch_size) + 1))
        # net = ViT(
        #    image_size = image_size,
        #    patch_size = args.patch_size,
        #    num_classes = num_classes,
        #    dim = 1024,
        #    depth = args.num_layers,
        #    heads = args.num_heads,
        #    mlp_dim = 2048,
        #    dropout = args.dropout,
        #    emb_dropout = 0.1,
        #    channels=channels

        #    )
    elif args.model == "CNN_MLP":
        cnn_mlp = CNN_MLP(args)
        print("----------------------use CNN_MLP----------------------")
    elif args.model == "CNN_PMI":
        cnn_pmi = CNN_PMI(args)
        print("----------------------use CNN_PMI----------------------")
    elif args.model == "CNN_TRHSW":
        cnn_trhsw = CNN_TRHSW(args)
        print("----------------------use CNN_TRHSW----------------------")


    # print(int((image_size*image_size) / (args.patch_size * args.patch_size)))
    class model(nn.Module):
        def __init__(self, net, image_size, patch_size, num_classes):
            super().__init__()
            # print(image_size)
            # print(patch_size)
            assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
            num_patches = (image_size // patch_size) ** 2
            patch_dim = channels * patch_size ** 2
            assert num_patches > MIN_NUM_PATCHES, f'your number of patches ({num_patches}) is way too small for attention to be effective (at least 16). Try decreasing your patch size'

            self.net = net
            self.patch_size = patch_size
            # print(patch_dim)
            self.patch_to_embedding = nn.Linear(patch_dim, args.h_dim)
            self.cls_token = nn.Parameter(torch.randn(1, 1, args.h_dim))

            self.mlp_head = nn.Linear(args.h_dim, num_classes)

        def forward(self, img, mask=None):
            p = self.patch_size
            # print(img.size())
            x = rearrange(img.to(device), 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)
            # print(x.size())
            # print(x.type())
            x = self.patch_to_embedding(x.to(device))   # (48->256)

            b, n, _ = x.shape
            # print(x.shape)

            cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
            x = torch.cat((cls_tokens, x), dim=1)  # (64,65,256)
            # print(x.size())

            x = self.net(x)

            x = self.mlp_head(x[:, 0])

            return x

    if args.model == "CNN_MLP":
        net = cnn_mlp
    elif args.model == "CNN_PMI":
        net = cnn_pmi
    elif args.model == "CNN_TRHSW":
        net = cnn_trhsw
    else:
        net = model(transformer, image_size, args.patch_size, num_classes)
    net = net.to(device)

    # 3.load model
    if os.path.exists('./checkpoint/' + args.name + '_ckpt.pth'):
        args.resume = True

    if False and args.resume:
        # Load checkpoint.
        # logging.info("==> Resuming from checkpoint..")
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/' + args.name + '_ckpt.pth')
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    # 4.Calculating parameters
    pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    try:
        rmc_params = sum(p.numel() for p in net.net.enc.self_attn.relational_memory.parameters() if p.requires_grad)
        print(rmc_params)
    except:
        pass
    # logging.info("Total number of parameters:{}".format(pytorch_total_params))
    total_params_in_mb = pytorch_total_params / 1_000_000
    print("Total trainable parameters in MB:", total_params_in_mb)

    # 5.Setting loss function
    if args.data == 'MNIST':
        pre_loss_fn = nn.Sigmoid()
    else:
        pre_loss_fn = nn.Identity()

    if args.data == "MNIST":
        criterion = nn.BCELoss()
    else:
        criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(net.parameters(), lr=args.lr)
    optimizer = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.09)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=1, eta_min=0.0001)
    # warm_up = True
    # scheduler = WarmupScheduler(optimizer=optimizer,
    #                             steps=1 if warm_up else 0,
    #                             multiplier=0.1 if warm_up else 1)

    # 6.start training and testing
    # logging.info("Starting Training...")
    decay_done = False
    for epoch in range(start_epoch, start_epoch + args.epochs):
        train_acc, train_loss = train(epoch)
        test_acc = test(epoch)
        print(train_acc,test_acc)
        summary_writer.add_scalars('Accuracy/train_test', {
            'train_acc': train_acc,
            'test_acc': test_acc
        }, epoch)

        # if train_loss < 0.1 and not decay_done:
        #     scheduler.decay_lr(0.5)
        #     decay_done = True
        # scheduler.step()
