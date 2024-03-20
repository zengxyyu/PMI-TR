import os, sys
PATH_CUR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(PATH_CUR)
sys.path.append('../')
from argparse import ArgumentParser
import logging
import json
from pathlib import Path
import shutil
from tqdm import tqdm
from typing import Dict
import numpy as np
from tensorboardX import SummaryWriter
import random
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import argparse
from torch.utils.data import TensorDataset, DataLoader


from babi.data_preprocess.preprocess import parse
from sort_of_clevr_and_babi.baselines.sam import qamodel
from sort_of_clevr_and_babi.baselines.sam.utils import WarmupScheduler
from torch.autograd import Variable
from model import RN, CNN_MLP
from transformers import TransformerEncoder
from einops import rearrange, repeat
from transformer_utilities.set_transformer import SetTransformer
from sort_of_clevr_and_babi.baselines.sam.qamodel_zxy import InputModule
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
logger = logging.getLogger(__name__)

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


# 此处开始为新增内容
class Transformer(BasicModel):
    def __init__(self,  args, model_config):
        super(Transformer, self).__init__(args, 'Transformer')

        h_dim = args.embed_dim
        num_classes = model_config["vocab_size"]  # 88/10
        if args.functional:
            # 置换不变性 FFFT
            print('USING SET TRANSFORMER')
            self.net = SetTransformer(h_dim, dim_hidden = 512, num_inds = args.mem_slots)
        else:
            self.net = TransformerEncoder(
                            h_dim,
                            512,
                            num_layers = args.num_layers,
                            num_heads = 8,  # default=4
                            dropout = 0.1,  # # default=0.1
                            share_parameters = args.share_vanilla_parameters,
                            shared_memory_attention = args.shared_memory_attention,
                            use_topk = args.use_topk,
                            topk = args.topk,
                            mem_slots = args.mem_slots,
                            null_attention = args.null_attention,
                            num_steps = int(model_config["vocab_size"] + 1 + 18) )

        # self.patch_size = patch_size
        # print(patch_dim)
        self.word_embed = nn.Embedding(num_embeddings=model_config["vocab_size"],
                                       embedding_dim=model_config["symbol_size"])
        # self.patch_to_embedding = nn.Linear(64, h_dim)
        # self.question_to_embedding = nn.Linear(64, h_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, h_dim))

        if args.functional:
            self.mlp_head = nn.Linear(512, num_classes)
        else:
            self.mlp_head = nn.Linear(h_dim, num_classes)
        self.optimizer = optim.Adam(self.parameters(), lr=args.lr)
        # self.optimizer = optim.AdamW(self.parameters(), lr=args.lr, weight_decay=1e-4)
        self.input_module = InputModule(model_config)

    def forward(self, story, query):
        story_embed, query_embed = self.input_module(story, query)  # 维度(xx->64)
        # x = self.patch_to_embedding(story_embed).to(device)
        # q = self.question_to_embedding(query_embed).to(device)
        q= query_embed.unsqueeze(1)
        x = torch.cat((story_embed, q), dim = 1).to(device)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)  # (64,25+1+1,256)
        # 计算x 要么为transformer的输出，要么为transformer+GRU的输出
        x = self.net(x)   # 输出的x为(64,27,256)
        x = F.log_softmax(self.mlp_head(x[:,0]), dim = 1)  # (64,10)
        return x


def train(args, force: bool = False) -> None:
    # Create serialization dir
    dir_path = Path(args.serialization_path)
    print(dir_path)
    if dir_path.exists() and force:
        shutil.rmtree(dir_path)
    if not dir_path.exists():
        dir_path.mkdir(parents=True, exist_ok=False)
    model_path = dir_path / "model.pt"
    config_path = dir_path / "config.json"
    writer = SummaryWriter(log_dir=str(dir_path))

    # Read config
    with open(args.config_file, "r") as fp:
        config = json.load(fp)
    data_config = config["data"]
    trainer_config = config["trainer"]
    model_config = config["model"]
    optimizer_config = config["optimizer"]
    # Load data
    if data_config["task-id"]=="all":
        task_ids = range(1,21)
    else:
        task_ids = [data_config["task-id"]]
    word2id = None
    train_data_loaders = {}
    valid_data_loaders = {}
    test_data_loaders = {}

    num_train_batches = num_valid_batches = num_test_batches = 0
    max_seq = 0
    for i in task_ids:
        train_raw_data, valid_raw_data, test_raw_data, word2id = parse(data_config["data_path"],
                                                                       str(i), word2id=word2id,
                                                                       use_cache=True, cache_dir_ext="")
        # train_epoch_size = train_raw_data[0].shape[0]
        # valid_epoch_size = valid_raw_data[0].shape[0]
        # test_epoch_size = test_raw_data[0].shape[0]
        #
        # max_story_length = np.max(train_raw_data[1])
        # max_sentences = train_raw_data[0].shape[1]
        max_seq = max(max_seq, train_raw_data[0].shape[2])
        # max_q = train_raw_data[0].shape[1]
        # valid_batch_size = valid_epoch_size // 73  # like in the original implementation
        # test_batch_size = test_epoch_size // 73
        valid_batch_size = trainer_config["batch_size"]  # like in the original implementation
        test_batch_size = trainer_config["batch_size"]



        train_dataset = TensorDataset(*[torch.LongTensor(a) for a in train_raw_data[:-1]])
        valid_dataset = TensorDataset(*[torch.LongTensor(a) for a in valid_raw_data[:-1]])
        test_dataset = TensorDataset(*[torch.LongTensor(a) for a in test_raw_data[:-1]])

        train_data_loader = DataLoader(train_dataset, batch_size=trainer_config["batch_size"], shuffle=True, drop_last=True)
        valid_data_loader = DataLoader(valid_dataset, batch_size=valid_batch_size, drop_last=True)
        test_data_loader = DataLoader(test_dataset, batch_size=test_batch_size, drop_last=True)

        train_data_loaders[i] = [iter(train_data_loader), train_data_loader]
        valid_data_loaders[i] = valid_data_loader
        test_data_loaders[i] = test_data_loader

        num_train_batches += len(train_data_loader)
        num_valid_batches += len(valid_data_loader)
        num_test_batches += len(test_data_loader)

    print(f"total train data: {num_train_batches*trainer_config['batch_size']}")
    print(f"total valid data: {num_valid_batches*valid_batch_size}")
    print(f"total test data: {num_test_batches*test_batch_size}")
    print(f"voca size {len(word2id)}")

    model_config["vocab_size"] = len(word2id)
    model_config["max_seq"] = max_seq
    model_config["symbol_size"] =  args.embed_dim # 原来为64

    # Create model
    # model = qamodel.QAmodel(model_config).to(device)
    model = Transformer(args, model_config).to(device)
    print(model)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # optimizer = optim.Adam(model.parameters(),
    #                        lr=optimizer_config["lr"], betas=(optimizer_config["beta1"], optimizer_config["beta2"]))
    # optimizer = Nadam(model.parameters(),
    #                        lr=optimizer_config["lr"], betas=(optimizer_config["beta1"], optimizer_config["beta2"]))

    # optimizer = optim.RMSprop(model.parameters(), lr=1e-4, momentum=0.9)

    loss_fn = nn.CrossEntropyLoss(reduction='none')
    warm_up = optimizer_config.get("warm_up", False)

    scheduler = WarmupScheduler(optimizer=optimizer,
                                steps=optimizer_config["warm_up_steps"] if warm_up else 0,
                                multiplier=optimizer_config["warm_up_factor"] if warm_up else 1)

    decay_done = False
    max_acc = 0

    with config_path.open("w") as fp:
        json.dump(config, fp, indent=4)
    # if args.eval_test:
    #     print(f"testing ... load {model_path.absolute()}")
    #     # model.load_state_dict(torch.load(model_path.absolute()))
    #     model.load_state_dict(torch.load(model_path.absolute(), map_location=device))
    #
    #     # Evaluation on test data
    #     model.eval()
    #     correct = 0
    #     test_loss = 0
    #     with torch.no_grad():
    #         total_test_samples = 0
    #         single_task_acc = [0] * len(test_data_loaders)
    #         for k, te in test_data_loaders.items():
    #             test_data_loader = te
    #             task_acc = 0
    #             single_task_samples = 0
    #             for story, story_length, query, answer in tqdm(test_data_loader):
    #                 logits = model(story.to(device), query.to(device))
    #                 answer = answer.to(device)
    #                 correct_batch = (torch.argmax(logits, dim=-1) == answer).sum()
    #                 correct += correct_batch.item()
    #                 task_acc += correct_batch.item()
    #                 loss = loss_fn(logits, answer)
    #                 test_loss += loss.sum().item()
    #                 total_test_samples+=story.shape[0]
    #                 single_task_samples+=story.shape[0]
    #             print(f"validate acc task {k}: {task_acc/single_task_samples}")
    #             single_task_acc[k - 1] = task_acc/single_task_samples
    #         test_acc = correct / total_test_samples
    #         test_loss = test_loss / total_test_samples
    #     print(f"Test accuracy: {test_acc:.3f}, loss: {test_loss:.3f}")
    #     print(f"test avg: {np.mean(single_task_acc)}")
    #     raise True

    for i in range(trainer_config["epochs"]):
        logging.info(f"##### EPOCH: {i} #####")
        # 1.Train
        model.train()
        correct = 0
        train_loss = 0
        for _ in tqdm(range(num_train_batches)):
            if len(train_data_loaders) == 1:
                # 获取字典中唯一的键
                loader_i = next(iter(train_data_loaders))
                try:
                    story, story_length, query, answer = next(train_data_loaders[loader_i][0])
                except StopIteration:
                    train_data_loaders[loader_i][0] = iter(train_data_loaders[loader_i][1])
                    story, story_length, query, answer = next(train_data_loaders[loader_i][0])
            else:
                loader_i = random.randint(0,len(train_data_loaders)-1)+1
                try:
                    story, story_length, query, answer = next(train_data_loaders[loader_i][0])
                except StopIteration:
                    train_data_loaders[loader_i][0] = iter(train_data_loaders[loader_i][1])
                    story, story_length, query, answer = next(train_data_loaders[loader_i][0])
            optimizer.zero_grad()

            logits = model(story.to(device), query.to(device))  # logits(B,88) story(B,10,40)  query(B,40)
            answer = answer.to(device)  # (128,)
            correct_batch = (torch.argmax(logits, dim=-1) == answer).sum()
            correct += correct_batch.item()

            loss = loss_fn(logits, answer)   # logits(B,88)
            train_loss += loss.sum().item()
            loss = loss.mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), optimizer_config["max_gradient_norm"])
            # nn.utils.clip_grad_value_(model.parameters(), 10)

            optimizer.step()
            scheduler.step()

        train_acc = correct / (num_train_batches*trainer_config["batch_size"])
        train_loss = train_loss / (num_train_batches*trainer_config["batch_size"])

        # 2.Validation
        model.eval()
        correct = 0
        valid_loss = 0
        with torch.no_grad():
            total_valid_samples = 0
            for k, va in valid_data_loaders.items():
                valid_data_loader = va
                task_acc = 0
                single_valid_samples = 0
                for story, story_length, query, answer in valid_data_loader:
                    logits = model(story.to(device), query.to(device))
                    answer = answer.to(device)
                    correct_batch = (torch.argmax(logits, dim=-1) == answer).sum()
                    correct += correct_batch.item()
                    loss = loss_fn(logits, answer)
                    valid_loss += loss.sum().item()
                    task_acc += correct_batch.item()
                    total_valid_samples+= story.shape[0]
                    single_valid_samples+= story.shape[0]
                print(f"validate acc task {k}: {task_acc/single_valid_samples}")
            valid_acc = correct / total_valid_samples
            valid_loss = valid_loss / total_valid_samples
            if valid_acc>max_acc:
                print(f"saved model...{model_path}")
                torch.save(model.state_dict(), model_path.absolute())
                max_acc = valid_acc

        # 3.test
        # print(f"testing ... load {model_path.absolute()}")
        # model.load_state_dict(torch.load(model_path.absolute()))
        # Evaluation on test data
        model.eval()
        correct = 0
        test_loss = 0
        with torch.no_grad():
            total_test_samples = 0
            single_task_acc = [0] * len(test_data_loaders)
            for k, te in test_data_loaders.items():
                test_data_loader = te
                task_acc = 0
                single_task_samples = 0
                # for story, story_length, query, answer in tqdm(test_data_loader):
                for story, story_length, query, answer in test_data_loader:
                    logits = model(story.to(device), query.to(device))
                    answer = answer.to(device)
                    correct_batch = (torch.argmax(logits, dim=-1) == answer).sum()
                    correct += correct_batch.item()
                    task_acc += correct_batch.item()
                    loss = loss_fn(logits, answer)
                    test_loss += loss.sum().item()
                    total_test_samples += story.shape[0]
                    single_task_samples += story.shape[0]
                print(f"test acc task {k}: {task_acc / single_task_samples}")
                if len(test_data_loaders) == 1:
                    single_task_acc[0] = task_acc / single_task_samples
                else:
                    single_task_acc[k - 1] = task_acc / single_task_samples
            test_acc = correct / total_test_samples
            test_loss = test_loss / total_test_samples
        # print(f"Test accuracy: {test_acc:.3f}, loss: {test_loss:.3f}")
        # print(f"test avg: {np.mean(single_task_acc)}")
        # raise True

        writer.add_scalars("accuracy", {"train": train_acc,
                                        "validation": valid_acc,"test": test_acc}, i)
        writer.add_scalars("loss", {"train": train_loss,
                                    "validation": valid_loss, "test": test_loss}, i)

        logging.info(f"\nTrain accuracy: {train_acc:.3f}, loss: {train_loss:.3f}"
                     f"\nValid accuracy: {valid_acc:.3f}, loss: {valid_loss:.3f}"
                     f"\nTest accuracy: {test_acc:.3f}, loss: {test_loss:.3f}")
        if optimizer_config.get("decay", False) and valid_loss < optimizer_config["decay_thr"] and not decay_done:
            scheduler.decay_lr(optimizer_config["decay_factor"])
            decay_done = True

    # 计算可训练参数的总数量
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params_in_mb = total_params / 1_000_000
    print("Total trainable parameters in MB:", total_params_in_mb)
    writer.close()

def test(args, force: bool = False) -> None:
    # Create serialization dir
    dir_path = Path(args.serialization_path)
    print(dir_path)
    if dir_path.exists() and force:
        shutil.rmtree(dir_path)
    if not dir_path.exists():
        dir_path.mkdir(parents=True, exist_ok=False)
    model_path = dir_path / "model.pt"
    config_path = dir_path / "config.json"
    writer = SummaryWriter(log_dir=str(dir_path))

    # Read config
    with open(args.config_file, "r") as fp:
        config = json.load(fp)
    data_config = config["data"]
    trainer_config = config["trainer"]
    model_config = config["model"]
    # Load data
    if data_config["task-id"]=="all":
        task_ids = range(1,21)
    else:
        task_ids = [data_config["task-id"]]
    word2id = None
    test_data_loaders = {}

    num_test_batches = 0
    max_seq = 0
    for i in task_ids:
        train_raw_data, valid_raw_data, test_raw_data, word2id = parse(data_config["data_path"],
                                                                       str(i), word2id=word2id,
                                                                       use_cache=True, cache_dir_ext="")
        max_seq = max(max_seq, train_raw_data[0].shape[2])
        test_batch_size = trainer_config["batch_size"]
        test_dataset = TensorDataset(*[torch.LongTensor(a) for a in test_raw_data[:-1]])
        test_data_loader = DataLoader(test_dataset, batch_size=test_batch_size, drop_last=True)
        test_data_loaders[i] = test_data_loader
        num_test_batches += len(test_data_loader)

    print(f"total test data: {num_test_batches*test_batch_size}")
    print(f"voca size {len(word2id)}")

    model_config["vocab_size"] = len(word2id)
    model_config["max_seq"] = max_seq
    model_config["symbol_size"] = args.embed_dim # 原来为64
    # Create model
    model = Transformer(args, model_config).to(device)
    print(model)

    loss_fn = nn.CrossEntropyLoss(reduction='none')

    with config_path.open("w") as fp:
        json.dump(config, fp, indent=4)
    # if args.eval_test:
    #     print(f"testing ... load {model_path.absolute()}")
    #     # model.load_state_dict(torch.load(model_path.absolute()))
    #     model.load_state_dict(torch.load(model_path.absolute(), map_location=device))
    #
    #     # Evaluation on test data
    #     model.eval()
    #     correct = 0
    #     test_loss = 0
    #     with torch.no_grad():
    #         total_test_samples = 0
    #         single_task_acc = [0] * len(test_data_loaders)
    #         for k, te in test_data_loaders.items():
    #             test_data_loader = te
    #             task_acc = 0
    #             single_task_samples = 0
    #             for story, story_length, query, answer in tqdm(test_data_loader):
    #                 logits = model(story.to(device), query.to(device))
    #                 answer = answer.to(device)
    #                 correct_batch = (torch.argmax(logits, dim=-1) == answer).sum()
    #                 correct += correct_batch.item()
    #                 task_acc += correct_batch.item()
    #                 loss = loss_fn(logits, answer)
    #                 test_loss += loss.sum().item()
    #                 total_test_samples+=story.shape[0]
    #                 single_task_samples+=story.shape[0]
    #             print(f"validate acc task {k}: {task_acc/single_task_samples}")
    #             single_task_acc[k - 1] = task_acc/single_task_samples
    #         test_acc = correct / total_test_samples
    #         test_loss = test_loss / total_test_samples
    #     print(f"Test accuracy: {test_acc:.3f}, loss: {test_loss:.3f}")
    #     print(f"test avg: {np.mean(single_task_acc)}")
    #     raise True
    # 3.test
    print(f"testing ... load {model_path.absolute()}")
    model.load_state_dict(torch.load(model_path.absolute()))
    # state_dict = torch.load(model_path.absolute(), map_location=torch.device('cpu'))
    # model.load_state_dict(state_dict)
    model.eval()
    correct = 0
    test_loss = 0
    with torch.no_grad():
        total_test_samples = 0
        single_task_acc = [0] * len(test_data_loaders)
        for k, te in test_data_loaders.items():
            test_data_loader = te
            task_acc = 0
            single_task_samples = 0
            # for story, story_length, query, answer in tqdm(test_data_loader):
            for story, story_length, query, answer in test_data_loader:
                logits = model(story.to(device), query.to(device))
                answer = answer.to(device)
                correct_batch = (torch.argmax(logits, dim=-1) == answer).sum()
                correct += correct_batch.item()
                task_acc += correct_batch.item()
                loss = loss_fn(logits, answer)
                test_loss += loss.sum().item()
                total_test_samples += story.shape[0]
                single_task_samples += story.shape[0]
            print(f"test acc task {k}: {task_acc / single_task_samples}")
            single_task_acc[k - 1] = task_acc / single_task_samples
        test_acc = correct / total_test_samples
        test_loss = test_loss / total_test_samples
    logging.info(f"\nTest accuracy: {test_acc:.5f}, loss: {test_loss:.3f}")
    test_error = 100 * (1-test_acc)
    print(f"\nTest Error(%): {test_error:.3f}")
    # 计算可训练参数的总数量
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params_in_mb = total_params / 1_000_000
    print("Total trainable parameters in MB:", total_params_in_mb)
    writer.close()

# def train(config: Dict[str, Dict],
#           serialization_path: str,
#           eval_test: bool = False,
#           force: bool = False) -> None:
#     # Create serialization dir
#     dir_path = Path(serialization_path)
#     print(dir_path)
#     if dir_path.exists() and force:
#         shutil.rmtree(dir_path)
#     if not dir_path.exists():
#         dir_path.mkdir(parents=True, exist_ok=False)
#     model_path = dir_path / "model.pt"
#     config_path = dir_path / "config.json"
#     writer = SummaryWriter(log_dir=str(dir_path))
#
#     # Read config
#     data_config = config["data"]
#     trainer_config = config["trainer"]
#     model_config = config["model"]
#     optimizer_config = config["optimizer"]
#     # Load data
#     if data_config["task-id"]=="all":
#         task_ids = range(1,21)
#     else:
#         task_ids = [data_config["task-id"]]
#     # train_raw_data, valid_raw_data, test_raw_data, word2id = parse_all(data_config["data_path"],list(range(1,21)))
#     word2id = None
#     train_data_loaders = {}
#     valid_data_loaders = {}
#     test_data_loaders = {}
#
#     num_train_batches = num_valid_batches = num_test_batches = 0
#     max_seq = 0
#     for i in task_ids:
#         train_raw_data, valid_raw_data, test_raw_data, word2id = parse(data_config["data_path"],
#                                                                        str(i), word2id=word2id,
#                                                                        use_cache=True, cache_dir_ext="")
#         train_epoch_size = train_raw_data[0].shape[0]
#         valid_epoch_size = valid_raw_data[0].shape[0]
#         test_epoch_size = test_raw_data[0].shape[0]
#
#         max_story_length = np.max(train_raw_data[1])
#         max_sentences = train_raw_data[0].shape[1]
#         max_seq = max(max_seq, train_raw_data[0].shape[2])
#         max_q = train_raw_data[0].shape[1]
#         valid_batch_size = valid_epoch_size // 73  # like in the original implementation
#         test_batch_size = test_epoch_size // 73
#
#
#
#         train_dataset = TensorDataset(*[torch.LongTensor(a) for a in train_raw_data[:-1]])
#         valid_dataset = TensorDataset(*[torch.LongTensor(a) for a in valid_raw_data[:-1]])
#         test_dataset = TensorDataset(*[torch.LongTensor(a) for a in test_raw_data[:-1]])
#
#         train_data_loader = DataLoader(train_dataset, batch_size=trainer_config["batch_size"], shuffle=True)
#         valid_data_loader = DataLoader(valid_dataset, batch_size=valid_batch_size)
#         test_data_loader = DataLoader(test_dataset, batch_size=test_batch_size)
#
#         train_data_loaders[i] = [iter(train_data_loader), train_data_loader]
#         valid_data_loaders[i] = valid_data_loader
#         test_data_loaders[i] = test_data_loader
#
#         num_train_batches += len(train_data_loader)
#         num_valid_batches += len(valid_data_loader)
#         num_test_batches += len(test_data_loader)
#
#     print(f"total train data: {num_train_batches*trainer_config['batch_size']}")
#     print(f"total valid data: {num_valid_batches*valid_batch_size}")
#     print(f"total test data: {num_test_batches*test_batch_size}")
#     print(f"voca size {len(word2id)}")
#
#     model_config["vocab_size"] = len(word2id)
#     model_config["max_seq"] = max_seq
#     model_config["symbol_size"] = 64
#     # Create model
#     # model = qamodel.QAmodel(model_config).to(device)
#     model = Transformer(args).to(device)
#     print(model)
#     optimizer = optim.Adam(model.parameters(),
#                            lr=optimizer_config["lr"], betas=(optimizer_config["beta1"], optimizer_config["beta2"]))
#     # optimizer = Nadam(model.parameters(),
#     #                        lr=optimizer_config["lr"], betas=(optimizer_config["beta1"], optimizer_config["beta2"]))
#
#     # optimizer = optim.RMSprop(model.parameters(), lr=1e-4, momentum=0.9)
#
#     loss_fn = nn.CrossEntropyLoss(reduction='none')
#     warm_up = optimizer_config.get("warm_up", False)
#
#     scheduler = WarmupScheduler(optimizer=optimizer,
#                                 steps=optimizer_config["warm_up_steps"] if warm_up else 0,
#                                 multiplier=optimizer_config["warm_up_factor"] if warm_up else 1)
#
#     decay_done = False
#     max_acc = 0
#
#     with config_path.open("w") as fp:
#         json.dump(config, fp, indent=4)
#     if eval_test:
#         print(f"testing ... load {model_path.absolute()}")
#         model.load_state_dict(torch.load(model_path.absolute()))
#         # Evaluation on test data
#         model.eval()
#         correct = 0
#         test_loss = 0
#         with torch.no_grad():
#             total_test_samples = 0
#             single_task_acc = [0] * len(test_data_loaders)
#             for k, te in test_data_loaders.items():
#                 test_data_loader = te
#                 task_acc = 0
#                 single_task_samples = 0
#                 for story, story_length, query, answer in tqdm(test_data_loader):
#                     logits = model(story.to(device), query.to(device))
#                     answer = answer.to(device)
#                     correct_batch = (torch.argmax(logits, dim=-1) == answer).sum()
#                     correct += correct_batch.item()
#                     task_acc += correct_batch.item()
#                     loss = loss_fn(logits, answer)
#                     test_loss += loss.sum().item()
#                     total_test_samples+=story.shape[0]
#                     single_task_samples+=story.shape[0]
#                 print(f"validate acc task {k}: {task_acc/single_task_samples}")
#                 single_task_acc[k - 1] = task_acc/single_task_samples
#             test_acc = correct / total_test_samples
#             test_loss = test_loss / total_test_samples
#         print(f"Test accuracy: {test_acc:.3f}, loss: {test_loss:.3f}")
#         print(f"test avg: {np.mean(single_task_acc)}")
#         raise True
#
#     for i in range(trainer_config["epochs"]):
#         logging.info(f"##### EPOCH: {i} #####")
#         # Train
#         model.train()
#         correct = 0
#         train_loss = 0
#         for _ in tqdm(range(num_train_batches)):
#             loader_i = random.randint(0,len(train_data_loaders)-1)+1
#             try:
#                 story, story_length, query, answer = next(train_data_loaders[loader_i][0])
#             except StopIteration:
#                 train_data_loaders[loader_i][0] = iter(train_data_loaders[loader_i][1])
#                 story, story_length, query, answer = next(train_data_loaders[loader_i][0])
#             optimizer.zero_grad()
#
#             logits = model(story.to(device), query.to(device))  # logits(B,88) story(B,10,40)  query(B,40)
#             answer = answer.to(device)  # (128,)
#             correct_batch = (torch.argmax(logits, dim=-1) == answer).sum()
#             correct += correct_batch.item()
#
#             loss = loss_fn(logits, answer)   # logits(B,88)
#             train_loss += loss.sum().item()
#             loss = loss.mean()
#             loss.backward()
#             nn.utils.clip_grad_norm_(model.parameters(), optimizer_config["max_gradient_norm"])
#             # nn.utils.clip_grad_value_(model.parameters(), 10)
#
#             optimizer.step()
#             scheduler.step()
#
#         train_acc = correct / (num_train_batches*trainer_config["batch_size"])
#         train_loss = train_loss / (num_train_batches*trainer_config["batch_size"])
#
#         # Validation
#         model.eval()
#         correct = 0
#         valid_loss = 0
#         with torch.no_grad():
#             total_valid_samples = 0
#             for  k, va in valid_data_loaders.items():
#                 valid_data_loader = va
#                 task_acc = 0
#                 single_valid_samples = 0
#                 for story, story_length, query, answer in valid_data_loader:
#                     logits = model(story.to(device), query.to(device))
#                     answer = answer.to(device)
#                     correct_batch = (torch.argmax(logits, dim=-1) == answer).sum()
#                     correct += correct_batch.item()
#                     loss = loss_fn(logits, answer)
#                     valid_loss += loss.sum().item()
#                     task_acc += correct_batch.item()
#                     total_valid_samples+= story.shape[0]
#                     single_valid_samples+= story.shape[0]
#                 print(f"validate acc task {k}: {task_acc/single_valid_samples}")
#             valid_acc = correct / total_valid_samples
#             valid_loss = valid_loss / total_valid_samples
#             if valid_acc>max_acc:
#                 print(f"saved model...{model_path}")
#                 torch.save(model.state_dict(), model_path.absolute())
#                 max_acc = valid_acc
#
#         writer.add_scalars("accuracy", {"train": train_acc,
#                                         "validation": valid_acc}, i)
#         writer.add_scalars("loss", {"train": train_loss,
#                                     "validation": valid_loss}, i)
#
#         logging.info(f"\nTrain accuracy: {train_acc:.3f}, loss: {train_loss:.3f}"
#                      f"\nValid accuracy: {valid_acc:.3f}, loss: {valid_loss:.3f}")
#         if optimizer_config.get("decay", False) and valid_loss < optimizer_config["decay_thr"] and not decay_done:
#             scheduler.decay_lr(optimizer_config["decay_factor"])
#             decay_done = True
#
#
#     writer.close()

def str2bool(v):
    """Method to map string to bool for argument parser"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":
    parser = ArgumentParser(description="Training script.")
    parser.add_argument("--config-file", type=str, metavar='PATH', default="./babi/configs/config_all.json",
                        help="Path to the model config file")
    parser.add_argument("--serialization-path", type=str, metavar='PATH', default="./Params_88_10_7_9_0.0002_topk5_PMITR/",
                        help="Serialization directory path")
    parser.add_argument("--eval-test", default=False, action='store_true',
                        help="Whether to eval model on test dataset after training (default: False)")
    parser.add_argument("--logging-level", type=str, metavar='LEVEL', default=20, choices=range(10, 51, 10),
                        help="Logging level (default: 20)")
    parser.add_argument('--model', type=str, choices=['RN', 'CNN_MLP', 'Transformer'], default='Transformer',
                        help='resume from model stored')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: 20)')
    parser.add_argument('--lr', type=float, default=0.0002, metavar='LR',
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
    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=8)
    parser.add_argument('--num_heads', type=int, default=4)
    # 层之间参数是否共享，TR+HC和ISAB 不会共享，其他都是共享
    parser.add_argument('--share_vanilla_parameters', type=str2bool, default=True)  # default=False
    parser.add_argument('--use_topk', type=str2bool, default=False)  # default=False
    parser.add_argument('--topk', type=int, default=5)  # default=3
    parser.add_argument('--shared_memory_attention', type=str2bool, default=True)  # default=False
    parser.add_argument('--mem_slots', type=int, default=8)  # default=4
    parser.add_argument('--use_long_men', type=str2bool, default=True,
                        help='ues long-term memory or not')
    parser.add_argument('--long_mem_segs', type=int, default=5)
    parser.add_argument('--long_mem_aggre', type=str2bool, default=False,
                        help='uses cross-attention between workspace and LTM or not')
    parser.add_argument('--use_wm_inference', type=str2bool, default=False,
                        help='WM involvement during inference or not')
    parser.add_argument('--seed', type=int, default=1)  # default=0
    parser.add_argument('--functional', type=str2bool, default=False,
                        help='ues set_transformer or not')  # default=False

    parser.add_argument('--save_dir', type=str, default='model_zxycuda')
    parser.add_argument('--null_attention', type=str2bool, default=False)

    # 3.使用 parse_args() 解析添加的参数
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(args.seed)
    logging.basicConfig(level=args.logging_level)

    # with open(args.config_file, "r") as fp:
    #     config = json.load(fp)

    # train(config, args.serialization_path, args.eval_test)
    if args.eval_test:
        test(args)
    else:
        train(args)