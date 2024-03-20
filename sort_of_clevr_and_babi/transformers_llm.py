import torch
import torch.nn as nn
#from transformer import TransformerEncoder
import types
import math
import numpy as np
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # 将1替换为要使用的GPU索引
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# SimpleNamespace为扩展版的字典
args = types.SimpleNamespace()
args.use_module_communication = 'true'
args.encoder_embed_dim = 512
args.encoder_attention_heads = 8 # was 8
args.attention_dropout = 0.1
args.topk_ratio = 1.0
args.dropout = 0.2
args.encoder_normalize_before = True
args.encoder_ffn_embed_dim = 2048
args.use_nfm = 'false'
args.shared_memory_attention = False
args.self_attention = True
args.mem_slots = 4
args.use_topk = False
args.topk = 3
args.num_steps = 5

from transformer_utilities.transformer_layer import TransformerEncoderLayer, TransformerEncoderLayerVanilla
from transformer_utilities.pos_enc import PositionEncoder
from transformer_utilities.GroupLinearLayer import GroupLinearLayer
import math


class SelectAttention(nn.Module):
    def __init__(self, d_read, d_write, d_k = 16, num_read = 5, num_write = 5, share_query = False, share_key = False):
        super(SelectAttention, self).__init__()
        '''
        num_read和num_write 设定了每个时间步要读/写几个向量
        share_query: 是否共享读操作的查询矩阵
        share_key: 是否共享写操作的键矩阵
        '''
        if not share_key:
            self.gll_write = GroupLinearLayer(d_write,d_k, num_write)
        else:
            self.gll_write = nn.Linear(d_write, d_k)

        if not share_query:
            self.gll_read = GroupLinearLayer(d_read,d_k, num_read)
        else:
            self.gll_read = nn.Linear(d_read, d_k)

        self.temperature = math.sqrt(d_k)

    def forward(self, q, k):
        read = self.gll_read(q)
        write = self.gll_write(k)
        return torch.bmm(read, write.permute(0, 2, 1)) / self.temperature

class TransformerEncoder(nn.Module):

    def __init__(self,
                 embed_dim,
                 ffn_dim,
                 n_token=1000,
                 num_layers = 6,
                 num_heads = 1,
                 dropout = 0.1,
                 functional = False,
                 shared_memory_attention = False,
                 shared_memory_percentage = 0.1,
                 share_parameters = False,
                 mem_slots = 4,
                 num_attention_schemas = 3,
                 num_gru_schemas = 3,
                 schema_specific = False,
                 use_topk = False,
                 topk = 3,
                 num_steps = 5,
                 use_long_men=True,
                 long_mem_segs=5,
                 long_mem_aggre=False,
                 null_attention = False,
                 regressive = False):
        super().__init__()

        if schema_specific and (num_gru_schemas != num_attention_schemas):
            print('Cannot use schema specific as num_gru_schemas != num_attention_schemas, continuing without')
            self.schema_specific = False
        else:
            self.schema_specific = schema_specific

        args.mem_slots = mem_slots
        args.encoder_embed_dim = embed_dim
        args.encoder_ffn_embed_dim = ffn_dim
        args.encoder_attention_heads = num_heads
        args.dropout = dropout
        args.shared_memory_attention = shared_memory_attention
        args.num_steps = num_steps
        args.null_attention = null_attention
        args.regressive = regressive
        args.use_long_men = use_long_men
        args.long_mem_segs = long_mem_segs
        args.long_mem_aggre = long_mem_aggre


        self.num_layers = num_layers
        self.shared_memory_attention = shared_memory_attention
        self.shared_memory_percentage = shared_memory_percentage

        print('transformer embed_dim', embed_dim)
        self.functional = functional
        print('functional? '+str(self.functional))
        if not self.functional:
            layer_lst = []
            args.use_topk = use_topk
            args.topk = topk


            args.encoder_embed_dim = embed_dim
            self.share_parameters = share_parameters
            if share_parameters:
                self.enc = TransformerEncoderLayerVanilla(args)
            else:
                layer_lst = []
                for i in range(self.num_layers):
                    layer_lst.append(TransformerEncoderLayerVanilla(args))
                    print('flmklsd')
                self.layers = nn.ModuleList(layer_lst)
        else:
            #args.encoder_embed_dim = inp_dim
            #print('init_layer initialize')
            #self.init_layer = TransformerEncoderLayerVanilla(args=args, out_proj=h_dim)
            print('NUM GRU SCHEMAS:' + str(num_gru_schemas))
            print('NUM Attention SCHEMAS:' + str(num_attention_schemas))
            print('SCHEMA SPECIFIC:' + str(self.schema_specific))
            args.use_topk = use_topk
            args.topk = topk
            print('inp_att initialize')
            self.num_gru_schemas = num_gru_schemas
            self.num_att_schemas = num_attention_schemas
            self.schema_stats = np.zeros(self.num_gru_schemas)
            args.self_attention = True
            # num_attention_schemas用于控制 nn.ModuleList 中 TransformerEncoderLayerVanilla模块的数量。
            self.inp_att =  nn.ModuleList([TransformerEncoderLayerVanilla(args=args) for _ in range(num_attention_schemas)])
            self.select_attention_inp_att = SelectAttention( args.encoder_embed_dim, args.encoder_embed_dim, num_read = 1, num_write = num_attention_schemas)
            print('gru initialize')
            hidden_dim = args.encoder_embed_dim


            self.gru_pool = nn.ModuleList([nn.GRUCell(hidden_dim, hidden_dim) for _ in range(num_gru_schemas)])
            #args.self_attention = True
            #self.state_att = TransformerEncoderLayerVanilla(args=args)
            self.select_attention = SelectAttention( hidden_dim + hidden_dim, hidden_dim, num_read = 1, num_write = num_gru_schemas)

        self.pe = PositionEncoder(args.encoder_embed_dim)
        self.pe_state = PositionEncoder(args.encoder_embed_dim)

    def forward(self, x, mask = None, num_layers = None):
        if not self.functional:
            # 1.functional=false,仅仅只有transformer的情况 TR+HSW走此路
            if self.shared_memory_attention:
                memory_size = int(self.shared_memory_percentage * x.size(0))
                # (2,64,256)
                memory = torch.randn(memory_size, 1, x.size(2)).repeat(1 ,x.size(1), 1).to(x.device)
            else:
                memory = None

            if self.shared_memory_attention:
                # 如果共享网络参数，那么就需要使用 self.enc 中的 self-attention 层来初始化记忆。
                if self.share_parameters:
                    if self.enc.self_attn.memory is not None:
                        # 初始化记忆
                        self.enc.self_attn.init_memory(x.size(1), x.size(0), x.device)#.memory = self.enc.self_attn.memory.detach()
                else:
                    # 如果不共享参数，就需要对每一层的 self-attention 层分别进行初始化。
                    for layer in self.layers:
                        if layer.self_attn.memory is not None:
                            layer.self_attn.init_memory(x.size(1), x.device) #.memory = layer.self_attn.memory.detach()

            for i in range(self.num_layers):   # 4层
                # 若共享参数
                if self.share_parameters:
                    x, memory = self.enc(x, mask, memory = memory)  # memory每一层num_layers都不变
                else:
                    # 进入transformer_layer.py的forward函数
                    x, memory = self.layers[i](x, mask, memory = memory)
            return x

        else:
            # 2.transformer+GRU的情况
            # functional = true 则使用包括Gumbel-Softmax注意力和GRU层在内的更复杂的计算
            T, B, D = x.size()  # T-time_step B-batch_size D-feature_dim
            if num_layers is None:
                num_layers = self.num_layers
            #state = self.pe_state(torch.randn(x.size()).to(x.device))

            if self.shared_memory_attention:
                memory_size = int(self.shared_memory_percentage * x.size(0))
                memory_inp = torch.randn(memory_size, 1, x.size(2)).repeat(1, x.size(1), 1).to(x.device)
                memory_state = torch.randn(memory_size, 1, x.size(2)).repeat(1, x.size(1), 1).to(x.device)
            else:
                memory_inp = None
                memory_state = None

            if self.shared_memory_attention:
                for inp_att in self.inp_att:
                    # self.inp_att 列表包含了编码器层中的所有自注意力机制子模块
                    if inp_att.self_attn.memory is not None:
                        inp_att.self_attn.init_memory(x.size(1), x.device) # memory = inp_att.self_attn.memory.detach()

            # 开始进入for循环
            for i in range(0, num_layers):
                # 1.求gru_in, 该张量即为当前层的 GRU 层的输入
                gru_ins = []
                for inp_att in self.inp_att:
                    # 进入 transformer_layer.py的forward() 返回的为hx(gru_in)和next_memory(memory_inp,在后面没有用到)
                    # gru_in为transformer一层中(一个完整的encoder_1模块)输出的隐藏状态hx
                    # x为问题和图片编码+位置编码向量
                    gru_in, memory_inp = inp_att(x, mask, memory = memory_inp)
                    gru_ins.append(gru_in.permute(1, 0, 2))

                gru_ins = torch.stack(gru_ins, dim = 2)
                gru_ins = gru_ins.reshape(B * T, -1, D)

                x = x.permute(1, 0, 2)
                x = x.reshape(B * T, -1).unsqueeze(1)

                attn_scores_inp_att = self.select_attention_inp_att(x, gru_ins)  # 求x与gru_ins的相似度
                attn_scores_inp_att = attn_scores_inp_att.squeeze(1)
                attn_scores_inp_att = torch.nn.functional.gumbel_softmax(attn_scores_inp_att, dim = 1, hard = True, tau = 0.5)
                attn_scores_inp_att = attn_scores_inp_att.unsqueeze(-1)

                gru_in = (gru_ins * attn_scores_inp_att).sum(dim = 1)  # gru_in = x与gru_ins的相似度再点乘gru_ins
                gru_in = gru_in.reshape(B, T, -1)
                gru_in = gru_in.reshape(B * T, -1)
                x = x.reshape(B, T, -1)
                x = x.reshape(B * T, -1)

                # 2.求gru_outs 即最终的return x = gru_outs_hidden
                gru_outs = []
                for gru in self.gru_pool:
                    # GRU处理地方   gru_in实际为经过transformer_layer.py后的隐状态hx的变形
                   gru_outs.append(gru(gru_in, x))
                gru_outs = torch.stack(gru_outs, dim = 1)
                # 求 attn_scores
                selector = torch.cat((gru_in, x), dim = 1).unsqueeze(1)
                if not self.schema_specific:
                    # schema_specific =false 决定如何求attn_scores与self.schema_stats
                    attn_scores = self.select_attention(selector, gru_outs)   #求cat(gru_in, x)与gru_outs的相似度
                    attn_scores = attn_scores.squeeze(1)
                    attn_scores = torch.nn.functional.gumbel_softmax(attn_scores, dim = 1, tau = 1.0, hard = True)
                    # attn_scores.clone().detach()的作用是创建一个新的张量，该张量具有与原始张量相同的值和形状，
                    # 但是从计算图中分离出来，因此它可以避免梯度传播到它上面。
                    att_argmax = torch.sum(attn_scores.clone().detach(), dim = 0).cpu().numpy()
                    self.schema_stats += att_argmax
                    attn_scores = attn_scores.unsqueeze(-1)
                else:
                    attn_scores = attn_scores_inp_att  # attn_scores_inp_att为x与gru_ins的相似度
                    att_argmax = torch.sum(attn_scores.squeeze(-1).clone().detach(), dim = 0).cpu().numpy()
                    self.schema_stats += att_argmax

                gru_outs = (gru_outs * attn_scores).sum(dim = 1)    # gru_outs = cat(gru_in, x)与gru_outs的相似度 再点乘gru_outs
                gru_outs_hidden = gru_outs.reshape(B, T, -1)
                gru_outs_hidden = gru_outs_hidden.permute(1, 0, 2)
                #gru_outs_hidden, memory_state = self.state_att(gru_outs_hidden, mask, memory = memory_state)
                #gru_in = gru_in.reshape(B, T, -1).permute(1, 0, 2)
                #x = gru_in
                x = gru_outs_hidden

            return x.permute(1,0,2)

    def print_schema_stats(self):
        total = np.sum(self.schema_stats)
        for k in range(self.schema_stats.shape[0]):
            print('schema ' + str(k) + ' used ' + str(self.schema_stats[k]) + ' out of ' + str(total) + ' times')


    def reset_schema_stats(self):
        self.schema_stats = np.zeros(self.num_gru_schemas)


if __name__ == "__main__":
    # x = torch.randn(8, 20, 256).cuda()
    x = torch.randn(8, 20, 256)
    import time
    # TE1 = TransformerEncoder(256, 512, num_layers = 1, functional = False, num_gru_schemas = 3, num_attention_schemas = 3, schema_specific = False, shared_memory_attention = True, mem_slots = 8, num_steps = 20).cuda()
    # TTT: shared_memory_attention=True  use-top-k share_vanilla_parameters  F:functional=False
    TE1 = TransformerEncoder(256, 512, num_layers=1, functional=False, num_gru_schemas=3, num_attention_schemas=3,
                             schema_specific=False, shared_memory_attention=True, mem_slots=8, num_steps=20)
    t1 = time.time()
    print(TE1)
    for i in range(5):
        x = TE1(x)
    print(time.time() - t1)

    # x = torch.randn(8, 20, 256).cuda()
    x = torch.randn(8, 20, 256)
    # TE2 = TransformerEncoder(256, 512, num_layers = 1, functional = False, num_gru_schemas = 3, num_attention_schemas = 3, schema_specific = True, shared_memory_attention = False, mem_slots = 8, num_steps = 20).cuda()
    TE2 = TransformerEncoder(256, 512, num_layers=1, functional=False, num_gru_schemas=3, num_attention_schemas=3,
                             schema_specific=True, shared_memory_attention=False, mem_slots=8, num_steps=20)
    t1 = time.time()
    for i in range(5):
        x = TE2(x)
    print(time.time() - t1)
