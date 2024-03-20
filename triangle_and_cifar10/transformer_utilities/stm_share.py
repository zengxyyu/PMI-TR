import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 将1替换为要使用的GPU索引
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def op_att(q, k, v):
    qq = q.unsqueeze(2).repeat(1, 1, k.shape[1], 1)
    kk = k.unsqueeze(1).repeat(1, q.shape[1], 1, 1)
    # BxNXNxd_kq BxNxNxd_v --> BxNXNxd_kqxd_v
    output = torch.matmul(torch.tanh(qq*kk).unsqueeze(4), v.unsqueeze(1).repeat(1, q.shape[1], 1, 1).unsqueeze(3))
    # print(output.shape)
    output = torch.sum(output, dim=2)  # BxNxd_kqxd_v
    # print(output.shape)
    return output

def sdp_att(q,k,v):
    dot_product = torch.matmul(q, k.permute(0, 2, 1))
    weights = F.softmax(dot_product, dim=-1)

    # output is [B, H, N, V]
    output = torch.matmul(weights, v)
    return output

class MLP(nn.Module):
    def __init__(self, in_dim=28*28,  out_dim=10, hid_dim=-1, layers=1):
        super(MLP, self).__init__()
        self.layers = layers
        if hid_dim<=0:
            self.layers=-1
        if self.layers<0:
            hid_dim=out_dim
        self.fc1 = nn.Linear(in_dim, hid_dim)
        # linear layer (n_hidden -> hidden_2)
        if self.layers>0:
            self.fc2h = nn.ModuleList([nn.Linear(hid_dim, hid_dim)]*self.layers)
        # linear layer (n_hidden -> 10)
        if self.layers>=0:
            self.fc3 = nn.Linear(hid_dim, out_dim)

    def forward(self, x):
        o = self.fc1(x)
        if self.layers>0:
            for l in range(self.layers):
                o = self.fc2h[l](o)
        if self.layers >= 0:
            o = self.fc3(o)
        return o

class STM(nn.Module):
    def __init__(self, input_size, output_size, step = 1, num_slot=8,
                 mlp_size = 128, slot_size = 96, rel_size = 96,
                 out_att_size=64, rd=True,
                 init_alphas=[None,None,None],
                 learn_init_mem=True, mlp_hid=-1, num_heads=4, topk=3):
        super(STM, self).__init__()
        self.mlp_size = mlp_size
        self.slot_size = slot_size
        self.rel_size = rel_size
        self.rnn_hid = slot_size
        self.num_slot = num_slot
        self.step = step
        self.rd = rd
        self.learn_init_mem = learn_init_mem
        self.num_heads = num_heads  # zxy
        self.head_dim = slot_size // num_heads
        self.value_size = self.head_dim
        self.key_size = 32
        self.null_attention = False
        self.use_topk = True
        use_topk_ = True
        self.topk = topk

        self.out_att_size = out_att_size

        # self.qkv_projector = nn.ModuleList([nn.Linear(slot_size, num_slot*3)]*step)
        self.qkv_projector = nn.ModuleList([nn.Linear(num_slot, num_slot * 3)] * step)
        # self.qkv_layernorm = nn.ModuleList([nn.LayerNorm([num_slot, num_slot * 3])] * step)
        self.qkv_layernorm = nn.ModuleList([nn.LayerNorm([slot_size, num_slot*3])]*step)
        self.query_proj = nn.Linear(self.slot_size, self.key_size * self.num_heads)
        self.key_proj = nn.Linear(self.slot_size, self.key_size * self.num_heads)
        self.value_proj = nn.Linear(self.slot_size, self.value_size * self.num_heads)

        if init_alphas[0] is None:
            self.alpha1 = [nn.Parameter(torch.zeros(1))] * step
            for ia, a in enumerate(self.alpha1):
                setattr(self, 'alpha1' + str(ia), self.alpha1[ia])
        else:
            self.alpha1 = [init_alphas[0]]* step

        if init_alphas[1] is None:
            self.alpha2 = [nn.Parameter(torch.zeros(1))] * step
            for ia, a in enumerate(self.alpha2):
                setattr(self, 'alpha2' + str(ia), self.alpha2[ia])
        else:
            self.alpha2 = [init_alphas[1]] * step

        if init_alphas[2] is None:
            self.alpha3 = [nn.Parameter(torch.zeros(1))] * step
            for ia, a in enumerate(self.alpha3):
                setattr(self, 'alpha3' + str(ia), self.alpha3[ia])
        else:
            self.alpha3 = [init_alphas[2]] * step


        self.input_projector = MLP(input_size, slot_size, hid_dim=mlp_hid)
        self.input_projector2 = MLP(input_size, slot_size, hid_dim=mlp_hid)
        self.input_projector3 = MLP(input_size, num_slot, hid_dim=mlp_hid)


        self.input_gate_projector = nn.Linear(self.slot_size, self.slot_size*2)
        self.memory_gate_projector = nn.Linear(self.slot_size, self.slot_size*2)
        # trainable scalar gate bias tensors
        self.forget_bias = nn.Parameter(torch.tensor(1., dtype=torch.float32))
        self.input_bias = nn.Parameter(torch.tensor(0., dtype=torch.float32))

        self.rel_projector = nn.Linear(slot_size*slot_size, rel_size)
        self.rel_projector2 = nn.Linear(num_slot * slot_size, slot_size)
        self.rel_projector3 = nn.Linear(num_slot * rel_size, out_att_size)

        self.mlp = nn.Sequential(
            nn.Linear(out_att_size, self.mlp_size),
            nn.ReLU(),
            nn.Linear(self.mlp_size, self.mlp_size),
            nn.ReLU(),
        )

        self.out = nn.Linear(self.mlp_size, output_size)

        if self.learn_init_mem:
            self.register_parameter('item_memory_state_bias',
                                    torch.nn.Parameter(torch.Tensor(self.slot_size, self.slot_size).to(device)))
            self.register_parameter('rel_memory_state_bias', torch.nn.Parameter(
                torch.Tensor(self.num_slot, self.slot_size, self.slot_size).to(device)))
            stdev = 1 / (np.sqrt(self.slot_size + self.slot_size))
            nn.init.uniform_(self.item_memory_state_bias, -stdev, stdev)
            stdev = 1 / (np.sqrt(self.slot_size + self.slot_size + self.num_slot))
            nn.init.uniform_(self.rel_memory_state_bias, -stdev, stdev)

    def create_new_state(self, batch_size):
        if self.learn_init_mem:
            read_heads = torch.zeros(batch_size, self.out_att_size).to(device)
            item_memory_state = self.item_memory_state_bias.clone().repeat(batch_size, 1, 1)
            rel_memory_state = self.rel_memory_state_bias.clone().repeat(batch_size, 1, 1, 1)
        else:
            item_memory_state = torch.stack(
                [torch.zeros(self.slot_size, self.slot_size) for _ in range(batch_size)]).to(device)
            read_heads =  torch.zeros(batch_size, self.out_att_size).to(device)
            rel_memory_state = torch.stack(
                [torch.zeros(self.num_slot, self.slot_size, self.slot_size) for _ in range(batch_size)]).to(device)

        return read_heads, item_memory_state, rel_memory_state

    def compute_gates(self, inputs, memory):
        memory = torch.tanh(memory)
        if len(inputs.shape) == 3:
            if inputs.shape[1] > 1:
                raise ValueError(
                    "input seq length is larger than 1. create_gate function is meant to be called for each step, with input seq length of 1")
            inputs = inputs.view(inputs.shape[0], -1)

            gate_inputs = self.input_gate_projector(inputs)  # 128->256 slot_size->slot_size*2
            gate_inputs = gate_inputs.unsqueeze(dim=1)   # (128,1,256)
            gate_memory = self.memory_gate_projector(memory)  # 128->256  (128,128,256)
        else:
            raise ValueError("input shape of create_gate function is 2, expects 3")

        gates = gate_memory + gate_inputs
        gates = torch.split(gates, split_size_or_sections=int(gates.shape[2] / 2), dim=2)  # gates为二元组, 均为(128,128,128)
        input_gate, forget_gate = gates
        assert input_gate.shape[2] == forget_gate.shape[2]

        input_gate = torch.sigmoid(input_gate + self.input_bias)
        forget_gate = torch.sigmoid(forget_gate + self.forget_bias)

        return input_gate, forget_gate

    def compute(self, input_step, prev_state, Mi):  # input_step(64,256)
        hid = prev_state[0]  # (128,64)
        # Mi = prev_state[1]  # (128,128,128)  改为BND形式
        rel_memory_state = prev_state[2]  # (128,8,128,128)

        # Mi = self.multihead_attention(input_step, item_memory_state)

        # 1.transform input
        # controller_outp = self.input_projector(input_step)  # (40->128) (128,128)为公式的Xt
        controller_outp2 = self.input_projector2(input_step)  # (40->128)
        controller_outp3 = self.input_projector3(input_step)  # (40->8) controller_outp3(B64,卡槽数目8)

        X = torch.matmul(input_step.unsqueeze(2), input_step.unsqueeze(1))
        # 使用 torch.einsum 函数执行张量乘法运算，通过使用合适的维度标签，我们可以自由地控制输入张量和输出张量的形状。
        controller_outp3 = F.softmax(controller_outp3, dim=-1)  # (B64,8)
        Vtr = torch.einsum('bnd,bndf->bnf', Mi, rel_memory_state)  # (128,128)
        X2 = torch.einsum('bnd,bf->bnf', Vtr, controller_outp2)  # (128,128,128)原为BDD  BND
        for i in range(self.step):
            qkv = self.qkv_projector[i]((Mi + X2).permute(0,2,1))  # 公式12的SAM()内容  BDD->BDN*3(8*3=24)  BDN->BD3*N
            qkv = self.qkv_layernorm[i](qkv)
            qkv = qkv.permute(0, 2, 1)  # Bx3Nxd  N为Mr卡槽的个数 D为卡槽记忆的维度，可以与输入维度相等或者小于
            q, k, v = torch.split(qkv, [self.num_slot] * 3, 1)  # BxNxd
            R0 = op_att(q, k, v)  # BxNxdxd
            rel_memory_state = self.alpha1[i] * R0 + rel_memory_state  # zxy公式12应该为此


        # 5.Mr transfer to output 公式14  r_vec(64,768=8*96)
        r_vec = self.rel_projector(rel_memory_state.view(rel_memory_state.shape[0],
                                                         rel_memory_state.shape[1],
                                                         -1)).view(input_step.shape[0], -1)
        out = self.rel_projector3(r_vec)  # (B64,out_dim256)


        return out, (out, Mi, rel_memory_state)  # (隐状态, Mi, Mr)

    # zxy
    def forward(self, input_step, Mi, hidden=None):   # input_step(N27,B64,in_dim)??  (64,27,256)
        if len(input_step.shape) == 3:
            hx = []
            # input_step(N,B,D)  (27,64,256)
            # self.init_sequence(input_step.shape[1])
            input = input_step.permute(1,0,2)  # (B,N,D)
            input_reshape = self.input_projector(input)  # 将input_step变为与Mi一样的维度
            Mi_new = self.multihead_attention(input_reshape, Mi)
            # self.previous_state[1] = Mi_new
            # 1.单个时间步计算方式
            input_reshape = input_reshape.permute(1,0,2)
            for i in range(input_reshape.shape[0]):
                # previous_state为三个元组 (128,64) (128,128,128) (128,8,128,128)
                # return 隐状态, (隐状态, Mi, Mr)
                hx_step, self.previous_state = self.compute(input_reshape[i], self.previous_state, Mi_new)
                hx.append(hx_step)
            hx = torch.stack(hx)  # 得到的hx(N,B,D)

        else:
            if hidden is not None:
                logit, hidden = self.compute(input_step, hidden)
            else:
                # 产生的logit为(B128,64)
                hx, self.previous_state = self.compute(input_step, self.previous_state)

        # mlp = self.mlp(logit)  # (B,64->128)
        # out = self.out(mlp)  # (B,128->out_size 8)   hx维度应该为(27,64,256)与输入一样 所以需要cat logit
        return hx, self.previous_state

    def init_sequence(self, batch_size):
        """Initializing the state."""
        self.previous_state = self.create_new_state(batch_size)

    def calculate_num_params(self):
        """Returns the total number of parameters."""
        num_params = 0
        for p in self.parameters():
            num_params += p.data.view(-1).size(0)
        return num_params

    def attend_over_memory(self, inputs, memory):
        """
        Perform multiheaded attention over `memory`.
            Args:
              memory: Current relational memory.
              inputs: Current inputs.
            Returns:
              The attended-over memory.
        """
        for _ in range(self.num_blocks):
            # RMC的A部分  (B64,num_slots8,D256)
            attended_memory = self.multihead_attention(inputs, memory)
            # Add a skip connection to the multiheaded attention's input.   残差连接+LayerNorm操作
            memory = self.attended_memory_layernorm(memory + attended_memory)

            # add a skip connection to the attention_mlp's input.
            attention_mlp = memory
            for i, l in enumerate(self.attention_mlp):
                attention_mlp = self.attention_mlp[i](attention_mlp)
                attention_mlp = F.relu(attention_mlp)
            memory = self.attended_memory_layernorm2(memory + attention_mlp)
            # memory = self.multihead_attention(memory, memory, use_topk_ = False, store_log = False)

        return memory

    def multihead_attention(self, input, memory, use_topk_ = True, store_log = True):
        """
        Perform multi-head attention from 'Attention is All You Need'.
        Implementation of the attention mechanism from
        https://arxiv.org/abs/1706.03762.
        Args:
          memory: Memory tensor to perform attention on. 用于集中注意力的记忆张量
        Returns:
          new_memory: New memory tensor.  返回新的张量
        """
        # 1.RMC的A部分用此函数生成新记忆, q为记忆M k,v应该为为R矩阵[M:A]
        # 2.广播过程 memory=input_reshape input=new_memory 用此产生新hx
        # input为(B64,num_slots8,D256)
        q = self.query_proj(memory)
        k = self.key_proj(input)
        v = self.value_proj(input)

        q = q.reshape(q.size(0), q.size(1), self.num_heads, -1).permute(0, 2, 1, 3)  # 2.(64,4,27,32)
        k = k.reshape(k.size(0), k.size(1), self.num_heads, -1).permute(0, 2, 1, 3)  # 2.(64,4,8,32)
        v = v.reshape(v.size(0), v.size(1), self.num_heads, -1).permute(0, 2, 1, 3)  # 2.(64,4,8,64)
        scores = torch.matmul(q, k.transpose(2, 3))  # 1.(64,4,8,27) 2.(64,4,27,8)

        scores = torch.softmax(scores, dim=-1)
        #if store_log:
        #    self.attn_log = scores[0]
        if not self.null_attention:
            # self.null_attention 为 false
            if self.use_topk and use_topk_:  # 对scores进行top-k筛选  TR+HSW在更新记忆RMC的A部分时会进入
                # 使scores中top-k个位置为1，其余位置为0。当属于更新记忆时,实现竞争写入,选取topk
                topk = torch.topk(scores, dim=-1, k=self.topk)
                mask = torch.zeros(scores.size()).to(scores.device)
                mask.scatter_(3, topk.indices, 1)
                scores = scores * mask
        output = torch.matmul(scores, v)
        # [B, H, N, V] => [B, N, H, V] => [B, N, H*V]
        output_transpose = output.permute(0, 2, 1, 3).contiguous()
        new_memory = output_transpose.view((output_transpose.shape[0], output_transpose.shape[1], -1))  # (64,8,256)
        return new_memory
    def initial_state(self, batch_size, trainable=False):
        """
        Creates the initial memory. 创建初始内存
        We should ensure each row of the memory is initialized to be unique,
        so initialize the matrix to be the identity. We then pad or truncate 填充或者压缩
        as necessary so that init_state is of size
        (batch_size, self.mem_slots, self.mem_size).
        Args:
          batch_size: The size of the batch.
          trainable: Whether the initial state is trainable. This is always True.
        Returns:
          init_state: A truncated or padded matrix of size 初始化状态
            (batch_size, self.mem_slots, self.mem_size).
        """
        if True:
            init_state = torch.stack([torch.eye(self.num_slot) for _ in range(batch_size)])

            # pad the matrix with zeros 用0填充矩阵
            if self.slot_size > self.num_slot:
                difference = self.slot_size - self.num_slot
                pad = torch.zeros((batch_size, self.num_slot, difference))
                init_state = torch.cat([init_state, pad], -1)

            # truncation. take the first 'self.slot_size' components
            elif self.slot_size < self.num_slot:
                init_state = init_state[:, :, :self.slot_size]

            return init_state
if __name__ == "__main__":

    N=64
    S=80
    B=32
    K = torch.ones((B, S, N))
    V = torch.ones((B, S, N))
    q = torch.ones((B, N))
    R = op_att(K,V,q)
    print(R.shape)
