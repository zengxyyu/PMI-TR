import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 将1替换为要使用的GPU索引
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def op_att(q, k, v):
    qq = q.unsqueeze(2).repeat(1, 1, k.shape[1], 1)
    kk = k.unsqueeze(1).repeat(1, q.shape[1], 1, 1)
    # BxNXNxd_kq BxNxNxd_v --> BxNXNxd_kqxd_v
    # qq * kk 表示进行逐元素的乘法运算  存在内存溢出的情况
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
                 out_att_size=64, rd=False,
                 init_alphas=[None,None,None],
                 learn_init_mem=True, mlp_hid=-1):
        super(STM, self).__init__()
        self.mlp_size = mlp_size
        self.slot_size = slot_size
        self.rel_size = rel_size
        self.rnn_hid = slot_size
        self.num_slot = num_slot
        self.step = step
        self.rd = rd
        self.learn_init_mem = learn_init_mem

        self.out_att_size = out_att_size

        self.qkv_projector = nn.ModuleList([nn.Linear(slot_size, num_slot*3)]*step)
        self.qkv_layernorm = nn.ModuleList([nn.LayerNorm([slot_size, num_slot*3])]*step)

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

    def compute(self, input_step, prev_state):  # input_step(64,256)
        # print("stm_zxy!!!!!!!!!!!!!!!!!")
        hid = prev_state[0]  # (64,256)
        item_memory_state = prev_state[1]  # (64,h_dim256,h_dim256)
        rel_memory_state = prev_state[2]  # (64,8,h_dim256,h_dim256)

        # 1.transform input
        # controller_outp = self.input_projector(input_step)  # (40->128) (256->256)  (64,256)为公式的Xt
        # controller_outp2 = self.input_projector2(input_step)  # (40->128) (256->256)
        controller_outp3 = self.input_projector3(input_step)  # (40->8)  (256->8)controller_outp3(64,8)

        # 2.Mi write 初级记忆的更新
        # 公式9 Bxdxd(64,256,256)  (64,128,1)*(64,1,128)=(B,slot_size,slot_size)
        X = torch.matmul(input_step.unsqueeze(2), input_step.unsqueeze(1))
        # 根据Xt和先前的Mi生成遗忘门和输入门,均为(B,slot_size256,slot_size256) 公式10的门控
        input_gate, forget_gate = self.compute_gates(input_step.unsqueeze(1), item_memory_state)
        if self.rd:
            # Mi write gating 公式10实现
            Mi = input_gate * torch.tanh(X)
            Mi += forget_gate * item_memory_state
        else:
            # Mi write 直接将X+item_memory_state去更新Mi
            # R = item_memory_state + torch.matmul(controller_outp.unsqueeze(2), controller_outp.unsqueeze(1))#Bxdxd
            Mi = item_memory_state + X  # Bxdxd

        # 3.Mr read  关系记忆的读取
        # 使用 torch.einsum 函数执行张量乘法运算，通过使用合适的维度标签，我们可以自由地控制输入张量和输出张量的形状。
        # controller_outp3 = F.softmax(controller_outp3, dim=-1)  # (64,8)
        # 公式11得到认知Vtr即controller_outp4  其维度为BD 与输入一样
        # Vtr = torch.einsum('bn,bd,bndf->bf', controller_outp3, input_step,
        #                                 rel_memory_state)  # (128,128)
        Vtr = torch.einsum('bdf,bndf->bf', Mi, rel_memory_state)  # (128,128)
        # 4.Mr的更新  开始公式12: 公式12的一部分, Vtr与f2(xt)外积(让认知与输入相乘)
        X2 = torch.einsum('bd,bf->bdf', Vtr, input_step)  # (128,128,128)
        for i in range(self.step):
            # 4.1 SAM操作  self.alpha2[i]*X2为全0, 形状不变
            qkv = self.qkv_projector[i](Mi + self.alpha2[i] * X2)  # 公式12的SAM()内容  D->N*3
            qkv = self.qkv_layernorm[i](qkv)
            qkv = qkv.permute(0, 2, 1)  # Bx3Nxd  N为Mr卡槽的个数 D为卡槽记忆的维度，可以与输入维度相等或者小于
            q, k, v = torch.split(qkv, [self.num_slot] * 3, 1)  # BxNxd
            # 执行外积操作函数op_att q k 逐元素相乘，再与v 进行matmul
            R0 = op_att(q, k, v)  # BxNxdxd
            # Mr write 公式12 与公式存在微小偏差
            rel_memory_state = self.alpha1[i] * rel_memory_state + R0
            # rel_memory_state = self.alpha1[i] * R0 + rel_memory_state  # zxy公式12应该为此

            # 4.2 Mr transfer to Mi  公式13 二次刷新Mi
            # zxy 根据公式13 R0应该改为rel_memory_state
            # R2 = self.rel_projector2(
            #     rel_memory_state.view(rel_memory_state.shape[0], -1, rel_memory_state.shape[3]).permute(0, 2, 1))
            R2 = self.rel_projector2(R0.view(R0.shape[0], -1, R0.shape[3]).permute(0, 2, 1))
            Mi = Mi + self.alpha3[i] * R2  # R为初步更新的Mi

        # 5.Mr transfer to output 公式14  r_vec(64,768=8*96)
        r_vec = self.rel_projector(rel_memory_state.view(rel_memory_state.shape[0],
                                                         rel_memory_state.shape[1],
                                                         -1)).view(input_step.shape[0], -1)
        out = self.rel_projector3(r_vec)  # (B64,out_dim256)

        return out, (out, Mi, rel_memory_state)  # (隐状态, Mi, Mr)

    # zxy
    def forward(self, input_step, hidden=None):   # input(Seq8,Batch_size128,in_dim)??  (64,27,256)
        if len(input_step.shape) == 3:
            hx = []
            # input_step(N,B,D)  (27,64,256)
            self.init_sequence(input_step.shape[1])
            for i in range(input_step.shape[0]):
                # previous_state为三个元组 (128,64) (128,128,128) (128,8,128,128)
                # return 隐状态, (隐状态, Mi, Mr)
                hx_step, self.previous_state = self.compute(input_step[i], self.previous_state)
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
    # def forward(self, input_step, hidden=None):  # input_step(128,40)
    #     if len(input_step.shape)==3:
    #         self.init_sequence(input_step.shape[1])
    #         for i in range(input_step.shape[0]):
    #             # previous_state为三个元组 (128,64) (128,128,128) (128,8,128,128)
    #             # return 隐状态, (隐状态, Mi, Mr)
    #             logit, self.previous_state = self.compute(input_step[i], self.previous_state)
    #     else:
    #         if hidden is not None:
    #             logit, hidden = self.compute(input_step, hidden)
    #         else:
    #             logit, self.previous_state = self.compute(input_step,  self.previous_state)
    #
    #     mlp = self.mlp(logit)
    #     out = self.out(mlp)
    #     return out, self.previous_state

    def init_sequence(self, batch_size):
        """Initializing the state."""
        self.previous_state = self.create_new_state(batch_size)

    def calculate_num_params(self):
        """Returns the total number of parameters."""
        num_params = 0
        for p in self.parameters():
            num_params += p.data.view(-1).size(0)
        return num_params

if __name__ == "__main__":

    N=64
    S=80
    B=32
    K = torch.ones((B, S, N))
    V = torch.ones((B, S, N))
    q = torch.ones((B, N))
    R = op_att(K,V,q)
    print(R.shape)
