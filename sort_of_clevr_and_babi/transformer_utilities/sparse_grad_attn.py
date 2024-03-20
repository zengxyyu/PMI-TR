'''
Giving an N x M attention matrix, returns the same matrix,
but performs masking to determine where to block gradients.
'''

import numpy
import torch
from torch.autograd import Variable

from .sparse_attn import Sparse_attention
# from sparse_attn import Sparse_attention


class blocked_grad(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, mask):
        ctx.save_for_backward(x, mask)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        x, mask = ctx.saved_tensors
        return grad_output * mask, mask * 0.0


class Sparse_grad_attention(torch.autograd.Function):
    # def __init__(self, top_k):
    #     super(Sparse_grad_attention,self).__init__()
    #
    #     self.sa = Sparse_attention(top_k=top_k)

    def forward(ctx, inp, sa):
        sparsified = sa(inp)
        ctx.save_for_backward(inp, sparsified)

        return inp

    def backward(ctx, grad_output):
        inp, sparsified = ctx.saved_tensors
        # print('sparsified', sparsified)
        return (grad_output) * (sparsified > 0.0).float()

class Sparse_grad_attention_zxy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, top_k):
        sa = Sparse_attention(top_k=top_k)
        sparsified = sa(inp)
        ctx.save_for_backward(inp, sparsified)

        return inp

    @staticmethod
    def backward(ctx, grad_output):
        inp, sparsified = ctx.saved_tensors
        # print('sparsified', sparsified)
        return (grad_output) * (sparsified > 0.0).float(), None


if __name__ == "__main__":
    k = 2
    x = torch.from_numpy(numpy.array([[[0.1, 0.0, 0.3, 0.2, 0.4],
                                       [0.5, 0.4, 0.1, 0.0, 0.0]]]))
    x = x.reshape((2, 5))

    # Sparse_attention稀疏注意力  方式1
    sa = Sparse_attention(k)
    x = Variable(x.data, requires_grad=True)
    (sa(x).sum()).backward()
    print('normal grad', x.grad)

    # 方式2
    x = Variable(x, requires_grad=True)
    print(x)
    sga = Sparse_grad_attention_zxy.apply(x, k)
    (sga.sum()).backward()
    print('sga_out:', x.grad)

