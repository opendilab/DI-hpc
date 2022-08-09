import imp
import torch
import torch.nn.functional as F
from typing import Optional
from collections import namedtuple
import hpc_rl_utils
class SOFTARGMAXFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, softmax_x, res, grad_x):

        inputs = [x]
        outputs = [softmax_x, res]


        hpc_rl_utils.SOFTARGMAXForward(inputs, outputs)

        bp_inputs = [softmax_x]
        bp_outputs = [grad_x]
        ctx.bp_inputs = bp_inputs
        ctx.bp_outputs = bp_outputs

        return res

    @staticmethod
    def backward(ctx, grad_res):
        inputs = [grad_res]
        for var in ctx.bp_inputs:
            inputs.append(var)
        outputs = ctx.bp_outputs

        hpc_rl_utils.SOFTARGMAXBackward(inputs, outputs)

        grad_x = outputs[0]
        return grad_x, None, None, None


class SoftArgmaxHPC(torch.nn.Module):
    def __init__(self, B, H, W):

        super().__init__()
        self.register_buffer('res', torch.zeros(B, 2))
        self.register_buffer('softmax_x', torch.zeros(B, 1, H, W))
        self.register_buffer('grad_x', torch.zeros(B, 1, H, W))

    def forward(self, x):
        """
        Overview:
            forward of Retrace
        Arguments:
            - q_values (:obj:`torch.FloatTensor`): :math:`(T + 1, B, N)`, 
            - v_pred (:obj:`torch.FloatTensor`): :math:`(T + 1, B, 1)`, 
            - rewards (:obj:`torch.FloatTensor`): :math:`(T, B)`
            - avtions (:obj:`torch.LongTensor`): :math:`(T, B)`
            - weights (:obj:`torch.FloatTensor`): :math:`(T, B)`
            - ratio (:obj:`torch.FloatTensor`): :math:`(T, B, N)`
        Returns:
            - q_retraces (:obj:`torch.FloatTensor`): math:`(T + 1, B, 1)`, 
        """


        assert(x.is_cuda)

        res = SOFTARGMAXFunction.apply(x, self.softmax_x, self.res, self.grad_x)

        return res