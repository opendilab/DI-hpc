import torch
import torch.nn.functional as F
from typing import Optional
from collections import namedtuple
import hpc_rl_utils

class GRUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, bg, g, Wzy, Uzx, Wgy, Ugrx, grad_Wzy, grad_Uzx, grad_Wgy, grad_Ugrx, grad_bg, h, z, TB, input_dim):

        inputs = [x, Wzy, Uzx, Wgy, Ugrx, bg]
        outputs = [g, h, z]

        hpc_rl_utils.GRUForward(inputs, outputs, TB, input_dim)

        bp_inputs = [h, z, x]
        bp_outputs = [grad_Wzy, grad_Uzx, grad_Wgy, grad_Ugrx, grad_bg]
        ctx.TB = TB
        ctx.input_dim = input_dim
        ctx.bp_inputs = bp_inputs
        ctx.bp_outputs = bp_outputs

        return g

    @staticmethod
    def backward(ctx, grad_g):
        inputs = [grad_g]
        for var in ctx.bp_inputs:
            inputs.append(var)
        outputs = ctx.bp_outputs

        hpc_rl_utils.GRUBackward(inputs, outputs, ctx.TB, ctx.input_dim)

        grad_Wzy = outputs[0]
        grad_Uzx = outputs[1]
        grad_Wgy = outputs[2]
        grad_Ugrx = outputs[3]
        grad_bg = outputs[4]
        return None, grad_bg, None, grad_Wzy, grad_Uzx, grad_Wgy, grad_Ugrx, None, None, None, None, None, None, None


class GRU(torch.nn.Module):
    """
    OverviewI:
        Implementation of Proximal Policy Optimization (arXiv:1707.06347) with value_clip and dual_clip

    Interface:
        __init__, forward
    """

    def __init__(self, T, B, input_dim, bg: float = 2.):
        r"""
        Overview
            initialization of PPO

        Arguments:
            - B (:obj:`int`): batch size
            - N (:obj:`int`): number of output
        """

        super().__init__()

        self.Wr = torch.nn.Linear(input_dim, input_dim, bias=False)
        self.Ur = torch.nn.Linear(input_dim, input_dim, bias=False)
        self.Wz = torch.nn.Linear(input_dim, input_dim, bias=False)
        self.Uz = torch.nn.Linear(input_dim, input_dim, bias=False)
        self.Wg = torch.nn.Linear(input_dim, input_dim, bias=False)
        self.Ug = torch.nn.Linear(input_dim, input_dim, bias=False)
        self.bg = torch.nn.Parameter(torch.full([input_dim], bg))  # bias


        self.sigmoid = torch.nn.Sigmoid()
        self.TB = T * B
        self.input_dim = input_dim


        self.register_buffer('grad_Wzy', torch.zeros(T*B, input_dim))
        self.register_buffer('grad_Uzx', torch.zeros(T*B, input_dim))
        self.register_buffer('grad_Wgy', torch.zeros(T*B, input_dim))
        self.register_buffer('grad_Ugrx', torch.zeros(T*B, input_dim))
        self.register_buffer('grad_bg', torch.zeros(input_dim))
        self.register_buffer('g', torch.zeros(T, B, input_dim))

        self.register_buffer('Wzy', torch.zeros(T*B, input_dim))
        self.register_buffer('Uzx', torch.zeros(T*B, input_dim))
        self.register_buffer('Wgy', torch.zeros(T*B, input_dim))
        self.register_buffer('Ugrx', torch.zeros(T*B, input_dim))

        self.register_buffer('h', torch.zeros(T*B, input_dim))
        self.register_buffer('z', torch.zeros(T*B, input_dim))



    def forward(self, x, y):
        """
        Overview:
            forward of PPO
        Arguments:
            - logit_new (:obj:`torch.FloatTensor`): :math:`(B, N)`, where B is batch size and N is action dim
            - logit_old (:obj:`torch.FloatTensor`): :math:`(B, N)`
            - action (:obj:`torch.LongTensor`): :math:`(B, )`
            - value_new (:obj:`torch.FloatTensor`): :math:`(B, )`
            - value_old (:obj:`torch.FloatTensor`): :math:`(B, )`
            - adv (:obj:`torch.FloatTensor`): :math:`(B, )`
            - return (:obj:`torch.FloatTensor`): :math:`(B, )`
            - weight (:obj:`torch.FloatTensor` or :obj:`None`): :math:`(B, )`
            - clip_ratio (:obj:`float`): the ppo clip ratio for the constraint of policy update, defaults to 0.2
            - use_value_clip (:obj:`bool`): whether to use clip in value loss with the same ratio as policy
            - dual_clip (:obj:`float`): a parameter c mentioned in arXiv:1912.09729 Equ. 5, shoule be in [1, inf),\
            defaults to 5.0, if you don't want to use it, set this parameter to None

        Returns:
            - ppo_loss (:obj:`namedtuple`): the ppo loss item, all of them are the differentiable 0-dim tensor
            - ppo_info (:obj:`namedtuple`): the ppo optim information for monitoring, all of them are Python scalar

        .. note::
            adv is already normalized value (adv - adv.mean()) / (adv.std() + 1e-8), and there are many
            ways to calculate this mean and std, like among data buffer or train batch, so we don't couple
            this part into ppo_error, you can refer to our examples for different ways.
        """

        assert(x.is_cuda)
        assert(y.is_cuda)
        r = self.sigmoid(self.Wr(y) + self.Ur(x))
        self.Wzy = self.Wz(y)
        self.Uzx = self.Uz(x)
        self.Wgy = self.Wg(y)
        self.Ugrx = self.Ug(torch.mul(r, x))

        g = GRUFunction.apply(x, self.bg.data, self.g, self.Wzy, self.Uzx, self.Wgy, self.Ugrx, self.grad_Wzy, self.grad_Uzx, self.grad_Wgy, self.grad_Ugrx, self.grad_bg, self.h, self.z, self.TB, self.input_dim)

        return g