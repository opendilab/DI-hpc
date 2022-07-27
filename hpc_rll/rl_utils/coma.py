import torch
import torch.nn.functional as F
from typing import Optional
from collections import namedtuple
import hpc_rl_utils

# hpc version only support cuda

coma_loss = namedtuple('coma_loss', ['policy_loss', 'q_value_loss', 'entropy_loss'])

class COMAFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logit, action, q_value, target_q_value, reward, weight,
            gamma, lambda_, reward_new, q_taken, target_q_taken, prob, adv, entropy, return_,
            logits_grad_logits, logits_grad_prob, logits_grad_entropy,
            grad_policy_loss_buf, grad_value_loss_buf, grad_entropy_loss_buf,
            policy_loss, value_loss, entropy_loss, grad_q_value, grad_logit):

        inputs = [logit, action, q_value, target_q_value, reward, weight]
        outputs = [reward_new, q_taken, target_q_taken, prob, adv, entropy, return_, 
        logits_grad_logits, logits_grad_prob, logits_grad_entropy,
        grad_policy_loss_buf, grad_value_loss_buf, grad_entropy_loss_buf,
            policy_loss, value_loss, entropy_loss]

        hpc_rl_utils.COMAForward(inputs, outputs, gamma, lambda_)

        bp_inputs = [grad_policy_loss_buf, grad_value_loss_buf, grad_entropy_loss_buf, adv, action,
                    logits_grad_logits, logits_grad_prob, logits_grad_entropy]
        bp_outputs = [grad_q_value, grad_logit]
        ctx.bp_inputs = bp_inputs
        ctx.bp_outputs = bp_outputs

        return policy_loss, value_loss, entropy_loss

    @staticmethod
    def backward(ctx, grad_policy_loss, grad_value_loss, grad_entropy_loss):
        inputs = [grad_policy_loss, grad_value_loss, grad_entropy_loss]
        for var in ctx.bp_inputs:
            inputs.append(var)
        outputs = ctx.bp_outputs

        hpc_rl_utils.COMABackward(inputs, outputs)

        grad_q_value = outputs[0]
        grad_logit = outputs[1]
        return grad_logit, None, grad_q_value, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None




class COMA(torch.nn.Module):
    """
    OverviewI:
        Implementation of COMA

    Interface:
        __init__, forward
    """

    def __init__(self, T, B, A, N):
        r"""
        Overview
            initialization of COMA

        Arguments:
            - T (:obj:`int`): time step
            - B (:obj:`int`): batch size
            - A (:obj:`int`): number of agent
            - N (:obj:`int`): number of choice
        """

        super().__init__()
        self.register_buffer('reward_new', torch.zeros(T,B,A))
        self.register_buffer('q_taken', torch.zeros(T,B,A))
        self.register_buffer('target_q_taken', torch.zeros(T,B,A))
        self.register_buffer('prob', torch.zeros(T,B,A))
        self.register_buffer('adv', torch.zeros(T,B,A))
        self.register_buffer('entropy', torch.zeros(T,B,A))
        self.register_buffer('return_', torch.zeros(T - 1,B,A))
        self.register_buffer('logits_grad_logits', torch.zeros(T,B,A, N))
        self.register_buffer('logits_grad_prob', torch.zeros(T,B,A, N))
        self.register_buffer('logits_grad_entropy', torch.zeros(T,B,A, N))
        self.register_buffer('grad_policy_loss_buf', torch.zeros(T, B, A))
        self.register_buffer('grad_value_loss_buf', torch.zeros(T, B, A))
        self.register_buffer('grad_entropy_loss_buf', torch.zeros(T, B, A))

        self.register_buffer('policy_loss', torch.zeros(1))
        self.register_buffer('value_loss', torch.zeros(1))
        self.register_buffer('entropy_loss', torch.zeros(1))

        self.register_buffer('grad_q_value', torch.zeros(T, B, A, N))
        self.register_buffer('grad_logit', torch.zeros(T, B, A, N))

    def forward(self, logit, action, q_value, target_q_value, reward, weight,
            gamma: float = 0.99,
            lambda_: float = 0.95,
            ):
        """
        Overview:
            forward of COMA
        Arguments:
            - logit (:obj:`torch.FloatTensor`): :math:`(T, B, A, N)`, where B is batch size and N is action dim
            - action (:obj:`torch.IntTensor`): :math:`(T, B, A)`
            - q_value (:obj:`torch.FloatTensor`): :math:`(T, B, A, N)`
            - target_q_value (:obj:`torch.FloatTensor`): :math:`(T, B, A, N)`
            - reward (:obj:`torch.FloatTensor`): :math:`(T, B)`
            - weight (:obj:`torch.FloatTensor` or :obj:`None`): :math:`(T, B, A)`

        Returns:
            - coma_loss (:obj:`namedtuple`): the coma loss item, all of them are the differentiable 0-dim tensor
        """

        assert(logit.is_cuda)
        assert(action.is_cuda)
        assert(q_value.is_cuda)
        assert(target_q_value.is_cuda)
        assert(reward.is_cuda)
        assert(weight.is_cuda)
        if weight is None:
            weight = torch.ones_like(action)
        else:
            assert(weight.is_cuda)

        policy_loss, value_loss, entropy_loss = COMAFunction.apply(
                logit, action, q_value, target_q_value, reward, weight,
                gamma, lambda_, self.reward_new, self.q_taken, self.target_q_taken, self.prob, self.adv, self.entropy, self.return_,
                self.logits_grad_logits, self.logits_grad_prob, self.logits_grad_entropy, self.grad_policy_loss_buf, self.grad_value_loss_buf, 
                self.grad_entropy_loss_buf, self.policy_loss, self.value_loss, self.entropy_loss, self.grad_q_value, self.grad_logit)

        return coma_loss(policy_loss, value_loss, entropy_loss)


