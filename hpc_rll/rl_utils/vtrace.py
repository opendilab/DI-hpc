import torch
import torch.nn.functional as F
from collections import namedtuple
import hpc_rl_utils

# hpc version only support cuda

hpc_vtrace_loss = namedtuple('hpc_vtrace_loss', ['policy_loss', 'value_loss', 'entropy_loss'])

class VtraceFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, target_output, behaviour_output,
            action, value, reward, weight, gamma, lambda_, rho_clip_ratio, c_clip_ratio, rho_pg_clip_ratio,
            target_output_prob, target_output_entropy,
            target_output_grad_logits, target_output_grad_prob, target_output_grad_entropy, behaviour_output_prob,
            importance_weights, returns, advantages, pg_loss, value_loss, entropy_loss, grad_value, grad_target_output):

        inputs = [target_output, behaviour_output, action, value, reward, weight]
        outputs = [target_output_prob, target_output_entropy,
            target_output_grad_logits, target_output_grad_prob, target_output_grad_entropy, behaviour_output_prob,
            importance_weights, returns, advantages, pg_loss, value_loss, entropy_loss]

        hpc_rl_utils.VTraceForward(inputs, outputs, gamma, lambda_, rho_clip_ratio, c_clip_ratio, rho_pg_clip_ratio)

        bp_inputs = [value, action, weight, returns, advantages, target_output_grad_logits, target_output_grad_prob, target_output_grad_entropy]
        bp_outputs = [grad_value, grad_target_output]
        ctx.bp_inputs = bp_inputs
        ctx.bp_outputs = bp_outputs

        return pg_loss, value_loss, entropy_loss

    @staticmethod
    def backward(ctx, grad_pg_loss, grad_value_loss, grad_entropy_loss):
        inputs = [grad_pg_loss, grad_value_loss, grad_entropy_loss]
        for var in ctx.bp_inputs:
            inputs.append(var)
        outputs = ctx.bp_outputs

        hpc_rl_utils.VTraceBackward(inputs, outputs)

        grad_value = outputs[0]
        grad_target_output = outputs[1]
        return grad_target_output, None, None, grad_value, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None

class VTrace(torch.nn.Module):
    """
    Overview:
        Implementation of vtrace(IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner\
        Architectures), (arXiv:1802.01561)

    Interface:
        __init__, forward
    """

    def __init__(self, T, B, N):
        r"""
        Overview
            initialization of Vtrace

        Arguments:
            - T (:obj:`int`): trajectory length
            - B (:obj:`int`): batch size
            - N (:obj:`int`): number of output
        """

        super().__init__()
        self.register_buffer('weight', torch.ones(T, B))
        self.register_buffer('target_output_prob', torch.zeros(T, B))
        self.register_buffer('target_output_entropy', torch.zeros(T, B))
        self.register_buffer('target_output_grad_logits', torch.zeros(T, B, N))
        self.register_buffer('target_output_grad_prob', torch.zeros(T, B, N))
        self.register_buffer('target_output_grad_entropy', torch.zeros(T, B, N))
        self.register_buffer('behaviour_output_prob', torch.zeros(T, B))
        self.register_buffer('importance_weights', torch.zeros(T, B))
        self.register_buffer('returns', torch.zeros(T, B))
        self.register_buffer('advantages', torch.zeros(T, B))
        self.register_buffer('pg_loss', torch.zeros(1))
        self.register_buffer('value_loss', torch.zeros(1))
        self.register_buffer('entropy_loss', torch.zeros(1))
        self.register_buffer('grad_value', torch.zeros(T + 1, B))
        self.register_buffer('grad_target_output', torch.zeros(T, B, N))

    def forward(self, target_output, behaviour_output, action, value, reward,
            weight = None,
            gamma: float = 0.99,
            lambda_: float = 0.95,
            rho_clip_ratio: float = 1.0,
            c_clip_ratio: float = 1.0,
            rho_pg_clip_ratio: float = 1.0
            ):
        """
        Overview:
            forward of Vtrace
        Arguments:
            - target_output (:obj:`torch.Tensor`): :math:`(T, B, N)`, the output taking the action by the current policy network,\
                usually this output is network output logit
            - behaviour_output (:obj:`torch.Tensor`): :math:`(T, B, N)`, the output taking the action by the behaviour policy network,\
                usually this output is network output logit, which is used to produce the trajectory(actor)
            - action (:obj:`torch.Tensor`): :math:`(T, B)`, the chosen action(index for the discrete action space) in trajectory,\
                i.e.: behaviour_action
            - value (:obj:`torch.Tensor`): :math:`(T + 1, B)`, estimation of the state value at step 0 to T
            - reward (:obj:`torch.Tensor`): :math:`(T, B)`, the returns from time step 0 to T-1
            - gamma: (:obj:`float`): the future discount factor, defaults to 0.95
            - lambda: (:obj:`float`): mix factor between 1-step (lambda_=0) and n-step, defaults to 1.0
            - rho_clip_ratio (:obj:`float`): the clipping threshold for importance weights (rho) when calculating\
                the baseline targets (vs)
            - c_clip_ratio (:obj:`float`): the clipping threshold for importance weights (c) when calculating\
                the baseline targets (vs)
            - rho_pg_clip_ratio (:obj:`float`): the clipping threshold for importance weights (rho) when calculating\
                the policy gradient advantage

        Returns:
            - trace_loss (:obj:`namedtuple`): the vtrace loss item, all of them are the differentiable 0-dim tensor
        """

        assert(target_output.is_cuda)
        assert(behaviour_output.is_cuda)
        assert(action.is_cuda)
        assert(value.is_cuda)
        assert(reward.is_cuda)
        if weight is None:
            weight = self.weight
        else:
            assert(weight.is_cuda)

        pg_loss, value_loss, entropy_loss = VtraceFunction.apply(target_output, behaviour_output,
                action, value, reward, weight, gamma, lambda_, rho_clip_ratio, c_clip_ratio, rho_pg_clip_ratio,
                self.target_output_prob, self.target_output_entropy,
                self.target_output_grad_logits, self.target_output_grad_prob, self.target_output_grad_entropy, self.behaviour_output_prob,
                self.importance_weights, self.returns, self.advantages,
                self.pg_loss, self.value_loss, self.entropy_loss, self.grad_value, self.grad_target_output)

        return hpc_vtrace_loss(pg_loss, value_loss, entropy_loss)
