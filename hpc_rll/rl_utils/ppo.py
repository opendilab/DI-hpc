import torch
import torch.nn.functional as F
from typing import Optional
from collections import namedtuple
import hpc_rl_utils

# hpc version only support cuda

hpc_ppo_loss = namedtuple('hpc_ppo_loss', ['policy_loss', 'value_loss', 'entropy_loss'])
hpc_ppo_info = namedtuple('hpc_ppo_info', ['approx_kl', 'clipfrac'])

class PPOFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits_new, logits_old, action, value_new, value_old, adv, return_, weight,
            clip_ratio, use_value_clip, dual_clip, logits_new_prob, logits_new_entropy, logits_new_grad_logits,
            logits_new_grad_prob, logits_new_grad_entropy, logit_old_prob,
            grad_policy_loss_buf, grad_value_loss_buf, grad_entropy_loss_buf,
            policy_loss, value_loss, entropy_loss, approx_kl, clipfrac, grad_value, grad_logits_new):

        inputs = [logits_new, logits_old, action, value_new, value_old, adv, return_, weight]
        outputs = [logits_new_prob, logits_new_entropy, logits_new_grad_logits,
                logits_new_grad_prob, logits_new_grad_entropy, logit_old_prob,
                grad_policy_loss_buf, grad_value_loss_buf, grad_entropy_loss_buf,
                policy_loss, value_loss, entropy_loss, approx_kl, clipfrac]

        hpc_rl_utils.PPOForward(inputs, outputs, use_value_clip, clip_ratio, dual_clip)

        bp_inputs = [grad_policy_loss_buf, grad_value_loss_buf, grad_entropy_loss_buf,
                logits_new_grad_logits, logits_new_grad_prob, logits_new_grad_entropy]
        bp_outputs = [grad_value, grad_logits_new]
        ctx.bp_inputs = bp_inputs
        ctx.bp_outputs = bp_outputs

        return policy_loss, value_loss, entropy_loss, approx_kl, clipfrac

    @staticmethod
    def backward(ctx, grad_policy_loss, grad_value_loss, grad_entropy_loss, grad_approx_kl, grad_clipfrac):
        inputs = [grad_policy_loss, grad_value_loss, grad_entropy_loss]
        for var in ctx.bp_inputs:
            inputs.append(var)
        outputs = ctx.bp_outputs

        hpc_rl_utils.PPOBackward(inputs, outputs)

        grad_value = outputs[0]
        grad_logits_new = outputs[1]
        return grad_logits_new, None, None, grad_value, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None

class PPOContinuousFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mu_new, sigma_new, mu_old, sigma_old, action, value_new, value_old, adv, return_, weight,
            clip_ratio, use_value_clip, dual_clip, new_prob, new_entropy, mu_new_grad_prob,
            sigma_new_grad_prob, sigma_new_grad_entropy, old_prob,
            grad_policy_loss_buf, grad_value_loss_buf, grad_entropy_loss_buf,
            policy_loss, value_loss, entropy_loss, approx_kl, clipfrac, grad_value, grad_mu_new, grad_sigma_new):

        inputs = [mu_new, sigma_new, mu_old, sigma_old, action, value_new, value_old, adv, return_, weight]
        outputs = [new_prob, new_entropy, mu_new_grad_prob,
                sigma_new_grad_prob, sigma_new_grad_entropy, old_prob,
                grad_policy_loss_buf, grad_value_loss_buf, grad_entropy_loss_buf,
                policy_loss, value_loss, entropy_loss, approx_kl, clipfrac]

        hpc_rl_utils.PPOContinuousForward(inputs, outputs, use_value_clip, clip_ratio, dual_clip)

        bp_inputs = [grad_policy_loss_buf, grad_value_loss_buf, grad_entropy_loss_buf,
                mu_new_grad_prob, sigma_new_grad_prob, sigma_new_grad_entropy]
        bp_outputs = [grad_value, grad_mu_new, grad_sigma_new]
        ctx.bp_inputs = bp_inputs
        ctx.bp_outputs = bp_outputs

        return policy_loss, value_loss, entropy_loss, approx_kl, clipfrac

    @staticmethod
    def backward(ctx, grad_policy_loss, grad_value_loss, grad_entropy_loss, grad_approx_kl, grad_clipfrac):
        inputs = [grad_policy_loss, grad_value_loss, grad_entropy_loss]
        for var in ctx.bp_inputs:
            inputs.append(var)
        outputs = ctx.bp_outputs

        hpc_rl_utils.PPOContinuousBackward(inputs, outputs)

        grad_value = outputs[0]
        grad_mu_new = outputs[1]
        grad_sigma_new = outputs[2] 
        return grad_mu_new, grad_sigma_new, None, None, None, grad_value, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None


class PPO(torch.nn.Module):
    """
    OverviewI:
        Implementation of Proximal Policy Optimization (arXiv:1707.06347) with value_clip and dual_clip

    Interface:
        __init__, forward
    """

    def __init__(self, B, N):
        r"""
        Overview
            initialization of PPO

        Arguments:
            - B (:obj:`int`): batch size
            - N (:obj:`int`): number of output
        """

        super().__init__()
        self.register_buffer('weight', torch.ones(B))
        self.register_buffer('logits_new_prob', torch.zeros(B))
        self.register_buffer('logits_new_entropy', torch.zeros(B))
        self.register_buffer('logits_new_grad_logits', torch.zeros(B, N))
        self.register_buffer('logits_new_grad_prob', torch.zeros(B, N))
        self.register_buffer('logits_new_grad_entropy', torch.zeros(B, N))
        self.register_buffer('logit_old_prob', torch.zeros(B))
        self.register_buffer('grad_policy_loss_buf', torch.zeros(B))
        self.register_buffer('grad_value_loss_buf', torch.zeros(B))
        self.register_buffer('grad_entropy_loss_buf', torch.zeros(B))

        self.register_buffer('policy_loss', torch.zeros(1))
        self.register_buffer('value_loss', torch.zeros(1))
        self.register_buffer('entropy_loss', torch.zeros(1))
        self.register_buffer('approx_kl', torch.zeros(1))
        self.register_buffer('clipfrac', torch.zeros(1))

        self.register_buffer('grad_value', torch.zeros(B))
        self.register_buffer('grad_logits_new', torch.zeros(B, N))

    def forward(self, logits_new, logits_old, action, value_new, value_old, adv, return_,
            weight = None,
            clip_ratio: float = 0.2,
            use_value_clip: bool = True,
            dual_clip: Optional[float] = None
            ):
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

        assert(logits_new.is_cuda)
        assert(logits_old.is_cuda)
        assert(action.is_cuda)
        assert(value_new.is_cuda)
        assert(value_old.is_cuda)
        assert(adv.is_cuda)
        assert(return_.is_cuda)
        if weight is None:
            weight = self.weight
        else:
            assert(weight.is_cuda)

        assert dual_clip is None or dual_clip > 1.0, "dual_clip value must be greater than 1.0, but get value: {}".format(dual_clip)

        if dual_clip is None:
            dual_clip = 0.0;

        policy_loss, value_loss, entropy_loss, approx_kl, clipfrac = PPOFunction.apply(
                logits_new, logits_old, action, value_new, value_old, adv, return_, weight,
                clip_ratio, use_value_clip, dual_clip,
                self.logits_new_prob, self.logits_new_entropy, self.logits_new_grad_logits,
                self.logits_new_grad_prob, self.logits_new_grad_entropy, self.logit_old_prob,
                self.grad_policy_loss_buf, self.grad_value_loss_buf, self.grad_entropy_loss_buf,
                self.policy_loss, self.value_loss, self.entropy_loss, self.approx_kl, self.clipfrac,
                self.grad_value, self.grad_logits_new)

        return hpc_ppo_loss(policy_loss, value_loss, entropy_loss), hpc_ppo_info(approx_kl.item(), clipfrac.item())

class PPOContinuous(torch.nn.Module):
    """
    OverviewI:
        Implementation of Proximal Policy Optimization (arXiv:1707.06347) with value_clip and dual_clip

    Interface:
        __init__, forward
    """

    def __init__(self, B, N):
        r"""
        Overview
            initialization of PPO

        Arguments:
            - B (:obj:`int`): batch size
            - N (:obj:`int`): number of output
        """

        super().__init__()
        self.register_buffer('weight', torch.ones(B))
        self.register_buffer('new_prob', torch.zeros(B))
        self.register_buffer('new_entropy', torch.zeros(B))
        self.register_buffer('mu_new_grad_prob', torch.zeros(B, N))
        self.register_buffer('sigma_new_grad_prob', torch.zeros(B, N))
        self.register_buffer('sigma_new_grad_entropy', torch.zeros(B, N))
        self.register_buffer('old_prob', torch.zeros(B))
        self.register_buffer('grad_policy_loss_buf', torch.zeros(B))
        self.register_buffer('grad_value_loss_buf', torch.zeros(B))
        self.register_buffer('grad_entropy_loss_buf', torch.zeros(B))

        self.register_buffer('policy_loss', torch.zeros(1))
        self.register_buffer('value_loss', torch.zeros(1))
        self.register_buffer('entropy_loss', torch.zeros(1))
        self.register_buffer('approx_kl', torch.zeros(1))
        self.register_buffer('clipfrac', torch.zeros(1))

        self.register_buffer('grad_value', torch.zeros(B))
        self.register_buffer('grad_mu_new', torch.zeros(B, N))
        self.register_buffer('grad_sigma_new', torch.zeros(B, N))

    def forward(self, mu_new, sigma_new, mu_old, sigma_old, action, value_new, value_old, adv, return_,
            weight = None,
            clip_ratio: float = 0.2,
            use_value_clip: bool = True,
            dual_clip: Optional[float] = None
            ):
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

        assert(mu_new.is_cuda)
        assert(sigma_new.is_cuda)
        assert(mu_old.is_cuda)
        assert(sigma_old.is_cuda)
        assert(action.is_cuda)
        assert(value_new.is_cuda)
        assert(value_old.is_cuda)
        assert(adv.is_cuda)
        assert(return_.is_cuda)
        if weight is None:
            weight = self.weight
        else:
            assert(weight.is_cuda)

        assert dual_clip is None or dual_clip > 1.0, "dual_clip value must be greater than 1.0, but get value: {}".format(dual_clip)

        if dual_clip is None:
            dual_clip = 0.0;

        policy_loss, value_loss, entropy_loss, approx_kl, clipfrac = PPOContinuousFunction.apply(
                mu_new, sigma_new, mu_old, sigma_old, action, value_new, value_old, adv, return_, weight,
                clip_ratio, use_value_clip, dual_clip,
                self.new_prob, self.new_entropy, self.mu_new_grad_prob,
                self.sigma_new_grad_prob, self.sigma_new_grad_entropy, self.old_prob,
                self.grad_policy_loss_buf, self.grad_value_loss_buf, self.grad_entropy_loss_buf,
                self.policy_loss, self.value_loss, self.entropy_loss, self.approx_kl, self.clipfrac,
                self.grad_value, self.grad_mu_new, self.grad_sigma_new)

        return hpc_ppo_loss(policy_loss, value_loss, entropy_loss), hpc_ppo_info(approx_kl.item(), clipfrac.item())

