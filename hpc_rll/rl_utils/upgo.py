import torch
import hpc_rl_utils

# hpc version only support cuda
# 需排除spe2d case

class UpgoFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, target_output, rho, action, reward, value, advantage, metric, loss, grad_buf, grad_target_output):
        inputs = [target_output, rho, action, reward, value]
        outputs = [advantage, metric, loss, grad_buf]
        hpc_rl_utils.UpgoForward(inputs, outputs)

        ctx.bp_inputs = [grad_buf, advantage]
        ctx.bp_outputs = [grad_target_output]

        return loss

    @staticmethod
    def backward(ctx, grad_loss):
        inputs = [grad_loss]
        for var in ctx.bp_inputs:
            inputs.append(var)
        outputs = ctx.bp_outputs

        hpc_rl_utils.UpgoBackward(inputs, outputs)
        grad_target_output = outputs[0]
        return grad_target_output, None, None, None, None, None, None, None, None, None

class UPGO(torch.nn.Module):
    """
    Overview:
        Computing UPGO loss given constant gamma and lambda. There is no special handling for terminal state value,
        if the last state in trajectory is the terminal, just pass a 0 as bootstrap_terminal_value.

    Interface:
        __init__, forward
    """

    def __init__(self, T, B, N):
        r"""
        Overview
            initialization of UPGO

        Arguments:
            - T (:obj:`int`): trajectory length
            - B (:obj:`int`): batch size
            - N (:obj:`int`): number of output
        """

        super().__init__()
        self.register_buffer('loss', torch.zeros(1))
        self.register_buffer('advantage', torch.zeros(T, B))
        self.register_buffer('metric', torch.zeros(T, B))
        self.register_buffer('grad_buf', torch.zeros(T, B, N))
        self.register_buffer('grad_target_output', torch.zeros(T, B, N))

    def forward(self, target_output, rhos, action, rewards, bootstrap_values):
        """
        Overview:
            forward of UPGO
        Arguments:
            - target_output (:obj:`torch.Tensor`): :math:`(T, B, N)`, the output computed by the target policy network
            - rhos (:obj:`torch.Tensor`): :math:`(T, B)`, the importance sampling ratio
            - action (:obj:`torch.Tensor`): :math:`(T, B)`, the action taken
            - rewards (:obj:`torch.Tensor`): :math:`(T, B)`, the returns from time step 0 to T-1
            - bootstrap_values (:obj:`torch.Tensor`): :math:`(T + 1, B)`, estimation of the state value at step 0 to T
        Returns:
            - loss (:obj:`torch.Tensor`): :math:`()`, 0-dim tensor, Computed importance sampled UPGO loss, averaged over the samples
        """
        assert(target_output.is_cuda)
        assert(rhos.is_cuda)
        assert(action.is_cuda)
        assert(rewards.is_cuda)
        assert(bootstrap_values.is_cuda)

        loss = UpgoFunction.apply(target_output, rhos, action, rewards, bootstrap_values,
                self.advantage, self.metric, self.loss, self.grad_buf, self.grad_target_output)
        return loss

