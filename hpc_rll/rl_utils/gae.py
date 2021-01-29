import torch
import hpc_rl_utils

# hpc version only support cuda

class GAEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, reward, gamma, lambda_, adv):

        inputs = [value, reward]
        outputs = [adv]
        hpc_rl_utils.GaeForward(inputs, outputs, gamma, lambda_)

        return adv

    @staticmethod
    def backward(ctx, grad_adv):
        return None, None, None, None, None

class GAE(torch.nn.Module):
    """
    Overview:
        Implementation of Generalized Advantage Estimator (arXiv:1506.02438)

    Interface:
        __init__, forward
    """
    def __init__(self, T, B):
        r"""
        Overview
            initialization of gae

        Arguments:
            - T (:obj:`int`): trajectory length
            - B (:obj:`int`): batch size
        """

        super().__init__()
        self.register_buffer('adv', torch.zeros(T, B))

    def forward(self, value, reward, gamma: float = 0.99, lambda_: float = 0.97) -> torch.FloatTensor:
        """
        Overview:
            forward of gae
        Arguments:
            - value (:obj:`torch.FloatTensor`): :math:`(T + 1, B)`, gae input data
            - reward (:obj:`torch.FloatTensor`): :math:`(T, B)`, gae input data
            - gamma (:obj:`float`): the future discount factor, should be in [0, 1], defaults to 0.99.
            - lambda (:obj:`float`): the gae parameter lambda, should be in [0, 1], defaults to 0.97, when lambda -> 0,\
            it induces bias, but when lambda -> 1, it has high variance due to the sum of terms.
        Returns:
            - adv (:obj:`torch.FloatTensor`): :math:`(T, B)`, the calculated advantage
    
        .. note::
            value_{T+1} should be 0 if this trajectory reached a terminal state(done=True), otherwise we use value
            function, this operation is implemented in actor for packing trajectory.
        """
        assert(value.is_cuda)
        assert(reward.is_cuda)

        return GAEFunction.apply(value, reward, gamma, lambda_, self.adv)

