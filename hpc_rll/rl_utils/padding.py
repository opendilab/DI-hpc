import torch
import hpc_rl_utils

# hpc version only support cuda

def Padding1D(inputs):
    n = len(inputs)
    max_shape = 0
    s = []
    for t in inputs:
        s.append(t.shape[0])
        if max_shape < t.shape[0]:
            max_shape = t.shape[0]
    shape = torch.Tensor(s)
    new_x = torch.zeros(n, max_shape)
    mask = torch.zeros(n, max_shape)
    hpc_rl_utils.Pad1DForward(inputs, shape, new_x, mask, max_shape)
    return new_x, mask, shape


def UnPadding1D(inputs, shape):
    n = shape.size(0)
    outputs = []
    for i in range(n):
        outputs.append(torch.zeros(int(shape[i])))
    hpc_rl_utils.Unpad1DForward(inputs, shape, outputs)
    return outputs

def Padding2D(inputs):
    n = len(inputs)
    max_shape0 = 0
    max_shape1 = 0
    s = []
    for t in inputs:
        s.append(t.shape[0])
        s.append(t.shape[1])
        if max_shape0 < t.shape[0]:
            max_shape0 = t.shape[0]
        if max_shape1 < t.shape[1]:
            max_shape1 = t.shape[1]
    shape = torch.Tensor(s)
    new_x = torch.zeros(n, max_shape0, max_shape1)
    mask = torch.zeros(n, max_shape0, max_shape1)
    hpc_rl_utils.Pad2DForward(inputs, shape, new_x, mask, max_shape0, max_shape1)
    return new_x, mask, shape

def UnPadding2D(inputs, shape):
    n = shape.size(0) / 2
    outputs = []
    for i in range(n):
        outputs.append(torch.zeros(int(shape[i * 2], int(shape[i * 2 + 1]))))
    hpc_rl_utils.Unpad1DForward(inputs, shape, outputs)
    return outputs


    

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

