import torch
from typing import Tuple
import hpc_torch_utils_network

# hpc version only support cuda

class ScatterConnectionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, location, output, grad_in, scatter_type):
        inputs = [input, location]
        outputs = [output]
        hpc_torch_utils_network.ScatterConnectionForward(inputs, outputs, scatter_type)

        ctx.bp_inputs = [location]
        ctx.bp_outputs = [grad_in]

        return output

    @staticmethod
    def backward(ctx, grad_out):
        inputs = [grad_out]
        for var in ctx.bp_inputs:
            inputs.append(var)
        outputs = ctx.bp_outputs

        hpc_torch_utils_network.ScatterConnectionBackward(inputs, outputs)
        grad_in = outputs[0]
        return grad_in, None, None, None, None

class ScatterConnection(torch.nn.Module):
    r"""
    Overview:
        Scatter feature to its corresponding location
        In alphastar, each entity is embedded into a tensor, these tensors are scattered into a feature map
        with map size

    Interface:
        __init__, forward
    """

    def __init__(self, B, M, N, H, W, scatter_type) -> None:
        r"""
        Overview
            initialization of scatter connection

        Arguments:
            - B (:obj:`int`): batch size
            - M (:obj:`int`): the number of entity
            - N (:obj:`int`): the dimension of entity attributes
            - H (:obj:`int`): height of spatial feature
            - W (:obj:`int`): width of spatial feature
            - scatter_type (:obj:`str`): add or cover, if two entities have same location, scatter type decides the
                first one should be covered or added to second one
        """

        super().__init__()
        self.B = B
        self.M = M
        self.N = N
        self.H = H
        self.W = W
        self.scatter_type = scatter_type
        assert self.scatter_type in ['cover', 'add']

        self.register_buffer('output', torch.zeros(B, N, H, W))
        self.register_buffer('grad_in', torch.zeros(B, M, N))

    def forward(self, x: torch.Tensor, location: torch.Tensor) -> torch.Tensor:
        """
            Overview:
                forward of scatter connection, scatter x into a spatial feature map
            Arguments:
                - x (:obj:`torch.FloatTensor`): :math: `(B, M, N)`, the input tensor
                - location (:obj:`torch.LongTensor`): :math: `(B, M, 2)`, each location should be (y, x)
            Returns:
                - output (:obj:`FloatTensor`): :math: `(B, N, H, W)`, the scattered feature map

            .. note::
                when there are some overlapping in locations, ``cover`` mode will result in the loss of information, we
                use the addition as temporal substitute.
        """

        assert(x.is_cuda)
        assert(location.is_cuda)

        output = ScatterConnectionFunction.apply(x, location, self.output, self.grad_in, self.scatter_type)
        return output

