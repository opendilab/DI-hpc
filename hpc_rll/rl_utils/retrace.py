import torch
import torch.nn.functional as F
from typing import Optional
from collections import namedtuple
import hpc_rl_utils

class Retrace(torch.nn.Module):
    def __init__(self, T, B, N):

        super().__init__()
        self.register_buffer('q_retraces', torch.zeros(T+1, B, 1))
        self.register_buffer('q_gather', torch.zeros(T, B, 1))
        self.register_buffer('ratio_gather', torch.zeros(T, B, 1))
        self.register_buffer('tmp_retraces', torch.zeros(B, 1))

    def forward(self, q_values, v_pred, rewards, actions, weights, ratio, gamma=0.99):
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

        assert(q_values.is_cuda)
        assert(v_pred.is_cuda)
        assert(rewards.is_cuda)
        assert(actions.is_cuda)
        assert(weights.is_cuda)
        assert(ratio.is_cuda)
        inputs = [q_values, v_pred, rewards, actions, weights, ratio]
        outputs = [self.q_retraces, self.q_gather, self.ratio_gather, self.tmp_retraces]
        hpc_rl_utils.RetraceForward(inputs, outputs, gamma)




        return self.q_retraces