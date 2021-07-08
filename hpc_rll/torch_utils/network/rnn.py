import math

import torch
import torch.nn as nn

import hpc_torch_utils_network

# hpc version only support cuda

class HPCLSTMFunction(torch.autograd.Function):
    @staticmethod

    def forward(ctx, x, wx, wh, bias, ln_gamma, ln_beta, h0, c0, xbuf, hbuf, hn, cn, ifog, ym,
        ln_in, ln_mean, ln_rstd, dropout_mask, dropout_threshold, dgate, dx, dwx, dwh, dbias, d_ln_gamma, d_ln_beta):

        inputs = [x, h0, c0, wx, wh, bias, ln_gamma, ln_beta]
        outputs = [xbuf, hbuf, hn, cn, ifog, ym, ln_in, ln_mean, ln_rstd, dropout_mask]
        hpc_torch_utils_network.LstmForward(inputs, outputs, dropout_threshold)

        bp_inputs = [x, h0, c0, wx, wh, hn, cn, ifog, ym, ln_in, ln_mean, ln_rstd, ln_gamma, dropout_mask]
        bp_outputs = [dgate, xbuf, hbuf, dx, dwx, dwh, dbias, d_ln_gamma, d_ln_beta]
        ctx.bp_inputs = bp_inputs
        ctx.bp_outputs = bp_outputs
        ctx.dropout_threshold = dropout_threshold

        seq_len = x.size(0)
        num_layers = h0.size(0)
        y = ym[num_layers - 1]
        h = hn[seq_len - 1]
        c = cn[seq_len - 1]
        return y, h, c

    @staticmethod
    def backward(ctx, dy, dh, dc):
        inputs = ctx.bp_inputs
        outputs = ctx.bp_outputs
        outputs.append(dy)
        outputs.append(dh)
        outputs.append(dc)
        dropout_threshold = ctx.dropout_threshold

        hpc_torch_utils_network.LstmBackward(inputs, outputs, dropout_threshold)

        dx = outputs[3]
        dwx = outputs[4]
        dwh = outputs[5]
        dbias = outputs[6]
        d_ln_gamma = outputs[7]
        d_ln_beta = outputs[8]
        return dx, dwx, dwh, dbias, d_ln_gamma, d_ln_beta, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None

class LSTM(nn.Module):
    r"""
    Overview:
        Implimentation of HPC LSTM cell

        .. note::
            for begainners, you can reference <https://zhuanlan.zhihu.com/p/32085405> to learn the basics about lstm

    Interface:
        __init__, forward
    """
    def __init__(self, seq_len, batch_size, input_size, hidden_size, num_layers = 1, norm_type='LN', dropout=0.0):
        r"""
        Overview:
            initializate the LSTM cell

        Arguments:
            - seq_len (:obj:`int`): length of the sequence
            - batch_size (:obj:`int`): size of the batch
            - input_size (:obj:`int`): size of the input vector
            - hidden_size (:obj:`int`): size of the hidden state vector
            - num_layers (:obj:`int`): number of lstm layers
            - norm_type (:obj:`str`): type of the normaliztion, (default: LN)
            - dropout (:obj:float):  dropout rate, default set to .0
        """
        super().__init__()

        # TODO only support layer norm now
        assert norm_type in ['LN']

        self.seq_len = seq_len
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.norm_type = norm_type
        self.dropout = dropout

        # init parameters
        gain = math.sqrt(1. / self.hidden_size)

        wxlist = []
        whlist = []
        biaslist = []
        dims = [input_size] + [hidden_size] * num_layers
        for l in range(num_layers):
            w = torch.zeros(dims[l] * (dims[l + 1] * 4))
            h = torch.zeros(hidden_size * (hidden_size * 4))
            b = torch.zeros(hidden_size * 4)
            torch.nn.init.uniform_(w, -gain, gain)
            torch.nn.init.uniform_(h, -gain, gain)
            torch.nn.init.uniform_(b, -gain, gain)
            wxlist.append(w)
            whlist.append(h)
            biaslist.append(b)
        wx = torch.flatten(torch.cat(wxlist))
        wh = torch.flatten(torch.cat(whlist))
        bias = torch.flatten(torch.cat(biaslist))
        self.register_parameter("wx", nn.Parameter(wx))
        self.register_parameter("wh", nn.Parameter(wh))
        self.register_parameter("bias", nn.Parameter(bias))

        self.register_parameter("ln_gamma", nn.Parameter(torch.ones(num_layers, hidden_size * 4 * 2)))
        self.register_parameter("ln_beta", nn.Parameter(torch.zeros(num_layers, hidden_size * 4 * 2)))

        # Note: only use to validation
        #self.load_params()

        # init buffers

        self.register_buffer('xbuf', torch.zeros(seq_len, batch_size, hidden_size * 4))
        self.register_buffer('hbuf', torch.zeros(batch_size, hidden_size * 4))
        self.register_buffer('ifog', torch.zeros(num_layers, seq_len, batch_size, hidden_size * 4))
        self.register_buffer('hn', torch.zeros(seq_len, num_layers, batch_size, hidden_size))
        self.register_buffer('cn', torch.zeros(seq_len, num_layers, batch_size, hidden_size))
        self.register_buffer('ym', torch.zeros(num_layers, seq_len, batch_size, hidden_size))

        self.register_buffer('ln_in', torch.zeros(num_layers, seq_len, batch_size, hidden_size * 4 * 2))
        self.register_buffer('ln_mean', torch.zeros(num_layers, seq_len, batch_size, batch_size * 2))
        self.register_buffer('ln_rstd', torch.zeros(num_layers, seq_len, batch_size * 2))

        self.register_buffer('dropout_mask', torch.zeros((num_layers - 1, seq_len, batch_size, hidden_size), dtype=torch.int32))

        self.register_buffer('dgate', torch.zeros(num_layers, seq_len, batch_size, hidden_size * 4))
        self.register_buffer('dx', torch.zeros(seq_len, batch_size, input_size))
        self.register_buffer('dwx', torch.zeros_like(self.wx))
        self.register_buffer('dwh', torch.zeros_like(self.wh))
        self.register_buffer('dbias', torch.zeros_like(self.bias))
        self.register_buffer('d_ln_gamma', torch.zeros_like(self.ln_gamma))
        self.register_buffer('d_ln_beta', torch.zeros_like(self.ln_beta))


    def load_params(self):
        input_dict = torch.load('origin.input')
        wx = input_dict["wx"]
        wh = input_dict["wh"]
        bias = input_dict["bias"]
        self.wx.data.copy_(wx.data)
        self.wh.data.copy_(wh.data)
        self.bias.data.copy_(bias.data)


    def forward(self, inputs, prev_state):
        r"""
        Overview:
            Take the previous state and the input and calculate the output and the nextstate
        Arguments:
            - inputs (:obj:`tensor`): :math: `(seq_len, batch_size, input_size)`, input vector of cell
            - prev_state (:obj:`tensor`): None or two tensors of :math: `(num_layers, batch_size, hidden_size)`, for h0 and c0
        Returns:
            - output (:obj:`tensor`): :math: `(seq_len, batch_size, hidden_size)`, output from lstm
            - next_state (:obj:`tensor`): two tensors of :math: `(num_layers, batch_size, hidden_size)`, hidden state from lstm
        """

        assert(inputs.is_cuda)

        if prev_state is None:
            num_directions = 1
            zeros = torch.zeros(num_directions * self.num_layers, self.batch_size, self.hidden_size, dtype=inputs.dtype, device=inputs.device)
            prev_state = (zeros, zeros)

        h0, c0 = prev_state
        assert(h0.is_cuda)
        assert(c0.is_cuda)

        y, h, c = HPCLSTMFunction.apply(inputs, self.wx, self.wh, self.bias, self.ln_gamma, self.ln_beta, h0, c0,
                self.xbuf, self.hbuf, self.hn, self.cn, self.ifog, self.ym,
                self.ln_in, self.ln_mean, self.ln_rstd, self.dropout_mask, self.dropout,
                self.dgate, self.dx, self.dwx, self.dwh, self.dbias, self.d_ln_gamma, self.d_ln_beta)
        output = y
        next_state = [h, c]
        return output, next_state

