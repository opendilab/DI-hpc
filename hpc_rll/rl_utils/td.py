import torch
import hpc_rl_utils
from typing import Optional

# hpc version only support cuda

class DistNStepTDFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, dist, next_n_dist, action, next_n_action, reward, done, weight, gamma, v_min, v_max,
            td_err, loss, buf, grad_dist):
        inputs = [dist, next_n_dist, action, next_n_action, reward, done, weight]
        outputs = [td_err, loss, buf]
        hpc_rl_utils.DistNStepTdForward(inputs, outputs, gamma, v_min, v_max)

        ctx.bp_inputs = [buf, action]
        ctx.bp_outputs = [grad_dist]

        return loss, td_err

    @staticmethod
    def backward(ctx, grad_loss, grad_td_err):
        inputs = [grad_loss]
        for var in ctx.bp_inputs:
            inputs.append(var)
        outputs = ctx.bp_outputs

        hpc_rl_utils.DistNStepTdBackward(inputs, outputs)
        grad_dist = outputs[0]
        return grad_dist, None, None, None, None, None, None, None, None, None, None, None, None, None


class DistNStepTD(torch.nn.Module):
    """
    Overview:
        Multistep (1 step or n step) td_error for distributed q-learning based algorithm

    Interface:
        __init__, forward
    """

    def __init__(self, T, B, N, n_atom):
        r"""
        Overview
            initialization of dist_nstep_td_error

        Arguments:
            - T (:obj:`int`): trajectory length
            - B (:obj:`int`): batch size
            - N (:obj:`int`): action dim
            - n_atom (:obj:`int`): the number of atom sample point
        """

        super().__init__()
        self.n_atom = n_atom

        self.register_buffer('weight', torch.ones(B))
        self.register_buffer('td_error_per_sample', torch.zeros(B))
        self.register_buffer('loss', torch.zeros(1))
        # B for reward x fp reward_factor, (B * n_atom) for fp proj_dist, and the same (B * n_atom) for bp grad
        self.register_buffer('buf', torch.zeros(B + B * n_atom))
        self.register_buffer('grad_dist', torch.zeros(B, N, n_atom))

    def forward(self, dist, next_n_dist, action, next_n_action, reward, done, weight,
            gamma: float,
            v_min: float,
            v_max: float
            ) -> torch.Tensor:
        """
        Overview:
            forward of dist_nstep_td_error
        Arguments:
            - dist (:obj:`torch.FloatTensor`): :math:`(B, N, n_atom)`
            - next_n_dist (:obj:`torch.FloatTensor`): :math:`(B, N, n_atom)`
            - action (:obj:`torch.LongTensor`): :math:`(B, )`
            - next_n_action (:obj:`torch.LongTensor`): :math:`(B, )`
            - reward (:obj:`torch.FloatTensor`): :math:`(T, B)`
            - done (:obj:`torch.BoolTensor`): :math:`(B, )`, whether done in last timestep
            - weight (:obj:`torch.FloatTensor` or None): :math:`(B, )`, the training sample weight
            - gamma (:obj:`float`): discount factor
            - v_min (:obj:`float`): value distribution minimum value
            - v_max (:obj:`float`): value distribution maximum value
        Returns:
            - loss (:obj:`torch.Tensor`): :math:`()`, 0-dim tensor
            - td_error_per_sample (:obj:`torch.Tensor`): :math:`(B, )`, nstep td error, 1-dim tensor
        Note:
            only support default mode:
            - criterion (:obj:`torch.nn.modules`): loss function criterion, default set to MSELoss(reduction='none')
        """

        assert(dist.is_cuda)
        assert(next_n_dist.is_cuda)
        assert(action.is_cuda)
        assert(next_n_action.is_cuda)
        assert(reward.is_cuda)
        assert(done.is_cuda)
        if weight is None:
            weight = self.weight
        else:
            assert(weight.is_cuda)

        batch_size = action.shape[0]
        batch_range = torch.arange(batch_size)
        assert (dist[batch_range, action] > 0.0).all(), ("dist act", dist[batch_range, action], "dist:", dist)

        loss, td_err = DistNStepTDFunction.apply(dist, next_n_dist, action, next_n_action, reward, done, weight, gamma, v_min, v_max,
                self.td_error_per_sample, self.loss, self.buf, self.grad_dist)

        return loss, td_err


class TDLambdaFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, reward, weight, gamma, lambda_,
            loss, grad_buf, grad_value):
        inputs = [value, reward, weight]
        outputs = [loss, grad_buf]
        hpc_rl_utils.TdLambdaForward(inputs, outputs, gamma, lambda_)

        ctx.bp_inputs = [grad_buf]
        ctx.bp_outputs = [grad_value]
        ctx.gamma = gamma
        ctx.lambda_ = lambda_

        return loss

    @staticmethod
    def backward(ctx, grad_loss):
        inputs = [grad_loss]
        for var in ctx.bp_inputs:
            inputs.append(var)
        outputs = ctx.bp_outputs

        hpc_rl_utils.TdLambdaBackward(inputs, outputs)
        grad_value = outputs[0]
        return grad_value, None, None, None, None, None, None, None


class TDLambda(torch.nn.Module):
    """
    Overview:
        Computing TD(lambda) loss given constant gamma and lambda.
        There is no special handling for terminal state value,
        if some state has reached the terminal, just fill in zeros for values and rewards beyond terminal
        (*including the terminal state*, values[terminal] should also be 0)

    Interface:
        __init__, forward
    """
    def __init__(self, T, B):
        r"""
        Overview
            initialization of TD(lambda)

        Arguments:
            - T (:obj:`int`): trajectory length
            - B (:obj:`int`): batch size
        """

        super().__init__()
        self.register_buffer('weight', torch.ones(B))
        self.register_buffer('loss', torch.zeros(1))
        self.register_buffer('grad_buf', torch.zeros(T, B))
        self.register_buffer('grad_value', torch.zeros(T + 1, B))

    def forward(self, value, reward, weight = None, gamma: float = 0.9, lambda_: float = 0.8) -> torch.Tensor:
        """
        Overview:
            forward of TD(lambda)
        Arguments:
            - value (:obj:`torch.FloatTensor`): :math:`(T + 1, B)`, the estimation of the state value at step 0 to T
            - reward (:obj:`torch.FloatTensor`): :math:`(T, B)`, the returns from time step 0 to T-1
            - weight (:obj:`torch.FloatTensor` or None): :math:`(B, )`, the training sample weight
            - gamma (:obj:`float`): constant discount factor gamma, should be in [0, 1], defaults to 0.9
            - lambda (:obj:`float`): constant lambda, should be in [0, 1], defaults to 0.8
        Returns:
            - loss (:obj:`torch.Tensor`): :math:`()`, 0-dim tensor, computed MSE loss, averaged over the batch
        """
        assert(value.is_cuda)
        assert(reward.is_cuda)
        if weight is None:
            weight = self.weight
        else:
            assert(weight.is_cuda)

        loss = TDLambdaFunction.apply(value, reward, weight, gamma, lambda_,
                self.loss, self.grad_buf, self.grad_value)
        return loss


class QNStepTDFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, next_n_q, action, next_n_action, reward, done, weight, gamma,
            td_err, loss, grad_buf, grad_q):
        inputs = [q, next_n_q, action, next_n_action, reward, done, weight]
        outputs = [td_err, loss, grad_buf]
        hpc_rl_utils.QNStepTdForward(inputs, outputs, gamma)

        ctx.bp_inputs = [grad_buf, action]
        ctx.bp_outputs = [grad_q]

        return loss, td_err

    @staticmethod
    def backward(ctx, grad_loss, grad_td_err):
        inputs = [grad_loss]
        for var in ctx.bp_inputs:
            inputs.append(var)
        outputs = ctx.bp_outputs

        hpc_rl_utils.QNStepTdBackward(inputs, outputs)
        grad_q = outputs[0]
        return grad_q, None, None, None, None, None, None, None, None, None, None, None


class QNStepTD(torch.nn.Module):
    """
    Overview:
        Multistep (1 step or n step) td_error for q-learning based algorithm

    Interface:
        __init__, forward
    """

    def __init__(self, T, B, N):
        r"""
        Overview
            initialization of q_nstep_td_error

        Arguments:
            - T (:obj:`int`): trajectory length
            - B (:obj:`int`): batch size
            - N (:obj:`int`): action dim
        """

        super().__init__()
        self.register_buffer('weight', torch.ones(B))
        self.register_buffer('td_error_per_sample', torch.zeros(B))
        self.register_buffer('loss', torch.zeros(1))
        self.register_buffer('grad_buf', torch.zeros(B))
        self.register_buffer('grad_q', torch.zeros(B, N))

    def forward(self, q, next_n_q, action, next_n_action, reward, done, weight, gamma: float) -> torch.Tensor:
        """
        Overview:
            forward of q_nstep_td_error
        Arguments:
            - q (:obj:`torch.FloatTensor`): :math:`(B, N)`
            - next_n_q (:obj:`torch.FloatTensor`): :math:`(B, N)`
            - action (:obj:`torch.LongTensor`): :math:`(B, )`
            - next_n_action (:obj:`torch.LongTensor`): :math:`(B, )`
            - reward (:obj:`torch.FloatTensor`): :math:`(T, B)`
            - done (:obj:`torch.BoolTensor`): :math:`(B, )`, whether done in last timestep
            - weight (:obj:`torch.FloatTensor` or None): :math:`(B, )`, the training sample weight
            - gamma (:obj:`float`): discount factor
        Returns:
            - loss (:obj:`torch.Tensor`): :math:`()`, 0-dim tensor
            - td_error_per_sample (:obj:`torch.Tensor`): :math:`(B, )`, nstep td error, 1-dim tensor
        Note:
            only support default mode:
            - criterion (:obj:`torch.nn.modules`): loss function criterion, default set to MSELoss(reduction='none')
        """

        assert(q.is_cuda)
        assert(next_n_q.is_cuda)
        assert(action.is_cuda)
        assert(next_n_action.is_cuda)
        assert(reward.is_cuda)
        assert(done.is_cuda)
        if weight is None:
            weight = self.weight
        else:
            assert(weight.is_cuda)

        loss, td_err = QNStepTDFunction.apply(q, next_n_q, action, next_n_action, reward, done, weight, gamma,
                self.td_error_per_sample, self.loss, self.grad_buf, self.grad_q)

        return loss, td_err


class QNStepTDRescaleFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, next_n_q, action, next_n_action, reward, done, weight, gamma,
            td_err, loss, grad_buf, grad_q):
        inputs = [q, next_n_q, action, next_n_action, reward, done, weight]
        outputs = [td_err, loss, grad_buf]
        hpc_rl_utils.QNStepTdRescaleForward(inputs, outputs, gamma)

        ctx.bp_inputs = [grad_buf, action]
        ctx.bp_outputs = [grad_q]

        return loss, td_err

    @staticmethod
    def backward(ctx, grad_loss, grad_td_err):
        inputs = [grad_loss]
        for var in ctx.bp_inputs:
            inputs.append(var)
        outputs = ctx.bp_outputs

        hpc_rl_utils.QNStepTdRescaleBackward(inputs, outputs)
        grad_q = outputs[0]
        return grad_q, None, None, None, None, None, None, None, None, None, None, None


class QNStepTDRescale(torch.nn.Module):
    """
    Overview:
        Multistep (1 step or n step) td_error for q-learning based algorithm with value rescaling

    Interface:
        __init__, forward
    """

    def __init__(self, T, B, N):
        r"""
        Overview
            initialization of q_nstep_td_error_with_rescale

        Arguments:
            - T (:obj:`int`): trajectory length
            - B (:obj:`int`): batch size
            - N (:obj:`int`): action dim
        """

        super().__init__()
        self.register_buffer('weight', torch.ones(B))
        self.register_buffer('td_error_per_sample', torch.zeros(B))
        self.register_buffer('loss', torch.zeros(1))
        self.register_buffer('grad_buf', torch.zeros(B))
        self.register_buffer('grad_q', torch.zeros(B, N))

    def forward(self, q, next_n_q, action, next_n_action, reward, done, weight, gamma: float) -> torch.Tensor:
        """
        Overview:
            forward of q_nstep_td_error_with_rescale
        Arguments:
            - q (:obj:`torch.FloatTensor`): :math:`(B, N)`
            - next_n_q (:obj:`torch.FloatTensor`): :math:`(B, N)`
            - action (:obj:`torch.LongTensor`): :math:`(B, )`
            - next_n_action (:obj:`torch.LongTensor`): :math:`(B, )`
            - reward (:obj:`torch.FloatTensor`): :math:`(T, B)`
            - done (:obj:`torch.BoolTensor`): :math:`(B, )`, whether done in last timestep
            - weight (:obj:`torch.FloatTensor` or None): :math:`(B, )`, the training sample weight
            - gamma (:obj:`float`): discount factor
        Returns:
            - loss (:obj:`torch.Tensor`): :math:`()`, 0-dim tensor
            - td_error_per_sample (:obj:`torch.Tensor`): :math:`(B, )`, nstep td error, 1-dim tensor
        Note:
            only support default mode:
            - criterion (:obj:`torch.nn.modules`): loss function criterion, default set to MSELoss(reduction='none')
            - trans_fn (:obj:`Callable`): value transfrom function, default to value_transform\
                (refer to hpc_rl/origin/td.py)
            - inv_trans_fn (:obj:`Callable`): value inverse transfrom function, default to value_inv_transform\
                (refer to hpc_rl/origin/td.py)
        """

        assert(q.is_cuda)
        assert(next_n_q.is_cuda)
        assert(action.is_cuda)
        assert(next_n_action.is_cuda)
        assert(reward.is_cuda)
        assert(done.is_cuda)
        if weight is None:
            weight = self.weight
        else:
            assert(weight.is_cuda)

        loss, td_err = QNStepTDRescaleFunction.apply(q, next_n_q, action, next_n_action, reward, done, weight, gamma,
                self.td_error_per_sample, self.loss, self.grad_buf, self.grad_q)

        return loss, td_err


class IQNNStepTDErrorFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, next_n_q, action, next_n_action, reward, done, replay_quantiles, weight, value_gamma, gamma, kappa,
            loss, td_err_per_sample, bellman_err_buf, quantile_huber_loss_buf, grad_buf, grad_q):
        inputs = [q, next_n_q, action, next_n_action, reward, done, replay_quantiles, weight, value_gamma]
        outputs = [loss, td_err_per_sample, bellman_err_buf, quantile_huber_loss_buf, grad_buf]
        hpc_rl_utils.IQNNStepTDErrorForward(inputs, outputs, gamma, kappa)

        ctx.bp_inputs = [grad_buf, weight, action]
        ctx.bp_outputs = [grad_q]

        return loss, td_err_per_sample

    @staticmethod
    def backward(ctx, grad_loss, grad_td_err_per_sample):
        inputs = [grad_loss]
        for var in ctx.bp_inputs:
            inputs.append(var)
        outputs = ctx.bp_outputs

        hpc_rl_utils.IQNNStepTDErrorBackward(inputs, outputs)
        grad_q = outputs[0]
        return grad_q, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None


class IQNNStepTDError(torch.nn.Module):

    """
    Overview:
        Multistep (1 step or n step) td_error with in IQN, \
                referenced paper Implicit Quantile Networks for Distributional Reinforcement Learning \
                <https://arxiv.org/pdf/1806.06923.pdf>
   Interface:
        __init__, forward
    """

    def __init__(self, tau, tauPrime, T, B, N):
        r"""
        Overview
            initialization of iqn_nstep_td_error

        Arguments:
            - tau (:obj:`int`): num of quantiles
            - tauPrime (:obj:`int`): num of quantiles
            - T (:obj:`int`): trajectory length
            - B (:obj:`int`): batch size
            - N (:obj:`int`): action dim
            - gamma (:obj:`float`): discount factor
        """

        super().__init__()
        self.tau = tau
        self.tauPrime = tauPrime
        self.T = T
        self.B = B
        self.N = N
        self.register_buffer('weight', torch.ones(B))
        self.register_buffer('td_error_per_sample', torch.zeros(B))
        self.register_buffer('loss', torch.zeros(1))
        self.register_buffer('value_gamma', torch.zeros(B))
        self.register_buffer('bellman_err_buf', torch.zeros(B, tauPrime, tau))
        self.register_buffer('quantile_huber_loss_buf', torch.zeros(B, tauPrime, tau))
        self.register_buffer('grad_buf', torch.zeros(B, tauPrime, tau))
        self.register_buffer('grad_q', torch.zeros(tau, B, N))

    def forward(self, q, next_n_q, action, next_n_action, reward, done, replay_quantiles,
            gamma: float, kappa: float = 1.0,
            weight: Optional[torch.Tensor] = None,
            value_gamma: Optional[torch.Tensor] = None,
            ) -> torch.Tensor:
        """
        Overview:
            forward of iqn_nstep_td_error
        Arguments:
            - q (:obj:`torch.FloatTensor`): :math:`(tau, B, N)` i.e. [tau x batch_size, action_dim]
            - next_n_q (:obj:`torch.FloatTensor`): :math:`(tau, B, N)`
            - action (:obj:`torch.LongTensor`): :math:`(B, )`
            - next_n_action (:obj:`torch.LongTensor`): :math:`(B, )`
            - reward (:obj:`torch.FloatTensor`): :math:`(T, B)`, where T is timestep(nstep
            - done (:obj:`torch.BoolTensor`) :math:`(B, )`, whether done in last timestep
            - replay_quantiles (:obj:`torch.FloatTensor`): :math:`(B)`
            - gamma (:obj:`float`): discount factor
            - kappa (:obj:`float`)
            - weight (:obj:`torch.FloatTensor` or None): :math:`(B, )`, the training sample weight
            - value_gamma (:obj:`torch.FloatTensor`): :math:`(B)`
        Returns:
            - loss (:obj:`torch.Tensor`): nstep td error, 0-dim tensor
            - td_error_per_sample (:obj:`torch.Tensor`): :math:`(B, )`, iqn nstep td error per sample
        """

        assert(q.is_cuda)
        assert(next_n_q.is_cuda)
        assert(action.is_cuda)
        assert(next_n_action.is_cuda)
        assert(reward.is_cuda)
        assert(done.is_cuda)
        assert(replay_quantiles.is_cuda)
        if weight is None:
            weight = self.weight
        else:
            assert(weight.is_cuda)
        if value_gamma is None:
            self.value_gamma.fill_(gamma ** self.T)
            value_gamma = self.value_gamma
        else:
            assert(value_gamma.is_cuda)

        loss, td_err_per_sample = IQNNStepTDErrorFunction.apply(q, next_n_q, action, next_n_action,
                reward, done, replay_quantiles, weight, value_gamma, gamma, kappa,
                self.loss, self.td_error_per_sample, self.bellman_err_buf, self.quantile_huber_loss_buf, self.grad_buf, self.grad_q)

        return loss, td_err_per_sample


class QRDQNNStepTDErrorFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, next_n_q, action, next_n_action, reward, done, weight, value_gamma, gamma,
            loss, td_err_per_sample, bellman_err_buf, quantile_huber_loss_buf, grad_buf, grad_q):
        inputs = [q, next_n_q, action, next_n_action, reward, done, weight, value_gamma]
        outputs = [loss, td_err_per_sample, bellman_err_buf, quantile_huber_loss_buf, grad_buf]
        hpc_rl_utils.QRDQNNStepTDErrorForward(inputs, outputs, gamma)

        ctx.bp_inputs = [grad_buf, weight, action]
        ctx.bp_outputs = [grad_q]

        return loss, td_err_per_sample

    @staticmethod
    def backward(ctx, grad_loss, grad_td_err_per_sample):
        inputs = [grad_loss]
        for var in ctx.bp_inputs:
            inputs.append(var)
        outputs = ctx.bp_outputs

        hpc_rl_utils.QRDQNNStepTDErrorBackward(inputs, outputs)
        grad_q = outputs[0]
        return grad_q, None, None, None, None, None, None, None, None, None, None, None, None, None, None


class QRDQNNStepTDError(torch.nn.Module):

    """
    Overview:
        Multistep (1 step or n step) td_error with in QRDQN
   Interface:
        __init__, forward
    """

    def __init__(self, tau, T, B, N):
        r"""
        Overview
            initialization of qrdqn_nstep_td_error

        Arguments:
            - tau (:obj:`int`): num of quantiles
            - T (:obj:`int`): trajectory length
            - B (:obj:`int`): batch size
            - N (:obj:`int`): action dim
            - gamma (:obj:`float`): discount factor
        """

        super().__init__()
        self.tau = tau
        self.T = T
        self.B = B
        self.N = N
        self.register_buffer('weight', torch.ones(B))
        self.register_buffer('td_error_per_sample', torch.zeros(B))
        self.register_buffer('loss', torch.zeros(1))
        self.register_buffer('value_gamma', torch.zeros(B))
        self.register_buffer('bellman_err_buf', torch.zeros(B, tau, tau))
        self.register_buffer('quantile_huber_loss_buf', torch.zeros(B, tau, tau))
        self.register_buffer('grad_buf', torch.zeros(B, tau))
        self.register_buffer('grad_q', torch.zeros(B, N, tau))

    def forward(self, q, next_n_q, action, next_n_action, reward, done,
            gamma: float,
            weight: Optional[torch.Tensor] = None,
            value_gamma: Optional[torch.Tensor] = None,
            ) -> torch.Tensor:
        """
        Overview:
            forward of qrdqn_nstep_td_error
        Arguments:
            - q (:obj:`torch.FloatTensor`): :math:`(B, N, tau)` i.e. [batch_size, action_dim, tau]
            - next_n_q (:obj:`torch.FloatTensor`): :math:`(B, N, tau)`
            - action (:obj:`torch.LongTensor`): :math:`(B, )`
            - next_n_action (:obj:`torch.LongTensor`): :math:`(B, )`
            - reward (:obj:`torch.FloatTensor`): :math:`(T, B)`, where T is timestep(nstep)
            - done (:obj:`torch.BoolTensor`) :math:`(B, )`, whether done in last timestep
            - gamma (:obj:`float`): discount factor
            - weight (:obj:`torch.FloatTensor` or None): :math:`(B, )`, the training sample weight
            - value_gamma (:obj:`torch.FloatTensor`): :math:`(B)`
        Returns:
            - loss (:obj:`torch.Tensor`): nstep td error, 0-dim tensor
            - td_error_per_sample (:obj:`torch.Tensor`): :math:`(B, )`, qrdqn nstep td error per sample
        """

        assert(q.is_cuda)
        assert(next_n_q.is_cuda)
        assert(action.is_cuda)
        assert(next_n_action.is_cuda)
        assert(reward.is_cuda)
        assert(done.is_cuda)
        if weight is None:
            weight = self.weight
        else:
            assert(weight.is_cuda)
        if value_gamma is None:
            self.value_gamma.fill_(gamma ** self.T)
            value_gamma = self.value_gamma
        else:
            assert(value_gamma.is_cuda)

        loss, td_err_per_sample = QRDQNNStepTDErrorFunction.apply(q, next_n_q, action, next_n_action,
                reward, done, weight, value_gamma, gamma,
                self.loss, self.td_error_per_sample, self.bellman_err_buf, self.quantile_huber_loss_buf, self.grad_buf, self.grad_q)

        return loss, td_err_per_sample

