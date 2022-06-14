from collections import namedtuple
import torch
import torch.nn.functional as F
import pdb

coma_data = namedtuple('coma_data', ['logit', 'action', 'q_value', 'target_q_value', 'reward', 'weight'])
coma_loss = namedtuple('coma_loss', ['policy_loss', 'q_value_loss', 'entropy_loss'])

def multistep_forward_view(
        bootstrap_values: torch.Tensor, rewards: torch.Tensor, gammas: float, lambda_: float #两个输入均为(T-1, B, A)
) -> torch.Tensor:
    r"""
    Overview:
        Same as trfl.sequence_ops.multistep_forward_view
        Implementing (12.18) in Sutton & Barto

        ```
        result[T-1] = rewards[T-1] + gammas[T-1] * bootstrap_values[T]
        for t in 0...T-2 :
        result[t] = rewards[t] + gammas[t]*(lambdas[t]*result[t+1] + (1-lambdas[t])*bootstrap_values[t+1])
        ```

        Assuming the first dim of input tensors correspond to the index in batch
        There is no special handling for terminal state value,
        if some state has reached the terminal, just fill in zeros for values and rewards beyond terminal
        (including the terminal state, which is, bootstrap_values[terminal] should also be 0)
    Arguments:
        - bootstrap_values (:obj:`torch.Tensor`): estimation of the value at *step 1 to T*, of size [T_traj, batchsize]
        - rewards (:obj:`torch.Tensor`): the returns from 0 to T-1, of size [T_traj, batchsize]
        - gammas (:obj:`torch.Tensor`): discount factor for each step (from 0 to T-1), of size [T_traj, batchsize]
        - lambda (:obj:`torch.Tensor`): determining the mix of bootstrapping vs further accumulation of \
            multistep returns at each timestep of size [T_traj, batchsize], the element for T-1 is ignored \
            and effectively set to 0, as there is no information about future rewards.
    Returns:
        - ret (:obj:`torch.Tensor`): Computed lambda return value \
            for each state from 0 to T-1, of size [T_traj, batchsize]
    """
    result = torch.empty_like(rewards)
    # Forced cutoff at the last one
    result[-1, :] = rewards[-1, :] + gammas[-1, :] * bootstrap_values[-1, :]
    discounts = gammas * lambda_
    for t in reversed(range(rewards.size()[0] - 1)):
        result[t, :] = rewards[t, :] \
                       + discounts[t, :] * result[t + 1, :] \
                       + (gammas[t, :] - discounts[t, :]) * bootstrap_values[t, :]

    return result

def generalized_lambda_returns(
        bootstrap_values: torch.Tensor, rewards: torch.Tensor, gammas: float, lambda_: float
) -> torch.Tensor:
    r"""
    Overview:
        Functional equivalent to trfl.value_ops.generalized_lambda_returns
        https://github.com/deepmind/trfl/blob/2c07ac22512a16715cc759f0072be43a5d12ae45/trfl/value_ops.py#L74
        Passing in a number instead of tensor to make the value constant for all samples in batch
    Arguments:
        - bootstrap_values (:obj:`torch.Tensor` or :obj:`float`):
          estimation of the value at step 0 to *T*, of size [T_traj+1, batchsize]
        - rewards (:obj:`torch.Tensor`): the returns from 0 to T-1, of size [T_traj, batchsize]
        - gammas (:obj:`torch.Tensor` or :obj:`float`):
          discount factor for each step (from 0 to T-1), of size [T_traj, batchsize]
        - lambda (:obj:`torch.Tensor` or :obj:`float`): determining the mix of bootstrapping
          vs further accumulation of multistep returns at each timestep, of size [T_traj, batchsize]
    Returns:
        - return (:obj:`torch.Tensor`): Computed lambda return value
          for each state from 0 to T-1, of size [T_traj, batchsize]
    """
    if not isinstance(gammas, torch.Tensor):
        gammas = gammas * torch.ones_like(rewards)
    if not isinstance(lambda_, torch.Tensor):
        lambda_ = lambda_ * torch.ones_like(rewards)
    bootstrap_values_tp1 =  bootstrap_values[1:, :] #target q_taken
    return multistep_forward_view(bootstrap_values_tp1, rewards, gammas, lambda_)




def coma_error(data: namedtuple, gamma: float, lambda_: float) -> namedtuple:
    """
    Overview:
        Implementation of COMA
    Arguments:
        - data (:obj:`namedtuple`): coma input data with fieids shown in ``coma_data``
    Returns:
        - coma_loss (:obj:`namedtuple`): the coma loss item, all of them are the differentiable 0-dim tensor
    Shapes:
        - logit (:obj:`torch.FloatTensor`): :math:`(T, B, A, N)`, where B is batch size A is the agent num, and N is \
            action dim
        - action (:obj:`torch.LongTensor`): :math:`(T, B, A)`
        - q_value (:obj:`torch.FloatTensor`): :math:`(T, B, A, N)`
        - target_q_value (:obj:`torch.FloatTensor`): :math:`(T, B, A, N)`
        - reward (:obj:`torch.FloatTensor`): :math:`(T, B)`
        - weight (:obj:`torch.FloatTensor` or :obj:`None`): :math:`(T ,B, A)`
        - policy_loss (:obj:`torch.FloatTensor`): :math:`()`, 0-dim tensor
        - value_loss (:obj:`torch.FloatTensor`): :math:`()`
        - entropy_loss (:obj:`torch.FloatTensor`): :math:`()`
    """
    logit, action, q_value, target_q_value, reward, weight = data
    if weight is None:
        weight = torch.ones_like(action)
    q_taken = torch.gather(q_value, -1, index=action.unsqueeze(-1)).squeeze(-1) #[T,B,A]
    target_q_taken = torch.gather(target_q_value, -1, index=action.unsqueeze(-1)).squeeze(-1) #[T,B,A]
    T, B, A = target_q_taken.shape
    reward = reward.unsqueeze(-1).expand_as(target_q_taken).reshape(T, -1) #[T, B] -> [T, B * A]
    target_q_taken = target_q_taken.reshape(T, -1)
    return_ = generalized_lambda_returns(target_q_taken, reward[:-1], gamma, lambda_) # [T - 1, B*A]
    return_ = return_.reshape(T - 1, B, A)
    q_value_loss = (F.mse_loss(return_, q_taken[:-1], reduction='none') * weight[:-1]).mean()

    dist = torch.distributions.categorical.Categorical(logits=logit)
    logp = dist.log_prob(action) # TBA
    baseline = (torch.softmax(logit, dim=-1) * q_value).sum(-1).detach() #TBA
    adv = (q_taken - baseline).detach()
    entropy_loss = (dist.entropy() * weight).mean()
    policy_loss = -(logp * adv * weight).mean()
    return coma_loss(policy_loss, q_value_loss, entropy_loss)
