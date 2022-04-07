import torch
def compute_q_retraces(
        q_values: torch.Tensor, # T+1, B, N
        v_pred: torch.Tensor, # T+1, B, 1
        rewards: torch.Tensor, # T, B
        actions: torch.Tensor, # T, B
        weights: torch.Tensor, # T, B
        ratio: torch.Tensor, # T, B, N
        gamma: float = 0.9
) -> torch.Tensor:
    rewards = rewards.unsqueeze(-1)  # shape T,B,1
    actions = actions.unsqueeze(-1)  # shape T,B,1
    weights = weights.unsqueeze(-1)  # shape T,B,1
    q_retraces = torch.zeros_like(v_pred)  # shape (T+1),B,1
    n_len = q_retraces.size()[0]  # T+1
    tmp_retraces = v_pred[-1, ...]  # shape B,1
    q_retraces[-1, ...] = v_pred[-1, ...]
    q_gather = q_values[0:-1, ...].gather(-1, actions)  # shape T,B,1
    ratio_gather = ratio.gather(-1, actions)  # shape T,B,1

    for idx in reversed(range(n_len - 1)):
        q_retraces[idx, ...] = rewards[idx, ...] + gamma * weights[idx, ...] * tmp_retraces
        tmp_retraces = ratio_gather[idx, ...].clamp(max=1.0) * (q_retraces[idx, ...] - q_gather[idx, ...]) + v_pred[idx, ...]
    # print(q_retraces.squeeze())
    return q_retraces  # shape (T+1),B,1