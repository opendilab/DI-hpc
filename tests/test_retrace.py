import time
import torch
from hpc_rll.rl_utils.retrace import Retrace
from hpc_rll.origin.retrace import compute_q_retraces
from testbase import mean_relative_error, times

use_cuda = True

def test_compute_q_retraces():
    T, B, N = 5, 5, 10
    q_values = torch.randn(T + 1, B, N)
    v_pred = torch.randn(T + 1, B, 1)
    rewards = torch.randn(T, B)
    ratio = torch.rand(T, B, N) * 0.4 + 0.8
    assert ratio.max() <= 1.2 and ratio.min() >= 0.8
    weights = torch.rand(T, B)
    actions = torch.randint(0, N, size=(T, B))
    hpc_q_values = q_values.clone().detach()
    hpc_v_pred = v_pred.clone().detach()
    hpc_rewards = rewards.clone().detach()
    hpc_ratio = ratio.clone().detach()
    hpc_weights = weights.clone().detach()
    hpc_actions = actions.clone().detach()
    hpc_Retrace = Retrace(T, B, N)

    if use_cuda:
        q_values = q_values.cuda()
        v_pred = v_pred.cuda()
        rewards = rewards.cuda()
        ratio = ratio.cuda()
        weights = weights.cuda()
        actions = actions.cuda()
        hpc_q_values = hpc_q_values.cuda()
        hpc_v_pred = hpc_v_pred.cuda()
        hpc_rewards = hpc_rewards.cuda()
        hpc_ratio = hpc_ratio.cuda()
        hpc_weights = hpc_weights.cuda()
        hpc_actions = hpc_actions.cuda()
        hpc_Retrace = hpc_Retrace.cuda()

    q_retraces_origin = compute_q_retraces(q_values, v_pred, rewards, actions, weights, ratio, gamma=0.99)
    q_retraces_hpc = hpc_Retrace(hpc_q_values, hpc_v_pred, hpc_rewards, hpc_actions, hpc_weights, hpc_ratio, gamma=0.99)
    

    assert q_retraces_origin.shape == (T + 1, B, 1)
    assert q_retraces_hpc.shape == (T + 1, B, 1)
    mre = mean_relative_error(torch.flatten(q_retraces_origin).cpu().detach().numpy(), torch.flatten(q_retraces_hpc).cpu().detach().numpy())
    print("q_retraces mean_relative_error: " + str(mre))


    for i in range(times):
        t = time.time()
        q_retraces_origin = compute_q_retraces(q_values, v_pred, rewards, actions, weights, ratio, gamma=0.99)
        if use_cuda:
            torch.cuda.synchronize()
        print('epoch: {}, origin retrace cost time: {}'.format(i, time.time() - t))

    for i in range(times):
        t = time.time()
        q_retraces_hpc = hpc_Retrace(hpc_q_values, hpc_v_pred, hpc_rewards, hpc_actions, hpc_weights, hpc_ratio, gamma=0.99)
        if use_cuda:
            torch.cuda.synchronize()
        print('epoch: {}, hpc retrace cost time: {}'.format(i, time.time() - t))




if __name__ == '__main__':
    test_compute_q_retraces()
    
    



