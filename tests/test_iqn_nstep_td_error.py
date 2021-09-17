import time
import torch
from hpc_rll.origin.td import iqn_nstep_td_error, iqn_nstep_td_data
from hpc_rll.rl_utils.td import IQNNStepTDError
from testbase import mean_relative_error, times

assert torch.cuda.is_available()
use_cuda = True

tau = 33
tauPrime = 34
T = 10
B = 64
N = 8
gamma = 0.95
kappa = 0.9

def iqn_val():
    ori_q = torch.randn(tau, B, N)
    ori_next_n_q = torch.randn(tauPrime, B, N)
    ori_action = torch.randint(0, N, size=(B, ))
    ori_next_n_action = torch.randint(0, N, size=(B, ))
    ori_reward = torch.randn(T, B)
    ori_done = torch.randn(B)
    ori_r_q = torch.randn(tau, B)
    ori_weight = torch.randn(B)
    ori_value_gamma = torch.randn(B)
 
    hpc_q = ori_q.clone().detach()
    hpc_next_n_q = ori_next_n_q.clone().detach()
    hpc_action = ori_action.clone().detach()
    hpc_next_n_action = ori_next_n_action.clone().detach()
    hpc_reward = ori_reward.clone().detach()
    hpc_done = ori_done.clone().detach()
    hpc_r_q = ori_r_q.clone().detach()
    hpc_weight = ori_weight.clone().detach()
    hpc_value_gamma = ori_value_gamma.clone().detach()
    hpc_iqn = IQNNStepTDError(tau, tauPrime, T, B, N)

    if use_cuda:
        ori_q = ori_q.cuda()
        ori_next_n_q = ori_next_n_q.cuda()
        ori_action = ori_action.cuda()
        ori_next_n_action = ori_next_n_action.cuda()
        ori_reward = ori_reward.cuda()
        ori_done = ori_done.cuda()
        ori_r_q = ori_r_q.cuda()
        ori_weight = ori_weight.cuda()
        ori_value_gamma = ori_value_gamma.cuda()

        hpc_q = hpc_q.cuda()
        hpc_next_n_q = hpc_next_n_q.cuda()
        hpc_action = hpc_action.cuda()
        hpc_next_n_action = hpc_next_n_action.cuda()
        hpc_reward = hpc_reward.cuda()
        hpc_done = hpc_done.cuda()
        hpc_r_q = hpc_r_q.cuda()
        hpc_weight = hpc_weight.cuda()
        hpc_value_gamma = hpc_value_gamma.cuda()
        hpc_iqn = hpc_iqn.cuda()

    ori_q.requires_grad_(True)
    ori_loss, ori_ = iqn_nstep_td_error(iqn_nstep_td_data(ori_q, ori_next_n_q, ori_action, ori_next_n_action, ori_reward, ori_done, ori_r_q, ori_weight), gamma, T, kappa, ori_value_gamma)
    ori_loss = ori_loss.mean()
    ori_loss.backward()
    if use_cuda:
        torch.cuda.synchronize()

    torch.cuda.cudart().cudaProfilerStart()
    hpc_q.requires_grad_(True)
    hpc_loss, hpc_ = hpc_iqn(hpc_q, hpc_next_n_q, hpc_action, hpc_next_n_action, hpc_reward, hpc_done, hpc_r_q, gamma, kappa, hpc_weight, hpc_value_gamma)
    hpc_loss = hpc_loss.mean()
    hpc_loss.backward()
    if use_cuda:
        torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()

    mre = mean_relative_error(torch.flatten(ori_loss).cpu().detach().numpy(), torch.flatten(hpc_loss).cpu().detach().numpy())
    print("iqn fp mean_relative_error: " + str(mre))
    mre = mean_relative_error(torch.flatten(ori_q.grad).cpu().detach().numpy(), torch.flatten(hpc_q.grad).cpu().detach().numpy())
    print("iqn bp mean_relative_error: " + str(mre))

def iqn_perf():
    ori_q = torch.randn(tau, B, N)
    ori_next_n_q = torch.randn(tauPrime, B, N)
    ori_action = torch.randint(0, N, size=(B, ))
    ori_next_n_action = torch.randint(0, N, size=(B, ))
    ori_reward = torch.randn(T, B)
    ori_done = torch.randn(B)
    ori_r_q = torch.randn(tau, B)
    ori_weight = torch.randn(B)
    ori_value_gamma = torch.randn(B)
 
    hpc_q = ori_q.clone().detach()
    hpc_next_n_q = ori_next_n_q.clone().detach()
    hpc_action = ori_action.clone().detach()
    hpc_next_n_action = ori_next_n_action.clone().detach()
    hpc_reward = ori_reward.clone().detach()
    hpc_done = ori_done.clone().detach()
    hpc_r_q = ori_r_q.clone().detach()
    hpc_weight = ori_weight.clone().detach()
    hpc_value_gamma = ori_value_gamma.clone().detach()
    hpc_iqn = IQNNStepTDError(tau, tauPrime, T, B, N)

    if use_cuda:
        ori_q = ori_q.cuda()
        ori_next_n_q = ori_next_n_q.cuda()
        ori_action = ori_action.cuda()
        ori_next_n_action = ori_next_n_action.cuda()
        ori_reward = ori_reward.cuda()
        ori_done = ori_done.cuda()
        ori_r_q = ori_r_q.cuda()
        ori_weight = ori_weight.cuda()
        ori_value_gamma = ori_value_gamma.cuda()

        hpc_q = hpc_q.cuda()
        hpc_next_n_q = hpc_next_n_q.cuda()
        hpc_action = hpc_action.cuda()
        hpc_next_n_action = hpc_next_n_action.cuda()
        hpc_reward = hpc_reward.cuda()
        hpc_done = hpc_done.cuda()
        hpc_r_q = hpc_r_q.cuda()
        hpc_weight = hpc_weight.cuda()
        hpc_iqn = hpc_iqn.cuda()
        hpc_value_gamma = hpc_value_gamma.cuda()

    ori_q.requires_grad_(True)
    for i in range(times):
        t = time.time()
        ori_loss, ori_ = iqn_nstep_td_error(iqn_nstep_td_data(ori_q, ori_next_n_q, ori_action, ori_next_n_action, ori_reward, ori_done, ori_r_q, ori_weight), gamma, T, kappa, ori_value_gamma)
        ori_loss = ori_loss.mean()
        ori_loss.backward()
        if use_cuda:
            torch.cuda.synchronize()
        print('epoch: {}, original iqn cost time: {}'.format(i, time.time() - t))

    #torch.cuda.cudart().cudaProfilerStart()
    hpc_q.requires_grad_(True)
    for i in range(times):
        t = time.time()
        hpc_loss, hpc_ = hpc_iqn(hpc_q, hpc_next_n_q, hpc_action, hpc_next_n_action, hpc_reward, hpc_done, hpc_r_q, gamma, kappa, hpc_weight, hpc_value_gamma)
        hpc_loss = hpc_loss.mean()
        hpc_loss.backward()
        if use_cuda:
            torch.cuda.synchronize()
        print('epoch: {}, hpc iqn cost time: {}'.format(i, time.time() - t))
    #torch.cuda.cudart().cudaProfilerStop()

    mre = mean_relative_error(torch.flatten(ori_loss).cpu().detach().numpy(), torch.flatten(hpc_loss).cpu().detach().numpy())
    print("iqn fp mean_relative_error: " + str(mre))
    mre = mean_relative_error(torch.flatten(ori_q.grad).cpu().detach().numpy(), torch.flatten(hpc_q.grad).cpu().detach().numpy())
    print("iqn bp mean_relative_error: " + str(mre))

if __name__ == '__main__':
    print("target problem: tau = {}, tauPrime = {}, T = {}, B = {}, N = {}, gamma = {}, kappa = {}".format(tau, tauPrime, T, B, N, gamma, kappa))
    print("================run iqn validation test================")
    iqn_val()
    print("================run iqn performance test================")
    iqn_perf()
