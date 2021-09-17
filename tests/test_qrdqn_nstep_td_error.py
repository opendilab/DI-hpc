import time
import torch
from hpc_rll.origin.td import qrdqn_nstep_td_error, qrdqn_nstep_td_data
from hpc_rll.rl_utils.td import QRDQNNStepTDError
from testbase import mean_relative_error, times

assert torch.cuda.is_available()
use_cuda = True

tau = 39
T = 10
B = 89
N = 67
gamma = 0.95

def qrdqn_val():
    ori_q = torch.randn(B, N, tau)
    ori_next_n_q = torch.randn(B, N, tau)
    ori_action = torch.randint(0, N, size=(B, ))
    ori_next_n_action = torch.randint(0, N, size=(B, ))
    ori_reward = torch.randn(T, B)
    ori_done = torch.randn(B)
    ori_weight = torch.randn(B)
    ori_value_gamma = torch.randn(B)
 
    hpc_q = ori_q.clone().detach()
    hpc_next_n_q = ori_next_n_q.clone().detach()
    hpc_action = ori_action.clone().detach()
    hpc_next_n_action = ori_next_n_action.clone().detach()
    hpc_reward = ori_reward.clone().detach()
    hpc_done = ori_done.clone().detach()
    hpc_weight = ori_weight.clone().detach()
    hpc_value_gamma = ori_value_gamma.clone().detach()
    hpc_qrdqn = QRDQNNStepTDError(tau, T, B, N)

    if use_cuda:
        ori_q = ori_q.cuda()
        ori_next_n_q = ori_next_n_q.cuda()
        ori_action = ori_action.cuda()
        ori_next_n_action = ori_next_n_action.cuda()
        ori_reward = ori_reward.cuda()
        ori_done = ori_done.cuda()
        ori_weight = ori_weight.cuda()
        ori_value_gamma = ori_value_gamma.cuda()

        hpc_q = hpc_q.cuda()
        hpc_next_n_q = hpc_next_n_q.cuda()
        hpc_action = hpc_action.cuda()
        hpc_next_n_action = hpc_next_n_action.cuda()
        hpc_reward = hpc_reward.cuda()
        hpc_done = hpc_done.cuda()
        hpc_weight = hpc_weight.cuda()
        hpc_value_gamma = hpc_value_gamma.cuda()
        hpc_qrdqn = hpc_qrdqn.cuda()

    ori_q.requires_grad_(True)
    ori_loss, ori_ = qrdqn_nstep_td_error(qrdqn_nstep_td_data(ori_q, ori_next_n_q, ori_action, ori_next_n_action, ori_reward, ori_done, tau, ori_weight), gamma, T, ori_value_gamma)
    ori_loss = ori_loss.mean()
    ori_loss.backward()
    if use_cuda:
        torch.cuda.synchronize()

    torch.cuda.cudart().cudaProfilerStart()
    hpc_q.requires_grad_(True)
    hpc_loss, hpc_ = hpc_qrdqn(hpc_q, hpc_next_n_q, hpc_action, hpc_next_n_action, hpc_reward, hpc_done, gamma, hpc_weight, hpc_value_gamma)
    hpc_loss = hpc_loss.mean()
    hpc_loss.backward()
    if use_cuda:
        torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()

    mre = mean_relative_error(torch.flatten(ori_loss).cpu().detach().numpy(), torch.flatten(hpc_loss).cpu().detach().numpy())
    print("qrdqn fp mean_relative_error: " + str(mre))
    mre = mean_relative_error(torch.flatten(ori_q.grad).cpu().detach().numpy(), torch.flatten(hpc_q.grad).cpu().detach().numpy())
    print("qrdqn bp mean_relative_error: " + str(mre))

def qrdqn_perf():
    ori_q = torch.randn(B, N, tau)
    ori_next_n_q = torch.randn(B, N, tau)
    ori_action = torch.randint(0, N, size=(B, ))
    ori_next_n_action = torch.randint(0, N, size=(B, ))
    ori_reward = torch.randn(T, B)
    ori_done = torch.randn(B)
    ori_weight = torch.randn(B)
    ori_value_gamma = torch.randn(B)
 
    hpc_q = ori_q.clone().detach()
    hpc_next_n_q = ori_next_n_q.clone().detach()
    hpc_action = ori_action.clone().detach()
    hpc_next_n_action = ori_next_n_action.clone().detach()
    hpc_reward = ori_reward.clone().detach()
    hpc_done = ori_done.clone().detach()
    hpc_weight = ori_weight.clone().detach()
    hpc_value_gamma = ori_value_gamma.clone().detach()
    hpc_qrdqn = QRDQNNStepTDError(tau, T, B, N)

    if use_cuda:
        ori_q = ori_q.cuda()
        ori_next_n_q = ori_next_n_q.cuda()
        ori_action = ori_action.cuda()
        ori_next_n_action = ori_next_n_action.cuda()
        ori_reward = ori_reward.cuda()
        ori_done = ori_done.cuda()
        ori_weight = ori_weight.cuda()
        ori_value_gamma = ori_value_gamma.cuda()

        hpc_q = hpc_q.cuda()
        hpc_next_n_q = hpc_next_n_q.cuda()
        hpc_action = hpc_action.cuda()
        hpc_next_n_action = hpc_next_n_action.cuda()
        hpc_reward = hpc_reward.cuda()
        hpc_done = hpc_done.cuda()
        hpc_weight = hpc_weight.cuda()
        hpc_value_gamma = hpc_value_gamma.cuda()
        hpc_qrdqn = hpc_qrdqn.cuda()

    ori_q.requires_grad_(True)
    for i in range(times):
        t = time.time()
        ori_loss, ori_ = qrdqn_nstep_td_error(qrdqn_nstep_td_data(ori_q, ori_next_n_q, ori_action, ori_next_n_action, ori_reward, ori_done, tau, ori_weight), gamma, T, ori_value_gamma)
        ori_loss = ori_loss.mean()
        ori_loss.backward()
        if use_cuda:
            torch.cuda.synchronize()
        print('epoch: {}, original qrdqn cost time: {}'.format(i, time.time() - t))

    #torch.cuda.cudart().cudaProfilerStart()
    hpc_q.requires_grad_(True)
    for i in range(times):
        t = time.time()
        hpc_loss, hpc_ = hpc_qrdqn(hpc_q, hpc_next_n_q, hpc_action, hpc_next_n_action, hpc_reward, hpc_done, gamma, hpc_weight, hpc_value_gamma)
        hpc_loss = hpc_loss.mean()
        hpc_loss.backward()
        if use_cuda:
            torch.cuda.synchronize()
        print('epoch: {}, hpc qrdqn cost time: {}'.format(i, time.time() - t))
    #torch.cuda.cudart().cudaProfilerStop()

    mre = mean_relative_error(torch.flatten(ori_loss).cpu().detach().numpy(), torch.flatten(hpc_loss).cpu().detach().numpy())
    print("qrdqn fp mean_relative_error: " + str(mre))
    mre = mean_relative_error(torch.flatten(ori_q.grad).cpu().detach().numpy(), torch.flatten(hpc_q.grad).cpu().detach().numpy())
    print("qrdqn bp mean_relative_error: " + str(mre))

if __name__ == '__main__':
    print("target problem: tau = {}, T = {}, B = {}, N = {}, gamma = {}".format(tau, T, B, N, gamma))
    print("================run qrdqn validation test================")
    qrdqn_val()
    print("================run qrdqn performance test================")
    qrdqn_perf()
