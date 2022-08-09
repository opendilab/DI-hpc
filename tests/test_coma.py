
import time
import torch
import torch.nn.functional as F

from hpc_rll.origin.coma import coma_data, coma_error
from hpc_rll.rl_utils.coma import COMA

from testbase import mean_relative_error, times

assert torch.cuda.is_available()
use_cuda = True

T, B, A, N = 2, 4, 8, 32

warm_up_times = 100


def generate_data(weight_is_None:bool = False, weight_is_oneslike:bool = False):
    ori_logit = torch.randn(T, B, A, N)
    ori_action = torch.randint(0, N, size=(T, B, A))
    ori_q_value = torch.randn(T, B, A, N)
    ori_target_q_value = torch.randn(T, B, A, N)
    ori_reward = torch.rand(T, B)
    if weight_is_None:
        ori_weight = None
    elif weight_is_oneslike:
        ori_weight = torch.ones_like(ori_action)
    else:
        ori_weight = torch.rand(T, B, A) + 1

    hpc_logit = ori_logit.clone().detach()
    hpc_action = ori_action.clone().detach()
    hpc_q_value = ori_q_value.clone().detach()
    hpc_target_q_value = ori_target_q_value.clone().detach()
    hpc_reward = ori_reward.clone().detach()
    if ori_weight is not None:
        hpc_weight = ori_weight.clone().detach()
    else:
        hpc_weight = None

    if use_cuda:
        ori_logit = ori_logit.cuda()
        ori_action = ori_action.cuda()
        ori_q_value = ori_q_value.cuda()
        ori_target_q_value = ori_target_q_value.cuda()
        ori_reward = ori_reward.cuda()
        if ori_weight is not None:
            ori_weight = ori_weight.cuda()

        hpc_logit = hpc_logit.cuda()
        hpc_action = hpc_action.cuda()
        hpc_q_value = hpc_q_value.cuda()
        hpc_target_q_value = hpc_target_q_value.cuda()
        hpc_reward = hpc_reward.cuda()
        if hpc_weight is not None:
            hpc_weight = hpc_weight.cuda()

    ori_logit.requires_grad_(True)
    ori_q_value.requires_grad_(True)

    hpc_logit.requires_grad_(True)
    hpc_q_value.requires_grad_(True)

    return ori_logit, ori_action, ori_q_value, ori_target_q_value, ori_reward, ori_weight, \
        hpc_logit, hpc_action, hpc_q_value, hpc_target_q_value, hpc_reward, hpc_weight


def coma_val(weight_is_None:bool = False, weight_is_oneslike:bool = False):
    hpc_coma_error = COMA(T, B, A, N)

    if use_cuda:
        hpc_coma_error.cuda()

    ori_logit, ori_action, ori_q_value, ori_target_q_value, ori_reward, ori_weight, \
        hpc_logit, hpc_action, hpc_q_value, hpc_target_q_value, hpc_reward, hpc_weight=generate_data(weight_is_None, weight_is_oneslike)

    ori_data = coma_data(ori_logit, ori_action, ori_q_value,
                         ori_target_q_value, ori_reward, ori_weight)
    ori_coma_loss = coma_error(ori_data, 1, 1)
    ori_total_loss = sum(ori_coma_loss)
    ori_total_loss.backward()

    hpc_data = coma_data(hpc_logit, hpc_action, hpc_q_value,
                         hpc_target_q_value, hpc_reward, hpc_weight)
    hpc_coma_loss = hpc_coma_error(hpc_logit, hpc_action, hpc_q_value,
                                   hpc_target_q_value, hpc_reward, hpc_weight,
                                   1, 1)
    hpc_total_loss = sum(hpc_coma_loss)
    hpc_total_loss.backward()

    mre = mean_relative_error(
        torch.flatten(ori_total_loss).cpu().detach().numpy(),
        torch.flatten(hpc_total_loss).cpu().detach().numpy())
    print("coma total loss mean_relative_error: " + str(mre))
    assert mre<0.0001
    mre = mean_relative_error(
        torch.flatten(ori_logit.grad).cpu().detach().numpy(),
        torch.flatten(hpc_logit.grad).cpu().detach().numpy())
    print("coma logits_new mean_relative_error: " + str(mre))
    assert mre<0.0001
    mre = mean_relative_error(
        torch.flatten(ori_q_value.grad).cpu().detach().numpy(),
        torch.flatten(hpc_q_value.grad).cpu().detach().numpy())
    print("coma q_value mean_relative_error: " + str(mre))
    assert mre<0.0001

def coma_perf(do_backward: bool = True):
    hpc_coma_error = COMA(T, B, A, N)

    if use_cuda:
        hpc_coma_error.cuda()

    ori_test_data = []
    hpc_test_data = []
    for i in range(times):
        ori_logit, ori_action, ori_q_value, ori_target_q_value, ori_reward, ori_weight, \
            hpc_logit, hpc_action, hpc_q_value, hpc_target_q_value, hpc_reward, hpc_weight=generate_data()
        ori_test_data.append((
            ori_logit,
            ori_action,
            ori_q_value,
            ori_target_q_value,
            ori_reward,
            ori_weight,
        ))
        hpc_test_data.append((
            hpc_logit,
            hpc_action,
            hpc_q_value,
            hpc_target_q_value,
            hpc_reward,
            hpc_weight,
        ))

    ori_logit, ori_action, ori_q_value, ori_target_q_value, ori_reward, ori_weight, \
        hpc_logit, hpc_action, hpc_q_value, hpc_target_q_value, hpc_reward, hpc_weight=generate_data()

    for i in range(warm_up_times):
        ori_data = coma_data(ori_logit, ori_action, ori_q_value,
                             ori_target_q_value, ori_reward, ori_weight)
        ori_coma_loss = coma_error(ori_data, 1, 1)
        ori_total_loss = sum(ori_coma_loss)
        ori_total_loss.backward()
        if use_cuda:
            torch.cuda.synchronize()

    for i in range(times):
        ori_logit, ori_action, ori_q_value, ori_target_q_value, ori_reward, ori_weight = ori_test_data[
            i]
        t0 = time.time()
        ori_data = coma_data(ori_logit, ori_action, ori_q_value,
                             ori_target_q_value, ori_reward, ori_weight)
        ori_coma_loss = coma_error(ori_data, 1, 1)
        if do_backward:
            ori_total_loss = sum(ori_coma_loss)
            ori_total_loss.backward()
        if use_cuda:
            torch.cuda.synchronize()
        t1 = time.time()
        print('epoch: {}, origin coma cost time: {}'.format(i, t1 - t0))

    for i in range(warm_up_times):
        hpc_data = coma_data(hpc_logit, hpc_action, hpc_q_value,
                             hpc_target_q_value, hpc_reward, hpc_weight)
        hpc_coma_loss = hpc_coma_error(hpc_logit, hpc_action, hpc_q_value,
                                       hpc_target_q_value, hpc_reward,
                                       hpc_weight, 1, 1)
        hpc_total_loss = sum(hpc_coma_loss)
        hpc_total_loss.backward()
        if use_cuda:
            torch.cuda.synchronize()

    for i in range(times):
        hpc_logit, hpc_action, hpc_q_value, hpc_target_q_value, hpc_reward, hpc_weight = hpc_test_data[
            i]
        t0 = time.time()
        hpc_data = coma_data(hpc_logit, hpc_action, hpc_q_value,
                             hpc_target_q_value, hpc_reward, hpc_weight)
        hpc_coma_loss = hpc_coma_error(hpc_logit, hpc_action, hpc_q_value,
                                       hpc_target_q_value, hpc_reward,
                                       hpc_weight, 1, 1)
        if do_backward:
            hpc_total_loss = sum(hpc_coma_loss)
            hpc_total_loss.backward()
        if use_cuda:
            torch.cuda.synchronize()
        t1 = time.time()
        print('epoch: {}, hpc coma cost time: {}'.format(i, t1 - t0))


if __name__ == '__main__':
    print("target problem: T = {}, B = {}, A= {}, N = {}".format(T, B, A, N))
    print("================run coma validation test================")
    coma_val()
    print("================run coma validation test (weight==None)================")
    coma_val(weight_is_None=True, weight_is_oneslike=False)
    print("================run coma validation test (weight==ones_like(action))================")
    coma_val(weight_is_None=False, weight_is_oneslike=True)
    print("================run coma performance test================")
    print("----------------run coma forward only----------------")
    coma_perf(do_backward=False)
    print("----------------run coma forward and backward----------------")
    coma_perf(do_backward=True)

