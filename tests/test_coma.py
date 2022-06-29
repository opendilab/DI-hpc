import numpy as np
import torch
from hpc_rll.origin.coma import coma_data, coma_error
from hpc_rll.rl_utils.coma import COMA

cuda = torch.device('cuda')


def test_coma():
    T, B, A, N = 128, 4, 8, 32

    hpc_coma = COMA(T, B, A, N).cuda()

    logit = torch.randn(T, B, A, N)
    action = torch.randint(0, N, size=(T, B, A))
    q_value = torch.randn(T, B, A, N)
    target_q_value = torch.randn(T, B, A, N)
    reward = torch.rand(T, B)
    weight = torch.rand(T, B, A) + 1

    hpc_logit = logit.clone().detach()
    hpc_action = action.clone().detach()
    hpc_q_value = q_value.clone().detach()
    hpc_target_q_value = target_q_value.clone().detach()
    hpc_reward = reward.clone().detach()
    hpc_weight = weight.clone().detach()

    logit = logit.cuda()
    action = action.cuda()
    q_value = q_value.cuda()
    target_q_value = target_q_value.cuda()
    reward = reward.cuda()
    weight = weight.cuda()

    hpc_logit = hpc_logit.cuda()
    hpc_action = hpc_action.cuda()
    hpc_q_value = hpc_q_value.cuda()
    hpc_target_q_value = hpc_target_q_value.cuda()
    hpc_reward = hpc_reward.cuda()
    hpc_weight = hpc_weight.cuda()



    logit.requires_grad_(True)
    q_value.requires_grad_(True)
    hpc_logit.requires_grad_(True)
    hpc_q_value.requires_grad_(True)

    data = coma_data(logit, action, q_value, target_q_value, reward, weight)
    hpc_data = coma_data(hpc_logit, hpc_action, hpc_q_value, hpc_target_q_value, hpc_reward, hpc_weight)

    hpc_coma_loss = hpc_coma(*hpc_data, 1, 1)
    ori_coma_loss = coma_error(data, 1, 1)


    ori_total_loss = sum(ori_coma_loss)
    ori_total_loss.backward()
    print("ori_coma_loss")
    print(ori_coma_loss)
    print("ori logit grad")
    print(logit.grad)
    print("ori q_value grad")
    print(q_value.grad)


    hpc_total_loss = sum(hpc_coma_loss)
    hpc_total_loss.backward()
    print("hpc_coma_loss")
    print(hpc_coma_loss)
    print("hpc logit grad")
    print(hpc_logit.grad)
    print("hpc q_value grad")
    print(hpc_q_value.grad)



if __name__ == "__main__":
    test_coma()
