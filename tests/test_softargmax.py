
import numpy as np
import torch
from hpc_rll.origin.soft_argmax import SoftArgmax
from hpc_rll.rl_utils.soft_argmax import SoftArgmaxHPC

cuda = torch.device('cuda')


def test_soft_argmax():
    B, C, H, W = 3, 1, 3, 3

    hpc_soft = SoftArgmaxHPC(B, H, W).cuda()
    soft = SoftArgmax().cuda()

    logit = torch.randn(B, C, H, W)
    hpc_logit = logit.clone().detach()
    logit = logit.cuda()
    hpc_logit = hpc_logit.cuda()



    logit.requires_grad_(True)
    hpc_logit.requires_grad_(True)



    
    ori_res = soft(logit)
    print("ori res: ")
    print(ori_res)
    ori_res = torch.sum(ori_res)
    ori_res.backward()
    print("ori logit grad: ", logit.grad)
    hpc_res = hpc_soft(hpc_logit)
    print("hpc res: ")
    print(hpc_res)
    hpc_res = torch.sum(hpc_res)
    hpc_res.requires_grad_(True)
    hpc_res.backward()
    print("hpc logit grad: ", hpc_logit.grad)




if __name__ == "__main__":
    test_soft_argmax()
