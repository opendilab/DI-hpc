import time
import torch
import numpy as np
from testbase import mean_relative_error, times
import hpc_models

assert torch.cuda.is_available()
use_cuda = True

times = 100

batch_size = 8
max_entity_num = 182
input_dim = 1024

lstm_seq_len = 1
lstm_batch_size = 8
lstm_input_size = 32
lstm_hidden_size = 32
lstm_num_layers = 1


def torch_update_ae(autoregressive_embedding, key_embeddings, sample_entity, max_entity_num, end_flag):
    bs = autoregressive_embedding.shape[0]
    autoregressive_embedding = autoregressive_embedding + key_embeddings[torch.arange(bs), sample_entity] * ~end_flag.unsqueeze(dim=1)
    return autoregressive_embedding

def actor_critic_update_ae_val():
    ori_ae = torch.randn(batch_size, input_dim)
    ori_ke = torch.randn(batch_size, max_entity_num, input_dim)
    ori_entity_num = torch.randint(max_entity_num - 2, max_entity_num, size=(batch_size, ))
    ori_sample_entity = []
    for i in range(batch_size):
        entity_num = ori_entity_num[i]
        ori_sample_entity.append(torch.randint(0, entity_num, size=(1, )))
    ori_sample_entity = torch.stack(ori_sample_entity, dim=0).squeeze(1)
    ori_end_flag = torch.zeros(batch_size).bool()

    hpc_ae = ori_ae.clone().detach()
    hpc_ke = ori_ke.clone().detach()
    hpc_entity_num = ori_entity_num.clone().detach()
    hpc_sample_entity = ori_sample_entity.clone().detach()
    hpc_end_flag = ori_end_flag.clone().detach()

    if use_cuda:
        ori_ae = ori_ae.cuda()
        ori_ke = ori_ke.cuda()
        ori_entity_num = ori_entity_num.cuda()
        ori_sample_entity = ori_sample_entity.cuda()
        ori_end_flag = ori_end_flag.cuda()

        hpc_ae = hpc_ae.cuda()
        hpc_ke = hpc_ke.cuda()
        hpc_entity_num = hpc_entity_num.cuda()
        hpc_sample_entity = hpc_sample_entity.cuda()
        hpc_end_flag = hpc_end_flag.cuda()

    ori_end_flag[ori_sample_entity == ori_entity_num] = 1
    ori_out = torch_update_ae(ori_ae, ori_ke, ori_sample_entity, ori_entity_num, ori_end_flag)

    hpc_end_flag[hpc_sample_entity == hpc_entity_num] = 1
    hpc_models.actor_critic_update_ae([hpc_ke, hpc_sample_entity, hpc_entity_num], [hpc_ae])
    hpc_out = hpc_ae
    if use_cuda:
        torch.cuda.synchronize()

    mre = mean_relative_error(torch.flatten(ori_out).cpu().detach().numpy(), torch.flatten(hpc_out).cpu().detach().numpy())
    print("actor critic update ae mean_relative_error: " + str(mre))
    #print("ori_out: " + str(ori_out))
    #print("hpc_out: " + str(hpc_out))

def actor_critic_update_ae_perf():
    ori_ae = torch.randn(batch_size, input_dim)
    ori_ke = torch.randn(batch_size, max_entity_num, input_dim)
    ori_entity_num = torch.randint(max_entity_num - 2, max_entity_num, size=(batch_size, ))
    ori_sample_entity = []
    for i in range(batch_size):
        entity_num = ori_entity_num[i]
        ori_sample_entity.append(torch.randint(0, entity_num, size=(1, )))
    ori_sample_entity = torch.stack(ori_sample_entity, dim=0).squeeze(1)
    ori_end_flag = torch.zeros(batch_size).bool()

    hpc_ae = ori_ae.clone().detach()
    hpc_ke = ori_ke.clone().detach()
    hpc_entity_num = ori_entity_num.clone().detach()
    hpc_sample_entity = ori_sample_entity.clone().detach()
    hpc_end_flag = ori_end_flag.clone().detach()

    if use_cuda:
        ori_ae = ori_ae.cuda()
        ori_ke = ori_ke.cuda()
        ori_entity_num = ori_entity_num.cuda()
        ori_sample_entity = ori_sample_entity.cuda()
        ori_end_flag = ori_end_flag.cuda()

        hpc_ae = hpc_ae.cuda()
        hpc_ke = hpc_ke.cuda()
        hpc_entity_num = hpc_entity_num.cuda()
        hpc_sample_entity = hpc_sample_entity.cuda()
        hpc_end_flag = hpc_end_flag.cuda()
 
    t = time.time()
    for i in range(times):
        ori_end_flag[ori_sample_entity == ori_entity_num] = 1
        ori_out = torch_update_ae(ori_ae, ori_ke, ori_sample_entity, ori_entity_num, ori_end_flag)
    if use_cuda:
        torch.cuda.synchronize()
    print('original update ae cost time: {}'.format(time.time() - t))
 
    t = time.time()
    for i in range(times):
        hpc_end_flag[hpc_sample_entity == hpc_entity_num] = 1
        hpc_models.actor_critic_update_ae([hpc_ke, hpc_sample_entity, hpc_entity_num], [hpc_ae])
        hpc_out = hpc_ae
    if use_cuda:
        torch.cuda.synchronize()
    print('hpc update ae cost time: {}'.format(time.time() - t))


def actor_critic_lstm_activation_val():
    ori_x = torch.randn(lstm_seq_len, lstm_batch_size, lstm_input_size)
    ori_h0 = torch.randn(lstm_num_layers, lstm_batch_size, lstm_hidden_size)
    ori_c0 = torch.randn(lstm_num_layers, lstm_batch_size, lstm_hidden_size)
    ori_lstm = torch.nn.LSTM(lstm_input_size, lstm_hidden_size, lstm_num_layers)

    hpc_x = ori_x.clone().detach()
    hpc_h0 = ori_h0.clone().detach()
    hpc_c0 = ori_c0.clone().detach()
    hpc_ih = torch.zeros(lstm_batch_size, lstm_hidden_size * 4)
    hpc_hh = torch.zeros(lstm_batch_size, lstm_hidden_size * 4)

    if use_cuda:
        ori_x = ori_x.cuda()
        ori_h0 = ori_h0.cuda()
        ori_c0 = ori_c0.cuda()
        ori_lstm = ori_lstm.cuda()

        hpc_x = hpc_x.cuda()
        hpc_h0 = hpc_h0.cuda()
        hpc_c0 = hpc_c0.cuda()
        hpc_ih = hpc_ih.cuda()
        hpc_hh = hpc_hh.cuda()

    ori_out, ori_state = ori_lstm.forward(ori_x, (ori_h0, ori_c0))

    hpc_wih0 = ori_lstm.weight_ih_l0
    hpc_whh0 = ori_lstm.weight_hh_l0
    hpc_bih0 = ori_lstm.bias_ih_l0
    hpc_bhh0 = ori_lstm.bias_hh_l0
    hpc_bias = hpc_bih0 + hpc_bhh0
    
    torch.matmul(hpc_x[0].detach(), hpc_wih0.transpose(0, 1).detach(), out = hpc_ih)
    torch.matmul(hpc_h0[0].detach(), hpc_whh0.transpose(0, 1).detach(), out = hpc_hh)
    hpc_models.actor_critic_lstm_activation([hpc_ih, hpc_hh, hpc_bias], [hpc_h0, hpc_c0])
    hpc_out = hpc_h0
    if use_cuda:
        torch.cuda.synchronize()

    mre = mean_relative_error(torch.flatten(ori_out).cpu().detach().numpy(), torch.flatten(hpc_out).cpu().detach().numpy())
    print("actor critic lstm activation mean_relative_error: " + str(mre))


def actor_critic_lstm_activation_perf():
    ori_x = torch.randn(lstm_seq_len, lstm_batch_size, lstm_input_size)
    ori_h0 = torch.randn(lstm_num_layers, lstm_batch_size, lstm_hidden_size)
    ori_c0 = torch.randn(lstm_num_layers, lstm_batch_size, lstm_hidden_size)
    ori_lstm = torch.nn.LSTM(lstm_input_size, lstm_hidden_size, lstm_num_layers)

    hpc_x = ori_x.clone().detach()
    hpc_h0 = ori_h0.clone().detach()
    hpc_c0 = ori_c0.clone().detach()
    hpc_ih = torch.zeros(lstm_batch_size, lstm_hidden_size * 4)
    hpc_hh = torch.zeros(lstm_batch_size, lstm_hidden_size * 4)

    if use_cuda:
        ori_x = ori_x.cuda()
        ori_h0 = ori_h0.cuda()
        ori_c0 = ori_c0.cuda()
        ori_lstm = ori_lstm.cuda()

        hpc_x = hpc_x.cuda()
        hpc_h0 = hpc_h0.cuda()
        hpc_c0 = hpc_c0.cuda()
        hpc_ih = hpc_ih.cuda()
        hpc_hh = hpc_hh.cuda()

    ori_out, ori_state = ori_lstm.forward(ori_x, (ori_h0, ori_c0))
    if use_cuda:
        torch.cuda.synchronize()

    hpc_wih0 = ori_lstm.weight_ih_l0
    hpc_whh0 = ori_lstm.weight_hh_l0
    hpc_bih0 = ori_lstm.bias_ih_l0
    hpc_bhh0 = ori_lstm.bias_hh_l0
    hpc_bias = hpc_bih0 + hpc_bhh0
    
    t = time.time()
    for i in range(times):
        ori_out, ori_state = ori_lstm.forward(ori_x, (ori_h0, ori_c0))
    if use_cuda:
        torch.cuda.synchronize()
    print('original lstm activation cost time: {}'.format(time.time() - t))

    t = time.time()
    for i in range(times):
        torch.matmul(hpc_x[0].detach(), hpc_wih0.transpose(0, 1).detach(), out = hpc_ih)
        torch.matmul(hpc_h0[0].detach(), hpc_whh0.transpose(0, 1).detach(), out = hpc_hh)
        hpc_models.actor_critic_lstm_activation([hpc_ih, hpc_hh, hpc_bias], [hpc_h0, hpc_c0])
        hpc_out = hpc_h0
    if use_cuda:
        torch.cuda.synchronize()
    print('hpc lstm activation cost time: {}'.format(time.time() - t))


def actor_critic_pre_sample_val():
    ori_x = torch.randn(lstm_seq_len, lstm_batch_size, lstm_hidden_size)
    ori_key = torch.randn(lstm_batch_size, max_entity_num, lstm_hidden_size)
    ori_mask = torch.zeros(lstm_batch_size, max_entity_num)

    ori_entity_num = torch.randint(max_entity_num - 2, max_entity_num, size=(lstm_batch_size, ))
    ori_sample_entity = []
    ori_mask = []
    for i in range(lstm_batch_size):
        entity_num = ori_entity_num[i]
        sample_entity = torch.randint(0, entity_num, size=(1, ))
        ori_sample_entity.append(torch.randint(0, entity_num, size=(1, )))

        mask = []
        for j in range(max_entity_num):
            if j < entity_num:
                mask.append(torch.ones(size=(1, )))
            else:
                mask.append(torch.zeros(size=(1, )))
        mask = torch.stack(mask, dim=0).squeeze(1)
        ori_mask.append(mask)

    ori_sample_entity = torch.stack(ori_sample_entity, dim=0).squeeze(1)
    ori_mask = torch.stack(ori_mask, dim=0)

    ori_mask[torch.arange(lstm_batch_size), ori_sample_entity] = 0

    hpc_x = ori_x.clone().detach()
    hpc_key = ori_key.clone().detach()
    hpc_mask = ori_mask.clone().detach()
    hpc_out = torch.zeros(lstm_batch_size, max_entity_num)

    if use_cuda:
        ori_x = ori_x.cuda()
        ori_key = ori_key.cuda()
        ori_mask = ori_mask.cuda()

        hpc_x = hpc_x.cuda()
        hpc_key = hpc_key.cuda()
        hpc_mask = hpc_mask.cuda()
        hpc_out = hpc_out.cuda()

    ori_mask = ori_mask.bool()
    hpc_mask = hpc_mask.bool()

    ori_queries = ori_x.permute(1, 0, 2)
    ori_query_result = ori_queries * ori_key
    ori_step_logits = ori_query_result.sum(dim=2)
    ori_step_logits = ori_step_logits.masked_fill(~ori_mask, -1e9)
    ori_step_logits = ori_step_logits.div(0.8)
    ori_out = ori_step_logits

    hpc_models.actor_critic_pre_sample([hpc_key, hpc_x, hpc_mask], [hpc_out])
    if use_cuda:
        torch.cuda.synchronize()

    mre = mean_relative_error(torch.flatten(ori_out).cpu().detach().numpy(), torch.flatten(hpc_out).cpu().detach().numpy())
    print("actor critic pre sample mean_relative_error: " + str(mre))
    assert np.allclose(ori_out.detach().cpu().numpy(), hpc_out.detach().cpu().numpy(), rtol=1e-5, atol=1e-5)


def actor_critic_pre_sample_perf():
    ori_x = torch.randn(lstm_seq_len, lstm_batch_size, lstm_hidden_size)
    ori_key = torch.randn(lstm_batch_size, max_entity_num, lstm_hidden_size)
    ori_mask = torch.zeros(lstm_batch_size, max_entity_num)

    ori_entity_num = torch.randint(max_entity_num - 2, max_entity_num, size=(lstm_batch_size, ))
    ori_sample_entity = []
    ori_mask = []
    for i in range(lstm_batch_size):
        entity_num = ori_entity_num[i]
        sample_entity = torch.randint(0, entity_num, size=(1, ))
        ori_sample_entity.append(torch.randint(0, entity_num, size=(1, )))

        mask = []
        for j in range(max_entity_num):
            if j < entity_num:
                mask.append(torch.ones(size=(1, )))
            else:
                mask.append(torch.zeros(size=(1, )))
        mask = torch.stack(mask, dim=0).squeeze(1)
        ori_mask.append(mask)

    ori_sample_entity = torch.stack(ori_sample_entity, dim=0).squeeze(1)
    ori_mask = torch.stack(ori_mask, dim=0)

    ori_mask[torch.arange(lstm_batch_size), ori_sample_entity] = 0

    hpc_x = ori_x.clone().detach()
    hpc_key = ori_key.clone().detach()
    hpc_mask = ori_mask.clone().detach()
    hpc_out = torch.zeros(lstm_batch_size, max_entity_num)

    if use_cuda:
        ori_x = ori_x.cuda()
        ori_key = ori_key.cuda()
        ori_mask = ori_mask.cuda()

        hpc_x = hpc_x.cuda()
        hpc_key = hpc_key.cuda()
        hpc_mask = hpc_mask.cuda()
        hpc_out = hpc_out.cuda()

    ori_mask = ori_mask.bool()
    hpc_mask = hpc_mask.bool()

    t = time.time()
    for i in range(times):
        ori_queries = ori_x.permute(1, 0, 2)
        ori_query_result = ori_queries * ori_key
        ori_step_logits = ori_query_result.sum(dim=2)
        ori_step_logits = ori_step_logits.masked_fill(~ori_mask, -1e9)
        ori_step_logits = ori_step_logits.div(0.8)
        ori_out = ori_step_logits
    if use_cuda:
        torch.cuda.synchronize()
    print('original pre sample cost time: {}'.format(time.time() - t))

    t = time.time()
    for i in range(times):
        hpc_models.actor_critic_pre_sample([hpc_key, hpc_x, hpc_mask], [hpc_out])
    if use_cuda:
        torch.cuda.synchronize()
    print('hpc pre sample cost time: {}'.format(time.time() - t))


if __name__ == '__main__':
    print("target problem: batch_size = {}, max_entity_num = {}, input_dim = {}".format(batch_size, max_entity_num, input_dim))
    print("================run actor critic update ae validation test================")
    actor_critic_update_ae_val()
    print("================run actor critic update ae performance test================")
    actor_critic_update_ae_perf()
    print("\n")

    print("target problem: lstm_batch_size = {}, lstm_seq_len = {}, lstm_input_size = {}, lstm_hidden_size = {}, lstm_num_layers = {}".format(
        lstm_batch_size, lstm_seq_len, lstm_input_size, lstm_hidden_size, lstm_num_layers))
    print("================run actor critic lstm activation validation test================")
    actor_critic_lstm_activation_val()
    print("================run actor critic lstm activation performance test================")
    actor_critic_lstm_activation_perf()
    print("\n")

    print("target problem: lstm_batch_size = {}, lstm_seq_len = {}, lstm_hidden_size = {}, max_entity_num = {}".format(
        lstm_batch_size, lstm_seq_len, lstm_hidden_size, max_entity_num))
    print("================run actor critic pre sample validation test================")
    actor_critic_pre_sample_val()
    print("================run actor critic pre sample performance test================")
    actor_critic_pre_sample_perf()
    print("\n")
