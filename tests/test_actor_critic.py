import time
import torch
from testbase import mean_relative_error, times
import hpc_models

assert torch.cuda.is_available()
use_cuda = True

times = 100

batch_size = 1
entity_num = 182
input_dim = 1024

lstm_seq_len = 1
lstm_batch_size = 1
lstm_input_size = 32
lstm_hidden_size = 32
lstm_num_layers = 1


def torch_update_ae(autoregressive_embedding, key_embeddings, sample_result, entity_num, end_flag):
    bs = autoregressive_embedding.shape[0]
    autoregressive_embedding = autoregressive_embedding + key_embeddings[torch.arange(bs), sample_result] * ~end_flag.unsqueeze(dim=1)
    return autoregressive_embedding


def hpc_update_ae(autoregressive_embedding, key_embeddings, sample_result, entity_num):
    hpc_models.actor_critic_update_ae([key_embeddings, sample_result, entity_num], [autoregressive_embedding])
    return autoregressive_embedding


def actor_critic_update_ae_val():
    ori_ae = torch.randn(batch_size, input_dim)
    ori_ke = torch.randn(batch_size, entity_num, input_dim)
    ori_sample_result = torch.randint(0, entity_num, size=(1, ))
    ori_true_entity_num = torch.randint(entity_num - 1, entity_num, size=(1, ))

    hpc_ae = ori_ae.clone().detach()
    hpc_ke = ori_ke.clone().detach()
    hpc_sample_result = ori_sample_result.clone().detach()
    hpc_true_entity_num = ori_true_entity_num.clone().detach()

    if use_cuda:
        ori_ae = ori_ae.cuda()
        ori_ke = ori_ke.cuda()
        ori_sample_result = ori_sample_result.cuda()
        ori_true_entity_num = ori_true_entity_num.cuda()

        hpc_ae = hpc_ae.cuda()
        hpc_ke = hpc_ke.cuda()
        hpc_sample_result = hpc_sample_result.cuda()
        hpc_true_entity_num = hpc_true_entity_num.cuda()

    bs = ori_ae.shape[0]
    end_flag = torch.zeros(bs).cuda().bool()
    end_flag[ori_sample_result == ori_true_entity_num] = 1

    ori_out = torch_update_ae(ori_ae, ori_ke, ori_sample_result, ori_true_entity_num, end_flag)
    #hpc_out = hpc_update_ae(ori_ae, ori_ke, ori_sample_result, ori_true_entity_num)
    hpc_models.actor_critic_update_ae([hpc_ke, hpc_sample_result, hpc_true_entity_num], [hpc_ae])
    hpc_out = hpc_ae
    if use_cuda:
        torch.cuda.synchronize()

    mre = mean_relative_error(torch.flatten(ori_out).cpu().detach().numpy(), torch.flatten(hpc_out).cpu().detach().numpy())
    print("actor critic update ae mean_relative_error: " + str(mre))
    #print("ori_out: " + str(ori_out))
    #print("hpc_out: " + str(hpc_out))

def actor_critic_update_ae_perf():
    ori_ae = torch.randn(batch_size, input_dim)
    ori_ke = torch.randn(batch_size, entity_num, input_dim)
    ori_sample_result = torch.randint(0, entity_num, size=(1, ))
    ori_true_entity_num = torch.randint(entity_num - 1, entity_num, size=(1, ))

    hpc_ae = ori_ae.clone().detach()
    hpc_ke = ori_ke.clone().detach()
    hpc_sample_result = ori_sample_result.clone().detach()
    hpc_true_entity_num = ori_true_entity_num.clone().detach()

    if use_cuda:
        ori_ae = ori_ae.cuda()
        ori_ke = ori_ke.cuda()
        ori_sample_result = ori_sample_result.cuda()
        ori_true_entity_num = ori_true_entity_num.cuda()

        hpc_ae = hpc_ae.cuda()
        hpc_ke = hpc_ke.cuda()
        hpc_sample_result = hpc_sample_result.cuda()
        hpc_true_entity_num = hpc_true_entity_num.cuda()

    bs = ori_ae.shape[0]
    end_flag = torch.zeros(bs).cuda().bool()
    end_flag[ori_sample_result == ori_true_entity_num] = 1

    ori_out = torch_update_ae(ori_ae, ori_ke, ori_sample_result, ori_true_entity_num, end_flag)
    hpc_out = hpc_update_ae(ori_ae, ori_ke, ori_sample_result, ori_true_entity_num)
    if use_cuda:
        torch.cuda.synchronize()

    t = time.time()
    for i in range(times):
        ori_out = torch_update_ae(ori_ae, ori_ke, ori_sample_result, ori_true_entity_num, end_flag)
        #ori_ae = ori_ae + ori_ke[torch.arange(bs), ori_sample_result] * ~end_flag.unsqueeze(dim=1)
    if use_cuda:
        torch.cuda.synchronize()
    print('original update ae cost time: {}'.format(time.time() - t))

    t = time.time()
    for i in range(times):
        hpc_out = hpc_update_ae(hpc_ae, hpc_ke, hpc_sample_result, hpc_true_entity_num)
        #hpc_models.actor_critic_update_ae([hpc_ke, hpc_sample_result, hpc_true_entity_num], [hpc_ae])
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
    ori_key = torch.randn(lstm_batch_size, entity_num, lstm_hidden_size)
    ori_mask = torch.randint(0, 2, size=(entity_num, ))
    hpc_x = ori_x.clone().detach()
    hpc_key = ori_key.clone().detach()
    hpc_mask = ori_mask.clone().detach()
    hpc_out = torch.zeros(lstm_batch_size, entity_num)

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
    #print("ori_mask: " + str(ori_mask))

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


def actor_critic_pre_sample_perf():
    ori_x = torch.randn(lstm_seq_len, lstm_batch_size, lstm_hidden_size)
    ori_key = torch.randn(lstm_batch_size, entity_num, lstm_hidden_size)
    ori_mask = torch.randint(0, 2, size=(entity_num, ))
    hpc_x = ori_x.clone().detach()
    hpc_key = ori_key.clone().detach()
    hpc_mask = ori_mask.clone().detach()
    hpc_out = torch.zeros(lstm_batch_size, entity_num)

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
    #print("ori_mask: " + str(ori_mask))
 
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
    print("target problem: batch_size = {}, entity_num = {}, input_dim = {}".format(batch_size, entity_num, input_dim))
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

    print("target problem: lstm_batch_size = {}, lstm_seq_len = {}, lstm_hidden_size = {}, entity_num = {}".format(
        lstm_batch_size, lstm_seq_len, lstm_hidden_size, entity_num))
    print("================run actor critic pre sample validation test================")
    actor_critic_pre_sample_val()
    print("================run actor critic pre sample performance test================")
    actor_critic_pre_sample_perf()
    print("\n")
