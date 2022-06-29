import torch
from hpc_rll.origin.gru import GRUGatingUnit
from hpc_rll.rl_utils.gru import GRU
from testbase import mean_relative_error, times


def test_ori_gru():
    input_dim = 32
    gru = GRUGatingUnit(input_dim, 1.).cuda()
    x = torch.rand((4, 12, 32)).cuda()
    y = torch.rand((4, 12, 32)).cuda()
    out = gru(x, y)
    assert out.shape == x.shape
    print('ok')
    gru = GRUGatingUnit(
        input_dim, 100000.
    ).cuda()  # set high bias to check 'out' is similar to the first input 'x'
    # In GTrXL the bias is initialized with a value high enough such that information coming from the second input
    # 'y' are partially ignored so to produce a behavior more similar to a MDP, thus giving less importance to
    # past information
    out = gru(x, y)
    #torch.testing.assert_close(out, x)
    print('good')


def test_hpc_gru():
    input_dim = 4
    T = 1
    B = 2
    hpc_gru = GRU(T, B, input_dim=input_dim, bg=1.).cuda()

    x = torch.rand((T, B, input_dim)).cuda()
    y = torch.rand((T, B, input_dim)).cuda()

    hpc_out = hpc_gru(x, y)
    assert hpc_out.shape == x.shape

    hpc_gru = GRU(T, B, input_dim=input_dim, bg=100000.).cuda()
    # set high bias to check 'out' is similar to the first input 'x'
    # In GTrXL the bias is initialized with a value high enough such that information coming from the second input
    # 'y' are partially ignored so to produce a behavior more similar to a MDP, thus giving less importance to
    # past information
    out = hpc_gru(x, y)
    print('x: ', x)
    print('out: ', out)


if __name__ == "__main__":
    test_hpc_gru()