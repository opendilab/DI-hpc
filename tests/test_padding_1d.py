import time
from functools import reduce
from operator import __mul__
import numpy as np
import torch
import torch.profiler
from hpc_rll.origin.padding import Padding1D, UnPadding1D


def _get_baseline(lengths, m):
    n = len(lengths)
    arr = [None] + sorted(lengths)  # if already in order, this step is not necessary

    def p(_first, _last):  # cost of [start, end] in arr
        return arr[_last] * (_last - _first + 1)

    # DP, time complex O(MN^2), space complex O(MN)
    f = {(0, 0): (0, 0)}
    for i, length_ in enumerate(arr[1:], start=1):
        for j in range(1, m + 1):
            ress = []
            for k in range(0, i):
                if (k, j - 1) in f:
                    last_cost, _ = f[(k, j - 1)]
                    ress.append((last_cost + p(k + 1, i), k))

            if ress:
                f[(i, j)] = min(ress)

    min_cost, _ = f[(n, m)]
    last_position, last_cnt = n, m
    positions = [n]
    while last_position > 0:
        _, last_position = f[(last_position, last_cnt)]
        last_cnt -= 1
        positions.append(last_position)

    assert len(positions) == m + 1
    return min_cost

    # print(min_cost)  # minimal cost
    # for i in range(0, m):  # solution
    #     start = positions[i] + 1
    #     end = positions[i + 1]
    #     cost = p(start, end)
    #     print(i, arr[start:end + 1], start, end, cost)


B = 64
range_1D = [32, 128]
cuda = torch.cuda.is_available()
print(f'CUDA is {"available" if cuda else "not available"}.')


# Do not use dict object on default values, it is changeable which may affect the following calls.
# Use Ellipsis(...) or None, and check it inside instead.
def test_padding_1D(times=5, scheme=None):
    if scheme is None:
        scheme = {'naive': [Padding1D, UnPadding1D]}

    if cuda:
        import hpc_rll.rl_utils.padding as H
        scheme['hpc'] = [H.Padding1D, H.UnPadding1D]
    # warm up
    for _ in range(10):
        tmp = torch.randn(128, 128)
        if cuda:
            tmp = tmp.cuda()
        torch.matmul(tmp, tmp)
    # test
    shapes = [(np.random.randint(range_1D[0], range_1D[1]),) for _ in range(B)]
    data = [torch.randn(*s) for s in shapes]
    if cuda:
        data = [d.cuda() for d in data]
    assert len(data) == B
    max_shape = [max(t) for t in list(zip(*shapes))]
    for name, [pad, unpad] in scheme.items():
        if name == 'hpc':
            assert cuda
        # with torch.profiler.profile(
        #     schedule=torch.profiler.schedule(wait=2, warmup=2, active=6, repeat=1),
        #     on_trace_ready=torch.profiler.tensorboard_trace_handler(
        #         dir_name='Reports'),
        #     activities=[torch.profiler.ProfilerActivity.CPU,
        #                 torch.profiler.ProfilerActivity.CUDA],
        #     profile_memory=True,
        #     with_stack=True,
        # ) as profiler:

        for i in range(times):
            t = time.time()
            padding_data, padding_mask, ori_shapes = pad(data)
            assert padding_data.shape == (B, *max_shape)
            assert padding_mask.shape == (B, *max_shape)
            unpadding_data = unpad(padding_data, ori_shapes)

            ori_lengths = [d.shape[0] for d in data]
            final_size = reduce(__mul__, padding_data.shape)
            expected_size = _get_baseline(ori_lengths, 1)
            assert final_size <= expected_size, \
                f'No more than {expected_size} expected, but {final_size} found actually.'

            print('epoch: {}, {} cost time: {}'.format(i, name, time.time() - t))
            # profiler.step()
            for item, new_item in zip(data, unpadding_data):
                assert item.eq(new_item).all()
        # if cuda: torch.cuda.synchronize()
        # print('epoch: {}, {} cost time: {}'.format(i, name, time.time() - t))
    print("test_padding_1D OK")

    for name, [pad, unpad] in scheme.items():
        if name == 'hpc':
            assert cuda
        for mode in ['sample', 'oracle']:
            # start_time = time.time()
            # with torch.profiler.profile(
            # schedule=torch.profiler.schedule(wait=2, warmup=2, active=6, repeat=1),
            # on_trace_ready=torch.profiler.tensorboard_trace_handler(
            #     dir_name='Reports'),
            # activities=[torch.profiler.ProfilerActivity.CPU,
            #             torch.profiler.ProfilerActivity.CUDA],
            # profile_memory=True,
            # with_stack=True,
            # ) as profiler:
            for i in range(times):
                t = time.time()
                padding_data, padding_mask, ori_shapes = pad(data, group=4, group_mode=mode)
                assert len(padding_data) <= 4
                assert len(padding_mask) <= 4
                assert len(ori_shapes) <= 4
                # for i, item in enumerate(padding_data):
                #     print('{} group {} data shape: {}'.format(name, i, item.shape))
                assert sum([len(item) for item in padding_data]) == B
                assert sum([len(item) for item in padding_mask]) == B
                unpadding_data = unpad(padding_data, ori_shapes)

                ori_lengths = [d.shape[0] for d in data]
                final_size = sum([reduce(__mul__, sp.shape) for sp in padding_data])
                expected_size = _get_baseline(ori_lengths, 4)
                assert final_size <= expected_size, \
                    f'No more than {expected_size} expected, but {final_size} found actually.'

                print('epoch: {}, {} {} cost time: {}'.format(i, name, mode, time.time() - t))
                sorted_data = sorted(data, key=lambda t: reduce(lambda x, y: x * y, t.shape))
                for item, new_item in zip(sorted_data, unpadding_data): assert item.eq(new_item).all()
                # profiler.step()

                # if cuda: torch.cuda.synchronize()
                # end = time.time()
                # print(f'{} {} Total Time Consumption : {end:.4f}')
            print("{} test_padding_1D group with {} OK".format(name, mode))

        same_data = [torch.randn(32) for _ in range(B)]
        padding_data, padding_mask, ori_shapes = pad(same_data, group=4)
        assert len(padding_data) == 1
        assert padding_data[0].shape == (B, 32)
        print("{} test_padding_1D same data group OK".format(name))

if __name__ == '__main__':
    test_padding_1D()