import time
import numpy as np
import torch
import torch.profiler
from functools import reduce
from hpc_rll.origin.padding import Padding1D, UnPadding1D, Padding2D, UnPadding2D, Padding3D, UnPadding3D
import tensorboard

B = 64
range_1D = [32, 128]
range_2D = [[48, 80], [32, 64]]
range_3D = [[24, 32], [24, 32], [32, 40]]
cuda = torch.cuda.is_available()


def test_padding_1D(times=5, scheme={'naive': [Padding1D, UnPadding1D]}):
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
    shapes = [(np.random.randint(range_1D[0], range_1D[1]), ) for _ in range(B)]
    data = [torch.randn(*s) for s in shapes]
    if cuda:
        data = [d.cuda() for d in data]
    assert len(data) == B
    max_shape = [max(t) for t in list(zip(*shapes))]
    for name, [pad, unpad] in scheme.items():
        if name == 'hpc': assert cuda
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
            print('epoch: {}, {} cost time: {}'.format(i, name, time.time() - t))
                #profiler.step()
            for item, new_item in zip(data, unpadding_data):
                assert item.eq(new_item).all()
        #if cuda: torch.cuda.synchronize()
                #print('epoch: {}, {} cost time: {}'.format(i, name, time.time() - t))
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
                print('epoch: {}, {} {} cost time: {}'.format(i, name, mode, time.time() - t))
                sorted_data = sorted(data, key=lambda t: reduce(lambda x, y: x * y, t.shape))
                for item, new_item in zip(sorted_data, unpadding_data): assert item.eq(new_item).all()
                    #profiler.step()
                    

                #if cuda: torch.cuda.synchronize()
                #end = time.time()
                #print(f'{} {} Total Time Consumption : {end:.4f}')
            print("{} test_padding_1D group with {} OK".format(name, mode))

        same_data = [torch.randn(32) for _ in range(B)]
        padding_data, padding_mask, ori_shapes = pad(same_data, group=4)
        assert len(padding_data) == 1
        assert padding_data[0].shape == (B, 32)
        print("{} test_padding_1D same data group OK".format(name))


def test_padding_2D(times=5, scheme={'naive': [Padding2D, UnPadding2D]}):
    if cuda:
        import hpc_rll.rl_utils.padding as H
        scheme['hpc'] = [H.Padding2D, H.UnPadding2D]
    for _ in range(10):
        tmp = torch.randn(128, 128)
        if cuda:
            tmp = tmp.cuda()
        torch.matmul(tmp, tmp)
    shapes = [
        (np.random.randint(range_2D[0][0], range_2D[0][1]), np.random.randint(range_2D[1][0], range_2D[1][1]))
        for _ in range(B)
    ]
    data = [torch.randn(*s) for s in shapes]
    if cuda:
        data = [d.cuda() for d in data]
    assert len(data) == B
    max_shape = [max(t) for t in list(zip(*shapes))]

    for name, [pad, unpad] in scheme.items():
        if name == 'hpc': assert cuda
        for i in range(times):
            t = time.time()
            padding_data, padding_mask, ori_shapes = pad(data)
            assert padding_data.shape == (B, *max_shape)
            assert padding_mask.shape == (B, *max_shape)
            unpadding_data = unpad(padding_data, ori_shapes)
            if cuda: torch.cuda.synchronize()
            print('epoch: {}, {} cost time: {}'.format(i, name, time.time() - t))
            for item, new_item in zip(data, unpadding_data):
                assert item.eq(new_item).all()
    print("test_padding_2D OK")
    
    for name, [pad, unpad] in scheme.items():
        if name == 'hpc':
            assert cuda
        else: continue
        for mode in ['sample', 'oracle']:
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
                print('epoch: {}, {} {} cost time: {}'.format(i, name, mode, time.time() - t))
                sorted_data = sorted(data, key=lambda t: reduce(lambda x, y: x * y, t.shape))
                for item, new_item in zip(sorted_data, unpadding_data): 
                    assert item.eq(new_item).all()
            print("{} test_padding_2D group with {} OK".format(name, mode))

        same_data = [torch.randn(32,32) for _ in range(B)]
        padding_data, padding_mask, ori_shapes = pad(same_data, group=4)
        assert len(padding_data) == 1
        assert padding_data[0].shape == (B, 32, 32)
        print("{} test_padding_2D same data group OK".format(name))


def test_padding_3D(times=5, scheme={'naive': [Padding3D, UnPadding3D]}):
    if cuda:
        import hpc_rll.rl_utils.padding as H
        scheme['hpc'] = [H.Padding3D, H.UnPadding3D]
    for _ in range(10):
        tmp = torch.randn(128, 128)
        if cuda:
            tmp = tmp.cuda()
        torch.matmul(tmp, tmp)
    shapes = [
        (
            np.random.randint(range_3D[0][0], range_3D[0][1]), np.random.randint(range_3D[1][0], range_3D[1][1]),
            np.random.randint(range_3D[2][0], range_3D[2][1])
        ) for _ in range(B)
    ]
    data = [torch.randn(*s) for s in shapes]
    if cuda:
        data = [d.cuda() for d in data]
    assert len(data) == B
    max_shape = [max(t) for t in list(zip(*shapes))]
    for name, [pad, unpad] in scheme.items():
        if name == 'hpc': assert cuda
        for i in range(times):
            t = time.time()
            padding_data, padding_mask, ori_shapes = pad(data)
            
            assert padding_data.shape == (B, *max_shape)
            assert padding_mask.shape == (B, *max_shape)
            unpadding_data = unpad(padding_data, ori_shapes)
            if cuda: torch.cuda.synchronize()
            print('epoch: {}, {} cost time: {}'.format(i, name, time.time() - t))
            for item, new_item in zip(data, unpadding_data):
                
                assert item.eq(new_item).all()
    print("test_padding_3D OK")
    
    for name, [pad, unpad] in scheme.items():
        if name == 'hpc':
            assert cuda
        else: continue
        for mode in ['sample', 'oracle']:
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
                print('epoch: {}, {} {} cost time: {}'.format(i, name, mode, time.time() - t))
                sorted_data = sorted(data, key=lambda t: reduce(lambda x, y: x * y, t.shape))
                for item, new_item in zip(sorted_data, unpadding_data): 
                    assert item.eq(new_item).all()
            print("{} test_padding_3D group with {} OK".format(name, mode))

        same_data = [torch.randn(32,32,32) for _ in range(B)]
        padding_data, padding_mask, ori_shapes = pad(same_data, group=4)
        assert len(padding_data) == 1
        assert padding_data[0].shape == (B, 32, 32, 32)
        print("{} test_padding_3D same data group OK".format(name))


if __name__ == "__main__":
    #test_padding_1D()
    #test_padding_2D()
    test_padding_3D()
