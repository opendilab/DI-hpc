from fcntl import DN_DELETE
from typing import Tuple, List, Union
from functools import reduce
import torch
import numpy as np
import hpc_rl_utils

# hpc version only support cuda

def cum(t: List[int]) -> int:
    return reduce(lambda x, y: x * y, t)


def oracle_split_group(x: List[torch.Tensor], group: int) -> Tuple[List[Tuple], List[int]]:
    arr = [None] + [cum(t.shape) for t in x]
    N, M = len(arr) - 1, group

    def p(start, end):  # cost of [start, end] in arr
        return arr[end] * (end - start + 1)  # total cost is enough
        # return arr[end] * (end - start + 1) - sum(arr[start:end + 1])

    # DP, time complex O(MN^2), space complex O(MN)
    f = {(0, 0): (0, 0)}
    for i, length_ in enumerate(arr[1:], start=1):
        for j in range(1, M + 1):
            ress = []
            for k in range(0, i):
                if (k, j - 1) in f:
                    last_cost, _ = f[(k, j - 1)]
                    ress.append((last_cost + p(k + 1, i), k))

            if ress:
                f[(i, j)] = min(ress)

    min_cost, _ = f[(N, M)]
    last_position, last_cnt = N, M
    positions = [N]
    while last_position > 0:
        _, last_position = f[(last_position, last_cnt)]
        last_cnt -= 1
        positions.append(last_position)

    assert len(positions) == M + 1
    positions = positions[::-1]

    # print(min_cost)  # minimal cost
    # for i in range(0, M):  # solution
    #     start = positions[i] + 1
    #     end = positions[i + 1]
    #     cost = p(start, end)
    #     print(i, arr[start:end + 1], start, end, cost)
    shapes = [x[i - 1].shape for i in positions[1:]]
    return shapes, positions

def Padding1D(inputs: List[torch.Tensor], mode='constant', value: int = 0, group: int = 1, group_mode='sample'):
    assert mode in ['constant'], mode
    assert group_mode in ['sample', 'oracle'], group_mode
    assert group >= 1, group
    if group > 1:
        inputs = sorted(inputs, key=lambda t: cum(t.shape))
        if group_mode == 'sample':
            sampled_idx = np.random.choice(len(inputs), group - 1)
            group_shape = [t.shape for i, t in enumerate(inputs) if i in sampled_idx]
            group_shape += [inputs[-1].shape]  # max shape
            print('sample group_shape', group_shape)
            group_shape = list(set(group_shape))  # remove repeat shape
            group_shape = sorted(group_shape, key=lambda t: cum(t))
            group_shape_idx = 0
            group_idx = [0]
            for i, t in enumerate(inputs):
                if cum(t.shape) > cum(group_shape[group_shape_idx]):
                    group_idx.append(i)
                    group_shape_idx += 1
            group_idx.append(len(inputs))
        elif group_mode == 'oracle':
            group_shape, group_idx = oracle_split_group(inputs, group)
            print('group_shape', group_shape)
        assert len(group_idx) == len(group_shape) + 1
        s = [t.shape[0] for t in inputs]
        max_shape = [s[0] for s in group_shape]
        group_num = [(group_idx[i+1] - group_idx[i]) for i in range(len(group_shape))]
        new_x = []
        mask = []
        shapes = []
        group_id = []
        k = 0
        for i in range(len(group_num)):
            new_x.append(torch.zeros(group_num[i], max_shape[i]))
            mask.append(torch.zeros(group_num[i], max_shape[i]))
            shape = []
            for j in range(group_num[i]):
                shape.append(inputs[k].shape)
                group_id.append(i)
                k = k + 1
            shapes.append(shape)
        assert len(group_id) == len(inputs)
        hpc_rl_utils.GroupPad1DForward(inputs, torch.Tensor(s), new_x, mask, torch.Tensor(max_shape), torch.Tensor(group_id), torch.Tensor(group_idx), value)
        return [tuple(new_x), tuple(mask), tuple(shapes)]
    else:
        n = len(inputs)
        max_shape = 0
        shapes = [t.shape for t in inputs]
        s = []
        for t in inputs:
            s.append(t.shape[0])
            if max_shape < t.shape[0]:
                max_shape = t.shape[0]
        new_x = torch.zeros(n, max_shape)
        mask = torch.zeros(n, max_shape)
        hpc_rl_utils.Pad1DForward(inputs, torch.Tensor(s), new_x, mask, max_shape, value)
        return new_x, mask, shapes


def UnPadding1D(inputs, shape):
    n = len(shape)
    outputs = []
    for i in range(n):
        outputs.append(torch.zeros(shape[i][0]))
    hpc_rl_utils.Unpad1DForward(inputs, shape, outputs)
    return outputs

def Padding2D(inputs: List[torch.Tensor], mode='constant', value: int = 0, group: int = 1, group_mode='sample'):
    assert mode in ['constant'], mode
    assert group_mode in ['sample', 'oracle'], group_mode
    assert group >= 1, group
    if group > 1:
        inputs = sorted(inputs, key=lambda t: cum(t.shape))
        if group_mode == 'sample':
            sampled_idx = np.random.choice(len(inputs), group - 1)
            group_shape = [t.shape for i, t in enumerate(inputs) if i in sampled_idx]
            group_shape += [inputs[-1].shape]  # max shape
            print('sample group_shape', group_shape)
            group_shape = list(set(group_shape))  # remove repeat shape
            group_shape = sorted(group_shape, key=lambda t: cum(t))
            group_shape_idx = 0
            group_idx = [0]
            for i, t in enumerate(inputs):
                if cum(t.shape) > cum(group_shape[group_shape_idx]):
                    group_idx.append(i)
                    group_shape_idx += 1
            group_idx.append(len(inputs))
        elif group_mode == 'oracle':
            group_shape, group_idx = oracle_split_group(inputs, group)
            print('group_shape', group_shape)
        assert len(group_idx) == len(group_shape) + 1
        s = []
        max_shape = []
        for t in inputs:
            s.append(t.shape[0])
            s.append(t.shape[1])
        for t in group_shape:
            max_shape.append(t[0])
            max_shape.append(t[1])
        group_num = [(group_idx[i+1] - group_idx[i]) for i in range(len(group_shape))]
        new_x = []
        mask = []
        shapes = []
        group_id = []
        k = 0
        for i in range(len(group_num)):
            new_x.append(torch.zeros(group_num[i], max_shape[i * 2], max_shape[i * 2 + 1]))
            mask.append(torch.zeros(group_num[i], max_shape[i * 2], max_shape[i * 2 + 1]))
            shape = []
            for j in range(group_num[i]):
                shape.append(inputs[k].shape)
                group_id.append(i)
                k = k + 1
            shapes.append(shape)
        assert len(group_id) == len(inputs)
        hpc_rl_utils.GroupPad2DForward(inputs, torch.Tensor(s), new_x, mask, torch.Tensor(max_shape), torch.Tensor(group_id), torch.Tensor(group_idx), value)
        return [tuple(new_x), tuple(mask), tuple(shapes)]
    else:
        n = len(inputs)
        max_shape0 = 0
        max_shape1 = 0
        s = []
        shapes = [t.shape for t in inputs]
        for t in inputs:
            s.append(t.shape[0])
            s.append(t.shape[1])
            if max_shape0 < t.shape[0]:
                max_shape0 = t.shape[0]
            if max_shape1 < t.shape[1]:
                max_shape1 = t.shape[1]
        new_x = torch.zeros(n, max_shape0, max_shape1)
        mask = torch.zeros(n, max_shape0, max_shape1)
        hpc_rl_utils.Pad2DForward(inputs, torch.Tensor(s), new_x, mask, max_shape0, max_shape1, value)
        return new_x, mask, shapes

def UnPadding2D(inputs, shape):
    n = len(shape)
    outputs = []
    for i in range(n):
        outputs.append(torch.zeros(shape[i][0], shape[i][1]))
    hpc_rl_utils.Unpad2DForward(inputs, shape, outputs)
    return outputs

def Padding3D(inputs: List[torch.Tensor], mode='constant', value: int = 0, group: int = 1, group_mode='sample'):
    assert mode in ['constant'], mode
    assert group_mode in ['sample', 'oracle'], group_mode
    assert group >= 1, group
    if group > 1:
        inputs = sorted(inputs, key=lambda t: cum(t.shape))
        if group_mode == 'sample':
            sampled_idx = np.random.choice(len(inputs), group - 1)
            group_shape = [t.shape for i, t in enumerate(inputs) if i in sampled_idx]
            group_shape += [inputs[-1].shape]  # max shape
            print('sample group_shape', group_shape)
            group_shape = list(set(group_shape))  # remove repeat shape
            group_shape = sorted(group_shape, key=lambda t: cum(t))
            group_shape_idx = 0
            group_idx = [0]
            for i, t in enumerate(inputs):
                if cum(t.shape) > cum(group_shape[group_shape_idx]):
                    group_idx.append(i)
                    group_shape_idx += 1
            group_idx.append(len(inputs))
        elif group_mode == 'oracle':
            group_shape, group_idx = oracle_split_group(inputs, group)
            print('group_shape', group_shape)
        assert len(group_idx) == len(group_shape) + 1
        s = []
        max_shape = []
        for t in inputs:
            s.append(t.shape[0])
            s.append(t.shape[1])
            s.append(t.shape[2])
        for t in group_shape:
            max_shape.append(t[0])
            max_shape.append(t[1])
            max_shape.append(t[2])
        group_num = [(group_idx[i+1] - group_idx[i]) for i in range(len(group_shape))]
        new_x = []
        mask = []
        shapes = []
        group_id = []
        k = 0
        for i in range(len(group_num)):
            new_x.append(torch.zeros(group_num[i], max_shape[i * 3], max_shape[i * 3 + 1], max_shape[i * 3 + 2]))
            mask.append(torch.zeros(group_num[i], max_shape[i * 3], max_shape[i * 3 + 1], max_shape[i * 3 + 2]))
            shape = []
            for j in range(group_num[i]):
                shape.append(inputs[k].shape)
                group_id.append(i)
                k = k + 1
            shapes.append(shape)
        assert len(group_id) == len(inputs)
        hpc_rl_utils.GroupPad3DForward(inputs, torch.Tensor(s), new_x, mask, torch.Tensor(max_shape), torch.Tensor(group_id), torch.Tensor(group_idx), value)
        return [tuple(new_x), tuple(mask), tuple(shapes)]
    else:
        n = len(inputs)
        max_shape0 = 0
        max_shape1 = 0
        max_shape2 = 0
        s = []
        shapes = [t.shape for t in inputs]
        for t in inputs:
            s.append(t.shape[0])
            s.append(t.shape[1])
            s.append(t.shape[2])
            if max_shape0 < t.shape[0]:
                max_shape0 = t.shape[0]
            if max_shape1 < t.shape[1]:
                max_shape1 = t.shape[1]
            if max_shape2 < t.shape[2]:
                max_shape2 = t.shape[2]
        new_x = torch.zeros(n, max_shape0, max_shape1, max_shape2)
        mask = torch.zeros(n, max_shape0, max_shape1, max_shape2)
        hpc_rl_utils.Pad3DForward(inputs, torch.Tensor(s), new_x, mask, max_shape0, max_shape1, max_shape2, value)
        return new_x, mask, shapes

def UnPadding3D(inputs, shape):
    n = len(shape)
    outputs = []
    for i in range(n):
        outputs.append(torch.zeros(shape[i][0], shape[i][1], shape[i][2]))
    hpc_rl_utils.Unpad3DForward(inputs, shape, outputs)
    return outputs


