from fcntl import DN_DELETE
from typing import Tuple, List, Union
from functools import reduce
import torch
import numpy as np
import hpc_rl_utils

# hpc version only support cuda

def cum(t: List[int]) -> int:
    return reduce(lambda x, y: x * y, t)

def Padding1D(inputs: List[torch.Tensor], mode='constant', value: int = 0, group: int = 1, group_mode='sample'):
    assert mode in ['constant'], mode
    assert group_mode in ['sample', 'oracle'], group_mode
    assert group >= 1, group
    if group > 1:
        inputs = sorted(inputs, key=lambda t: cum(t.shape))
        if group_mode == 'sample':
            res = hpc_rl_utils.sample_split_group(inputs, group)
            group_idx = res[-1]
            group_shape = res[:-1]
        elif group_mode == 'oracle':
            res = hpc_rl_utils.oracle_split_group(inputs, group)
            group_idx = res[-1]
            group_shape = res[:-1]
        assert len(group_idx) == len(group_shape) + 1
        max_shape = [s[0] for s in group_shape]
        group_num = [(group_idx[i+1] - group_idx[i]) for i in range(len(group_shape))]
        shapes = []
        group_id = []
        k = 0
        for i in range(len(group_num)):
            shape = []
            for j in range(group_num[i]):
                shape.append(inputs[k].shape[0])
                group_id.append(i)
                k = k + 1
            shapes.append(shape)
        assert len(group_id) == len(inputs)
        result = hpc_rl_utils.GroupPad1DForward(inputs, group_num, max_shape, group_id, group_idx, value)
        new_x = result[0]
        mask = result[1]
        return [tuple(new_x), tuple(mask), tuple(shapes)]
    else:
        shapes = [t.shape[0] for t in inputs]
        result = hpc_rl_utils.Pad1DForward(inputs, value)
        new_x = result[0]
        mask = result[1]
        return new_x, mask, shapes


def UnPadding1D(x: Union[torch.Tensor, List[torch.Tensor]],
                shapes: Union[List, List[List]]) -> List[torch.Tensor]:
    if isinstance(x, torch.Tensor):
        return hpc_rl_utils.Unpad1DForward(x, shapes)
    else:
        ret = []
        for t, s in zip(x, shapes):
            ret.append(hpc_rl_utils.Unpad1DForward(t, s))
        return sum(ret, [])

def Padding2D(inputs: List[torch.Tensor], mode='constant', value: int = 0, group: int = 1, group_mode='sample'):
    assert mode in ['constant'], mode
    assert group_mode in ['sample', 'oracle'], group_mode
    assert group >= 1, group
    if group > 1:
        inputs = sorted(inputs, key=lambda t: cum(t.shape))
        if group_mode == 'sample':
            res = hpc_rl_utils.sample_split_group(inputs, group)
            group_idx = res[-1]
            group_shape = res[:-1]
        elif group_mode == 'oracle':
            res = hpc_rl_utils.oracle_split_group(inputs, group)
            group_idx = res[-1]
            group_shape = res[:-1]
        assert len(group_idx) == len(group_shape) + 1
        max_shape = []
        for s in group_shape:
            max_shape.append(s[0])
            max_shape.append(s[1])
        group_cnt = [(group_idx[i+1] - group_idx[i]) for i in range(len(group_shape))]
        shapes = []
        group_id = []
        k = 0
        for i in range(len(group_cnt)):
            shape = []
            for j in range(group_cnt[i]):
                shape.append(inputs[k].shape[0])
                shape.append(inputs[k].shape[1])
                group_id.append(i)
                k = k + 1
            shapes.append(shape)
        assert len(group_id) == len(inputs)
        result = hpc_rl_utils.GroupPad2DForward(inputs, group_cnt, max_shape, group_id, group_idx, value)
        new_x = result[0]
        mask = result[1]
        return [tuple(new_x), tuple(mask), tuple(shapes)]
    else:
        shapes = []
        for t in inputs:
            shapes.append(t.shape[0])
            shapes.append(t.shape[1])
        result = hpc_rl_utils.Pad2DForward(inputs, value)
        new_x = result[0]
        mask = result[1]
        return new_x, mask, shapes


def UnPadding2D(x: Union[torch.Tensor, List[torch.Tensor]],
                shapes: Union[List, List[List]]) -> List[torch.Tensor]:
    if isinstance(x, torch.Tensor):
        return hpc_rl_utils.Unpad2DForward(x, shapes)
    else:
        ret = []
        for t, s in zip(x, shapes):
            ret.append(hpc_rl_utils.Unpad2DForward(t, s))
        return sum(ret, [])

def Padding3D(inputs: List[torch.Tensor], mode='constant', value: int = 0, group: int = 1, group_mode='sample'):
    assert mode in ['constant'], mode
    assert group_mode in ['sample', 'oracle'], group_mode
    assert group >= 1, group
    if group > 1:
        inputs = sorted(inputs, key=lambda t: cum(t.shape))
        if group_mode == 'sample':
            res = hpc_rl_utils.sample_split_group(inputs, group)
            group_idx = res[-1]
            group_shape = res[:-1]
        elif group_mode == 'oracle':
            res = hpc_rl_utils.oracle_split_group(inputs, group)
            group_idx = res[-1]
            group_shape = res[:-1]
        assert len(group_idx) == len(group_shape) + 1
        max_shape = []
        for s in group_shape:
            max_shape.append(s[0])
            max_shape.append(s[1])
            max_shape.append(s[2])

        group_cnt = [(group_idx[i+1] - group_idx[i]) for i in range(len(group_shape))]
        shapes = []
        group_id = []
        k = 0
        for i in range(len(group_cnt)):
            shape = []
            for j in range(group_cnt[i]):
                shape.append(inputs[k].shape[0])
                shape.append(inputs[k].shape[1])
                shape.append(inputs[k].shape[2])
                group_id.append(i)
                k = k + 1
            shapes.append(shape)
        assert len(group_id) == len(inputs)
        result = hpc_rl_utils.GroupPad3DForward(inputs, group_cnt, max_shape, group_id, group_idx, value)
        new_x = result[0]
        mask = result[1]
        return [tuple(new_x), tuple(mask), tuple(shapes)]
    else:
        shapes = []
        for t in inputs:
            shapes.append(t.shape[0])
            shapes.append(t.shape[1])
            shapes.append(t.shape[2])
        result = hpc_rl_utils.Pad3DForward(inputs, value)
        new_x = result[0]
        mask = result[1]
        return new_x, mask, shapes


def UnPadding3D(x: Union[torch.Tensor, List[torch.Tensor]],
                shapes: Union[List, List[List]]) -> List[torch.Tensor]:
    if isinstance(x, torch.Tensor):
        return hpc_rl_utils.Unpad3DForward(x, shapes)
    else:
        ret = []
        for t, s in zip(x, shapes):
            ret.append(hpc_rl_utils.Unpad3DForward(t, s))
        return sum(ret, [])


