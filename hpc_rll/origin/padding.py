from typing import Tuple, List, Union
from functools import reduce
import numpy as np
import torch


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


def _Padding1D(x: List[torch.Tensor], value: int = 0) -> Tuple[torch.Tensor, torch.Tensor, List]:
    shapes = [t.shape for t in x]
    max_shape = [max(t) for t in list(zip(*shapes))]
    new_shape = [len(x)] + max_shape
    mask = torch.full(new_shape, fill_value=value, dtype=x[0].dtype, device=x[0].device)
    new_x = torch.full(new_shape, fill_value=value, dtype=x[0].dtype, device=x[0].device)
    for i in range(mask.shape[0]):
        idx = [i] + list(shapes[i])
        mask[idx[0], :idx[1]] = 1
        new_x[idx[0], :idx[1]] = x[i]
    return new_x, mask, shapes


def Padding1D(x: List[torch.Tensor], mode='constant', value: int = 0, group: int = 1, group_mode='sample') -> Tuple:
    assert mode in ['constant'], mode
    assert group_mode in ['sample', 'oracle'], group_mode
    assert group >= 1, group
    if group > 1:
        x = sorted(x, key=lambda t: cum(t.shape))
        if group_mode == 'sample':
            sampled_idx = np.random.choice(len(x), group - 1)
            group_shape = [t.shape for i, t in enumerate(x) if i in sampled_idx]
            group_shape += [x[-1].shape]  # max shape
            print('sample group_shape', group_shape)
            group_shape = list(set(group_shape))  # remove repeat shape
            group_shape = sorted(group_shape, key=lambda t: cum(t))
            group_shape_idx = 0
            group_idx = [0]
            for i, t in enumerate(x):
                if cum(t.shape) > cum(group_shape[group_shape_idx]):
                    group_idx.append(i)
                    group_shape_idx += 1
            group_idx.append(len(x))
        elif group_mode == 'oracle':
            group_shape, group_idx = oracle_split_group(x, group)
            print('group_shape', group_shape)
        assert len(group_idx) == len(group_shape) + 1

        ret = [_Padding1D(x[group_idx[i]:group_idx[i + 1]], value) for i in range(len(group_shape))]
        return list(zip(*ret))
    else:
        return _Padding1D(x, value)


def _UnPadding1D(x, shapes, deepcopy: bool = False):
    new_x = []
    for i in range(x.shape[0]):
        idx = [i] + list(shapes[i])
        item = x[idx[0], :idx[1]]
        if deepcopy:
            item = item.clone()
        new_x.append(item)
    return new_x


def UnPadding1D(x: Union[torch.Tensor, List[torch.Tensor]],
                shapes: Union[List, List[List]],
                deepcopy: bool = False) -> List[torch.Tensor]:
    if isinstance(x, torch.Tensor):
        return _UnPadding1D(x, shapes, deepcopy)
    else:
        ret = [_UnPadding1D(t, s, deepcopy) for t, s in zip(x, shapes)]
        return sum(ret, [])


def Padding2D(x: List[torch.Tensor],
              mode='constant',
              value: int = 0,
              group: int = 1) -> Tuple[torch.Tensor, torch.Tensor, List]:
    assert mode in ['constant'], mode
    assert group >= 1, group
    shapes = [t.shape for t in x]
    max_shape = [max(t) for t in list(zip(*shapes))]
    new_shape = [len(x)] + max_shape
    mask = torch.full(new_shape, fill_value=value, dtype=x[0].dtype, device=x[0].device)
    new_x = torch.full(new_shape, fill_value=value, dtype=x[0].dtype, device=x[0].device)
    for i in range(mask.shape[0]):
        idx = [i] + list(shapes[i])
        mask[idx[0], :idx[1], :idx[2]] = 1
        new_x[idx[0], :idx[1], :idx[2]] = x[i]
    return new_x, mask, shapes


def UnPadding2D(x: torch.Tensor, shapes: List, deepcopy: bool = False) -> List[torch.Tensor]:
    new_x = []
    for i in range(x.shape[0]):
        idx = [i] + list(shapes[i])
        item = x[idx[0], :idx[1], :idx[2]]
        if deepcopy:
            item = item.clone()
        new_x.append(item)
    return new_x


def Padding3D(x: List[torch.Tensor],
              mode='constant',
              value: int = 0,
              group: int = 1) -> Tuple[torch.Tensor, torch.Tensor, List]:
    assert mode in ['constant'], mode
    assert group >= 1, group
    shapes = [t.shape for t in x]
    max_shape = [max(t) for t in list(zip(*shapes))]
    new_shape = [len(x)] + max_shape
    mask = torch.full(new_shape, fill_value=value, dtype=x[0].dtype, device=x[0].device)
    new_x = torch.full(new_shape, fill_value=value, dtype=x[0].dtype, device=x[0].device)
    for i in range(mask.shape[0]):
        idx = [i] + list(shapes[i])
        mask[idx[0], :idx[1], :idx[2], :idx[3]] = 1
        new_x[idx[0], :idx[1], :idx[2], :idx[3]] = x[i]
    return new_x, mask, shapes


def UnPadding3D(x: torch.Tensor, shapes: List, deepcopy: bool = False) -> List[torch.Tensor]:
    new_x = []
    for i in range(x.shape[0]):
        idx = [i] + list(shapes[i])
        item = x[idx[0], :idx[1], :idx[2], :idx[3]]
        if deepcopy:
            item = item.clone()
        new_x.append(item)
    return new_x
