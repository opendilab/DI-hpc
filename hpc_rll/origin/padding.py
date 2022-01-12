from typing import Tuple, List
import torch


def Padding1D(x: List[torch.Tensor], mode='constant', value: int = 0, group: int = 1) -> Tuple[torch.Tensor, torch.Tensor, List]:
    assert mode in ['constant'], mode
    assert group >= 1, group
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


def UnPadding1D(x: torch.Tensor, shapes: List, deepcopy: bool = False) -> List[torch.Tensor]:
    new_x = []
    for i in range(x.shape[0]):
        idx = [i] + list(shapes[i])
        item = x[idx[0], :idx[1]]
        if deepcopy:
            item = item.clone()
        new_x.append(item)
    return new_x


def Padding2D(x: List[torch.Tensor], mode='constant', value: int = 0, group: int = 1) -> Tuple[torch.Tensor, torch.Tensor, List]:
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
