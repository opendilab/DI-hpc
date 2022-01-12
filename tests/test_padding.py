import time
import numpy as np
import torch
from hpcrll.origin.padding import Padding1D, UnPadding1D, Padding2D, UnPadding2D


B = 64
range_1D = [32, 128]
range_2D = [[48, 80], [32, 64]]
cuda = False


def test_padding_1D():
    shapes = [(np.random.randint(range_1D[0], range_1D[1]), ) for _ in range(B)]
    data = [torch.randn(*s) for s in shapes]
    if cuda:
        data = [d.cuda() for d in data]
    assert len(data) == B
    max_shape = [max(t) for t in list(zip(*shapes))]
    padding_data, padding_mask, ori_shapes = Padding1D(data)
    assert padding_data.shape == (B, *max_shape)
    assert padding_mask.shape == (B, *max_shape)
    unpadding_data = UnPadding1D(padding_data, ori_shapes)
    for item, new_item in zip(data, unpadding_data):
        assert item.eq(new_item).all()
    print("test_padding_1D OK")


def test_padding_2D():
    shapes = [(np.random.randint(range_2D[0][0], range_2D[0][1]), np.random.randint(range_2D[1][0], range_2D[1][1])) for _ in range(B)]
    data = [torch.randn(*s) for s in shapes]
    if cuda:
        data = [d.cuda() for d in data]
    assert len(data) == B
    max_shape = [max(t) for t in list(zip(*shapes))]
    padding_data, padding_mask, ori_shapes = Padding2D(data)
    assert padding_data.shape == (B, *max_shape)
    assert padding_mask.shape == (B, *max_shape)
    unpadding_data = UnPadding2D(padding_data, ori_shapes)
    for item, new_item in zip(data, unpadding_data):
        assert item.eq(new_item).all()
    print("test_padding_2D OK")


if __name__ == "__main__":
    test_padding_1D()
    test_padding_2D()
