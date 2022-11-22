import torch
import torch.nn as nn

from building_blocks.layers import Conv2d


# Used for identifying intrinsic modules used in quantization
class _FusedModule(torch.nn.Sequential):
    pass


class ConvAct2d(_FusedModule):
    r"""This is a sequential container which calls the Conv2d and ReLU modules.
    During quantization this will be replaced with the corresponding fused module."""
    def __init__(self, conv, act):
        assert type(conv) == Conv2d and (type(act) == nn.ReLU or type(act) == nn.Hardswish), \
            'Incorrect types for input modules{}{}'.format(
                type(conv), type(act))
        super().__init__(conv, act)