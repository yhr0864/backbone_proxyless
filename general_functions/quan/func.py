import torch as t
from building_blocks import layers


class QuanConv2d(layers.misc.Conv2d):
    def __init__(self, m: layers.misc.Conv2d, quan_w_fn=None, quan_a_fn=None):
        assert type(m) == layers.misc.Conv2d
        super().__init__(m.in_channels, m.out_channels, m.kernel_size,
                         stride=m.stride,
                         padding=m.padding,
                         dilation=m.dilation,
                         groups=m.groups,
                         bias=True if m.bias is not None else False,
                         padding_mode=m.padding_mode)
        self.quan_w_fn = quan_w_fn
        self.quan_a_fn = quan_a_fn
        self.weight = t.nn.Parameter(m.weight.detach())
        self.quan_w_fn.init_from(m.weight)
        if m.bias is not None:
            self.bias = t.nn.Parameter(m.bias.detach())

    def forward(self, x):
        quantized_weight = self.quan_w_fn(self.weight)
        x = self.quan_a_fn(x)
        return self._conv_forward(x, quantized_weight, self.bias)


class QuanAct(t.nn.Module):
    def __init__(self, act, quan_a_fn=None):
        super().__init__()
        self.act = act
        self.quan_a_fn = quan_a_fn

    def forward(self, x):
        return self.quan_a_fn(self.act(x))


if __name__=='__main__':
    import torch
    from building_blocks.layers import Conv2d
    from general_functions.quan.utils import quantizer
    from supernet_functions.config_for_supernet import CONFIG_SUPERNET
    QuanModuleMapping = {
        t.nn.Conv2d: QuanConv2d
    }

    op = QuanConv2d(m=Conv2d(3,3,1),
                          quan_w_fn=quantizer(CONFIG_SUPERNET['quan']['weight']),
                          quan_a_fn=quantizer(CONFIG_SUPERNET['quan']['act']))

    torch.nn.init.kaiming_normal_(op.weight, mode="fan_out", nonlinearity="relu")
    input = torch.randn(2,3,10,10)
    print(op(input).shape)