# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from __future__ import absolute_import, division, print_function, unicode_literals

import copy
import logging
import math
from collections import OrderedDict

import numpy as np

from general_functions.loss import compute_loss
from general_functions.quan import QuanConv2d, QuanAct, quantizer
from supernet_functions.config_for_supernet import CONFIG_SUPERNET

import torch
import torch.nn as nn
from .layers import (BatchNorm2d, Conv2d, FrozenBatchNorm2d, interpolate)
from .modeldef import MODEL_ARCH, Test_model_arch

logger = logging.getLogger(__name__)


def _py2_round(x):
    return math.floor(x + 0.5) if x >= 0.0 else math.ceil(x - 0.5)


def _get_divisible_by(num, divisible_by, min_val):
    ret = int(num)
    if divisible_by > 0 and num % divisible_by != 0:
        ret = int((_py2_round(num / divisible_by) or min_val) * divisible_by)
    return ret


# include all the operations
PRIMITIVES = {
    "none": lambda C_in, C_out, expansion, stride, prune, **kwargs: Zero(
        stride
    ),
    "skip": lambda C_in, C_out, expansion, stride, prune, **kwargs: Identity(
        C_in, C_out, stride
    ),
    "ir_k3_re": lambda C_in, C_out, expansion, stride, prune, **kwargs: IRFBlock(
        C_in, C_out, stride, expansion, prune, kernel=3, nl="relu", **kwargs
    ),
    "ir_k3_hs": lambda C_in, C_out, expansion, stride, prune, **kwargs: IRFBlock(
        C_in, C_out, stride, expansion, prune, kernel=3, nl="hswish", **kwargs
    ),
    "ir_k3_r2_re": lambda C_in, C_out, expansion, stride, prune, **kwargs: IRFBlock(
        C_in, C_out, stride, expansion, prune, kernel=3, nl="relu", dil=2, **kwargs
    ),
    "ir_k3_r2_hs": lambda C_in, C_out, expansion, stride, prune, **kwargs: IRFBlock(
        C_in, C_out, stride, expansion, prune, kernel=3, nl="hswish", dil=2, **kwargs
    ),
    "ir_k3_r3_re": lambda C_in, C_out, expansion, stride, prune, **kwargs: IRFBlock(
        C_in, C_out, stride, expansion, prune, kernel=3, nl="relu", dil=3, **kwargs
    ),
    "ir_k5_re": lambda C_in, C_out, expansion, stride, prune, **kwargs: IRFBlock(
        C_in, C_out, stride, expansion, prune, kernel=5, nl="relu", **kwargs
    ),
    "ir_k5_hs": lambda C_in, C_out, expansion, stride, prune, **kwargs: IRFBlock(
        C_in, C_out, stride, expansion, prune, kernel=5, nl="hswish", **kwargs
    ),
    "ir_k5_r2_re": lambda C_in, C_out, expansion, stride, prune, **kwargs: IRFBlock(
        C_in, C_out, stride, expansion, prune, kernel=5, nl="relu", dil=2, **kwargs
    ),
    "ir_k5_r2_hs": lambda C_in, C_out, expansion, stride, prune, **kwargs: IRFBlock(
        C_in, C_out, stride, expansion, prune, kernel=5, nl="hswish", dil=2, **kwargs
    ),
    "ir_k5_r3_re": lambda C_in, C_out, expansion, stride, prune, **kwargs: IRFBlock(
        C_in, C_out, stride, expansion, prune, kernel=5, nl="relu", dil=3, **kwargs
    ),
    "ir_k7_re": lambda C_in, C_out, expansion, stride, prune, **kwargs: IRFBlock(
        C_in, C_out, stride, expansion, prune, kernel=7, nl="relu", **kwargs
    ),
    "ir_k7_hs": lambda C_in, C_out, expansion, stride, prune, **kwargs: IRFBlock(
        C_in, C_out, stride, expansion, prune, kernel=7, nl="hswish", **kwargs
    ),
}


class Zero(nn.Module):
    def __init__(self, stride):
        super(Zero, self).__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.)
        return x[:, :, ::self.stride, ::self.stride].mul(0.)

    @property
    def module_list(self):
        return False


class Identity(nn.Module):
    def __init__(self, C_in, C_out, stride):
        super(Identity, self).__init__()
        self.output_depth = C_out
        self.moduleList = nn.ModuleList([
            ConvBNRelu(
                C_in,
                C_out,
                kernel=1,
                stride=stride,
                pad=0,
                no_bias=1,
                use_relu="relu",
                bn_type="bn",
            )]) if C_in != C_out or stride != 1 else None

    def forward(self, x):
        if self.moduleList:
            out = self.moduleList[0](x)
        else:
            out = x
        return out


class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        """Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]"""
        N, C, H, W = x.size()
        g = self.groups
        assert C % g == 0, "Incompatible group size {} for input channel {}".format(
            g, C
        )
        return (
            x.view(N, g, int(C / g), H, W)
                .permute(0, 2, 1, 3, 4)
                .contiguous()
                .view(N, C, H, W)
        )


class ConvBNRelu(nn.Sequential):
    def __init__(
            self,
            input_depth,
            output_depth,
            kernel,
            stride,
            pad,
            no_bias,
            use_relu,
            bn_type,
            group=1,
            dil=1,
            quant=True,
            *args,
            **kwargs
    ):
        super(ConvBNRelu, self).__init__()

        assert use_relu in ["relu", "hswish", None, False]
        if isinstance(bn_type, (list, tuple)):
            assert len(bn_type) == 2
            assert bn_type[0] == "gn"
            gn_group = bn_type[1]
            bn_type = bn_type[0]
        assert bn_type in ["bn", "af", "gn", None]
        assert stride in [1, 2, 4]
        assert dil in [1, 2, 3, None]

        op = Conv2d(
            input_depth,
            output_depth,
            kernel_size=kernel,
            stride=stride,
            padding=pad,
            dilation=dil,
            bias=not no_bias,
            groups=group,
            *args,
            **kwargs
        )

        if quant:
            op = QuanConv2d(op,
                            quan_w_fn=quantizer(CONFIG_SUPERNET['quan']['weight']),
                            quan_a_fn=quantizer(CONFIG_SUPERNET['quan']['act']))

        nn.init.kaiming_normal_(op.weight, mode="fan_out", nonlinearity="relu")
        if op.bias is not None:
            nn.init.constant_(op.bias, 0.0)
        self.add_module("conv", op)

        if bn_type == "bn":
            bn_op = BatchNorm2d(output_depth)
        elif bn_type == "gn":
            bn_op = nn.GroupNorm(num_groups=gn_group, num_channels=output_depth)
        elif bn_type == "af":
            bn_op = FrozenBatchNorm2d(output_depth)
        if bn_type is not None:
            self.add_module("bn", bn_op)

        if use_relu == "relu":
            act = nn.ReLU(inplace=True)
            # if quant:
            #     act = QuanAct(act, quan_a_fn=quantizer(CONFIG_SUPERNET['quan']['act']))
            self.add_module("act", act)
        elif use_relu == "hswish":
            act = nn.Hardswish(inplace=True)
            # if quant:
            #     act = QuanAct(act, quan_a_fn=quantizer(CONFIG_SUPERNET['quan']['act']))
            self.add_module("act", act)


class SEModule(nn.Module):
    reduction = 4

    def __init__(self, C):
        super(SEModule, self).__init__()
        mid = max(C // self.reduction, 8)
        conv1 = Conv2d(C, mid, 1, 1, 0)
        conv2 = Conv2d(mid, C, 1, 1, 0)

        self.op = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), conv1, nn.ReLU(inplace=True), conv2, nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.op(x)


class Upsample(nn.Module):
    def __init__(self, scale_factor, mode, align_corners=None):
        super(Upsample, self).__init__()
        self.scale = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        return interpolate(
            x, scale_factor=self.scale, mode=self.mode,
            align_corners=self.align_corners
        )

    @property
    def module_list(self):
        return False


class IRFBlock(nn.Module):
    def __init__(
            self,
            input_depth,
            output_depth,
            stride,
            expansion=None,
            prune=None,
            bn_type="bn",
            kernel=3,
            nl="relu",
            dil=1,
            width_divisor=1,
            shuffle_type=None,
            pw_group=1,
            se=False,
            dw_skip_bn=False,
            dw_skip_relu=False
    ):
        super(IRFBlock, self).__init__()

        assert kernel in [1, 3, 5, 7], kernel

        self.use_res_connect = stride == 1 and input_depth == output_depth
        self.output_depth = output_depth

        self.module_list = nn.ModuleList()

        if expansion:
            mid_depth = int(input_depth * expansion)
            mid = mid_depth
        elif prune:
            mid_depth = prune[0]
            mid = prune[1]
        else:
            raise ValueError("neither given expansion nor mid")

        mid_depth = _get_divisible_by(mid_depth, width_divisor, width_divisor)

        # pw
        self.pw = ConvBNRelu(
            input_depth,
            mid_depth,
            kernel=1,
            stride=1,
            pad=0,
            no_bias=1,
            use_relu=nl,
            bn_type=bn_type,
            group=pw_group,
        )
        self.module_list.append(self.pw)
        # dw
        self.dw = ConvBNRelu(
            mid_depth,
            mid,
            kernel=kernel,
            stride=stride,
            pad=(kernel // 2) * dil,
            dil=dil,
            group=mid_depth,
            no_bias=1,
            use_relu=nl if not dw_skip_relu else None,
            bn_type=bn_type if not dw_skip_bn else None,
        )
        self.module_list.append(self.dw)
        # pw-linear
        self.pwl = ConvBNRelu(
            mid,
            output_depth,
            kernel=1,
            stride=1,
            pad=0,
            no_bias=1,
            use_relu=False,
            bn_type=bn_type,
            group=pw_group,
        )
        self.module_list.append(self.pwl)

        self.shuffle_type = shuffle_type
        if shuffle_type is not None:
            self.shuffle = ChannelShuffle(pw_group)

        self.se4 = SEModule(output_depth) if se else nn.Sequential()

        self.output_depth = output_depth

    def forward(self, x):
        y = self.module_list[0](x) # pw
        if self.shuffle_type == "mid":
            y = self.shuffle(y)

        y = self.module_list[1](y) # dw
        y = self.module_list[2](y) # pwl
        if self.use_res_connect:
            y += x
        #y = self.se4(y)
        return y


class SampledNet(nn.Module):
    def __init__(self, arch_def, num_anchors, num_cls,
                 layer_parameters,
                 layer_parameters_head26,
                 layer_parameters_head13,
                 layer_parameters_fpn,
                 yolo_layer26,
                 yolo_layer13,
                 prune_para=None):
        super(SampledNet, self).__init__()
        self.module_list = nn.ModuleList()
        self.first = ConvBNRelu(input_depth=3, output_depth=16, kernel=3, stride=2,
                                pad=3 // 2, no_bias=1, use_relu="relu", bn_type="bn")

        self.module_list.append(self.first) # i=0

        operations = lambda x: {op_name[0]: PRIMITIVES[op_name[0]] for op_name in x}

        self.backbones = arch_def['block_op_type_backbone']
        self.head26 = arch_def['block_op_type_head26']
        self.head13 = arch_def['block_op_type_head13']
        self.fpn = arch_def['block_op_type_fpn']

        for i, op_name in enumerate(self.backbones):
            self.module_list.append(operations(self.backbones)[op_name[0]](*layer_parameters[i]))  # i=1~11

        # preprocess: input_depth = layer_parameters[-1][1]
        #             output_depth = layer_parameters_fpn[0][1] 与fpn所有同步
        
        if prune_para is None:
            self.preprocess = ConvBNRelu(input_depth=1024,
                                         output_depth=512,
                                         kernel=1, stride=1,
                                         pad=0, no_bias=1, use_relu="relu", bn_type="bn")
            
            self.break_conv1x1 = ConvBNRelu(input_depth=512,
                                        output_depth=512,
                                        kernel=1, stride=1,
                                        pad=0, no_bias=1, use_relu="relu", bn_type="bn")
            
        else:
            in_dp_break = None
            for i in range(1, len(prune_para)):
                prev = prune_para[i-1]
                curr = prune_para[i]
                if curr[0] == 8: # 第8层未被剪去
                    in_dp_break = curr[1]
                elif curr[0] == 12:
                    in_dp_prep = prev[1]
                    out_dp_prep = curr[1]
                elif curr[0] == 14:
                    out_dp_break = curr[1]
            if in_dp_break is None:
                for i in range(1, len(prune_para)):
                    prev = prune_para[i-1]
                    curr = prune_para[i]
                    if curr[0] > 8:
                        in_dp_break = prev[1]
                        
            self.preprocess = ConvBNRelu(input_depth=in_dp_prep,
                                         output_depth=out_dp_prep,
                                         kernel=1, stride=1,
                                         pad=0, no_bias=1, use_relu="relu", bn_type="bn")
        # break_conv1x1: input_depth = layer_parameters[7][1]
        #                output_depth = layer_parameters_fpn[0][1] 与fpn所有同步

            self.break_conv1x1 = ConvBNRelu(input_depth=in_dp_break,
                                            output_depth=out_dp_break,
                                            kernel=1, stride=1,
                                            pad=0, no_bias=1, use_relu="relu", bn_type="bn")

        self.module_list.append(self.preprocess) # i=12

        self.upsample = Upsample(scale_factor=2, mode="nearest")

        self.module_list.append(self.upsample) # i=13
        self.module_list.append(self.break_conv1x1) # i=14

        for i, op_name in enumerate(self.fpn):
            self.module_list.append(operations(self.fpn)[op_name[0]](*layer_parameters_fpn[i]))  # i=15~22

        # convert1: input_depth = layer_parameters_fpn[任意][1] 任意：前提是有out_channel而不是none或skip
        #           output_depth = layer_parameters_head[0][0][0] head26的第一个的输入
        
        if prune_para is None:
            self.convert1 = ConvBNRelu(input_depth=512,
                                       output_depth=256,
                                       kernel=1, stride=1,
                                       pad=0, no_bias=1, use_relu="relu", bn_type="bn")

            # convert2: input_depth = layer_parameters_fpn[任意][1] 任意：前提是有out_channel而不是none或skip
            #           output_depth = layer_parameters_head[1][0][0] head13的第一个的输入

            self.convert2 = ConvBNRelu(input_depth=512,
                                       output_depth=256,
                                       kernel=1, stride=1,
                                       pad=0, no_bias=1, use_relu="relu", bn_type="bn")
        else:
            for i in range(len(prune_para)):
                curr = prune_para[i]
                if 15 <= curr[0] <= 22:
                    in_dp_cvt = curr[1]
                elif curr[0] == 23:
                    out_dp_cvt1 = curr[1]
                elif curr[0] == 24:
                    out_dp_cvt2 = curr[1]
                    
            self.convert1 = ConvBNRelu(input_depth=in_dp_cvt,
                                   output_depth=out_dp_cvt1,
                                   kernel=1, stride=1,
                                   pad=0, no_bias=1, use_relu="relu", bn_type="bn")

            self.convert2 = ConvBNRelu(input_depth=in_dp_cvt,
                                       output_depth=out_dp_cvt2,
                                       kernel=1, stride=1,
                                       pad=0, no_bias=1, use_relu="relu", bn_type="bn")  
                
        self.module_list.append(self.convert1)  # i=23
        self.module_list.append(self.convert2)  # i=24

        for i, op_name in enumerate(self.head26):
            self.module_list.append(operations(self.head26)[op_name[0]](*layer_parameters_head26[i]))  # i=25~29

        for i, op_name in enumerate(self.head13):
            self.module_list.append(operations(self.head13)[op_name[0]](*layer_parameters_head13[i]))  # i=30~34
        
        if prune_para is None:
            self.head_converter26 = ConvBNRelu(input_depth=256,
                                               output_depth=num_anchors * (num_cls + 5),
                                               kernel=1, stride=1,
                                               pad=0, no_bias=1, use_relu="relu", bn_type="bn")

            self.head_converter13 = ConvBNRelu(input_depth=256,
                                               output_depth=num_anchors * (num_cls + 5),
                                               kernel=1, stride=1,
                                               pad=0, no_bias=1, use_relu="relu", bn_type="bn")
        else:
            for i in range(len(prune_para)):
                curr = prune_para[i]
                if 25 <= curr[0] <= 29:
                    in_dp_26 = curr[1]
                elif 30 <= curr[0] <= 34:
                    in_dp_13 = curr[1]
            
            self.head_converter26 = ConvBNRelu(input_depth=in_dp_26,
                                               output_depth=num_anchors * (num_cls + 5),
                                               kernel=1, stride=1,
                                               pad=0, no_bias=1, use_relu="relu", bn_type="bn")

            self.head_converter13 = ConvBNRelu(input_depth=in_dp_13,
                                               output_depth=num_anchors * (num_cls + 5),
                                               kernel=1, stride=1,
                                               pad=0, no_bias=1, use_relu="relu", bn_type="bn")

        self.module_list.append(self.head_converter26)  # i=35
        self.module_list.append(self.head_converter13)  # i=36

        self.yololayer_26 = yolo_layer26
        self.yololayer_13 = yolo_layer13

        self.module_list.append(self.yololayer_26) # i=37
        self.module_list.append(self.yololayer_13) # i=38

        self.yolo_layers = [self.yololayer_26, self.yololayer_13]

    def forward(self, x):
        img_size = x.size(2)

        # first layer
        x = self.module_list[0](x)

        # backbones
        for i in range(len(self.backbones)): # i=1~11
            x = self.module_list[i+1](x)
            if i == 7: # FPN26
                fpn26 = x
        fpn13 = x
        id_preprocess = 1 + len(self.backbones)
        fpn13 = self.module_list[id_preprocess](fpn13) # i=12

        # FPN
        # hid_layer26 = fpn26 + upsample(fpn13)
        # hid_layer13 = fpn13 + downsample(fpn26)
        id_upsample = id_preprocess + 1
        id_break = 1 + id_upsample  # i=14
        id_fpn = id_break + 1 # i=15

        fpn26 = self.module_list[id_break](fpn26)

        fpn_mixop0 = self.module_list[id_fpn](fpn26) # fpn0
        fpn_mixop2 = self.module_list[id_fpn+2](fpn13) # fpn2
        hid_layer26 = fpn_mixop0 + self.module_list[id_upsample](fpn_mixop2)

        fpn_mixop1 = self.module_list[id_fpn+1](fpn26) # fpn1
        fpn_mixop3 = self.module_list[id_fpn+3](fpn13) # fpn3
        hid_layer13 = fpn_mixop1 + fpn_mixop3

        # fpn26 = hid_layer26 + upsample(hid_layer13)
        # fpn13 = hid_layer13 + downsample(hid_layer26)
        fpn_mixop4 = self.module_list[id_fpn+4](hid_layer26) # fpn4
        fpn_mixop6 = self.module_list[id_fpn+6](hid_layer13) # fpn6
        fpn26 = fpn_mixop4 + self.module_list[id_upsample](fpn_mixop6)

        fpn_mixop5 = self.module_list[id_fpn+5](hid_layer26) # fpn5
        fpn_mixop7 = self.module_list[id_fpn+7](hid_layer13) # fpn7
        fpn13 = fpn_mixop5 + fpn_mixop7

        id_convert1 = id_fpn + len(self.fpn) # i=23
        id_convert2 = id_convert1 + 1  # i=24

        # convert ch: 512 -> 256
        fpn26 = self.module_list[id_convert1](fpn26)
        fpn13 = self.module_list[id_convert2](fpn13)

        # head
        id_head26 = id_convert2 + 1
        id_head13 = id_head26 + len(self.head26)
        for i in range(len(self.head26)):
            fpn26 = self.module_list[id_head26+i](fpn26) # i=25~29
            fpn13 = self.module_list[id_head13+i](fpn13) # i=30~34

        # yolo_layer
        id_head_converter26 = id_head13 + len(self.head13) # i=35
        id_head_converter13 = id_head_converter26 + 1  # i=36

        yololayer_26 = self.module_list[id_head_converter26](fpn26)
        yololayer_13 = self.module_list[id_head_converter13](fpn13)

        id_yololayer26 = id_head_converter13 + 1 # i=37
        id_yololayer13 = id_yololayer26 + 1 # i=38

        output26 = self.module_list[id_yololayer26](yololayer_26, img_size)
        output13 = self.module_list[id_yololayer13](yololayer_13, img_size)
        yolo_output = [output26, output13]
        return yolo_output if self.training else torch.cat(yolo_output, 1)


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()

    def forward(self, outs, targets, model):
        ce, loss_components = compute_loss(outs, targets, model)
        return ce, loss_components


if __name__ == '__main__':
    #from supernet_functions.lookup_table_builder import (LookUpTable, SEARCH_SPACE_BACKBONE, SEARCH_SPACE_HEAD, SEARCH_SPACE_FPN,
    #                                                     YOLO_LAYER_26, YOLO_LAYER_13)
    #arch = 'test_net'
    #arch_def = Test_model_arch[arch]
    #print(arch_def['block_op_type_backbone'])
    #backbones = arch_def['block_op_type_backbone']

    #operations = {op_name[0]: PRIMITIVES[op_name[0]] for op_name in backbones}
   #print(operations)

    #layer_parameters, _ = LookUpTable._generate_layers_parameters(search_space_backbone=SEARCH_SPACE_BACKBONE)
    #print(*layer_parameters)
    #operations = lambda part: {op_name[0]: PRIMITIVES[op_name[0]] for op_name in part}

    #operations = {op_name[0]: PRIMITIVES[op_name[0]] for op_name in backbones}
    #print([operations(backbones)[op_name[0]](*layer_parameters[i]) for i, op_name in enumerate(backbones)])  # Modulelist for backbone
    # op = ConvBNRelu(
    #     3,
    #     10,
    #     kernel=1,
    #     stride=1,
    #     pad=1,
    #     no_bias=1,
    #     use_relu='relu',
    #     bn_type='bn'
    # )
    # for i, m in enumerate(op.modules()):
    #     print(i)
    #     print(m)
    # input = torch.randn(1,3,10,10)
    # out = op(input)
    # print(out)
    # convqt = QuanConv2d(Conv2d(3,10,1,1), quan_w_fn=quantizer(CONFIG_SUPERNET['quan']['weight']))
    # print("normal conv: ", Conv2d(3,10,1,1))
    # print("QT-conv: ", convqt)
    # #print(convqt(input))
    # act = nn.Hardswish(inplace=True)
    # actqt = QuanAct(act, quan_a_fn=quantizer(CONFIG_SUPERNET['quan']['act']))
    # input2 = torch.randn(5,5)
    # print(input2)
    # print(actqt(input2))
    input = torch.randn(1, 3, 320, 320)
    for k in PRIMITIVES:
        op = PRIMITIVES[k](3, 16, 6, 1)
        #print(op)
        torch.onnx.export(op,  # model being run
                          input,  # model input (or a tuple for multiple inputs)
                          "./onnx/{}.onnx".format(k),  # where to save the model (can be a file or file-like object)
                          export_params=True,  # store the trained parameter weights inside the model file
                          # the ONNX version to export the model to
                          opset_version=7,
                          verbose=True,
                          input_names=['input'],  # the model's input names
                          output_names=['output'],  # the model's output names
                          dynamic_axes=None,
                         )
