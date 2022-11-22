import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import math
import numpy as np
from queue import Queue
from collections import OrderedDict
from itertools import chain
from building_blocks.builder import ConvBNRelu, Upsample
from general_functions.loss import compute_loss
from supernet_functions.config_for_supernet import CONFIG_SUPERNET


def detach_variable(inputs):
    if isinstance(inputs, tuple):
        return tuple([detach_variable(x) for x in inputs])
    else:
        x = inputs.detach()
        x.requires_grad = inputs.requires_grad
        return x


def delta_ij(i, j):
    if i == j:
        return 1
    else:
        return 0


class MixedOperation(nn.Module):
    MODE = None  # full, two, None, full_v2

    def __init__(self, layer_parameters, proposed_operations, latency):
        super(MixedOperation, self).__init__()
        ops_names = [op_name for op_name in proposed_operations]
        
        self.ops = nn.ModuleList([proposed_operations[op_name](*layer_parameters)
                                  for op_name in ops_names])

        self.AP_path_alpha = Parameter(torch.Tensor(self.n_choices))  # architecture parameters
        self.AP_path_wb = Parameter(torch.Tensor(self.n_choices))  # binary gates

        self.active_index = [0]
        self.inactive_index = None

        self.log_prob = None
        self.current_prob_over_ops = None

        self.latency = [latency[op_name] for op_name in ops_names]
        #self.thetas = nn.Parameter(torch.Tensor([1.0 / len(ops_names) for i in range(len(ops_names))]))

    @property
    def n_choices(self):
        return len(self.ops)

    @property
    def probs_over_ops(self):
        probs = F.softmax(self.AP_path_alpha, dim=0)  # softmax to probability
        return probs

    @property
    def chosen_index(self):
        probs = self.probs_over_ops.data.cpu().numpy()
        index = int(np.argmax(probs))
        return index, probs[index]

    @property
    def chosen_op(self):
        index, _ = self.chosen_index
        return self.candidate_ops[index]

    @property
    def active_op(self):
        """ assume only one path is active """
        return self.ops[self.active_index[0]]

    @property
    def random_op(self):
        index = np.random.choice([_i for _i in range(self.n_choices)], 1)[0]
        return self.candidate_ops[index]

    def set_chosen_op_active(self):
        chosen_idx, _ = self.chosen_index
        self.active_index = [chosen_idx]
        self.inactive_index = [_i for _i in range(0, chosen_idx)] + \
                              [_i for _i in range(chosen_idx + 1, self.n_choices)]

    def forward(self, x, latency_to_accumulate):
        if MixedOperation.MODE == 'full' or MixedOperation.MODE == 'two':
            output = 0
            for _i in self.active_index: # [0]
                oi = self.ops[_i](x)
                latency = self.latency[_i]
                output = output + self.AP_path_wb[_i] * oi
                latency_to_accumulate = latency_to_accumulate + self.AP_path_alpha[_i] * latency
            for _i in self.inactive_index:
                oi = self.ops[_i](x)
                latency = self.latency[_i]
                output = output + self.AP_path_wb[_i] * oi.detach()
                latency_to_accumulate = latency_to_accumulate + self.AP_path_alpha[_i] * latency.detach()
        else:
            output = self.active_op(x)
            for _i in range(self.n_choices):
                latency_to_accumulate = latency_to_accumulate + self.latency[_i] * self.AP_path_alpha[_i]

        return output, latency_to_accumulate

    def binarize(self):
        """ prepare: active_index, inactive_index, AP_path_wb, log_prob (optional), current_prob_over_ops (optional) """
        self.log_prob = None
        # reset binary gates
        self.AP_path_wb.data.zero_() # [0,0,0,0,0]
        # binarize according to probs
        probs = self.probs_over_ops # 归一化后的结构参数
        if MixedOperation.MODE == 'two':
            # sample two ops according to `probs`
            sample_op = torch.multinomial(probs.data, 2, replacement=False) # 根据probs大小采样，采出2个，不放回,返回下标 [2, 3]
            # 将采出的2个结构参数归一化 [0.4, 0.6]
            probs_slice = F.softmax(torch.stack([
                self.AP_path_alpha[idx] for idx in sample_op
            ]), dim=0)
            self.current_prob_over_ops = torch.zeros_like(probs)
            for i, idx in enumerate(sample_op):
                self.current_prob_over_ops[idx] = probs_slice[i] #将之前归一化后的2个结构参数存进，其余为0 [0, 0, 0.4, 0.6, 0]
            # chose one to be active and the other to be inactive according to probs_slice
            c = torch.multinomial(probs_slice.data, 1)[0]  # 0 or 1
            active_op = sample_op[c].item() # 激活op对应下标 3
            inactive_op = sample_op[1 - c].item()
            self.active_index = [active_op] # [3]
            self.inactive_index = [inactive_op] # [2]
            # set binary gate
            self.AP_path_wb.data[active_op] = 1.0 # [0,0,0,1,0]
        else:
            sample = torch.multinomial(probs.data, 1)[0].item() # 只采1个: 2
            self.active_index = [sample] # [2]
            self.inactive_index = [_i for _i in range(0, sample)] + \
                                  [_i for _i in range(sample + 1, self.n_choices)] # [0,1,3,4]
            self.log_prob = torch.log(probs[sample])
            self.current_prob_over_ops = probs
            # set binary gate
            self.AP_path_wb.data[sample] = 1.0 # [0,0,1,0,0]
        # avoid over-regularization
        for _i in range(self.n_choices):
            for name, param in self.ops[_i].named_parameters():
                param.grad = None

    def set_arch_param_grad(self):
        binary_grads = self.AP_path_wb.grad.data
        if self.active_op.is_zero_layer():
            self.AP_path_alpha.grad = None
            return
        if self.AP_path_alpha.grad is None:
            self.AP_path_alpha.grad = torch.zeros_like(self.AP_path_alpha.data)
        if MixedOperation.MODE == 'two':
            involved_idx = self.active_index + self.inactive_index # [3, 2]
            probs_slice = F.softmax(torch.stack([                  # [0.6, 0.4]
                self.AP_path_alpha[idx] for idx in involved_idx
            ]), dim=0).data
            for i in range(2):
                for j in range(2):
                    origin_i = involved_idx[i]
                    origin_j = involved_idx[j]
                    # dL / da
                    self.AP_path_alpha.grad.data[origin_i] += \
                        binary_grads[origin_j] * probs_slice[j] * (delta_ij(i, j) - probs_slice[i])
                    #       grad                       pj                 delta             pi
                    # grad[3] += bi_grad[3]*0.6*(1-0.6) for i=j=0
                    # grad[3] += bi_grad[2]*0.4*(0-0.6) for i=0, j=1
                    # grad[2] += bi_grad[3]*0.6*(0-0.4) for i=1, j=0
                    # grad[2] += bi_grad[2]*0.4*(1-0.4) for i=j=1

            for _i, idx in enumerate(self.active_index): # [3] -> [(0, alpha[3])]
                self.active_index[_i] = (idx, self.AP_path_alpha.data[idx].item())
            for _i, idx in enumerate(self.inactive_index): # [2] -> [(0, alpha[2])]
                self.inactive_index[_i] = (idx, self.AP_path_alpha.data[idx].item())
        else: # full
            probs = self.probs_over_ops.data
            for i in range(self.n_choices):
                for j in range(self.n_choices):
                    self.AP_path_alpha.grad.data[i] += binary_grads[j] * probs[j] * (delta_ij(i, j) - probs[i])
        return

    def rescale_updated_arch_param(self):
        if not isinstance(self.active_index[0], tuple):
            assert self.active_op.is_zero_layer()
            return
        involved_idx = [idx for idx, _ in (self.active_index + self.inactive_index)]
        old_alphas = [alpha for _, alpha in (self.active_index + self.inactive_index)]
        new_alphas = [self.AP_path_alpha.data[idx] for idx in involved_idx]

        offset = math.log(
            sum([math.exp(alpha) for alpha in new_alphas]) / sum([math.exp(alpha) for alpha in old_alphas])
        )

        for idx in involved_idx:
            self.AP_path_alpha.data[idx] -= offset


class YOLOLayer(nn.Module):
    """Detection layer
    if training: return x[bs,3,13,13,cls+5]
           else: return x[bs,3*13*13,cls+5]
    """

    def __init__(self, anchors, num_classes):
        super(YOLOLayer, self).__init__()
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.no = num_classes + 5  # number of outputs per anchor
        self.grid = torch.zeros(1)  # TODO

        anchors = torch.tensor(list(chain(*anchors))).float().view(-1, 2)
        self.register_buffer('anchors', anchors)
        self.register_buffer(
            'anchor_grid', anchors.clone().view(1, -1, 1, 1, 2))
        self.stride = None

    def forward(self, x, img_size):
        stride = img_size // x.size(2) #416//13=32 input_img与featuremap的比例
        self.stride = stride
        bs, _, ny, nx = x.shape  # x(bs,75,13,13) to x(bs,3,25,13,13) to x(bs,3,13,13,25)
        x = x.view(bs, self.num_anchors, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

        if not self.training:  # inference, 非训练时执行,生成回归参数用于画bbox,训练时只用求loss,在loss.py中生成
            if self.grid.shape[2:4] != x.shape[2:4]:
                self.grid = self._make_grid(nx, ny).to(x.device) #grid_cell左上角？ [1,1,13,13,2]

            x[..., 0:2] = (x[..., 0:2].sigmoid() + self.grid) * stride  # featuremap上的xy放缩至原图
            x[..., 2:4] = torch.exp(x[..., 2:4]) * self.anchor_grid # wa*exp(tw), ha*exp(th)
            x[..., 4:] = x[..., 4:].sigmoid() #to
            x = x.view(bs, -1, self.no) #[bs,3*13*13,25]

        return x

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


class Stochastic_SuperNet(nn.Module):
    def __init__(self, lookup_table):
        super(Stochastic_SuperNet, self).__init__()
        self._redundant_modules = None
        self._unused_modules = None

        self.backbone_len = lookup_table.cnt_layers
        self.fpn_len = lookup_table.fpn_edges
        self.head_len = lookup_table.head_layers

        self.module_list = nn.ModuleList()

        # self.first identical to 'add_first' in the building_blocks/builder.py
        self.first = ConvBNRelu(input_depth=3, output_depth=16, kernel=3, stride=2,
                                pad=3 // 2, no_bias=1, use_relu="relu", bn_type="bn")
        self.module_list.append(self.first)  # i=0

        # TBS-backbone
        for layer_id in range(self.backbone_len): # i=1~11
            self.module_list.append(MixedOperation(
                                       lookup_table.layers_parameters[layer_id],
                                       lookup_table.lookup_table_operations,
                                       lookup_table.lookup_table_latency[layer_id]))

        self.preprocess = ConvBNRelu(input_depth=1024,
                                     output_depth=512,
                                     kernel=1, stride=1,
                                     pad=0, no_bias=1, use_relu="relu", bn_type="bn")

        self.break_conv1x1 = ConvBNRelu(input_depth=512,
                                        output_depth=512,
                                        kernel=1, stride=1,
                                        pad=0, no_bias=1, use_relu="relu", bn_type="bn")

        self.module_list.append(self.preprocess)  # i=12

        self.upsample = Upsample(scale_factor=2, mode="nearest")
        self.module_list.append(self.upsample)  # i=13
        self.module_list.append(self.break_conv1x1) # i=14

        # TBS-FPN
        for edge_id in range(self.fpn_len): # i=15~22
            self.module_list.append(MixedOperation(
                                         lookup_table.layers_parameters_fpn[edge_id],
                                         lookup_table.lookup_table_operations_fpn,
                                         lookup_table.lookup_table_latency_fpn[edge_id]))

        self.convert1 = ConvBNRelu(input_depth=512,
                                  output_depth=256,
                                  kernel=1, stride=1,
                                  pad=0, no_bias=1, use_relu="relu", bn_type="bn")

        self.convert2 = ConvBNRelu(input_depth=512,
                                  output_depth=256,
                                  kernel=1, stride=1,
                                  pad=0, no_bias=1, use_relu="relu", bn_type="bn")

        self.module_list.append(self.convert1) # i=23
        self.module_list.append(self.convert2) # i=24

        # TBS-head
        for layer_id in range(self.head_len): # i=25~29 for head26
            self.module_list.append(MixedOperation(
                                       lookup_table.layers_parameters_head[layer_id],
                                       lookup_table.lookup_table_operations_head,
                                       lookup_table.lookup_table_latency_head[layer_id]))

        for layer_id in range(self.head_len): # i=30~34 for head13
            self.module_list.append(MixedOperation(
                                       lookup_table.layers_parameters_head[layer_id],
                                       lookup_table.lookup_table_operations_head,
                                       lookup_table.lookup_table_latency_head[layer_id]))

        # channel converter
        self.head_converter26 = ConvBNRelu(input_depth=256,
                                  output_depth=lookup_table.num_anchors*(lookup_table.num_cls+5),
                                  kernel=1, stride=1,
                                  pad=0, no_bias=1, use_relu="relu", bn_type="bn")

        self.head_converter13 = ConvBNRelu(input_depth=256,
                                  output_depth=lookup_table.num_anchors * (lookup_table.num_cls + 5),
                                  kernel=1, stride=1,
                                  pad=0, no_bias=1, use_relu="relu", bn_type="bn")

        self.module_list.append(self.head_converter26)  # i=35
        self.module_list.append(self.head_converter13)  # i=36

        # yolo_layers
        self.yololayer_26 = YOLOLayer(lookup_table.anchors_26, lookup_table.num_cls)
        self.yololayer_13 = YOLOLayer(lookup_table.anchors_13, lookup_table.num_cls)
        self.module_list.append(self.yololayer_26)  # i=37
        self.module_list.append(self.yololayer_13)  # i=38

        self.yolo_layers = [self.yololayer_26, self.yololayer_13]

    def forward(self, x, latency_to_accumulate):
        img_size = x.size(2)

        y = self.module_list[0](x)

        for i in range(self.backbone_len): # i=1~11
            y, latency_to_accumulate = self.module_list[i+1](y, latency_to_accumulate)
            if i == 7: # FPN26
                fpn26 = y
        fpn13 = y # FPN13

        # fpn13 ch: 1024 -> 512
        id_preprocess = 1 + self.backbone_len

        fpn13 = self.module_list[id_preprocess](fpn13) # i=12

        # FPN
        # hid_layer26 = fpn26 + upsample(fpn13)
        # hid_layer13 = fpn13 + downsample(fpn26)
        id_upsample = id_preprocess + 1
        id_break = 1 + id_upsample # i=14
        id_fpn = id_break + 1 # i=15

        fpn26 = self.module_list[id_break](fpn26)

        fpn_mixop0, latency_to_accumulate = self.module_list[id_fpn](fpn26, latency_to_accumulate)
        fpn_mixop2, latency_to_accumulate = self.module_list[id_fpn+2](fpn13, latency_to_accumulate)
        hid_layer26 = fpn_mixop0 + self.module_list[id_upsample](fpn_mixop2)

        fpn_mixop1, latency_to_accumulate = self.module_list[id_fpn+1](fpn26, latency_to_accumulate)
        fpn_mixop3, latency_to_accumulate = self.module_list[id_fpn+3](fpn13, latency_to_accumulate)
        hid_layer13 = fpn_mixop1 + fpn_mixop3

        # fpn26 = hid_layer26 + upsample(hid_layer13)
        # fpn13 = hid_layer13 + downsample(hid_layer26)
        fpn_mixop4, latency_to_accumulate = self.module_list[id_fpn+4](hid_layer26, latency_to_accumulate)
        fpn_mixop6, latency_to_accumulate = self.module_list[id_fpn+6](hid_layer13, latency_to_accumulate)
        fpn26 = fpn_mixop4 + self.module_list[id_upsample](fpn_mixop6)

        fpn_mixop5, latency_to_accumulate = self.module_list[id_fpn+5](hid_layer26, latency_to_accumulate)
        fpn_mixop7, latency_to_accumulate = self.module_list[id_fpn+7](hid_layer13, latency_to_accumulate)
        fpn13 = fpn_mixop5 + fpn_mixop7

        # convert ch: 512 -> 256
        id_convert1 = id_fpn + self.fpn_len  # i=23
        id_convert2 = id_convert1 + 1 # i=24
        fpn26 = self.module_list[id_convert1](fpn26)
        fpn13 = self.module_list[id_convert2](fpn13)

        # head
        id_head26 = id_convert2 + 1 # i=25
        id_head13 = id_head26 + self.head_len

        for i in range(self.head_len):
            fpn26, latency_to_accumulate = self.module_list[id_head26+i](fpn26, latency_to_accumulate) # i=25~29
            fpn13, latency_to_accumulate = self.module_list[id_head13+i](fpn13, latency_to_accumulate) # i=30~34

        # for mixed_op_26, mixed_op_13 in zip(self.head26, self.head13):
        #     fpn26, latency_to_accumulate = mixed_op_26(fpn26, latency_to_accumulate)
        #     fpn13, latency_to_accumulate = mixed_op_13(fpn13, latency_to_accumulate)

        id_head_converter26 = id_head13 + self.head_len  # i=35
        id_head_converter13 = id_head_converter26 + 1  # i=36

        yololayer_26 = self.module_list[id_head_converter26](fpn26)
        yololayer_13 = self.module_list[id_head_converter13](fpn13)

        id_yololayer26 = id_head_converter13 + 1  # i=37
        id_yololayer13 = id_yololayer26 + 1  # i=38

        output26 = self.module_list[id_yololayer26](yololayer_26, img_size)
        output13 = self.module_list[id_yololayer13](yololayer_13, img_size)
        yolo_output = [output26, output13]
        return yolo_output if self.training else torch.cat(yolo_output, 1), latency_to_accumulate

    """ weight parameters, arch_parameters & binary gates """

    def architecture_parameters(self):
        for name, param in self.named_parameters():
            if 'AP_path_alpha' in name:
                yield param

    def binary_gates(self):
        for name, param in self.named_parameters():
            if 'AP_path_wb' in name:
                yield param

    def weight_parameters(self):
        for name, param in self.named_parameters():
            if 'AP_path_alpha' not in name and 'AP_path_wb' not in name:
                yield param

    """ architecture parameters related methods """

    @property
    def redundant_modules(self):
        if self._redundant_modules is None:
            module_list = []
            for m in self.modules():
                if m.__str__().startswith('MixedOperation'):
                    module_list.append(m)
            self._redundant_modules = module_list
        return self._redundant_modules

    def init_arch_params(self, init_type='normal', init_ratio=1e-3):
        for param in self.architecture_parameters():
            if init_type == 'normal':
                param.data.normal_(0, init_ratio)
            elif init_type == 'uniform':
                param.data.uniform_(-init_ratio, init_ratio)
            else:
                raise NotImplementedError

    def reset_binary_gates(self):
        """
        对所有mixededge进行二值化
        """
        for m in self.redundant_modules:
            try:
                m.binarize()
            except AttributeError:
                print(type(m), ' do not support binarize')

    def set_arch_param_grad(self):
        for m in self.redundant_modules:
            try:
                m.set_arch_param_grad()
            except AttributeError:
                print(type(m), ' do not support `set_arch_param_grad()`')

    def rescale_updated_arch_param(self):
        for m in self.redundant_modules:
            try:
                m.rescale_updated_arch_param()
            except AttributeError:
                print(type(m), ' do not support `rescale_updated_arch_param()`')

    """ training related methods """

    def unused_modules_off(self):
        """
        关闭未选中的Module,即将未选中的candidate置为None
        """
        self._unused_modules = []
        for m in self.redundant_modules:
            unused = {}
            if MixedOperation.MODE in ['full', 'two', 'full_v2']:
                involved_index = m.active_index + m.inactive_index # [2] + [3] = [2, 3]
            else:
                involved_index = m.active_index # [2]
            for i in range(m.n_choices):
                if i not in involved_index:
                    unused[i] = m.candidate_ops[i]
                    m.candidate_ops[i] = None
            self._unused_modules.append(unused)

    def unused_modules_back(self):
        """
        再次将不使用的modules放回
        """
        if self._unused_modules is None:
            return
        for m, unused in zip(self.redundant_modules, self._unused_modules):
            for i in unused:
                m.candidate_ops[i] = unused[i]
        self._unused_modules = None

    def set_chosen_op_active(self):
        for m in self.redundant_modules:
            try:
                m.set_chosen_op_active()
            except AttributeError:
                print(type(m), ' do not support `set_chosen_op_active()`')


class SupernetLoss(nn.Module):
    def __init__(self):
        super(SupernetLoss, self).__init__()
        self.alpha = CONFIG_SUPERNET['loss']['alpha']
        self.beta = CONFIG_SUPERNET['loss']['beta']
    
    def forward(self, outs, targets, latency, model):
        
        #ce = self.weight_criterion(outs, targets)
        ce, loss_components = compute_loss(outs, targets, model)
        lat = torch.log(latency ** self.beta)
        loss = self.alpha * ce * lat
        return loss, ce, lat, loss_components #.unsqueeze(0)

if __name__=="__main__":
    from lookup_table_builder import LookUpTable
    from torch.autograd import Variable
    from general_functions.utils import weights_init
    #device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    lookup_table = LookUpTable(calculate_latency=CONFIG_SUPERNET['lookup_table']['create_from_scratch'])
    #print(lookup_table.lookup_table_operations)
    #print(lookup_table.layers_parameters)
    #print(lookup_table.lookup_table_latency)
    #print(lookup_table.num_cls)
    #print(lookup_table.num_anchors)
    #print(lookup_table.head_layers)
    # mixop = MixedOperation(lookup_table.layers_parameters[0],
    #                        lookup_table.lookup_table_operations,
    #                        lookup_table.lookup_table_latency[0])
    model = Stochastic_SuperNet(lookup_table)
    model = model.apply(weights_init)
    model.to(device)
    model.train()

    latency_to_accumulate = Variable(torch.Tensor([[0.0]]), requires_grad=True)

    #temperature = CONFIG_SUPERNET['train_settings']['init_temperature']

    loss = SupernetLoss()

    input = torch.randn(1, 3, 640, 640).to(device)
    target = torch.randn(2, 6).to(device)

    count = 0

    # if model.module_list[1].__str__().startswith('MixedOperation'):
    #     print(model.module_list[1].chosen_op)

    #print(model.module_list[1])
    # for m in model.modules():
    #     if m.__str__().startswith('MixedOperation'):
    #         for n in m.modules():
    #             if isinstance(n, ConvBNRelu):
    #                 print(n[1])

    #print(count)

    #out, latency_to_accumulate = model(input, latency_to_accumulate)

    #Loss, ce, lat, loss_components = loss(out, target, latency_to_accumulate, model)

    #print("Loss: ", Loss)
    #print("ce: ", ce)
    #print("lat: ", lat)

    #print(len(out))
    #print(out[0].shape)
    #print(out[1].shape)
    #print(model)

    for id, module in enumerate(model.module_list):
        if 1 <= id <= 34:
            for m in module.modules():
                if isinstance(m, ConvBNRelu):
                    print(id)

    ###############################################################################################
    #visualize
    # x = (input, temperature, latency_to_accumulate)
    # torch.onnx.export(model,  # model being run
    #                   x,  # model input (or a tuple for multiple inputs)
    #                   "./model.onnx",  # where to save the model (can be a file or file-like object)
    #                   export_params=True,  # store the trained parameter weights inside the model file
    #                   # the ONNX version to export the model to
    #                   opset_version=7,
    #                   do_constant_folding=True,  # whether to execute constant folding for optimization
    #                   input_names=['input'],  # the model's input names
    #                   output_names=['output'],  # the model's output names
    #                  )
    ########################################################
    ###LOSS
    #criterion = SupernetLoss()




