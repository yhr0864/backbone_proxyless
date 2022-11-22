import torch
from terminaltables import AsciiTable
from copy import deepcopy
import numpy as np
import torch.nn.functional as F
from building_blocks.builder import ConvBNRelu


def get_sr_flag(epoch, sr):
    # return epoch >= 5 and sr
    return sr


def gather_bn_weights(model):
    size_list = []
    bn_modules = []
    for idx, module in enumerate(model.module_list):
        if 1 <= idx <= 34:
            for m in module.modules():
                if isinstance(m, ConvBNRelu):
                    bn_module = m[1]
                    bn_modules.append(bn_module)
                    size_list.append(bn_module.weight.data.shape[0])

    #size_list = [module_list[idx][1].weight.data.shape[0] for idx in prune_idx] # [32, 64, 128,...]
    # 将所有BN的所有channel展开放一起
    bn_weights = torch.zeros(sum(size_list))
    index = 0
    for idx, size in enumerate(size_list):
        bn_weights[index:(index + size)] = bn_modules[idx].weight.data.abs().clone()
        index += size

    return bn_weights


def pack_filter_para(start_id, end_id, num_filters):
    para = [[] for i in range(end_id-start_id+1)]
    for i in num_filters:
        if start_id <= i[0] <= end_id:
            para[i[0]-start_id].append(i[1])
    for i in para:
        if len(i) == 0:
            i.append(None)
    return para


def generate_searchspace(para, first_input):
    channel_size = []
    input_shape = [first_input]
    prune = []
    for i in para:
        channel_size.append(i[-1])
        if len(i) > 2:
            prune.append((i[0], i[1]))
        else:
            prune.append(None)
    
    if channel_size[0] is None:
        channel_size[0] = first_input

    for i in range(1, len(channel_size)):
        prev = channel_size[i-1]
        current = channel_size[i]
        if current is None:
            channel_size[i] = prev

    for i, j in enumerate(channel_size):
        if i != len(channel_size)-1:
            input_shape.append(j)
    return input_shape, channel_size, prune


class BNOptimizer:
    @staticmethod
    def updateBN(sr_flag, model, s):
        if sr_flag:
            for idx, module in enumerate(model.module_list):
                if 1 <= idx <= 34:
                    for m in module.modules():
                        if isinstance(m, ConvBNRelu):
                            bn_module = m[1]  # 1对应BN
                            bn_module.weight.grad.data.add_(s * torch.sign(bn_module.weight.data))  # L1


def BN_preprocess(model):

    fusion_modules = [[], [], [], []]  # [[(false, pwl),(tru, pwl),...()], [head26], [head13]]
    fusion_modules_in_block = [[], [], [], []] # [[(1, pw, dw),(2, pw, dw),...], ]
    
    fusion_modules[0].append((None, model.module_list[0]))  # backbone第一层不参与剪枝，仅用于占位

    fusion_modules[1].append((None, model.module_list[12])) # fpn与backbone的2个连接处
    fusion_modules[1].append((None, model.module_list[14]))

    fusion_modules[2].append((None, model.module_list[23]))  # head与fpn的2个连接处
    fusion_modules[3].append((None, model.module_list[24]))

    for idx, module in enumerate(model.module_list):
        if 1 <= idx <= 11:  # backbone
            if module.__str__().startswith('IRFBlock'):
                current_module = [m for m in module.children()]
                fusion_modules[0].append((module.use_res_connect, current_module[-2]))
                fusion_modules_in_block[0].append((idx, current_module[-4], current_module[-3]))
            else:
                if module.moduleList:
                    current_module = module.moduleList[0]
                    fusion_modules[0].append((None, current_module))

        elif 15 <= idx <= 22: # fpn
            if module.__str__().startswith('IRFBlock'):
                current_module = [m for m in module.children()]
                fusion_modules[1].append((module.use_res_connect, current_module[-2]))
                fusion_modules_in_block[1].append((idx, current_module[-4], current_module[-3]))
            else:
                if not module.__str__().startswith('Zero') and module.moduleList:
                    current_module = module.moduleList[0]
                    fusion_modules[1].append((None, current_module))

        elif 25 <= idx <= 29:  # head26
            if module.__str__().startswith('IRFBlock'):
                current_module = [m for m in module.children()]
                fusion_modules[2].append((module.use_res_connect, current_module[-2]))
                fusion_modules_in_block[2].append((idx, current_module[-4], current_module[-3]))
            else:
                if module.moduleList:
                    current_module = module.moduleList[0]
                    fusion_modules[2].append((None, current_module))
        elif 30 <= idx <= 34:  # head13
            if module.__str__().startswith('IRFBlock'):
                current_module = [m for m in module.children()]
                fusion_modules[3].append((module.use_res_connect, current_module[-2]))
                fusion_modules_in_block[3].append((idx, current_module[-4], current_module[-3]))
            else:
                if module.moduleList:
                    current_module = module.moduleList[0]
                    fusion_modules[3].append((None, current_module))

    def sychronBN(*bn):
        sum = torch.zeros_like(bn[0].weight.data)
        count = 0
        for bni in bn:
            sum += bni.weight.data.abs()
            count += 1
        for bni in bn:
            bni.weight.data = sum / count

    # 同步BN前
    # print(fusion_modules[0])
    # print(fusion_modules[0][5][1][1].weight.data)
    # print(fusion_modules[0][6][1][1].weight.data)
    for i in range(len(fusion_modules[0]) - 1, 0, -1):  # backbone 倒叙判断
        current = fusion_modules[0][i]  # 第i个元组
        prev = fusion_modules[0][i - 1]
        if current[0]:
            if prev[0]:  # 如果前一个也是res则三个同步
                sychronBN(current[1][1], prev[1][1], fusion_modules[0][i - 2][1])
            else:
                sychronBN(current[1][1], prev[1][1])
    # 同步BN后
    # print(fusion_modules[0][5][1][1].weight.data)
    # print(fusion_modules[0][6][1][1].weight.data)

    # 对于fpn直接全部同步
    sychronBN(*[m[1][1] for m in fusion_modules[1]])

    # 对于head直接全部同步
    sychronBN(*[m[1][1] for m in fusion_modules[2]])
    sychronBN(*[m[1][1] for m in fusion_modules[3]])
    
    # 对于IRFBlock前两层同步
    for modules in fusion_modules_in_block:
        for m in modules:
            sychronBN(m[1][1], m[2][1])


def obtain_quantiles(bn_weights, num_quantile=5):

    sorted_bn_weights, i = torch.sort(bn_weights)
    total = sorted_bn_weights.shape[0]
    quantiles = sorted_bn_weights.tolist()[-1::-total//num_quantile][::-1]
    print("\nBN weights quantile:")
    quantile_table = [
        [f'{i}/{num_quantile}' for i in range(1, num_quantile+1)],
        ["%.3f" % quantile for quantile in quantiles]
    ]
    print(AsciiTable(quantile_table).table)

    return quantiles


def obtain_bn_mask(bn_module, thre):

    thre = thre.cuda()
    mask = bn_module.weight.data.abs().ge(thre).float()

    return mask


def id_trace_back(idx, filters_mask):
    """
    找到上一层非skip,none的id
    """
    i = idx
    mask_id = [mask[0] for mask in filters_mask]
    while not i in mask_id:
        i -= 1
    return i


def get_input_mask(idx, filters_mask):
    """
    返回上一层的mask
    """

    if idx == 0:
        return [np.ones(3)]

    ### 特殊情况e.g. skip
    if 1 <= idx <= 12: # fpn之前
        i = id_trace_back(idx - 1, filters_mask)
        filters = [[mask[1]] for mask in filters_mask if mask[0] == i]
        return filters
    elif idx == 14 or idx == 35:
        i = id_trace_back(idx - 6, filters_mask)
        filters = [[mask[1]] for mask in filters_mask if mask[0] == i]
        return filters
    # fpn与convert
    elif 15 <= idx <= 24: # 上一层为id=14因为同步故id=14,12与所有fpn剪枝相同
        filters = [[mask[1]] for mask in filters_mask if mask[0] == 14]
        return filters
    elif idx == 25:
        filters = [[mask[1]] for mask in filters_mask if mask[0] == 23]
        return filters
    elif idx == 30:
        filters = [[mask[1]] for mask in filters_mask if mask[0] == 24]
        return filters
    elif 26 <= idx <= 29:
        i = id_trace_back(idx - 1, filters_mask)
        filters = [[mask[1]] for mask in filters_mask if mask[0] == i]
        return filters
    elif 31 <= idx <= 34:
        i = id_trace_back(idx - 1, filters_mask)
        filters = [[mask[1]] for mask in filters_mask if mask[0] == i]
        return filters
    elif idx == 36:
        i = id_trace_back(idx - 2, filters_mask)
        filters = [[mask[1]] for mask in filters_mask if mask[0] == i]
        return filters
          

def load_weights_from_loose_model(compact_model, loose_model, filters_mask):
    for idx, module in enumerate(loose_model.module_list):
        loose_bn_modules = []
        compact_bn_modules = []
        loose_conv_modules = []
        compact_conv_modules = []

        filters = [mask[1] for mask in filters_mask if mask[0] == idx]
        out_channel_idx = [np.argwhere(f)[:, 0].tolist() for f in filters] # [[pw],[dw],[pwl]]
        
        # if idx < 3:
        #     print(idx)
        #     # print(filters)
        #     print(out_channel_idx)
        
        if 1 <= idx <= 34:
            if idx == 13: # upsample layer
                continue
            input_mask = get_input_mask(idx, filters_mask) # [mask1, mask2, mask3]
            
            # print(input_mask)
            
            in_channel_idx = np.argwhere(input_mask[-1])[:, 0].tolist() # 确保只是最后一个的输出作为下层输入（pw输入）
            
            # print("in ch id")
            # print(in_channel_idx)
            
            for m in module.modules():
                if isinstance(m, ConvBNRelu):
                    loose_bn_modules.append(m[1])
                    loose_conv_modules.append(m[0])
            for m in compact_model.module_list[idx].modules():
                if isinstance(m, ConvBNRelu):
                    compact_bn_modules.append(m[1])
                    compact_conv_modules.append(m[0])
            
            # print("cmp bn mod", len(compact_bn_modules))
            # print(compact_bn_modules)
            # print("loose bn mod", len(loose_bn_modules))
            # print(loose_bn_modules)
            # print("out ch id", len(out_channel_idx))
            # print(out_channel_idx)
            
            for compact_bn, loose_bn, out_id in zip(compact_bn_modules, loose_bn_modules, out_channel_idx):
                compact_bn.weight.data = loose_bn.weight.data[out_id].clone()
                compact_bn.bias.data = loose_bn.bias.data[out_id].clone()
                compact_bn.running_mean.data = loose_bn.running_mean.data[out_id].clone()
                compact_bn.running_var.data = loose_bn.running_var.data[out_id].clone()

            # 即上一层最后一个和当前层前两个channel作为本层3个(pw,dw,pwl)的输入channel
            # if idx == 3:
            #     print(in_channel_idx) # []
            #     print("out ch id")
            #     print(out_channel_idx)
            #     print("the first 2 out ch id")
            #     print(out_channel_idx[:2]) # [[],[]]
            if len(out_channel_idx) > 2:
                in_conv_channel = [in_channel_idx, out_channel_idx[0], out_channel_idx[1]] # [[in_id],[pw],[dw]]
            else:
                in_conv_channel = [in_channel_idx, out_channel_idx]

            for compact_conv, loose_conv, input_id, out_conv_id in zip(compact_conv_modules,
                                                                       loose_conv_modules,
                                                                       in_conv_channel,
                                                                       out_channel_idx):
                # if idx == 3:
                #     print("in conv ch")
                #     print(in_conv_channel)
                #     print("cmp conv")
                #     print(compact_conv)
                #     print("los conv")
                #     print(loose_conv)
                #     print("in id")
                #     print(input_id)
                #     print("out id")
                #     print(out_conv_id)
                #     print("loose w", loose_conv.weight.data.shape)
                #     print("cmp w", compact_conv.weight.data.shape)
                if loose_conv.weight.data.shape[1] == 1: # 深度卷积输入ch维度为1
                    compact_conv.weight.data = loose_conv.weight.data[out_conv_id, :, :, :].clone()
                else:    
                    tmp = loose_conv.weight.data[:, input_id, :, :].clone() # [:, pw, :, :]
                    compact_conv.weight.data = tmp[out_conv_id, :, :, :].clone() # [dw, :, :, :]
          
        else: # 其余无需剪枝层则按原样导入数据（1和最后层）
            if isinstance(module, ConvBNRelu):
                compact = compact_model.module_list[idx]
                loose = module
                
                input_mask = get_input_mask(idx, filters_mask)  # [mask]
                
                in_channel_idx = np.argwhere(input_mask[-1])[:, 0].tolist() # 确保只是最后一个的输出作为下层输入

                compact_bn, loose_bn = compact[1], loose[1]
                compact_conv, loose_conv = compact[0], loose[0]

                compact_bn.weight.data = loose_bn.weight.data[out_channel_idx[0]].clone()
                compact_bn.bias.data = loose_bn.bias.data[out_channel_idx[0]].clone()
                compact_bn.running_mean.data = loose_bn.running_mean.data[out_channel_idx[0]].clone()
                compact_bn.running_var.data = loose_bn.running_var.data[out_channel_idx[0]].clone()

                # tmp = loose_conv.weight.data[:, in_channel_idx, :, :].clone()
                # compact_conv.weight.data = tmp[out_channel_idx, :, :, :].clone()
                compact_conv.weight.data = loose_conv.weight.data[:, in_channel_idx, :, :].clone()
                
              
