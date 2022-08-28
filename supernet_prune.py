from models import *
from utils.utils import *
import torch
import numpy as np
from copy import deepcopy
from supernet_functions.training_functions_supernet import TrainerSupernet
from terminaltables import AsciiTable
import time
from general_functions.prune_utils import *
from general_functions.utils import load
from supernet_functions.lookup_table_builder import LookUpTable
from supernet_functions.model_supernet import Stochastic_SuperNet
from supernet_functions.config_for_supernet import CONFIG_SUPERNET


#%%
#################################################
### 载入subnet(可将此部分代码全部合并到arch_main_file种)
#################################################

eval_model = TrainerSupernet._validate(model, loader, epoch, img_size)
obtain_num_parameters = lambda model:sum([param.nelement() for param in model.parameters()])

origin_model_metric = eval_model(model)
origin_nparameters = obtain_num_parameters(model)

#CBL_idx, Conv_idx, prune_idx= parse_module_defs(model.module_defs)
# 将所有能剪的层的channel的bn权重保存


class PrunedModel:
    @staticmethod
    def get_highest_thre(model):
        bn_weights = gather_bn_weights(model)

        sorted_bn = torch.sort(bn_weights)[0]

        # 避免剪掉所有channel的最高阈值(每个BN层的gamma的最大值的最小值即为阈值上限)
        highest_thre = []
        for id, module in enumerate(model.module_list):
            if 1 <= id <= 34:
                for m in module.modules():
                    if isinstance(m, ConvBNRelu):
                        bn_module = m[1]
                        highest_thre.append(bn_module.weight.data.abs().max().item())

        # for idx in prune_idx:
        #     highest_thre.append(model.module_list[idx][1].weight.data.abs().max().item())
        highest_thre = min(highest_thre) # 当超过该thres时，会至少有一层所有channel被剪去

        # 找到highest_thre对应的下标对应的百分比
        percent_limit = (sorted_bn==highest_thre).nonzero().item()/len(bn_weights) # 极限剪枝率，即之后设定不允许超过该值

        print(f'Threshold should be less than {highest_thre:.4f}.')
        print(f'The corresponding prune ratio is {percent_limit:.3f}.')
        return highest_thre

    @staticmethod
    def prune_and_eval(model, eval_model, percent=.0):
        bn_weights = gather_bn_weights(model)
        sorted_bn = torch.sort(bn_weights)[0]

        model_copy = deepcopy(model)
        thre_index = int(len(sorted_bn) * percent)
        thre = sorted_bn[thre_index]

        print(f'Channels with Gamma value less than {thre:.4f} are pruned!')

        remain_num = 0

        for id, module in enumerate(model_copy.module_list):
            if 1 <= id <= 34:
                for m in module.modules():
                    if isinstance(m, ConvBNRelu):
                        bn_module = m[1]
                        mask = obtain_bn_mask(bn_module, thre)

                        remain_num += int(mask.sum())
                        bn_module.weight.data.mul_(mask) # 将所有低于thres的全变为0

        mAP = eval_model(model_copy)[2].mean()

        print(f'Number of channels has been reduced from {len(sorted_bn)} to {remain_num}')
        print(f'Prune ratio: {1-remain_num/len(sorted_bn):.3f}')
        print(f'mAP of the pruned model is {mAP:.4f}')

        return thre

    # percent = 0.85
    # threshold = prune_and_eval(model, sorted_bn, percent)

    @staticmethod
    def obtain_filters_mask(model, thre):

        pruned = 0
        total = 0
        num_filters = []
        filters_mask = []

        for id, module in enumerate(model.module_list):
            if 1 <= id <= 34:
                for m in module.modules():
                    if isinstance(m, ConvBNRelu):
                        bn_module = m[1]
                        mask = obtain_bn_mask(bn_module, thre).cpu().numpy()
                        remain = int(mask.sum())  # 当前层保留的channel数量
                        pruned = pruned + mask.shape[0] - remain  # 累积待剪枝的channel数

                        if remain == 0:
                            print("Channels would be all pruned!")
                            raise Exception

                        print(f'layer index: {idx:>3d} \t total channel: {mask.shape[0]:>4d} \t '
                              f'remaining channel: {remain:>4d}')
                        total += mask.shape[0]
                        num_filters.append((id, remain))
                        filters_mask.append((id, mask.copy()))
            else:
                if isinstance(module, ConvBNRelu):
                    bn_module = module[1]
                    mask = np.ones(bn_module.weight.data.shape)
                    remain = mask.shape[0]

                    total += mask.shape[0]
                    num_filters.append((id, remain))
                    filters_mask.append((id, mask.copy()))

        prune_ratio = pruned / total # 实际剪枝率
        print(f'Prune channels: {pruned}\tPrune ratio: {prune_ratio:.3f}')

        return num_filters, filters_mask


num_filters, filters_mask = obtain_filters_mask(model, threshold)

#%%
CBLidx2mask = {idx: mask for idx, mask in zip(CBL_idx, filters_mask)}

pruned_model = prune_model_keep_size(model, prune_idx, CBL_idx, CBLidx2mask)

eval_model(pruned_model)


#%%
# 从这里开始才真正将剪掉的channel从模型中去除

# num_filters就为最终结果

compact_module_defs = deepcopy(model.module_defs)
for idx, num in zip(CBL_idx, num_filters):
    assert compact_module_defs[idx]['type'] == 'convolutional'
    compact_module_defs[idx]['filters'] = str(num) # 替换config文件中的channel数为剪枝后的,之后重新生成剪枝后的模型

#%%
compact_model = Darknet([model.hyperparams.copy()] + compact_module_defs).to(device)
compact_nparameters = obtain_num_parameters(compact_model)

init_weights_from_loose_model(compact_model, pruned_model, CBL_idx, Conv_idx, CBLidx2mask)

#%%
random_input = torch.rand((1, 3, model.img_size, model.img_size)).to(device)

def obtain_avg_forward_time(input, model, repeat=200):

    model.eval()
    start = time.time()
    with torch.no_grad():
        for i in range(repeat):
            output = model(input)
    avg_infer_time = (time.time() - start) / repeat

    return avg_infer_time, output

pruned_forward_time, pruned_output = obtain_avg_forward_time(random_input, pruned_model)
compact_forward_time, compact_output = obtain_avg_forward_time(random_input, compact_model)

diff = (pruned_output-compact_output).abs().gt(0.001).sum().item()
if diff > 0:
    print('Something wrong with the pruned model!')

#%%
# 在测试集上测试剪枝后的模型, 并统计模型的参数数量
compact_model_metric = eval_model(compact_model)

#%%
# 比较剪枝前后参数数量的变化、指标性能的变化
metric_table = [
    ["Metric", "Before", "After"],
    ["mAP", f'{origin_model_metric[2].mean():.6f}', f'{compact_model_metric[2].mean():.6f}'],
    ["Parameters", f"{origin_nparameters}", f"{compact_nparameters}"],
    ["Inference", f'{pruned_forward_time:.4f}', f'{compact_forward_time:.4f}']
]
print(AsciiTable(metric_table).table)

#%%
# 生成剪枝后的cfg文件并保存模型
pruned_cfg_name = opt.model_def.replace('/', f'/prune_{percent}_')
pruned_cfg_file = write_cfg(pruned_cfg_name, [model.hyperparams.copy()] + compact_module_defs)
print(f'Config file has been saved: {pruned_cfg_file}')

compact_model_name = opt.model.replace('/', f'/prune_{percent}_')
torch.save(compact_model.state_dict(), compact_model_name)
print(f'Compact model has been saved: {compact_model_name}')
