import torch
import numpy as np
from copy import deepcopy
from terminaltables import AsciiTable
from torch.autograd import Variable

from general_functions.prune_utils import *
from general_functions.utils import ap_per_class, xywh2xyxy, non_max_suppression, get_batch_statistics

from supernet_functions.config_for_supernet import CONFIG_SUPERNET


def eval_model(model, loader, img_size):
        model.eval()
        labels = []
        sample_metrics = []

        for step, (_, images, targets) in enumerate(loader):
            images, targets = images.cuda(), targets.cuda()
            # Extract labels
            labels += targets[:, 1].tolist()
            # Rescale target
            targets[:, 2:] = xywh2xyxy(targets[:, 2:])
            targets[:, 2:] *= img_size

            images = Variable(images, requires_grad=False)

            with torch.no_grad():
                outs = model(images)
                outs = non_max_suppression(outs, conf_thres=CONFIG_SUPERNET['valid_settings']['conf_thres'],
                                           iou_thres=CONFIG_SUPERNET['valid_settings']['nms_thres'])

            sample_metrics += get_batch_statistics(outs, targets.cpu(),
                                                   iou_threshold=CONFIG_SUPERNET['valid_settings']['iou_thres'])

        if len(sample_metrics) == 0:  # No detections over whole validation set.
            print("---- No detections over whole validation set ----")
            return None

        true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
        metrics_output = ap_per_class(true_positives, pred_scores, pred_labels, labels)

        if metrics_output is not None:
            precision, recall, AP, f1, ap_class = metrics_output
        else:
            print(" mAP not measured (no detections found by model)")
            return
        return AP.mean().item()
    
    
class PrunedModel:
    @staticmethod
    def get_highest_thre(model):
        bn_weights = gather_bn_weights(model)

        sorted_bn = torch.sort(bn_weights)[0]

        # 避免剪掉所有channel的最高阈值(每个BN层的gamma的最大值的最小值即为阈值上限)
        highest_thre = []
        for idx, module in enumerate(model.module_list):
            if 1 <= idx <= 34:
                for m in module.modules():
                    if isinstance(m, ConvBNRelu):
                        bn_module = m[1]
                        highest_thre.append(bn_module.weight.data.abs().max().item())

        # for idx in prune_idx:
        #     highest_thre.append(model.module_list[idx][1].weight.data.abs().max().item())
        highest_thre = min(highest_thre) # 当超过该thres时，会至少有一层所有channel被剪去

        # 找到highest_thre对应的下标对应的百分比   
        percent_limit = (sorted_bn==highest_thre).nonzero().min().item()/len(bn_weights) # 极限剪枝率，即之后设定不允许超过该值

        print(f'Threshold should be less than {highest_thre:.4f}.')
        print(f'The corresponding prune ratio is {percent_limit:.3f}.')
        return highest_thre, percent_limit

    @staticmethod
    def prune_and_eval(model, test_loader, img_size, percent=.0):
        bn_weights = gather_bn_weights(model)
        sorted_bn = torch.sort(bn_weights)[0]

        model_copy = deepcopy(model)
        thre_index = int(len(sorted_bn) * percent)
        thre = sorted_bn[thre_index]

        print(f'Channels with Gamma value less than {thre:.4f} are pruned!')

        remain_num = 0

        for idx, module in enumerate(model_copy.module_list):
            if 1 <= idx <= 34:
                for m in module.modules():
                    if isinstance(m, ConvBNRelu):
                        bn_module = m[1]
                        mask = obtain_bn_mask(bn_module, thre)

                        remain_num += int(mask.sum())
                        bn_module.weight.data.mul_(mask) # 将所有低于thres的全变为0

        mAP = eval_model(model_copy, test_loader, img_size)

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

        for idx, module in enumerate(model.module_list):
            if 1 <= idx <= 34:
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
                        num_filters.append((idx, remain))
                        filters_mask.append((idx, mask.copy()))
            else:
                if isinstance(module, ConvBNRelu):
                    bn_module = module[1]
                    mask = np.ones(bn_module.weight.data.shape)
                    remain = mask.shape[0]

                    total += mask.shape[0]
                    num_filters.append((idx, remain))
                    filters_mask.append((idx, mask.copy()))

        prune_ratio = pruned / total # 实际剪枝率
        print(f'Prune channels: {pruned}\tPrune ratio: {prune_ratio:.3f}')

        return num_filters, filters_mask
