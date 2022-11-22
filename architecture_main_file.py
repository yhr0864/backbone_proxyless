import torch
from torch import nn
import numpy as np
from tensorboardX import SummaryWriter
import argparse
from collections import OrderedDict

from general_functions.prune_utils import BN_preprocess, pack_filter_para, generate_searchspace, init_weights_from_loose_model
from general_functions.utils import (worker_seed_set, parse_data_config, load_classes, get_logger, weights_init, load,
                                     create_directories_from_list, check_tensor_in_list, writh_new_ARCH_to_modeldef)
from building_blocks.builder import FBNet, Loss, ConvBNRelu, Identity
from building_blocks.modeldef import MODEL_ARCH, Test_model_arch

from supernet_main_file import _create_data_loader, _create_test_data_loader

from architecture_functions.training_functions import TrainerArch
from architecture_functions.config_for_arch import CONFIG_ARCH
from supernet_functions.config_for_supernet import CONFIG_SUPERNET
from supernet_functions.lookup_table_builder import (LookUpTable, SEARCH_SPACE_BACKBONE, SEARCH_SPACE_HEAD, SEARCH_SPACE_FPN,
                                                     YOLO_LAYER_26, YOLO_LAYER_13, extract_anchors)
from supernet_functions.model_supernet import YOLOLayer, Stochastic_SuperNet
from supernet_prune import PrunedModel


parser = argparse.ArgumentParser("architecture")
parser.add_argument('--architecture_name', type=str, default='',
                    help='You can choose architecture from the building_blocks/modeldef.py')
parser.add_argument("-d", "--data", type=str, default="./config/detrac.data",
                    help="Path to data config file (.data)")
parser.add_argument("--n_cpu", type=int, default=0,
                    help="Number of cpu threads to use during batch generation")
args = parser.parse_args()


def makeYOLOLayer(yolo_layer_26, yolo_layer_13):
    anchor26 = extract_anchors(yolo_layer_26)
    anchor13 = extract_anchors(yolo_layer_13)
    num_cls = yolo_layer_26['classes']
    return YOLOLayer(anchor26, num_cls), YOLOLayer(anchor13, num_cls)


def get_model(arch):
    #assert arch in MODEL_ARCH
    #arch_def = MODEL_ARCH[arch]

    # for test only
    ##############################
    assert arch in Test_model_arch
    arch_def = Test_model_arch[arch]
    ##############################
    yolo_layer26, yolo_layer13 = makeYOLOLayer(YOLO_LAYER_26, YOLO_LAYER_13)
    layer_parameters, _ = LookUpTable._generate_layers_parameters(search_space=SEARCH_SPACE_BACKBONE)
    layer_parameters_head, _ = LookUpTable._generate_layers_parameters(search_space=SEARCH_SPACE_HEAD)
    layer_parameters_fpn, _ = LookUpTable._generate_layers_parameters(search_space=SEARCH_SPACE_FPN)

    model = FBNet(arch_def, num_anchors=len(YOLO_LAYER_26['mask']), num_cls=YOLO_LAYER_26['classes'],
                  layer_parameters=layer_parameters,
                  layer_parameters_head26=layer_parameters_head,
                  layer_parameters_head13=layer_parameters_head,
                  layer_parameters_fpn=layer_parameters_fpn,
                  yolo_layer26=yolo_layer26,
                  yolo_layer13=yolo_layer13,
                  connect_layer_para=None)
    return model

def get_pruned_model(arch, num_filters):
    assert arch in MODEL_ARCH
    arch_def = MODEL_ARCH[arch]

    yolo_layer26, yolo_layer13 = makeYOLOLayer(YOLO_LAYER_26, YOLO_LAYER_13)
    backbone_para = pack_filter_para(1, 11, num_filters)
    fpn_para = pack_filter_para(15, 22, num_filters)
    head26_para = pack_filter_para(25, 29, num_filters)
    head13_para = pack_filter_para(30, 34, num_filters)

    input_shape_backbone, channel_size_backbone, prune_backbone = generate_searchspace(backbone_para, first_input=16)
    PRUNED_SEARCH_SPACE_BACKBONE = OrderedDict()
    PRUNED_SEARCH_SPACE_BACKBONE['input_shape'] = input_shape_backbone
    PRUNED_SEARCH_SPACE_BACKBONE['channel_size'] = channel_size_backbone
    PRUNED_SEARCH_SPACE_BACKBONE['prune'] = prune_backbone
    PRUNED_SEARCH_SPACE_BACKBONE['stride'] = SEARCH_SPACE_BACKBONE['stride']

    input_fpn = pack_filter_para(14, 14, num_filters)[0][0]
    input_shape_fpn, channel_size_fpn, prune_fpn = generate_searchspace(fpn_para, first_input=input_fpn)
    PRUNED_SEARCH_SPACE_FPN = OrderedDict()
    PRUNED_SEARCH_SPACE_FPN['input_shape'] = input_shape_fpn
    PRUNED_SEARCH_SPACE_FPN['channel_size'] = channel_size_fpn
    PRUNED_SEARCH_SPACE_FPN['prune'] = prune_fpn
    PRUNED_SEARCH_SPACE_FPN['stride'] = SEARCH_SPACE_FPN['stride']

    input_head26 = pack_filter_para(23, 23, num_filters)[0][0]
    input_head13 = pack_filter_para(24, 24, num_filters)[0][0]
    input_shape_head26, channel_size_head26, prune_head26 = generate_searchspace(head26_para, first_input=input_head26)
    input_shape_head13, channel_size_head13, prune_head13 = generate_searchspace(head13_para, first_input=input_head13)
    PRUNED_SEARCH_SPACE_HEAD26 = OrderedDict()
    PRUNED_SEARCH_SPACE_HEAD13 = OrderedDict()
    PRUNED_SEARCH_SPACE_HEAD26['input_shape'] = input_shape_head26
    PRUNED_SEARCH_SPACE_HEAD13['input_shape'] = input_shape_head13
    PRUNED_SEARCH_SPACE_HEAD26['channel_size'] = channel_size_head26
    PRUNED_SEARCH_SPACE_HEAD13['channel_size'] = channel_size_head13
    PRUNED_SEARCH_SPACE_HEAD26['prune'] = prune_head26
    PRUNED_SEARCH_SPACE_HEAD13['prune'] = prune_head13
    PRUNED_SEARCH_SPACE_HEAD26['stride'] = SEARCH_SPACE_HEAD['stride']
    PRUNED_SEARCH_SPACE_HEAD13['stride'] = SEARCH_SPACE_HEAD['stride']

    layer_parameters, _ = LookUpTable._generate_layers_parameters(search_space=PRUNED_SEARCH_SPACE_BACKBONE)
    layer_parameters_fpn, _ = LookUpTable._generate_layers_parameters(search_space=PRUNED_SEARCH_SPACE_FPN)
    layer_parameters_head26, _ = LookUpTable._generate_layers_parameters(search_space=PRUNED_SEARCH_SPACE_HEAD26)
    layer_parameters_head13, _ = LookUpTable._generate_layers_parameters(search_space=PRUNED_SEARCH_SPACE_HEAD13)

    model = FBNet(arch_def, num_anchors=len(YOLO_LAYER_26['mask']), num_cls=YOLO_LAYER_26['classes'],
                  layer_parameters=layer_parameters,
                  layer_parameters_head26=layer_parameters_head26,
                  layer_parameters_head13=layer_parameters_head13,
                  layer_parameters_fpn=layer_parameters_fpn,
                  yolo_layer26=yolo_layer26,
                  yolo_layer13=yolo_layer13,
                  connect_layer_para=None)
    return model


def main():
    manual_seed = 1
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)
    torch.backends.cudnn.benchmark = True
    
    create_directories_from_list([CONFIG_ARCH['logging']['path_to_tensorboard_logs']])
    
    logger = get_logger(CONFIG_ARCH['logging']['path_to_log_file'])
    writer = SummaryWriter(log_dir=CONFIG_ARCH['logging']['path_to_tensorboard_logs'])

    data_config = parse_data_config(args.data)
    train_path = data_config["train"]
    valid_path = data_config["valid"]

    #### DataLoading
    train_loader = _create_data_loader(train_path,
                                       CONFIG_ARCH['dataloading']['batch_size'],
                                       CONFIG_ARCH['dataloading']['img_size'],
                                       args.n_cpu)

    valid_loader = _create_test_data_loader(valid_path,
                                            CONFIG_ARCH['dataloading']['batch_size'],
                                            CONFIG_ARCH['dataloading']['img_size'],
                                            args.n_cpu)

    #### Model
    arch = args.architecture_name
    model = get_model(arch).cuda()

    #### Load Parameters
    lookup_table = LookUpTable()
    supernet = Stochastic_SuperNet(lookup_table=lookup_table)
    checkpoint = torch.load(CONFIG_SUPERNET['train_settings']['path_to_save_model'])
    supernet.load_state_dict(checkpoint["state_dict"])

    del_keys_backbone = []
    rev_keys_backbone = []
    del_keys_fpn = []
    rev_keys_fpn = []
    del_keys_head = []
    rev_keys_head = []

    supernet_copy = supernet.state_dict().copy()

    def key_process(key, start, end, del_keys, rev_keys):
        global chosen_id
        for i in range(start, end):
            if key.split('.')[1] == str(i) and key.split('.')[-1] == 'AP_path_alpha':
                chosen_id = np.argmax(supernet_copy[key].cpu().numpy())
                # print(chosen_id)
            if len(key.split('.')) > 3:
                if key.split('.')[1] == str(i) and key.split('.')[3] != str(chosen_id):
                    del_keys.append(key)
                elif key.split('.')[1] == str(i) and key.split('.')[3] == str(chosen_id):
                    rev_keys.append(key)

    for key in supernet_copy.keys():
        key_process(key, 1, 12, del_keys_backbone, rev_keys_backbone)  # backbone

        key_process(key, 15, 23, del_keys_fpn, rev_keys_fpn)  # fpn

        key_process(key, 25, 35, del_keys_head, rev_keys_head)  # head

    # delete the unchosen parameters
    def key_revise(supernet, del_keys, rev_keys):
        for del_key in del_keys:
            del supernet[del_key]

        # revise the name of rev_keys to match the sub-model
        for k in rev_keys:
            or_name = k.split('.')
            new_names = or_name[0:2] + or_name[4:]
            new_name = ''
            for i in range(len(new_names)):
                new_name += new_names[i]
                if i != len(new_names) - 1:
                    new_name += '.'

            supernet[new_name] = supernet[k]
            del supernet[k]

    key_revise(supernet_copy, del_keys_backbone, rev_keys_backbone)
    key_revise(supernet_copy, del_keys_fpn, rev_keys_fpn)
    key_revise(supernet_copy, del_keys_head, rev_keys_head)

    missing_keys, unexpected_keys = model.load_state_dict(state_dict=supernet_copy, strict=False)

    # save the sub-model
    torch.save(model.state_dict(), CONFIG_ARCH['sub-model-saving'])

    # 接下来可以剪枝
    ##############
    #第一步先BN同步
    ##############
    BN_preprocess(model)

    #######
    # prune
    #######
    # get highest prune ratio
    highest_thre = PrunedModel.get_highest_thre(model)
    # get the tres and evaluate pruned model
    threshold = PrunedModel.prune_and_eval(model, eval_model, percent=highest_thre-0.01)
    # get num_filters for re-constructing the pruned model
    num_filters, filters_mask = PrunedModel.obtain_filters_mask(model, threshold)
    # rebuild the model
    # generate the new layer para.
    pruned_model = get_pruned_model(arch, num_filters)

    # reload parameters
    init_weights_from_loose_model(pruned_model, model, filters_mask)

    model = nn.DataParallel(pruned_model, [0])

    #### Loss and Optimizer
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                                lr=CONFIG_ARCH['optimizer']['lr'],
                                momentum=CONFIG_ARCH['optimizer']['momentum'],
                                weight_decay=CONFIG_ARCH['optimizer']['weight_decay'])
    criterion = Loss().cuda()
    
    #### Scheduler
    if CONFIG_ARCH['train_settings']['scheduler'] == 'MultiStepLR':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                    milestones=CONFIG_ARCH['train_settings']['milestones'],
                                                    gamma=CONFIG_ARCH['train_settings']['lr_decay'])  
    elif CONFIG_ARCH['train_settings']['scheduler'] == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                               T_max=CONFIG_ARCH['train_settings']['cnt_epochs'],
                                                               eta_min=0.001, last_epoch=-1)
    else:
        logger.info("Please, specify scheduler in architecture_functions/config_for_arch")
        
    
    #### Training Loop
    trainer = TrainerArch(criterion, optimizer, scheduler, logger, writer)
    trainer.train_loop(train_loader, valid_loader, model) 
    
if __name__ == "__main__":
    #main()
    def weights_init(m):
        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.kaiming_uniform_(m.weight.data)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, torch.nn.BatchNorm2d):
            torch.nn.init.normal_(m.weight.data)
            #m.weight.data.fill_(0.5)
            m.bias.data.zero_()
    arch = 'test_net'
    model = get_model(arch)
    model = model.apply(weights_init)
    model.train()

    #print(model)


    input = torch.randn(1, 3, 416, 416)
    #out = model(input)
    #
    #print(out[0].shape)
    #print(out[1].shape)

    print(model.module_list[8])

    ###############################################################################################
    # visualize
    ## First step: convert .pt to .onnx
    # torch.onnx.export(model,  # model being run
    #                   input,  # model input (or a tuple for multiple inputs)
    #                   "./model_arch.onnx",  # where to save the model (can be a file or file-like object)
    #                   export_params=True,  # store the trained parameter weights inside the model file
    #                   # the ONNX version to export the model to
    #                   opset_version=13,
    #                   do_constant_folding=True,  # whether to execute constant folding for optimization
    #                   input_names=['input'],  # the model's input names
    #                   output_names=['output'],  # the model's output names
    #                  )
    ## Second step: visualize with netron
    # import netron
    # netron.start('./model_arch.onnx')
    ########################################################


    #### QUANTIZE TEST ####
    #######################
    #### FX GRAPH MODE ####
    ###################################################################################################
    # import copy
    # from torch.quantization import get_default_qconfig
    # from torch.quantization.quantize_fx import prepare_fx, convert_fx
    # from torch.fx import symbolic_trace
    #
    #
    # float_model = copy.deepcopy(model)
    # #symbolic_traced: torch.fx.GraphModule = symbolic_trace(float_model)
    # float_model.cpu()
    # float_model.eval()
    # qconfig = get_default_qconfig("fbgemm")
    # qconfig_dict = {"": qconfig}
    #
    # def calibrate(model, data_loader):
    #     model.eval()
    #     with torch.no_grad():
    #         for idx, (image, target) in enumerate(data_loader):
    #             model(image)
    #             if idx == int(0.5 * len(data_loader)):
    #                 break
    #
    # prepared_model = prepare_fx(float_model, qconfig_dict)  # fuse modules and insert observers
    # #calibrate(prepared_model, train_loader)  # run calibration on sample data
    # quantized_model = convert_fx(prepared_model)  # convert the calibrated model to a quantized model
    ###################################################################################################
    # from building_blocks.layers import Conv2d, BatchNorm2d
    # from general_functions.fuse.fuser_method_mappings import fuse_conv_bn, fuse_conv_bn_relu
    # from general_functions.fuse.fuse_modules import fuse_modules
    # import copy
    #
    # model_fp32 = copy.deepcopy(model)
    # model_fp32.cpu()
    # model_fp32.eval()
    # model_fp32.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    #
    #
    # fuse_custom_config_dict = {
    #     # Additional fuser_method mapping
    #     "additional_fuser_method_mapping": {
    #         (Conv2d, BatchNorm2d): fuse_conv_bn,
    #         (Conv2d, BatchNorm2d, nn.ReLU): fuse_conv_bn_relu,
    #         (Conv2d, BatchNorm2d, nn.Hardswish): fuse_conv_bn_relu,
    #     },
    # }

    #### FUSION ####
    # for i, m in enumerate(model_fp32.module_list):
    #     if isinstance(m, nn.Sequential):
    #         if len(m) == 3:
    #             model_fp32.module_list[i] = fuse_modules(m, ['conv', 'bn', 'act'],
    #                                                      fuse_custom_config_dict=fuse_custom_config_dict)
    #         if len(m) == 2:
    #             model_fp32.module_list[i] = fuse_modules(m, ['conv', 'bn'],
    #                                                      fuse_custom_config_dict=fuse_custom_config_dict)
    #         #print(i)
    #         #print(m)
    #     else:
    #         #print(i)
    #         try:
    #             if m.module_list:
    #                 #print(m.module_list[0])
    #                 if len(m.module_list) == 3:
    #                     m.module_list[0] = fuse_modules(m.module_list[0], ['conv', 'bn', 'act'],
    #                                                     fuse_custom_config_dict=fuse_custom_config_dict)
    #
    #                     m.module_list[1] = fuse_modules(m.module_list[1], ['conv', 'bn', 'act'],
    #                                                     fuse_custom_config_dict=fuse_custom_config_dict)
    #
    #                     m.module_list[2] = fuse_modules(m.module_list[2], ['conv', 'bn'],
    #                                                     fuse_custom_config_dict=fuse_custom_config_dict)
    #                 elif len(m.module_list) == 1:
    #                     m.module_list[0] = fuse_modules(m.module_list[0], ['conv', 'bn', 'act'],
    #                                                     fuse_custom_config_dict=fuse_custom_config_dict)
    #         except:
    #             pass
    # 标准卷积才能量化
    # model_fp32_prepared = torch.quantization.prepare(model_fp32)
    # model_fp32_prepared(input)
    # model_int8 = torch.quantization.convert(model_fp32_prepared)
    # print(model_int8)
    # model_fp32.module_list[1].module_list[0] = fuse_modules(model_fp32.module_list[1].module_list[0], ['conv', 'bn', 'act'],
    #                                                      fuse_custom_config_dict=fuse_custom_config_dict)

    #print(model_fp32.module_list[1].module_list[0])
    #print(model_fp32.module_list[1])
    #model_fp32_fused = torch.quantization.fuse_modules(model_fp32, ['conv', 'bn', 'act'])


    # from torch.ao.quantization.fuser_method_mappings import fuse_conv_bn, fuse_conv_bn_relu
    # from building_blocks.layers import Conv2d, BatchNorm2d
    # """
    # 单纯融合conv-bn没有问题, 但是对于conv-bn-relu有问题
    # """
    # m1 = Conv2d(10, 20, 3)
    # b1 = BatchNorm2d(20)
    # r1 = nn.ReLU(inplace=False)
    #
    # m1.eval()
    # b1.eval()
    # r1.eval()
    #
    # m2 = fuse_conv_bn(m1, b1)
    #
    # class ConvReLU2d(nn.Sequential):
    #     def __init__(self, conv, relu):
    #         assert type(conv) == Conv2d and type(relu) == nn.ReLU, \
    #             'Incorrect types for input modules{}{}'.format(
    #                 type(conv), type(relu))
    #         super().__init__(conv, relu)
    #
    # fused_module = ConvReLU2d
    # m4 = fused_module(m2, r1)
    # print(m4)