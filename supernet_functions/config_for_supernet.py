import numpy as np

CONFIG_SUPERNET = {
    'gpu_settings' : {
        'gpu_ids' : [0]
    },
    'lookup_table' : {
        'create_from_scratch' : False,
        'path_to_lookup_table' : './lookup_table.txt',
        'path_to_lookup_table_head' : './lookup_table_head.txt',
        'path_to_lookup_table_fpn' : './lookup_table_fpn.txt',
        'number_of_runs' : 50 # each operation run number_of_runs times and then we will take average
    },
    'logging' : {
        'path_to_log_file' : './supernet_functions/logs/logger/',
        'path_to_tensorboard_logs' : './supernet_functions/logs/tb'
    },
    'binary_mode' : 'two',
    'dataloading' : {
        'img_size' : 416,
        'batch_size' : 256,
        'w_share_in_train' : 0.8,
        'path_to_save_data' : './cifar10_data'
    },
    'optimizer' : {
        # SGD parameters for w
        'w_lr' : 0.1,
        'w_momentum' : 0.9,
        'w_weight_decay' : 1e-4,
        # Adam parameters for thetas
        'thetas_lr' : 0.01,
        'thetas_weight_decay' : 5 * 1e-4
    },
    'loss' : {
        'alpha' : 0.2,
        'beta' : 0.6
    },
    'train_settings' : {
        'cnt_epochs' : 180, # 90
        'train_thetas_from_the_epoch' : 10,
        'print_freq' : 50,
        'path_to_save_model' : './supernet_functions/logs/best_model.pth',
        'path_to_save_current_model' : './supernet_functions/logs/current_model.pth',
        # for Gumbel Softmax
        'init_temperature' : 5.0,
        'exp_anneal_rate' : np.exp(-0.045)
    },
    'valid_settings' : {
        'iou_thres' : 0.5,
        'conf_thres' : 0.1,
        'nms_thres' : 0.5
    },
    'quan' : {
        'act' : {
            'mode': 'lsq',
            # Bit width of quantized activation
            'bit': 8,
            # Each output channel uses its own scaling factor
            'per_channel': False,
            # Whether to use symmetric quantization
            'symmetric': False,
            # Quantize all the numbers to non-negative
            'all_positive': True
        },
        'weight' : {
            'mode': 'lsq',
            # Bit width of quantized activation
            'bit': 8,
            # Each output channel uses its own scaling factor
            'per_channel': True,
            # Whether to use symmetric quantization
            'symmetric': False,
            # Quantize all the numbers to non-negative
            'all_positive': False
        },
        'excepts' : {
            # Specify quantized bit width for some layers, like this:
            'conv1' : {
                'weight' : {'bit' : None},
                'act' : {'all_positive' : False}
            }
        }
    },
    'prune' : {
        'sr' : True,
        'scale_sparse_rate' : 0.01
    }
}