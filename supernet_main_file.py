import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter
from scipy.special import softmax
import argparse

from general_functions.augmentations import AUGMENTATION_TRANSFORMS
from general_functions.transforms import DEFAULT_TRANSFORMS
from general_functions.datasets import ListDataset
from general_functions.utils import (worker_seed_set, parse_data_config, get_logger, weights_init, load,
                                     create_directories_from_list, check_tensor_in_list, writh_new_ARCH_to_modeldef)

from supernet_functions.lookup_table_builder import LookUpTable
from supernet_functions.model_supernet import Stochastic_SuperNet, SupernetLoss
from supernet_functions.training_functions_supernet import TrainerSupernet
from supernet_functions.config_for_supernet import CONFIG_SUPERNET

from building_blocks.modeldef import MODEL_ARCH
from building_blocks.builder import ConvBNRelu, Upsample

parser = argparse.ArgumentParser("action")
parser.add_argument('--train_or_sample', type=str, default='',
                    help='train means training of the SuperNet, sample means sample from SuperNet\'s results')
parser.add_argument("-d", "--data", type=str, default="./config/detrac.data",
                    help="Path to data config file (.data)")
parser.add_argument("--n_cpu", type=int, default=0,
                    help="Number of cpu threads to use during batch generation")
parser.add_argument("--resume", type=str, default=None,
                    help="Resume training path")
parser.add_argument('--architecture_name', type=str, default='',
                    help='Name of an architecture to be sampled')
args = parser.parse_args()


def _create_data_loader(img_path, batch_size, img_size, n_cpu, multiscale_training=False):
    """Creates a DataLoader for training.

    :param img_path: Path to file containing all paths to training images.
    :type img_path: str
    :param batch_size: Size of each image batch
    :type batch_size: int
    :param img_size: Size of each image dimension for yolo
    :type img_size: int
    :param n_cpu: Number of cpu threads to use during batch generation
    :type n_cpu: int
    :param multiscale_training: Scale images to different sizes randomly
    :type multiscale_training: bool
    :return: Returns DataLoader
    :rtype: DataLoader
    """
    dataset = ListDataset(
        img_path,
        img_size=img_size,
        multiscale=multiscale_training,
        transform=AUGMENTATION_TRANSFORMS)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
        worker_init_fn=worker_seed_set)
    return dataloader


def _create_test_data_loader(img_path, batch_size, img_size, n_cpu):
    """
    Creates a DataLoader for test.

    :param img_path: Path to file containing all paths to validation images.
    :type img_path: str
    :param batch_size: Size of each image batch
    :type batch_size: int
    :param img_size: Size of each image dimension for yolo
    :type img_size: int
    :param n_cpu: Number of cpu threads to use during batch generation
    :type n_cpu: int
    :return: Returns DataLoader
    :rtype: DataLoader
    """
    dataset = ListDataset(img_path, img_size=img_size, multiscale=False, transform=DEFAULT_TRANSFORMS)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn)
    return dataloader


def train_supernet(resume=False):
    manual_seed = 1
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)
    torch.backends.cudnn.benchmark = True

    data_config = parse_data_config(args.data)
    train_path = data_config["train"]
    valid_path = data_config["valid"]
    test_path = data_config["test"]

    create_directories_from_list([CONFIG_SUPERNET['logging']['path_to_tensorboard_logs']])

    logger = get_logger(CONFIG_SUPERNET['logging']['path_to_log_file'])
    writer = SummaryWriter(log_dir=CONFIG_SUPERNET['logging']['path_to_tensorboard_logs'])

    #### LookUp table consists all information about layers
    lookup_table = LookUpTable(calculate_latency=CONFIG_SUPERNET['lookup_table']['create_from_scratch'])

    #### DataLoading
    train_w_loader = _create_data_loader(train_path,
                                         CONFIG_SUPERNET['dataloading']['batch_size'],
                                         CONFIG_SUPERNET['dataloading']['img_size'],
                                         args.n_cpu)

    train_thetas_loader = _create_data_loader(valid_path,
                                              CONFIG_SUPERNET['dataloading']['batch_size'],
                                              CONFIG_SUPERNET['dataloading']['img_size'],
                                              args.n_cpu)

    test_loader = _create_test_data_loader(test_path,
                                           CONFIG_SUPERNET['dataloading']['batch_size'],
                                           CONFIG_SUPERNET['dataloading']['img_size'],
                                           args.n_cpu)

    #### Model
    model = Stochastic_SuperNet(lookup_table).cuda()
    model = model.apply(weights_init)
    model = nn.DataParallel(model, device_ids=[0])

    #### Loss, Optimizer and Scheduler
    criterion = SupernetLoss().cuda()

    thetas_params = model.architecture_parameters() # AP_path_alpha
    weights_params = model.weight_parameters() # parameters except AP_path_alpha & AP_path_wb

    w_optimizer = torch.optim.SGD(params=weights_params,
                                  lr=CONFIG_SUPERNET['optimizer']['w_lr'],
                                  momentum=CONFIG_SUPERNET['optimizer']['w_momentum'],
                                  weight_decay=CONFIG_SUPERNET['optimizer']['w_weight_decay'])

    theta_optimizer = torch.optim.Adam(params=thetas_params,
                                       lr=CONFIG_SUPERNET['optimizer']['thetas_lr'],
                                       weight_decay=CONFIG_SUPERNET['optimizer']['thetas_weight_decay'])

    last_epoch = -1
    w_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(w_optimizer,
                                                             T_max=CONFIG_SUPERNET['train_settings']['cnt_epochs'],
                                                             last_epoch=last_epoch)

    if resume:
        checkpoint = torch.load(resume)
        model.load_state_dict(checkpoint["state_dict"])
        w_optimizer.load_state_dict(checkpoint["w_optimizer"])
        theta_optimizer.load_state_dict(checkpoint["theta_optimizer"])
        w_scheduler = checkpoint["w_scheduler"]

    #### Training Loop
    trainer = TrainerSupernet(criterion, w_optimizer, theta_optimizer, w_scheduler, logger, writer)
    trainer.train_loop(train_w_loader, train_thetas_loader, test_loader, model)


# Arguments:
# hardsampling=True means get operations with the largest weights
#             =False means apply softmax to weights and sample from the distribution
# unique_name_of_arch - name of architecture. will be written into building_blocks/modeldef.py
#                       and can be used in the training by train_architecture_main_file.py
def sample_architecture_from_the_supernet(unique_name_of_arch):
    logger = get_logger(CONFIG_SUPERNET['logging']['path_to_log_file'])

    lookup_table = LookUpTable()
    model = Stochastic_SuperNet(lookup_table).cuda()
    model = nn.DataParallel(model)

    checkpoint = torch.load(CONFIG_SUPERNET['train_settings']['path_to_save_model'])
    model.load_state_dict(checkpoint["state_dict"])

    ops_names_backbone = [op_name for op_name in lookup_table.lookup_table_operations]
    ops_names_head = [op_name for op_name in lookup_table.lookup_table_operations_head]
    ops_names_fpn = [op_name for op_name in lookup_table.lookup_table_operations_fpn]

    arch_operations_backbone = []
    arch_operations_head26 = []
    arch_operations_head13 = []
    arch_operations_fpn = []

    for i, layer in enumerate(model.module_list[1:11]):
        arch_operations_backbone.append(ops_names_backbone[np.argmax(layer.AP_path_alpha.detach().cpu().numpy())])
    for layer in model.module_list[15:22]:
        arch_operations_fpn.append(ops_names_fpn[np.argmax(layer.AP_path_alpha.detach().cpu().numpy())])
    for layer26, layer13 in zip(model.module_list[24:28], model.module_list[29:33]):
        arch_operations_head26.append(ops_names_head[np.argmax(layer26.AP_path_alpha.detach().cpu().numpy())])
        arch_operations_head13.append(ops_names_head[np.argmax(layer13.AP_path_alpha.detach().cpu().numpy())])

    arch_operations = {
        'arch_operations_backbone': arch_operations_backbone,
        'arch_operations_head26': arch_operations_head26,
        'arch_operations_head13': arch_operations_head13,
        'arch_operations_fpn': arch_operations_fpn
    }

    logger.info("Sampled Architecture: " + " - ".join(arch_operations))
    writh_new_ARCH_to_modeldef(arch_operations, my_unique_name_for_ARCH=unique_name_of_arch)
    logger.info("CONGRATULATIONS! New architecture " + unique_name_of_arch
                + " was written into building_blocks/modeldef.py")


if __name__ == "__main__":
    assert args.train_or_sample in ['train', 'sample']
    if args.train_or_sample == 'train':
        train_supernet(resume=args.resume)
    elif args.train_or_sample == 'sample':
        assert args.architecture_name != '' and args.architecture_name not in MODEL_ARCH
        hardsampling = False if args.hardsampling_bool_value in ['False', '0'] else True
        sample_architecture_from_the_supernet(unique_name_of_arch=args.architecture_name)
