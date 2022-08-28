import torch
import time
import numpy as np

from torch.autograd import Variable
from general_functions.prune_utils import BNOptimizer
from general_functions.utils import save, ap_per_class, xywh2xyxy, non_max_suppression, get_batch_statistics
from architecture_functions.config_for_arch import CONFIG_ARCH
from supernet_functions.config_for_supernet import CONFIG_SUPERNET


class TrainerArch:
    def __init__(self, criterion, optimizer, scheduler, logger, writer):
        self.logger = logger
        self.writer = writer
        
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        
        self.path_to_save_model = CONFIG_ARCH['train_settings']['path_to_save_model']
        self.cnt_epochs         = CONFIG_ARCH['train_settings']['cnt_epochs']
        self.print_freq         = CONFIG_ARCH['train_settings']['print_freq']
        
    def train_loop(self, train_loader, valid_loader, model):
        best_mAP = 0.0

        for epoch in range(self.cnt_epochs):
            
            self.writer.add_scalar('learning_rate', self.optimizer.param_groups[0]['lr'], epoch)
            
            #if epoch and epoch % self.lr_decay_period == 0:
            #    self.optimizer.param_groups[0]['lr'] *= self.lr_decay

            # training
            self._train(train_loader, model, epoch)
            # validation
            mAP = self._validate(valid_loader, model, epoch, img_size=CONFIG_ARCH['dataloading']['img_size'])

            if best_mAP < mAP:
                best_mAP = mAP
                self.logger.info("Best mAP by now. Save model")
                save(model, self.path_to_save_model)

            self.scheduler.step()
    
    def _train(self, loader, model, epoch):
        model.train()

        for step, (_, images, targets) in enumerate(loader):
            images, targets = images.cuda(non_blocking=True), targets.cuda(non_blocking=True)

            outs = model(images)
            loss, loss_components = self.criterion(outs, targets)
            loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

            self._train_logging(loss, loss_components, step, epoch, len_loader=len(loader))


    def _validate(self, loader, model, epoch, img_size):
        model.eval()
        start_time = time.time()

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

            sample_metrics += get_batch_statistics(outs, targets,
                                                   iou_threshold=CONFIG_SUPERNET['valid_settings']['iou_thres'])

        if len(sample_metrics) == 0:  # No detections over whole validation set.
            print("---- No detections over whole validation set ----")
            return None

        true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
        metrics_output = ap_per_class(true_positives, pred_scores, pred_labels, labels)

        if metrics_output is not None:
            precision, recall, AP, f1, ap_class = metrics_output
            self._valid_logging(start_time=start_time, epoch=epoch, metrics_output=metrics_output)
        else:
            self.logger.info(
                " mAP not measured (no detections found by model) for {:3d}/{} Time {:.2f}".format(
                    epoch + 1, self.cnt_epochs, time.time() - start_time))
            return
        return AP.mean()

    def _valid_logging(self, start_time, epoch, metrics_output):
        precision, recall, AP, f1, ap_class = metrics_output

        self.writer.add_scalar('valid_precision', precision.mean(), epoch)
        self.writer.add_scalar('valid_recall', recall.mean(), epoch)
        self.writer.add_scalar('valid_mAP', AP.mean(), epoch)
        self.writer.add_scalar('valid_f1', f1.mean(), epoch)

        self.logger.info("valid : [{:3d}/{}] Final Precision {:.4%}, Time {:.2f}".format(
            epoch + 1, self.cnt_epochs, AP.mean(), time.time() - start_time))

    def _train_logging(self, loss, loss_components, step, epoch, len_loader):

        self.writer.add_scalar('total_loss', loss.mean(), epoch)
        self.writer.add_scalar('iou_loss', loss_components[0].mean(), epoch)
        self.writer.add_scalar('obj_loss', loss_components[1].mean(), epoch)
        self.writer.add_scalar('cls_loss', loss_components[2].mean(), epoch)

        if (step > 1 and step % self.print_freq == 0) or step == len_loader - 1:
            self.logger.info("training: [{:3d}/{}] Step {:03d}/{:03d} Loss {:.3f} "
                             "iou_loss {:.3f}, obj_loss {:.3f}, cls_loss {:.3f}".format(
                                 epoch + 1, self.cnt_epochs, step, len_loader - 1, loss,
                                 loss_components[0], loss_components[1], loss_components[2]))
