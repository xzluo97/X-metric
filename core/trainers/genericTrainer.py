# -*- coding: utf-8 -*-
"""
Unified modules for network training, validation and testing.

@author: Xinzhe Luo
"""

import os
from core import utils
import math
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


class Trainer(object):
    def __init__(self, net, verbose=0, **kwargs):
        self.verbose = verbose
        self.kwargs = kwargs
        self.net = net
        self.save_path = self.kwargs.get("save_path", './Train')
        if not os.path.exists(self.save_path):
            logging.info("Allocating directory: %s" % os.path.abspath(self.save_path))
            os.makedirs(self.save_path)

        self.logger = self.kwargs.get("logger", logging)
        self.optimizer_name = self.kwargs.get("optimizer_name", "Adam")
        self.lr = self.kwargs.get("learning_rate", 1e-3)
        self.weight_decay = self.kwargs.get("weight_decay", 0)
        self.scheduler_name = self.kwargs.get("scheduler_name", None)

        if verbose == 1:
            self.logger.info(self.net)


    def _get_writer(self, save_path=None):
        if save_path is None:
            save_path = self.save_path
        return SummaryWriter(log_dir=os.path.join(save_path, 'runs/'))

    def _get_optimizer(self, params, **kwargs):
        optimizer_name = kwargs.pop('optimizer_name', None)
        lr = kwargs.pop('lr', self.lr)
        if optimizer_name is None:
            optimizer_name = self.optimizer_name
        if optimizer_name == 'Adam':
            optimizer = optim.Adam(params, lr=lr, weight_decay=self.weight_decay, **kwargs)
        elif optimizer_name == 'AdamW':
            optimizer = optim.AdamW(params, lr=lr, weight_decay=self.weight_decay, **kwargs)
        elif optimizer_name == 'SGD':
            optimizer = optim.SGD(params, lr=lr, weight_decay=self.weight_decay, **kwargs)
        elif optimizer_name == 'RMSprop':
            optimizer = optim.RMSprop(params, lr=lr, weight_decay=self.weight_decay, **kwargs)
        else:
            raise ValueError("Unknown optimizer name: %s" % self.optimizer_name)

        return optimizer

    def _get_scheduler(self, optimizer, **kwargs):
        if self.scheduler_name is None:
            return None

        elif self.scheduler_name == 'CyclicLR':
            self.base_lr = self.kwargs.get("base_lr", 1e-5)
            self.max_lr = self.kwargs.get("max_lr", 1e-4)
            scheduler = optim.lr_scheduler.CyclicLR(optimizer, self.base_lr, self.max_lr,
                                                    mode='exp_range', gamma=0.99999, cycle_momentum=False)
        elif self.scheduler_name == 'OneCycleLR':
            self.max_lr = self.kwargs.get("max_lr", 1e-4)
            scheduler = optim.lr_scheduler.OneCycleLR(optimizer, self.max_lr, **kwargs)
        else:
            raise NotImplementedError

        return scheduler

    def train(self, train_dataset, valid_dataset=None, epochs=100, display_step=1, device='cuda:0', **kwargs):
        training_iters = len(train_dataset)
        batch_size = self.kwargs.get("batch_size", 1)
        num_workers = self.kwargs.get("num_workers", 4)
        self.logger.info("------ Number of training iterations per epoch %s ------" % math.ceil(training_iters / batch_size))

        train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
        self.logger.info("------ Start optimization with criterion {:}, optimizer {:}, lr {:.2e}, epochs {:}, "
                         "weight_decay {:.2e} ------".format(self.criterion_name, self.optimizer_name, self.lr, epochs,
                                                             self.weight_decay))

        train_metrics = {"Loss": {}}
        for k in self.metrics.keys():
            train_metrics[k] = {}

        for epoch in range(epochs):
            running_loss = 0.
            for step, data in enumerate(train_loader):
                inputs, labels = data

                self.optimizer.zero_grad()

                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

                if step % display_step == (display_step - 1):
                    predictions = utils.get_predictor(outputs)
                    train_metrics['Loss'][step] = loss.item()
                    for k, v in self.metrics.items():
                        train_metrics[k][step] = v(labels, predictions).item()

                    self.output_minibatch_stats(epoch, step, train_metrics)

            self.logger.info("[Train] Epoch: {:}, Average Loss: {:.4f}, "
                             "Learning Rate: {:.1e}".format(epoch, running_loss / training_iters, self.lr))
