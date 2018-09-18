# Copyright (c) 2018, Curious AI Ltd. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""Train CIFAR-10 with 1000 or 4000 labels and all training images. Evaluate against test set."""

import sys
import logging

import torch

import main_mean_teacher
from mean_teacher.cli import parse_dict_args
from mean_teacher.run_context import RunContext

LOG = logging.getLogger('runner')


def parameters():
    defaults = {
        # Technical details
        'workers': 2,
        'checkpoint_epochs': 20,

        # Data
        'dataset': 'cifar100',
        'train_subdir': 'train+val',
        'eval_subdir': 'test',

        # Data sampling
        'base_batch_size': 128,
        'base_labeled_batch_size': 31,

        # Architecture
        'arch': 'cifar_cnn13',

        # Costs
        'consistency_type': 'mse',
        'consistency_rampup': 5,
        'consistency': 100.0,
        'logit_distance_cost': 0.01,
        'weight_decay': 2e-4,

        # Optimization
        'lr_rampup': 0,
        'base_lr': 0.2,     # lr: cifar100 = cifar10 * 2
        'nesterov': True,
    }

    # 10000 labels:
    for data_seed in range(10, 11):
        yield {
            **defaults,
            'title': '10000-label cifar-100 mt',
            'n_labels': 10000,
            'data_seed': data_seed,
            'epochs': 300,
            'lr_rampdown_epochs': 350,
            'ema_decay': 0.97,
        }


def run(title, base_batch_size, base_labeled_batch_size, base_lr, n_labels,
        data_seed, **kwargs):
    LOG.info('run title: %s, data seed: %d', title, data_seed)

    ngpu = torch.cuda.device_count()
    assert ngpu > 0, "Expecting at least one GPU, found none."

    adapted_args = {
        'batch_size':
        base_batch_size * ngpu,
        'labeled_batch_size':
        base_labeled_batch_size * ngpu,
        'lr':
        base_lr * ngpu,
        'labels':
        'data-local/labels/cifar100/{}_balanced_labels/{:02d}.txt'.format(
            n_labels, data_seed),
    }
    context = RunContext(__file__, "{}_{}".format(n_labels, data_seed))
    main_mean_teacher.args = parse_dict_args(**adapted_args, **kwargs)
    main_mean_teacher.main(context)


if __name__ == "__main__":
    for run_params in parameters():
        run(**run_params)
