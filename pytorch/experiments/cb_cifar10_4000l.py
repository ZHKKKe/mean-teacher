import sys
import logging

import torch

import main_compite_buddy
from mean_teacher.cli import parse_dict_args
from mean_teacher.run_context import RunContext

LOG = logging.getLogger('runner')

def parameters():
    defaults = {
        # Technical details
        'workers': 2,
        'checkpoint_epochs': 20,

        # Data
        'dataset': 'cifar10',
        'train_subdir': 'train+val',
        'eval_subdir': 'test',

        # Data sampling
        'base_batch_size': 128,
        'base_labeled_batch_size': 31,

        # Architecture
        'arch': 'cifar_shakeshake26',

        # Costs
        'consistency_type': 'mse',
        'consistency_rampup': 10,
        'consistency': 100.0,
        'logit_distance_cost': -1,
        'weight_decay': 2e-4,

        # Optimization
        'lr_rampup': 0,
        'base_lr': 0.05,
        'nesterov': True,
    }

    # 4000 labels:
    for data_seed in range(10, 11):
        yield {
            **defaults,
            'title': '4000-label cifar10 cb',
            'n_labels': 4000,
            'data_seed': data_seed,
            'epochs': 300,
            'lr_rampdown_epochs': 350,
        }

def run(title, base_batch_size, base_labeled_batch_size, base_lr, n_labels, data_seed, **kwargs):
    LOG.info('run title: %s, data seed: %d', title, data_seed)
    ngpu = torch.cuda.device_count()
    assert ngpu > 0, "Expecting at least one GPU, found none."
    adapted_args = {
        'batch_size': base_batch_size * ngpu,
        'labeled_batch_size': base_labeled_batch_size * ngpu,
        'lr': base_lr * ngpu,
        'labels': 'data-local/labels/cifar10/{}_balanced_labels/{:02d}.txt'.format(n_labels, data_seed),
    }
    context = RunContext(__file__, "{}_{}".format(n_labels, data_seed))
    main_compite_buddy.args = parse_dict_args(**adapted_args, **kwargs)
    main_compite_buddy.main(context)

if __name__ == '__main__':
    for run_params in parameters():
        run(**run_params)