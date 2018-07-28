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
        'base_batch_size': 100,
        'base_labeled_batch_size': 50,

        # Architecture
        'arch': 'cifar_cnn13_k2',

        # Costs
        'consistency_type': 'mse',
        'consistency_rampup': 5,
        'consistency': 50.0,
        'logit_distance_cost': -1,
        'weight_decay': 1e-4,

        # Optimization
        'lr_rampup': 0,
        'base_lr': 0.1,
        'nesterov': True,

        'disc_lr': 0.0001,
        'gen_lr': 0.001,
        # 'disc_lr_rampup': 5,
        'train_generator': True,
        
        'same_net_init': True,
        'reverse_fake': False,

        # EMA loss competition
        'ema_loss': 0.5,
        'epoch_init_ema_loss': True,
        # 'ema_decay': 0.97,

        'draw_curve': True,
    }

    # 4000 labels:
    for data_seed in range(10, 11):
        yield {
            **defaults,
            'title': '4000-label cifar10 cb',
            'n_labels': 4000,
            'data_seed': data_seed,
            'epochs': 180,
            'lr_rampdown_epochs': 210,
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