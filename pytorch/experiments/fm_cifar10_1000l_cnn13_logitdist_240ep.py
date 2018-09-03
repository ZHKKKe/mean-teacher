import sys
import logging

import torch

import main_four_models
from mean_teacher.cli import parse_dict_args
from mean_teacher.run_context import RunContext

LOG = logging.getLogger('main')
fh = logging.FileHandler('fm_cifar10_1000l_cnn13_logitdist_240ep.log')
fh.setLevel(logging.INFO)
LOG.addHandler(fh)

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
        'arch': 'cifar_cnn13',

        # Costs
        'consistency_type': 'mse',
        'consistency_rampup': 5,
        'consistency': 100.0,
        'logit_distance_cost': .01,
        'weight_decay': 1e-4,

        # Optimization
        'lr_rampup': 0,
        'base_lr': 0.05,
        'nesterov': True,

        # EMA loss competition
        'ema_loss': 0.5,
        'epoch_init_ema_loss': False, 

        'ema_decay': 0.97,
        'ema_model_judge': True,

        'as_co_train_lr': True,

        'draw_curve': False,
        
        'logits_disc': False,
        'disc_lr': 0.0, 
    }

    # 4000 labels:
    for data_seed in range(10, 11):
        yield {
            **defaults,
            'title': '1000-label cifar10 fm',
            'n_labels': 1000,
            'data_seed': data_seed,
            'epochs': 240,
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
    main_four_models.args = parse_dict_args(**adapted_args, **kwargs)
    main_four_models.main(context)

if __name__ == '__main__':
    for run_params in parameters():
        run(**run_params)