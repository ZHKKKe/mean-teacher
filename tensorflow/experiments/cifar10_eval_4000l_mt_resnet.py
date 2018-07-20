# Copyright (c) 2018, Curious AI Ltd. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""CIFAR-10 final evaluation"""

import sys
import logging

from experiments.run_context import RunContext
import tensorflow as tf

from datasets import Cifar10ZCA
from mean_teacher.model_sgd import Model
from mean_teacher import minibatching

LOG = logging.getLogger('main')


def parameters():
    yield {
        'test_phase': True,
        'model_type': 'mean_teacher',
        'n_labeled': 4000,
        'data_seed': 2000
    }


def model_hyperparameters(model_type, n_labeled):
    return {
        'minibatch_size': 128,
        'n_labeled_per_batch': 31,
    }    

def run(test_phase, n_labeled, data_seed, model_type):
    hyperparams = model_hyperparameters(model_type, n_labeled)

    tf.reset_default_graph()
    model = Model(RunContext(__file__, data_seed))

    cifar = Cifar10ZCA(
        n_labeled=n_labeled, data_seed=data_seed, test_phase=test_phase)

    model['flip_horizontally'] = True
    model['normalize_input'] = False  # Keep ZCA information
    model['max_consistency_cost'] = 100.0

    model['ema_consistency'] = True
    model['apply_consistency_to_labeled'] = True
        
    # model['adam_beta_2_during_rampup'] = 0.999
    model['ema_decay_during_rampup'] = 0.999
    model['rampdown_length'] = 25000
    model['training_length'] = 150000
    model['ema_decay_during_rampup'] = 0.99
    model['ema_decay_after_rampup'] = 0.99
    model['max_learning_rate'] = 0.05

    model['rampup_length'] = 2500
    model['training_length'] = 150000
    model['lr_down_iters'] = 175000

    model['print_span'] = 50
    model['evaluation_span'] = 474

    training_batches = minibatching.training_batches(
        cifar.training, hyperparams['minibatch_size'], hyperparams['n_labeled_per_batch'])
    evaluation_batches_fn = minibatching.evaluation_epoch_generator(
        cifar.evaluation, hyperparams['minibatch_size'])

    tensorboard_dir = model.save_tensorboard_graph()
    LOG.info("Saved tensorboard graph to %r", tensorboard_dir)

    model.train(training_batches, evaluation_batches_fn)


if __name__ == "__main__":
    for run_params in parameters():
        run(**run_params)
