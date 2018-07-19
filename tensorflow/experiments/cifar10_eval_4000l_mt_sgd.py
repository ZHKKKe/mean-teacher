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
        'n_labeled_per_batch': 'vary',
        'max_consistency_cost': 100.0 * n_labeled / 50000,
        'apply_consistency_to_labeled': True,
        'ema_consistency': model_type == 'mean_teacher'
    }    

def run(test_phase, n_labeled, data_seed, model_type):
    minibatch_size = 100
    hyperparams = model_hyperparameters(model_type, n_labeled)

    tf.reset_default_graph()
    model = Model(RunContext(__file__, data_seed))

    cifar = Cifar10ZCA(
        n_labeled=n_labeled, data_seed=data_seed, test_phase=test_phase)

    model['flip_horizontally'] = True
    model['ema_consistency'] = hyperparams['ema_consistency']
    model['max_consistency_cost'] = hyperparams['max_consistency_cost']
    model['apply_consistency_to_labeled'] = hyperparams[
        'apply_consistency_to_labeled']
        
    # model['adam_beta_2_during_rampup'] = 0.999
    model['ema_decay_during_rampup'] = 0.999
    model['normalize_input'] = False  # Keep ZCA information
    model['rampdown_length'] = 25000
    model['training_length'] = 150000

    training_batches = minibatching.training_batches(
        cifar.training, minibatch_size, hyperparams['n_labeled_per_batch'])
    evaluation_batches_fn = minibatching.evaluation_epoch_generator(
        cifar.evaluation, minibatch_size)

    tensorboard_dir = model.save_tensorboard_graph()
    LOG.info("Saved tensorboard graph to %r", tensorboard_dir)

    model.train(training_batches, evaluation_batches_fn)


if __name__ == "__main__":
    for run_params in parameters():
        run(**run_params)
