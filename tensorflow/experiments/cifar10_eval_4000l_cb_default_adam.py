import sys
import logging

from experiments.run_context import RunContext
import tensorflow as tf

from datasets import Cifar10ZCA
from mean_teacher.cb_model import CBModel
from mean_teacher import minibatching

LOG = logging.getLogger('main')
# fh = logging.FileHandler('/home/sensetime/Desktop/cb_log.log')
# LOG.addHandler(fh)


def parameters():
    yield {
        'test_phase': True,
        'model_type': 'competitive-buddy',
        'n_labeled': 4000,
        'data_seed': 2000
    }

def model_hyperparameters(model_type, n_labeled):
    return {
        'n_labeled_per_batch': 'vary',
        'max_consistency_cost': 100.0 * n_labeled / 50000,
        'labeled_consistency': True,
        'ema_loss': True,
        'ema_scale': 0.5,
    }

def run(test_phase, n_labeled, data_seed, model_type):
    minibatch_size = 100
    hyperparams = model_hyperparameters(model_type, n_labeled)

    tf.reset_default_graph()
    cb_model = CBModel(RunContext(__file__, data_seed))
    cifar = Cifar10ZCA(n_labeled=n_labeled, data_seed=data_seed, test_phase=test_phase)

    cb_model['h_flip'] = True
    cb_model['ema_loss'] = hyperparams['ema_loss']
    cb_model['ema_scale'] = hyperparams['ema_scale']
    cb_model['max_consistency_cost'] = hyperparams['max_consistency_cost']
    cb_model['labeled_consistency'] = hyperparams['labeled_consistency']

    # Default Adam Setting-------------------------------------------------
    cb_model['adam_beta2_during_rampup'] = 0.999
    cb_model['adam_beta1_after_rampdown'] = 0.9
    cb_model['max_lr'] = 0.001
    # ---------------------------------------------------------------------

    cb_model['normalize_input'] = False  # Keep ZCA information
    cb_model['rampdown_steps'] = 25000
    cb_model['training_steps'] = 150000

    training_batches = minibatching.training_batches(
        cifar.training, minibatch_size, hyperparams['n_labeled_per_batch'])
    evaluation_batches_fn = minibatching.evaluation_epoch_generator(
        cifar.evaluation, minibatch_size)

    tensorboard_dir = cb_model.save_tensorboard_graph()
    LOG.info("Saved tensorboard graph to %r", tensorboard_dir)

    cb_model.train(training_batches, evaluation_batches_fn)


if __name__ == "__main__":
    for run_params in parameters():
        run(**run_params)
