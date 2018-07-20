import sys
import logging

from experiments.run_context import RunContext
import tensorflow as tf

from datasets import Cifar10ZCA
from mean_teacher.cb_model_resnet import CBModel
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
        'minibatch_size': 128,
        'n_labeled_per_batch': 31,
    }

def run(test_phase, n_labeled, data_seed, model_type):
    hyperparams = model_hyperparameters(model_type, n_labeled)

    tf.reset_default_graph()
    cb_model = CBModel(RunContext(__file__, data_seed))
    cifar = Cifar10ZCA(n_labeled=n_labeled, data_seed=data_seed, test_phase=test_phase)

    cb_model['h_flip'] = True
    cb_model['normalize_input'] = False  # Keep ZCA information
    
    cb_model['dropout_prob'] = 0.0

    cb_model['max_consistency_cost'] = 100.0
    cb_model['line_up_cons'] = False

    cb_model['ema_loss'] = True
    cb_model['ema_scale'] = 0.5
    cb_model['ema_init_batches'] = 474
    cb_model['epoch_ema_init'] = False

    cb_model['lr_down_iters'] = 175000
    cb_model['rampup_steps'] = 2500
    cb_model['training_steps'] = 150000

    cb_model['print_span'] = 50
    cb_model['evaluation_span'] = 474

    training_batches = minibatching.training_batches(
        cifar.training, hyperparams['minibatch_size'], hyperparams['n_labeled_per_batch'])
    evaluation_batches_fn = minibatching.evaluation_epoch_generator(
        cifar.evaluation, hyperparams['minibatch_size'])

    tensorboard_dir = cb_model.save_tensorboard_graph()
    LOG.info("Saved tensorboard graph to %r", tensorboard_dir)

    cb_model.train(training_batches, evaluation_batches_fn)


if __name__ == "__main__":
    for run_params in parameters():
        run(**run_params)
