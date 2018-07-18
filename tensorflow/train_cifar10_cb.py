import logging
from datetime import datetime

from experiments.run_context import RunContext
from datasets import Cifar10ZCA
from mean_teacher.cb_model import CBModel
from mean_teacher import minibatching

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger('main')

def run(data_seed=0):
    n_labeled = 4000

    cb_model = CBModel(RunContext(__file__, 0))
    cb_model['h_flip'] = True
    cb_model['normalize_input'] = False
    cb_model['rampdown_steps'] = 0
    cb_model['rampup_steps'] = 5000
    cb_model['training_steps'] = 40000
    cb_model['max_consistency_cost'] = 50.0

    tensorboard_dir = model.save_tensorboard_graph()
    LOG.info("Saved tensorboard graph to %r", tensorboard_dir)

    cifar = Cifar10ZCA(data_seed, n_labeled)
    training_batches = minibatching.training_batches(cifar.training, n_labeled_per_batch=50)
    evaluation_batches_fn = minibatching.evaluation_epoch_generator(cifar.evaluation)

    cb_model.train(train_batches, evaluation_batches_fn)

if __name__ == '__main__':
    run()