"Competitive buddy model"

import os
import logging
from collections import namedtuple

import tensorflow as tf
from tensorflow.contrib import slim, metrics
from tensorflow.contrib.metrics import streaming_mean

from . import nn
from . import weight_norm as wn
from . import string_utils
from .framework import *

LOG = logging.getLogger('main')
Hyperparam = namedtuple("Hyperparam", ['tensor', 'getter', 'setter'])


def train_control(global_step, print_span, evaluation_span, max_step, name=None):
    with tf.name_scope(name, "train_control"):
        return {
            "step": global_step,
            "time_to_print": tf.equal(tf.mod(global_step, print_span), 0),
            "time_to_evaluate": tf.equal(tf.mod(global_step, evaluation_span), 0),
            "time_to_stop": tf.greater_equal(global_step, max_step),
        }

def line_rampup(global_step, rampup_steps):
    def ramp():
        phase = tf.maximum(0.0, global_step) / tf.to_float(rampup_steps)
        return phase

    global_step = tf.to_float(global_step)
    rampup_steps = tf.to_float(rampup_steps)

    result = tf.cond(global_step < rampup_steps, ramp, lambda: tf.constant(1.0))
    return tf.identity(result, name='line_rampup')

def step_rampup(global_step, rampup_length):
    result = tf.cond(global_step < rampup_length, lambda: tf.constant(0.0),
                     lambda: tf.constant(1.0))
    return tf.identity(result, name="step_rampup")


def sigmoid_rampup(global_step, rampup_length):
    def ramp():
        phase = 1.0 - tf.maximum(0.0, global_step) / rampup_length
        return tf.exp(-5.0 * phase * phase)

    global_step = tf.to_float(global_step)
    rampup_length = tf.to_float(rampup_length)

    result = tf.cond(global_step < rampup_length, ramp, lambda: tf.constant(1.0))
    return tf.identity(result, name="sigmoid_rampup")


def sigmoid_rampdown(global_step, rampdown_length, training_length):
    def ramp():
        phase = 1.0 - tf.maximum(0.0, training_length - global_step) / rampdown_length
        return tf.exp(-12.5 * phase * phase)

    global_step = tf.to_float(global_step)
    rampdown_length = tf.to_float(rampdown_length)
    training_length = tf.to_float(training_length)

    result = tf.cond(global_step >= training_length - rampdown_length,
                     ramp, lambda: tf.constant(1.0))
    return tf.identity(result, name="sigmoid_rampdown")

def cosine_rampdown(global_step, rampdown_step, epoch_step):
    def ramp():
        phase = tf.multiply(tf.constant(3.141592658), global_step / rampdown_step)
        return tf.multiply(tf.constant(0.5), tf.add(tf.cos(phase), tf.constant(1.0)))

    global_step = tf.to_float(global_step)
    rampdown_step = tf.to_float(rampdown_step)

    result = tf.cond(global_step <= rampdown_step,
                     ramp, lambda:tf.constant(1.0))
    return tf.identity(result, name='cosine_rampdown')

def errors(logits, labels, name=None):
    """Compute error mean and whether each unlabeled example is erroneous

    Assume unlabeled examples have label == -1.
    Compute the mean error over unlabeled examples.
    Mean error is NaN if there are no unlabeled examples.
    Note that unlabeled examples are treated differently in cost calculation.
    """
    with tf.name_scope(name, "errors") as scope:
        applicable = tf.not_equal(labels, -1)
        labels = tf.boolean_mask(labels, applicable)
        logits = tf.boolean_mask(logits, applicable)
        predictions = tf.argmax(logits, -1)
        labels = tf.cast(labels, tf.int64)
        per_sample = tf.to_float(tf.not_equal(predictions, labels))
        mean = tf.reduce_mean(per_sample, name=scope)
        return mean, per_sample


def classification_costs(logits, labels, name=None):
    """Compute classification cost mean and classification cost per sample

    Assume unlabeled examples have label == -1. For unlabeled examples, cost == 0.
    Compute the mean over all examples.
    Note that unlabeled examples are treated differently in error calculation.
    """
    with tf.name_scope(name, "classification_costs") as scope:
        applicable = tf.not_equal(labels, -1)

        # Change -1s to zeros to make cross-entropy computable
        labels = tf.where(applicable, labels, tf.zeros_like(labels))

        # This will now have incorrect values for unlabeled examples
        per_sample = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)

        # Retain costs only for labeled
        per_sample = tf.where(applicable, per_sample, tf.zeros_like(per_sample))

        # Take mean over all examples, not just labeled examples.
        labeled_sum = tf.reduce_sum(per_sample)
        total_count = tf.to_float(tf.shape(per_sample)[0])
        mean = tf.div(labeled_sum, total_count, name=scope)

        return mean, per_sample

def cb_consistency_costs(logits1, logits2, cons_scale, mask, cons_trust, name=None):
    with tf.name_scope(name, 'cb_consistency_costs') as scope:
        num_classes = 10
        assert_shape(logits1, [None, num_classes])
        assert_shape(logits2, [None, num_classes])
        assert_shape(cons_scale, [])

        softmax1 = tf.nn.softmax(logits1)
        softmax2 = tf.nn.softmax(logits2)
        softmax2_stoped = tf.stop_gradient(softmax2)
        logits2_stoped = tf.stop_gradient(logits2)

        kl_cost_multiplier = 2 * (1 - 1 / num_classes) / num_classes ** 2 / cons_trust ** 2

        def pure_mse():
            costs = tf.reduce_mean((softmax1 - softmax2_stoped) ** 2, -1)
            return costs

        def pure_kl():
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits1, labels=softmax2_stoped)
            entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits2_stoped, labels=softmax2_stoped)
            costs = cross_entropy - entropy
            costs = costs * kl_cost_multiplier
            return costs

        def mixture_kl():
            with tf.control_dependencies([tf.assert_greater(cons_trust, 0.0), tf.assert_less(cons_trust, 1.0)]):
                uniform = tf.constant(1 / num_classes, shape=[num_classes])
                mixed_softmax1 = cons_trust * softmax1 + (1 - cons_trust) * uniform
                mixed_softmax2 = cons_trust * softmax2_stoped + (1 - cons_trust) * uniform
                costs = tf.reduce_sum(mixed_softmax2 * tf.log(mixed_softmax2 / mixed_softmax1), axis=1)
                costs = costs * kl_cost_multiplier
                return costs

        costs = tf.case([(tf.equal(cons_trust, 0.0), pure_mse),
                         (tf.equal(cons_trust, 1.0), pure_kl)],
                         default=mixture_kl)

        costs = costs * tf.to_float(mask) * cons_scale
        mean_cost = tf.reduce_mean(costs, name=scope)
        assert_shape(costs, [None])
        assert_shape(mean_cost, [])
        return mean_cost, costs

def total_costs(*all_costs, name=None):
    with tf.name_scope(name, "total_costs") as scope:
        for cost in all_costs:
            assert_shape(cost, [None])
        costs = tf.reduce_sum(all_costs, axis=1)
        mean_cost = tf.reduce_mean(costs, name=scope)
        return mean_cost, costs

def tower_resnet(inputs, is_training, dropout_prob, input_noise, normalize_input,
                   h_flip, translate, num_logits, is_init=False, name=None):
    LOG.info('Building ResNet.')
    from . import resnet_model as resnet
    with tf.name_scope(name, 'tower_resnet'):
        training_args = dict(is_training=is_training)
        training_mode_funcs = [nn.random_translate, nn.flip_randomly, nn.gaussian_noise,
                               slim.dropout, wn.fully_connected, wn.conv2d]

        with slim.arg_scope(training_mode_funcs, **training_args):
            x = inputs
            x = tf.cond(normalize_input, lambda: slim.layer_norm(x, scale=False, center=False, scope='normalize_inputs'), lambda: x)
            assert_shape(x, [None, 32, 32, 3])

            x = nn.flip_randomly(x, horizontally=h_flip, vertically=False, name='random_flip')
            x = tf.cond(translate, lambda: nn.random_translate(x, scale=2, name='random_translate'), lambda: x)
            x = nn.gaussian_noise(x, scale=input_noise, name='gaussian_noise')

            logits_1 = resnet.inference(x, 4, reuse=False)
            logits_2 = logits_1
            return logits_1, logits_2


def tower(inputs, is_training, dropout_prob, input_noise, normalize_input,
          h_flip, translate, num_logits, is_init=False, name=None):
    with tf.name_scope(name, "tower"):
        training_args = dict(is_training=is_training)
        default_conv_args = dict(padding='SAME', kernel_size=[3, 3], activation_fn=nn.lrelu, init=is_init)

        training_mode_funcs = [nn.random_translate, nn.flip_randomly, nn.gaussian_noise,
                               slim.dropout, wn.fully_connected, wn.conv2d]

        with slim.arg_scope([wn.conv2d], **default_conv_args), slim.arg_scope(training_mode_funcs, **training_args):
            x = inputs
            x = tf.cond(normalize_input, lambda: slim.layer_norm(x, scale=False, center=False, scope='normalize_inputs'), lambda: x)
            assert_shape(x, [None, 32, 32, 3])

            x = nn.flip_randomly(x, horizontally=h_flip, vertically=False, name='random_flip')
            x = tf.cond(translate, lambda: nn.random_translate(x, scale=2, name='random_translate'), lambda: x)
            x = nn.gaussian_noise(x, scale=input_noise, name='gaussian_noise')

            x = wn.conv2d(x, 128, scope='conv_1_1')
            x = wn.conv2d(x, 128, scope='conv_1_2')
            x = wn.conv2d(x, 128, scope='conv_1_3')
            x = slim.max_pool2d(x, [2, 2], scope='max_pool_1')
            x = slim.dropout(x, 1 - dropout_prob, scope='dropout_prob_1')
            assert_shape(x, [None, 16, 16, 128])

            x = wn.conv2d(x, 256, scope='conv_2_1')
            x = wn.conv2d(x, 256, scope='conv_2_2')
            x = wn.conv2d(x, 256, scope='conv_2_3')
            x = slim.max_pool2d(x, [2, 2], scope='max_pool_2')
            x = slim.dropout(x, 1 - dropout_prob, scope='dropout_prob_2')
            assert_shape(x, [None, 8, 8, 256])

            x = wn.conv2d(x, 512, padding='VALID', scope='conv_3_1')
            x = wn.conv2d(x, 256, kernel_size=[1, 1], scope='conv_3_2')
            x = wn.conv2d(x, 128, kernel_size=[1, 1], scope='conv_3_3')
            x = slim.avg_pool2d(x, [6, 6], scope='avg_pool_1')
            assert_shape(x, [None, 1, 1, 128])

            x = slim.flatten(x)
            assert_shape(x, [None, 128])

            primary_logits = wn.fully_connected(x, 10, init=is_init)
            secondary_logits = wn.fully_connected(x, 10, init=is_init)

            with tf.control_dependencies([tf.assert_greater_equal(num_logits, 1),
                                          tf.assert_less_equal(num_logits, 2)]):
                secondary_logits = tf.case([
                    (tf.equal(num_logits, 1), lambda: primary_logits),
                    (tf.equal(num_logits, 2), lambda: secondary_logits)
                    ], exclusive=True, default=lambda: primary_logits)

            assert_shape(primary_logits, [None, 10])
            assert_shape(secondary_logits, [None, 10])

            return primary_logits, secondary_logits

def inference(inputs, is_training, input_noise, dropout_prob, normalize_input, h_flip, translate, num_logits):
    tower_args = dict(inputs=inputs, is_training=is_training, input_noise=input_noise, dropout_prob=dropout_prob,
                      normalize_input=normalize_input, h_flip=h_flip, translate=translate, num_logits=num_logits)

    with tf.variable_scope('left'):
        class_logits_l, cons_logits_l = tower_resnet(**tower_args, is_init=True)
    with tf.variable_scope('right'):
        class_logits_r, cons_logits_r = tower_resnet(**tower_args, is_init=True)

    return (class_logits_l, cons_logits_l), (class_logits_r, cons_logits_r)


class CBModel:
    DEFAULT_HYPERPARAMS = {
        # Data augmentation
        'h_flip': False,
        'translate': True,

        # Whether to scale each input image to mean=0 and std=1 per channel
        # Use False if input is already normalized in some other way
        'normalize_input': True,

        # Architecture hyperparameters
        'input_noise': 0.15,
        'dropout_prob': 0.5,

        # Optimizer hyperparameters
        # 'max_lr': 0.003,
        'adam_beta1_before_rampdown': 0.9,
        'adam_beta1_after_rampdown': 0.5,
        'adam_beta2_during_rampup': 0.99,
        'adam_beta2_after_rampup': 0.999,
        # 'adam_epsilon': 1e-8,
        'max_lr': 0.05,
        'momentum': 0.9,
        'use_nesterov': True,
        'lr_down_iters': 175000,

        # Consistency hyperparameters
        'max_consistency_cost': 100.0,
        'cons_trust': 0.0,      # not use
        'labeled_consistency': True,
        'num_logits': 1,  # Either 1 or 2
        'line_up_cons': False,

        # Ema loss hyperparameters
        'ema_loss': True,
        'ema_scale': 0.5,
        'ema_max': 0.9,
        'ema_init_batches': 500,
        'ema_line_rampup': False,
        'epoch_ema_init': False,


        # Training schedule
        'rampup_steps': 50000,
        'rampdown_steps': 25000,
        'training_steps': 150000,

        # Output schedule
        'print_span': 20,
        'evaluation_span': 500,
    }

    def __init__(self, run_context=None):
        # set run context
        if run_context is not None:
            self.training_log = run_context.create_train_log('training')
            self.validation_log = run_context.create_train_log('validation')
            self.checkpoint_path = os.path.join(run_context.transient_dir, 'checkpoint')
            self.tensorboard_path = os.path.join(run_context.result_dir, 'tensorboard')

        # define place holder for inputs data batch
        with tf.name_scope('placeholders'):
            self.images = tf.placeholder(dtype=tf.float32, shape=(None, 32, 32, 3), name='images')
            self.labels = tf.placeholder(dtype=tf.int32, shape=(None, ), name='labels')
            self.is_training = tf.placeholder(dtype=tf.bool, shape=(), name='is_training')

        # define global variables
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.global_step2 = tf.Variable(0, trainable=False, name='global_step2')
        self.ema_loss_l = tf.Variable(0.0, trainable=False, name='ema_loss_l', dtype=tf.float32)
        self.ema_loss_r = tf.Variable(0.0, trainable=False, name='ema_loss_r', dtype=tf.float32)
        self.ema_loss_l_in = tf.placeholder(tf.float32, None, name='ema_loss_in_l')
        self.ema_loss_r_in = tf.placeholder(tf.float32, None, name='ema_loss_in_r')
        self.ema_loss_l_set = self.ema_loss_l.assign(self.ema_loss_l_in)
        self.ema_loss_r_set = self.ema_loss_r.assign(self.ema_loss_r_in)

        # add hyperparameters to 'init_to_init' collection
        tf.add_to_collection('init_to_init', self.global_step)
        tf.add_to_collection('init_to_init', self.global_step2)
        tf.add_to_collection('init_to_init', self.ema_loss_l)
        tf.add_to_collection('init_to_init', self.ema_loss_r)
        self.hyper = HyperparamVariables(self.DEFAULT_HYPERPARAMS)
        for var in self.hyper.variables.values():
            tf.add_to_collection('init_to_init', var)

        # calculate rampup/rampdown parameters
        with tf.name_scope('ramps'):
            step_up = step_rampup(self.global_step, self.hyper['rampup_steps'])
            sig_up = sigmoid_rampup(self.global_step, self.hyper['rampup_steps'])
            sig_down = sigmoid_rampdown(self.global_step, self.hyper['rampdown_steps'], self.hyper['training_steps'])
            line_up = line_rampup(self.global_step, self.hyper['rampup_steps'])
            cos_down = cosine_rampdown(self.global_step, self.hyper['lr_down_iters'], tf.constant(500))

            # self.learn_rate = tf.multiply(sig_up * sig_down, self.hyper['max_lr'], name='learn_rate')
            self.learn_rate = tf.multiply(cos_down, self.hyper['max_lr'], name='learn_rate')

            self.cons_scale = tf.cond(self.hyper['line_up_cons'],
                                      lambda: tf.multiply(line_up, self.hyper['max_consistency_cost'], name='cons_scale'),
                                      lambda: tf.multiply(sig_up, self.hyper['max_consistency_cost'], name='cons_scale'))

            self.adam_beta1 = tf.add(sig_down * self.hyper['adam_beta1_before_rampdown'],
                                     (1 - sig_down) * self.hyper['adam_beta1_after_rampdown'], name='adam_beta1')
            self.adam_beta2 = tf.add((1 - step_up) * self.hyper['adam_beta2_during_rampup'],
                                     step_up * self.hyper['adam_beta2_after_rampup'], name='adam_beta2')

            # add ema_loss rampup
            bias = tf.subtract(self.hyper['ema_max'], self.hyper['ema_scale'])
            up_ema_loss_scale = tf.add(self.hyper['ema_scale'], tf.multiply(line_up, bias))
            self.ema_scale = tf.cond(self.hyper['ema_line_rampup'],
                                     lambda: up_ema_loss_scale,
                                     lambda: self.hyper['ema_scale'])


        # build networks
        logits_l, logits_r = inference(inputs=self.images,
                                        is_training=self.is_training,
                                        input_noise=self.hyper['input_noise'],
                                        dropout_prob=self.hyper['dropout_prob'],
                                        normalize_input=self.hyper['normalize_input'],
                                        h_flip=self.hyper['h_flip'],
                                        translate=self.hyper['translate'],
                                        num_logits=self.hyper['num_logits'])
        self.class_logits_l, self.cons_logits_l = logits_l[0], logits_l[1]
        self.class_logits_r, self.cons_logits_r = logits_r[0], logits_r[1]

        with tf.name_scope('objectives'):
            self.mean_err_l, self.err_l = errors(self.class_logits_l, self.labels)
            self.mean_err_r, self.err_r = errors(self.class_logits_r, self.labels)

            # classify loss
            self.mean_class_loss_l, self.class_loss_l = \
                classification_costs(self.class_logits_l, self.labels)
            self.mean_class_loss_r, self.class_loss_r = \
                classification_costs(self.class_logits_r, self.labels)

            # consistency loss
            labeled_cons = self.hyper['labeled_consistency']
            cons_mask = tf.logical_or(tf.equal(self.labels, -1), labeled_cons)

            self.mean_cons_loss_l, self.cons_loss_l = cb_consistency_costs(self.cons_logits_l, self.class_logits_r,
                self.cons_scale, cons_mask, self.hyper['cons_trust'], name='cons_loss_l')
            self.mean_cons_loss_r, self.cons_loss_r = cb_consistency_costs(self.cons_logits_r, self.class_logits_l,
                self.cons_scale, cons_mask, self.hyper['cons_trust'], name='cons_loss_r')

            # update ema loss
            self.ema_loss_l = tf.cond(self.hyper['ema_loss'],
                                      lambda: tf.add(tf.multiply(self.ema_scale, self.ema_loss_l),
                                       tf.multiply(1 - self.ema_scale, self.mean_class_loss_l)),
                                       lambda: self.mean_class_loss_l)

            self.ema_loss_r = tf.cond(self.hyper['ema_loss'],
                                    lambda: tf.add(tf.multiply(self.ema_scale, self.ema_loss_r),
                                       tf.multiply(1 - self.ema_scale, self.mean_class_loss_r)),
                                      lambda: self.mean_class_loss_r)

            # total loss
            self.mean_loss_l, self.loss_l = tf.cond(
                tf.greater(self.ema_loss_l, self.ema_loss_r),
                lambda: total_costs(self.class_loss_l, self.cons_loss_l),
                lambda: total_costs(self.class_loss_l, tf.zeros(tf.shape(self.cons_loss_l)))
            )

            self.mean_loss_r, self.loss_r = tf.cond(
                tf.less(self.ema_loss_l, self.ema_loss_r),
                lambda: total_costs(self.class_loss_r, self.cons_loss_r),
                lambda: total_costs(self.class_loss_r, tf.zeros(tf.shape(self.cons_loss_r)))
            )


        with tf.name_scope('train_step'):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                LOG.info('SGD_CB_MODEL')
                # Note: just update self.global_step in l_model
                # self.train_step_op_l = nn.adam_optimizer(self.mean_loss_l,
                #                                          self.global_step,
                #                                          learning_rate=self.learn_rate,
                #                                          beta1=self.adam_beta1,
                #                                          beta2=self.adam_beta2,
                #                                          epsilon=self.hyper['adam_epsilon'])
                # self.train_step_op_r = nn.adam_optimizer(self.mean_loss_r,
                #                                          self.global_step2,
                #                                          learning_rate=self.learn_rate,
                #                                          beta1=self.adam_beta1,
                #                                          beta2=self.adam_beta2,
                #                                          epsilon=self.hyper['adam_epsilon'])
                self.train_step_op_l = nn.sgd_optimizer(self.mean_loss_l,
                                                        self.global_step,
                                                        learn_rate=self.learn_rate,
                                                        momentum=self.hyper['momentum'],
                                                        use_nesterov=True)
                self.train_step_op_r = nn.sgd_optimizer(self.mean_loss_r,
                                                        self.global_step2,
                                                        learn_rate=self.learn_rate,
                                                        momentum=self.hyper['momentum'],
                                                        use_nesterov=True)

        self.train_control = train_control(self.global_step, self.hyper['print_span'],
                                           self.hyper['evaluation_span'], self.hyper['training_steps'])

        self.train_metrics = {
            'learn_rate': self.learn_rate,
            'adam_beta1': self.adam_beta1,
            'adam_beta2': self.adam_beta2,
            'cons_scale': self.cons_scale,
            'error/l': self.mean_err_l,
            'error/r': self.mean_err_r,
            'class_loss/l': self.mean_class_loss_l,
            'class_loss/r': self.mean_class_loss_r,
            'cons_loss/l': self.mean_cons_loss_l,
            'cons_loss/r': self.mean_cons_loss_r,
            'ema_loss/l': self.ema_loss_l,
            'ema_loss/r': self.ema_loss_r,
            'total_loss/l': self.mean_loss_l,
            'total_loss/r': self.mean_loss_r,
        }

        with tf.variable_scope('validation_metrics') as metrics_scope:
            self.metric_values, self.metric_update_ops = metrics.aggregate_metric_map({
                'eval/error/l': streaming_mean(self.err_l),
                'eval/error/r': streaming_mean(self.err_r),
                'eval/class_loss/l': streaming_mean(self.class_loss_l),
                'eval/class_loss/r': streaming_mean(self.class_loss_r),
                # 'eval/cons_loss/l': streaming_mean(self.cons_loss_l),
                # 'eval/cons_loss/r': streaming_mean(self.cons_loss_r),
                # 'eval/ema_loss/l': self.ema_loss_l,
                # 'eval/ema_loss/r': self.ema_loss_r,
            })
            metric_variables = slim.get_local_variables(scope=metrics_scope.name)
            self.metric_init_op = tf.variables_initializer(metric_variables)

        self.result_formatter = string_utils.DictFormatter(
            order=['error/l', 'error/r', 'class_loss/l', 'class_loss/r',
                   'cons_loss/l', 'cons_loss/r', 'total_loss/l', 'total_loss/r', 'ema_loss/l', 'ema_loss/r'],
            default_format='{name}: {value:>8.6f}',
            separator=', ')
        self.result_formatter.add_format('error', '{name}: {value:>6.3%}')

        with tf.variable_scope('init_ema_loss') as ema_scope:
            self.ema_loss_values, self.init_ema_loss_op = metrics.aggregate_metric_map({
                'ema_loss_l': streaming_mean(self.mean_class_loss_l),
                'ema_loss_r': streaming_mean(self.mean_class_loss_r),
            })
            ema_variables = slim.get_local_variables(scope=ema_scope.name)
            self.ema_loss_init_op = tf.variables_initializer(ema_variables)
        self.ema_init_formatter = string_utils.DictFormatter(
            order=['init/ema_loss/l', 'init/ema_loss/r'],
            default_format='{name}: {value:>10.6f}',
            separator=', ')

        with tf.name_scope('initializers'):
            init_init_variables = tf.get_collection('init_to_init')
            train_init_variables = [var for var in tf.global_variables() if var not in init_init_variables]
            self.init_init_op = tf.variables_initializer(init_init_variables)
            self.train_init_op = tf.variables_initializer(train_init_variables)

        self.saver = tf.train.Saver()
        self.session = tf.Session()
        self.run(self.init_init_op)

    def __setitem__(self, key, value):
        self.hyper.assign(self.session, key, value)

    def __getitem__(self, key):
        return self.hyper.get(self.session, key)

    def get_train_control(self):
        return self.session.run(self.train_control)

    def run(self, *args, **kwargs):
        return self.session.run(*args, **kwargs)

    def save_checkpoint(self):
        path = self.saver.save(
            self.session, self.checkpoint_path, global_step=self.global_step)
        LOG.info("Saved checkpoint: %r", path)

    def save_tensorboard_graph(self):
        writer = tf.summary.FileWriter(self.tensorboard_path)
        writer.add_graph(self.session.graph)
        return writer.get_logdir()

    def feed_dict(self, batch, is_training=True):
        return {
            self.images: batch['x'],
            self.labels: batch['y'],
            self.is_training: is_training
        }

    def train(self, train_batches, eval_batches_fn):
        self.run(self.train_init_op, self.feed_dict(next(train_batches)))
        LOG.info('Model variables initialized')
        self.evaluate(eval_batches_fn)
        self.save_checkpoint()

        # init ema_loss
        # self.init_ema_loss(train_batches, init_batches=self.run(self.hyper['ema_init_batches']))
        self.run(self.ema_loss_init_op)

        init_idx = 0
        is_init_ema = True
        init_batches = self.run(self.hyper['ema_init_batches'])
        epoch_ema_init = self.run(self.hyper['epoch_ema_init'])

        for batch in train_batches:
            if is_init_ema:
                if init_idx >= init_batches:
                    results = self.run(self.ema_loss_values)
                    self.run(self.ema_loss_l_set, {self.ema_loss_l_in: results['ema_loss_l']})
                    self.run(self.ema_loss_r_set, {self.ema_loss_r_in: results['ema_loss_r']})

                    init_idx = 0
                    is_init_ema = False
                    LOG.info('Finish ema loss init.')
                else:
                    if init_idx == 0:
                        LOG.info('Init ema loss...')

                    self.run(self.init_ema_loss_op, feed_dict=self.feed_dict(batch, is_training=False))
                    init_idx += 1

                    if init_idx % self.run(self.hyper['print_span']) == 0:
                        results = self.run(self.ema_loss_values)
                        LOG.info('idx: %d , ema_loss_l: %f , ema_loss_r: %f' % (init_idx, results['ema_loss_l'], results['ema_loss_r']))

            if not is_init_ema:
                results, _, _ = self.run(
                    [self.train_metrics, self.train_step_op_l, self.train_step_op_r], self.feed_dict(batch))

                step_control = self.get_train_control()
                self.training_log.record(step_control['step'], {**results, **step_control})

                if step_control['time_to_print']:
                    LOG.info("step %5d: %s", step_control['step'], self.result_formatter.format_dict(results))
                if step_control['time_to_evaluate']:
                    self.evaluate(eval_batches_fn)
                    self.save_checkpoint()
                    if epoch_ema_init:
                        is_init_ema = True
                if step_control['time_to_stop']:
                    break

        self.evaluate(eval_batches_fn)
        self.save_checkpoint()


    def init_ema_loss(self, train_batches, init_batches=500):
        LOG.info('Calculate init ema_loss...')
        self.run(self.ema_loss_init_op)

        idx = 0
        for batch in train_batches:
            if idx >= init_batches:
                break
            self.run(self.init_ema_loss_op, feed_dict=self.feed_dict(batch, is_training=False))

            if idx % self.run(self.hyper['print_span']) == 0:
                results = self.run(self.ema_loss_values)
                LOG.info('idx: %d , ema_loss_l: %f , ema_loss_r: %f' % (idx, results['ema_loss_l'], results['ema_loss_r']))
            idx += 1

        results = self.run(self.ema_loss_values)
        # self.ema_loss_l.assign(results['ema_loss_l'])
        # self.ema_loss_r.assign(results['ema_loss_r'])
        self.run(self.ema_loss_l_set, {self.ema_loss_l_in: results['ema_loss_l']})
        self.run(self.ema_loss_r_set, {self.ema_loss_r_in: results['ema_loss_r']})

        LOG.info('init ema loss. %s', self.ema_init_formatter.format_dict(results))

    def evaluate(self, eval_batches_fn):
        self.run(self.metric_init_op)
        for batch in eval_batches_fn():
            self.run(self.metric_update_ops, feed_dict=self.feed_dict(batch, is_training=False))
        step = self.run(self.global_step)
        results = self.run(self.metric_values)
        self.validation_log.record(step, results)
        LOG.info("step %5d: %s", step, self.result_formatter.format_dict(results))
