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
    num_classes = 10
    with tf.name_scope(name, 'cb_consistency_costs') as scope:
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
        class_logits_l, cons_logits_l = tower(**tower_args, is_init=True)
    with tf.variable_scope('right'):
        class_logits_r, cons_logits_r = tower(**tower_args, is_init=True)

    return (class_logits_l, cons_logits_l), (class_logits_r, cons_logits_r)


class CBModel:
    DEFAULT_HYPERPARAMS = {
        # Data augmentation
        'h_flip': False,
        'translate': True,

        # Architecture hyperparameters
        'input_noise': 0.15,
        'dropout_prob': 0.5,

        # Optimizer hyperparameters
        'max_lr': 0.003,
        'adam_beta1_before_rampdown': 0.9,
        'adam_beta1_after_rampdown': 0.5,
        'adam_beta2_during_rampup': 0.99,
        'adam_beta2_after_rampup': 0.999,
        'adam_epsilon': 1e-8,

        # Consistency hyperparameters
        'max_consistency_cost': 100.0,
        'labeled_consistency': True,
        'num_logits': 1,  # Either 1 or 2

        # Whether to scale each input image to mean=0 and std=1 per channel
        # Use False if input is already normalized in some other way
        'normalize_input': True,

        # Training schedule
        'rampup_steps': 40000,
        'rampdown_steps': 25000,
        'training_steps': 150000,

        # Output schedule
        'print_span': 20,
        'evaluation_span': 500,
    }

    def __init__(self, run_context=None):
        if run_context is not None:
            self.training_log = run_context.create_train_log('training')
            self.validation_log = run_context.create_train_log('validation')
            self.checkpoint_path = os.path.join(run_context.transient_dir, 'checkpoint')
            self.tensorboard_path = os.path.join(run_context.result_dir, 'tensorboard')

        with tf.name_scope('placeholders'):
            self.images = tf.placeholder(dtype=tf.float32, shape=(None, 32, 32, 3), name='images')
            self.labels = tf.placeholder(dtype=tf.float32, shape=(None, ), name='labels')
            self.is_training = tf.placeholder(dtype=tf.bool, shape=(), name='is_training')

        # define global variables
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.ema_loss_l = tf.Variable(0, trainable=False, name='ema_loss_l')
        self.ema_loss_r = tf.Variable(0, trainable=False, name='ema_loss_r')
        tf.add_to_collection('init_to_init', self.global_step)
        tf.add_to_collection('init_to_init', self.ema_loss_l)
        tf.add_to_collection('init_to_init', self.ema_loss_r)

        self.hyper = HyperparamVariables(self.DEFAULT_HYPERPARAMS)
        for var in self.hyper.variables.values():
            tf.add_to_collection('init_to_init', var)

        # define rampup/rampdown parameters
        with tf.name_scope('ramps'):
            step_up = step_rampup(self.global_step, self.hyper['rampup_steps'])
            sig_up = sigmoid_rampup(self.global_step, self.hyper['rampup_steps'])
            sig_down = sigmoid_rampdown(self.global_step, self.hyper['rampdown_steps'], self.hyper['training_steps'])

            self.learn_rate = tf.multiply(sig_up * sig_down, self.hyper['max_lr'], name='learn_rate')
            self.cons_scale = tf.multiply(sig_up, self.hyper['max_consistency_cost'], name='cons_scale')
            self.adam_beta1 = tf.add(sig_down * self.hyper['adam_beta1_before_rampdown'],
                                     (1 - sig_down) * self.hyper['adam_beta1_after_rampdown'], name='adam_beta1')
            self.adam_beta2 = tf.add((1 - step_up) * self.hyper['adam_beta2_during_rampup'],
                                     step_up * self.hyper['adam_beta2_after_rampup'], name='adam_beta2')
            # TODO: add ema_loss rampup?

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

        with tf.name_scope('costs'):
            # classify loss
            self.mean_class_loss_l, self.class_loss_l = \
                classification_costs(self.class_logits_l, self.labels)
            self.mean_class_loss_r, self.class_loss_r = \
                classification_costs(self.class_logits_r, self.labels)

            # consistency loss
            labeled_cons = self.hyper['labeled_consistency']
            cons_mask = tf.logical_or(tf.equal(self.labels, -1), labeled_cons)

            mean_cons_loss_l, cons_loss_l = cb_consistency_costs(self.cons_logits_l, self.class_logits_r, 
                self.cons_scale, cons_mask, self.hyper['cons_trust'], name='cons_loss_l')
            mean_cons_loss_r, cons_loss_r = cb_consistency_costs(self.cons_logits_r, self.class_logits_l,
                self.cons_scale, cons_mask, self.hyper['cons_trust'], name='cons_loss_r')
                
            self.mean_loss_l, self.loss_l = tf.case([
                (self.ema_loss_l < self.ema_loss_r), lambda: self.class_loss_l,
                (self.ema_loss_l > self.ema_loss_r), lambda: total_costs(self.class_loss_l, cons_loss_l)],
                default=lambda: self.class_loss_l)

            self.mean_loss_r, self.loss_r = tf.case([
                (self.ema_loss_l < self.ema_loss_r), lambda: total_costs(self.class_loss_r, cons_loss_r),
                (self.ema_loss_l > self.ema_loss_r), lambda: self.class_loss_r],
                default=lambda: self.class_loss_r)

            