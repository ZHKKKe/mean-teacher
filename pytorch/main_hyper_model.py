import re
import argparse
import os
import shutil
import time
import math
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import torchvision.datasets

from mean_teacher import architectures, datasets, data, losses, ramps, cli
from mean_teacher.run_context import RunContext
from mean_teacher.data import NO_LABEL
from mean_teacher.utils import *

LOG = logging.getLogger('main')

args = None
best_prec1 = 0
global_step = 0

def adjust_ct_learning_rate(optimizer, epoch, step_in_epoch, total_steps_in_epoch):
    lr = args.lr
    epoch = epoch + step_in_epoch / total_steps_in_epoch
    lr *= ramps.ct_lr_rampdown(epoch, args.epochs)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def create_module(side, num_classes, arch_name, **kwargs):
    LOG.info('=> creating {pretrained} {side} model: {arch}'.format(
        pretrained='pre-trained' if args.pretrained else 'non-pre-trained',
        side=side,
        arch=arch_name))

    model_factory = architectures.__dict__[arch_name]
    model_params = dict(pretrained=args.pretrained, num_classes=num_classes)
    model = model_factory(**model_params)
    model = nn.DataParallel(model).cuda()
    return model


def create_data_loaders(train_transformation, eval_transformation, datadir, args):
    traindir = os.path.join(datadir, args.train_subdir)
    evaldir = os.path.join(datadir, args.eval_subdir)

    assert_exactly_one([args.exclude_unlabeled, args.labeled_batch_size])

    dataset = torchvision.datasets.ImageFolder(traindir, train_transformation)

    if args.labels:
        with open(args.labels) as f:
            labels = dict(line.split(' ') for line in f.read().splitlines())
        labeled_idxs, unlabeled_idxs = data.relabel_dataset(dataset, labels)

    if args.exclude_unlabeled:
        sampler = SubsetRandomSampler(labeled_idxs)
        batch_sampler = BatchSampler(sampler, args.batch_size, drop_last=True)
    elif args.labeled_batch_size:
        batch_sampler = data.TwoStreamBatchSampler(
            unlabeled_idxs, labeled_idxs, args.batch_size,
            args.labeled_batch_size)
    else:
        assert False, "labeled batch size {}".format(args.labeled_batch_size)

    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=args.workers,
        pin_memory=True)

    eval_loader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(evaldir, eval_transformation),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2 * args.workers,  # Needs images twice as fast
        pin_memory=True,
        drop_last=False)

    return train_loader, eval_loader


def accuracy(output, target, topk=(1, )):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    labeled_minibatch_size = max(target.ne(NO_LABEL).sum(), 1e-8)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / labeled_minibatch_size))
    return res

def save_checkpoint(state, is_best, dirpath, epoch):
    filename = 'checkpoint.{}.ckpt'.format(epoch)
    checkpoint_path = os.path.join(dirpath, filename)
    best_path = os.path.join(dirpath, 'best.ckpt')
    torch.save(state, checkpoint_path)
    LOG.info("--- checkpoint saved to %s ---" % checkpoint_path)
    if is_best:
        shutil.copyfile(checkpoint_path, best_path)
        LOG.info("--- checkpoint copied to %s ---" % best_path)

def validate(eval_loader, model, c_model, log, global_step, epoch, co_test=False, b_model=None, x=0.9, fc_idx=1):
    class_criterion = nn.CrossEntropyLoss(
        size_average=False, ignore_index=NO_LABEL).cuda()
    meters = AverageMeterSet()

    # switch to evaluate mode
    model.eval()
    c_model.eval()

    if co_test:
        b_model.eval()

    end = time.time()
    for i, (input, target) in enumerate(eval_loader):
        meters.update('data_time', time.time() - end)

        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(
            target.cuda(async=True), volatile=True)

        minibatch_size = len(target_var)
        labeled_minibatch_size = target_var.data.ne(NO_LABEL).sum()
        assert labeled_minibatch_size > 0
        meters.update('labeled_minibatch_size', labeled_minibatch_size)

        # compute output
        conv_out = model(input_var)
        if co_test:
            b_conv_out = b_model(input_var)
            # scales = [0.001, 0.005, 0.01, 0.999, 0.99, 0.995]
            # x = float(np.random.choice(scales, 1))
            # x = np.random.uniform(low=0.0, high=1.0)
            conv_out = x * conv_out + (1 - x) * b_conv_out
            # conv_out = torch.cat((x * conv_out, (1 - x) * b_conv_out), dim=1)

        output = c_model(conv_out, idx=fc_idx)

        softmax = F.softmax(output, dim=1)
        class_loss = class_criterion(output, target_var) / minibatch_size

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target_var.data, topk=(1, 5))
        meters.update('class_loss', class_loss.data[0], labeled_minibatch_size)
        meters.update('top1', prec1[0], labeled_minibatch_size)
        meters.update('error1', 100.0 - prec1[0], labeled_minibatch_size)
        meters.update('top5', prec5[0], labeled_minibatch_size)
        meters.update('error5', 100.0 - prec5[0], labeled_minibatch_size)

        # measure elapsed time
        meters.update('batch_time', time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            LOG.info('Test: [{0}/{1}]\t'
                     'Time {meters[batch_time]:.3f}\t'
                     'Data {meters[data_time]:.3f}\t'
                     'Class {meters[class_loss]:.4f}\t'
                     'Prec@1 {meters[top1]:.3f}\t'
                     'Prec@5 {meters[top5]:.3f}'.format(
                         i, len(eval_loader), meters=meters))

    LOG.info(' * Prec@1 {top1.avg:.3f}\tPrec@5 {top5.avg:.3f}'.format(
        top1=meters['top1'], top5=meters['top5']))
    log.record(
        epoch, {
            'step': global_step,
            **meters.values(),
            **meters.averages(),
            **meters.sums()
        })

    return meters['top1'].avg


def train_epoch(train_loader, l_model, r_model, c_model, l_optimizer, r_optimizer, c_optimizer, epoch, log):
    global global_step

    def sigmoid_rampup_ke(current, rampup_length, exp_scale):
        """Exponential rampup from https://arxiv.org/abs/1610.02242"""
        if rampup_length == 0:
            return 1.0
        else:
            current = np.clip(current, 0.0, rampup_length)
            phase = 1.0 - current / rampup_length
            return float(np.exp(exp_scale * phase * phase))

    def calculate_consistency_scale(epoch):
        return args.consistency * sigmoid_rampup_ke(epoch, args.consistency_rampup, args.consistency_rampup_exp)

    def calculate_cot_js_scale(epoch):
        return args.cot_js_scale * sigmoid_rampup_ke(epoch, args.cot_js_rampup, args.cot_js_rampup_exp)

    class_criterion = nn.CrossEntropyLoss(size_average=False, ignore_index=NO_LABEL).cuda()
    cot_criterion = losses.js_loss
    
    if args.consistency_type == 'mse':
        consistency_criterion = losses.softmax_mse_loss
    elif args.consistency_type == 'kl':
        consistency_criterion = losses.softmax_kl_loss
    else:
        assert False, args.consistency_type

    # residual_logit_criterion = losses.symmetric_mse_loss

    meters = AverageMeterSet()

    l_model.train()
    r_model.train()
    c_model.train()

    end = time.time()
    for i, ((l_input, r_input), target) in enumerate(train_loader):
        meters.update('data_time', time.time() - end)

        # adjust learning rate, just for ramp-down now
        adjust_ct_learning_rate(l_optimizer, epoch, i, len(train_loader))
        adjust_ct_learning_rate(r_optimizer, epoch, i, len(train_loader))
        meters.update('l_lr', l_optimizer.param_groups[0]['lr'])
        meters.update('r_lr', r_optimizer.param_groups[0]['lr'])

        l_input_var = torch.autograd.Variable(l_input)
        r_input_var = torch.autograd.Variable(r_input)
        target_var = torch.autograd.Variable(target.cuda(async=True))

        minibatch_size = len(target_var)
        labeled_minibatch_size = target_var.data.ne(NO_LABEL).sum()
        unlabeled_minibatch_size = minibatch_size - labeled_minibatch_size
        assert labeled_minibatch_size > 0
        meters.update('labeled_minibatch_size', labeled_minibatch_size)

        # scales = [0.001, 0.005, 0.01, 0.999, 0.99, 0.995]
        # alpha = float(np.random.choice(scales, 1))
        # beta = float(np.random.choice(scales, 1))

        # alpha = np.random.uniform(low=0.0, high=1.0)
        # beta = np.random.uniform(low=0.0, high=1.0)
        alpha = 0.9
        beta = 0.1
        if i % args.print_freq == 0:
            LOG.info('alpha: {0}\t beta: {1}\t'.format(alpha, beta))

        l_model_out1 = l_model(l_input_var)
        r_model_out1 = r_model(l_input_var)

        l_model_out2 = l_model(r_input_var)
        r_model_out2 = r_model(r_input_var)

        alpha_hyper_out = alpha * l_model_out1 + (1 - alpha) * r_model_out1
        beta_hyper_out = beta * l_model_out2 + (1 - beta) * r_model_out2
        # alpha_hyper_out = torch.cat((alpha * l_model_out1, (1 - alpha) * r_model_out1), dim=1)
        # beta_hyper_out = torch.cat((beta * l_model_out2, (1 - beta) * r_model_out2), dim=1)

        alpha_classify_out = c_model(alpha_hyper_out, idx=0)
        beta_classify_out = c_model(beta_hyper_out, idx=1)

        # now just use 2 output for cifar10 dataset
        if isinstance(alpha_classify_out, Variable):
            assert args.logit_distance_cost < 0
            alpha_logit1 = alpha_classify_out
            beta_logit1 = beta_classify_out
        else:
            assert len(alpha_classify_out) == 2
            assert len(alpha_classify_out) == 2
            alpha_logit1, alpha_logit2 = alpha_classify_out
            beta_logit1, beta_logit2 = beta_classify_out

        if args.logit_distance_cost >= 0:
            LOG.error('compite_buddy not support logit_distance_cost now.')

        alpha_class_logit, alpha_cons_logit = alpha_logit1, alpha_logit1
        beta_class_logit, beta_cons_logit = beta_logit1, beta_logit1

        alpha_class_loss = class_criterion(alpha_class_logit, target_var) / minibatch_size
        beta_class_loss = class_criterion(beta_class_logit, target_var) / minibatch_size
        meters.update('a_class_loss', alpha_class_loss.data[0])
        meters.update('b_class_loss', beta_class_loss.data[0])

        alpha_loss, beta_loss = alpha_class_loss, beta_class_loss

        assert not (np.isnan(alpha_loss.data[0]) or alpha_loss.data[0] > 1e5), 'A-Loss explosion: {}'.format(alpha_loss.data[0])
        assert not (np.isnan(beta_loss.data[0]) or beta_loss.data[0] > 1e5), 'B-Loss explosion: {}'.format(beta_loss.data[0])
        meters.update('a_loss', alpha_loss.data[0])
        meters.update('b_loss', beta_loss.data[0])

        l_prec1, l_prec5 = accuracy(alpha_class_logit.data, target_var.data, topk=(1, 5))
        meters.update('a_top1', l_prec1[0], labeled_minibatch_size)
        meters.update('a_error1', 100. - l_prec1[0], labeled_minibatch_size)
        meters.update('a_top5', l_prec5[0], labeled_minibatch_size)
        meters.update('a_error5', 100. - l_prec5[0], labeled_minibatch_size)

        r_prec1, r_prec5 = accuracy(beta_class_logit.data, target_var.data, topk=(1, 5))
        meters.update('b_top1', r_prec1[0], labeled_minibatch_size)
        meters.update('b_error1', 100. - r_prec1[0], labeled_minibatch_size)
        meters.update('b_top5', r_prec5[0], labeled_minibatch_size)
        meters.update('b_error5', 100. - r_prec5[0], labeled_minibatch_size)

        # cot js loss
        cot_js_weight = calculate_cot_js_scale(epoch)
        cot_js_loss = cot_js_weight * cot_criterion(alpha_class_logit[:unlabeled_minibatch_size], beta_class_logit[:unlabeled_minibatch_size])
        cot_js_loss /= unlabeled_minibatch_size
        meters.update('cot_loss', cot_js_loss.data[0])

        l_optimizer.zero_grad()
        r_optimizer.zero_grad()
        c_optimizer.zero_grad()

        loss = alpha_loss + beta_loss + cot_js_loss
        # loss = alpha_loss + beta_loss + co_js_loss
        loss.backward()
        # l_loss.backward()
        # r_loss.backward()

        c_optimizer.step()
        l_optimizer.step()
        r_optimizer.step()

        global_step += 1
        meters.update('batch_time', time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            LOG.info('Epoch: [{0}][{1}/{2}]\t'
                     'Batch-T {meters[batch_time]:.3f}\t'
                     'L-Class {meters[a_class_loss]:.4f}\t'
                     'R-Class {meters[b_class_loss]:.4f}\t'
                     'Cot {meters[cot_loss]:.4f}\n'
                     'L-Prec@1 {meters[a_top1]:.3f}\t'
                     'R-Prec@1 {meters[b_top1]:.3f}\t'
                     'L-Prec@5 {meters[a_top5]:.3f}\t'
                     'R-Prec@5 {meters[b_top5]:.3f}'.format(
                     epoch, i, len(train_loader), meters=meters))

            log.record(epoch + i / len(train_loader), {
                'step': global_step,
                **meters.values(),
                **meters.averages(),
                **meters.sums()})

def main(context):
    global best_prec1
    global global_step

    checkpoint_path = context.transient_dir
    training_log = context.create_train_log('training')
    l_validation_log = context.create_train_log('l_validation')
    r_validation_log = context.create_train_log('r_validation')

    dataset_config = datasets.__dict__[args.dataset]()
    num_classes = dataset_config.pop('num_classes')
    train_loader, eval_loader = create_data_loaders(**dataset_config, args=args)

    l_model = create_module(side='l', num_classes=num_classes, arch_name=args.arch + '_conv')
    r_model = create_module(side='r', num_classes=num_classes, arch_name=args.arch + '_conv')
    c_model = create_module(side='c', num_classes=num_classes, arch_name=args.arch + '_multi_fc')
    LOG.info(parameters_string(l_model))
    LOG.info(parameters_string(r_model))
    LOG.info(parameters_string(c_model))

    l_optimizer = torch.optim.SGD(params=l_model.parameters(),
                                  lr=args.lr,
                                  momentum=args.momentum,
                                  weight_decay=args.weight_decay,
                                  nesterov=args.nesterov)
    r_optimizer = torch.optim.SGD(params=r_model.parameters(),
                                  lr=args.lr,
                                  momentum=args.momentum,
                                  weight_decay=args.weight_decay,
                                  nesterov=args.nesterov)
    c_optimizer = torch.optim.SGD(params=c_model.parameters(),
                                  lr=args.lr,
                                  momentum=args.momentum,
                                  weight_decay=args.weight_decay,
                                  nesterov=args.nesterov)

    if args.resume:
        assert os.path.isfile(args.resume), '=> no checkpoint found at: {}'.format(args.resume)
        LOG.info('=> loading checkpoint: {}'.format(args.resume))

        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        global_step = checkpoint['global_step']
        best_prec1 = checkpoint['best_prec1']
        l_model.load_state_dict(checkpoint['l_model'])
        r_model.load_state_dict(checkpoint['r_model'])
        c_model.load_state_dict(checkpoint['c_model'])
        l_optimizer.load_state_dict(checkpoint['l_optimizer'])
        r_optimizer.load_state_dict(checkpoint['r_optimizer'])
        c_optimizer.load_state_dict(checkpoint['c_optimizer'])

        LOG.info('=> loaded checkpoint {} (epoch {})'.format(args.resume, checkpoint['epoch']))

    cudnn.benchmark = True

    # if args.evaluate:
    #     LOG.info('Evaluating the left model: ')
    #     validate(eval_loader, l_model, c_model, l_validation_log, global_step, args.start_epoch)
    #     LOG.info('Evaluating the right model: ')
    #     validate(eval_loader, r_model, c_model, r_validation_log, global_step, args.start_epoch)
    #     return

    for epoch in range(args.start_epoch, args.epochs):
        start_time = time.time()

        train_epoch(train_loader, l_model, r_model, c_model, l_optimizer, r_optimizer, c_optimizer, epoch, training_log)
        LOG.info('--- training epoch in {} seconds ---'.format(time.time()-start_time))

        is_best = False
        l_better = False
        if args.evaluation_epochs and (epoch + 1) % args.evaluation_epochs == 0:
            start_time = time.time()
            LOG.info('Evaluating the left model: ')
            l_prec1 = validate(eval_loader, l_model, c_model, l_validation_log, global_step, epoch + 1, co_test=True, b_model=r_model, x=0.9, fc_idx=0)
            LOG.info('Evaluating the right model: ')
            r_prec1 = validate(eval_loader, r_model, c_model, r_validation_log, global_step, epoch + 1, co_test=True, b_model=l_model, x=0.9, fc_idx=1)
            LOG.info('--- validation in {} seconds ---'.format(time.time() - start_time))
            better_prec1 = l_prec1 if l_prec1 > r_prec1 else r_prec1
            best_prec1 = max(better_prec1, best_prec1)
            is_best = better_prec1 > best_prec1

            if better_prec1 == l_prec1:
                l_better = True
                LOG.info('Left model work better.')
            else:
                LOG.info('Right model work better.')

        if args.checkpoint_epochs and (epoch + 1) % args.checkpoint_epochs == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'global_step': global_step,
                'better_model': -1 if l_better else 1,
                'arch': args.arch,
                'l_model': l_model.state_dict(),
                'r_model': r_model.state_dict(),
                'l_optimizer':l_optimizer.state_dict(),
                'r_optimizer':r_optimizer.state_dict(),
            }, is_best, checkpoint_path, epoch + 1)




if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    args = cli.parse_commandline_args()
    main(RunContext(__file__, 0))
