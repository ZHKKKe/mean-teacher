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

l_ema_loss = 0
r_ema_loss = 0


def create_compite_model(side, num_classes):
    LOG.info('=> creating {pretrained} {side} model: {arch}'.format(
        pretrained='pre-trained' if args.pretrained else 'non-pre-trained',
        side=side,
        arch=args.arch))

    model_factory = architectures.__dict__[args.arch]
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


def adjust_learning_rate(optimizer, epoch, step_in_epoch,
                         total_steps_in_epoch):
    lr = args.lr
    epoch = epoch + step_in_epoch / total_steps_in_epoch

    # LR warm-up to handle large minibatch sizes from https://arxiv.org/abs/1706.02677
    lr = ramps.linear_rampup(
        epoch, args.lr_rampup) * (args.lr - args.initial_lr) + args.initial_lr

    # Cosine LR rampdown from https://arxiv.org/abs/1608.03983 (but one cycle only)
    if args.lr_rampdown_epochs:
        assert args.lr_rampdown_epochs >= args.epochs
        lr *= ramps.cosine_rampdown(epoch, args.lr_rampdown_epochs)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

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

def validate(eval_loader, model, c_model, log, global_step, epoch, co_test=False, b_model=None):
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
            x = np.random.uniform(low=0.0, high=1.0)
            conv_out = x * conv_out + (1 - x) * b_conv_out
            # conv_out = torch.cat((x * conv_out, (1 - x) * b_conv_out), dim=1)

        output1, output2 = c_model(conv_out)

        softmax1, softmax2 = F.softmax(
            output1, dim=1), F.softmax(
                output2, dim=1)
        class_loss = class_criterion(output1, target_var) / minibatch_size

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output1.data, target_var.data, topk=(1, 5))
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

def calculate_train_ema_loss(train_loader, l_model, r_model):
    # loss calculate scale same as train
    # Note: loss calculate scale not same as validate?
    LOG.info('Calculate train ema loss initial value.')
    class_criterion = nn.CrossEntropyLoss(size_average=False, ignore_index=NO_LABEL).cuda()
    meters = AverageMeterSet()

    l_model.eval()
    r_model.eval()

    end = time.time()
    for i, ((l_input, r_input), target) in enumerate(train_loader):
        l_input_var = torch.autograd.Variable(l_input, volatile=True)
        r_input_var = torch.autograd.Variable(r_input, volatile=True)
        target_var = torch.autograd.Variable(target.cuda(async=True), volatile=True)

        minibatch_size = len(target_var)
        labeled_minibatch_size = target_var.data.ne(NO_LABEL).sum()
        assert labeled_minibatch_size > 0

        l_model_out = l_model(l_input_var)
        r_model_out = r_model(r_input_var)

        if isinstance(l_model_out, Variable):
            assert args.logit_distance_cost < 0
            l_output1 = l_model_out
            r_output1 = r_model_out
        else:
            assert len(l_model_out) == 2
            assert len(r_model_out) == 2
            l_output1, _ = l_model_out
            r_output1, _ = r_model_out

        l_class_loss = class_criterion(l_output1, target_var) / minibatch_size
        r_class_loss = class_criterion(r_output1, target_var) / minibatch_size
        meters.update('l_class_loss', l_class_loss.data[0])
        meters.update('r_class_loss', r_class_loss.data[0])

        # measure elapsed time
        meters.update('batch_time', time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            LOG.info('Iter [{0}]\t'
                     'Time {meters[batch_time]:.3f}\t'
                     'L_EMA_Loss: {meters[l_class_loss]:.4f}\t'
                     'R_EMA_Loss: {meters[r_class_loss]:.4f}'.format(i, meters=meters))

    LOG.info(' * L_EMA_LOSS {l.avg:.4f}\tR_EMA_LOSS {r.avg:.4f}'.format(
        l=meters['l_class_loss'], r=meters['r_class_loss']))

    return meters['l_class_loss'].avg, meters['r_class_loss'].avg

def train_epoch(train_loader, l_model, r_model, c_model, l_optimizer, r_optimizer, c_optimizer, epoch, log):
    global global_step
    global l_ema_loss
    global r_ema_loss

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

    def calculate_js_scale(epoch):
        return args.js_scale * sigmoid_rampup_ke(epoch, args.js_rampup, args.consistency_rampup_exp)

    class_criterion = nn.CrossEntropyLoss(size_average=False, ignore_index=NO_LABEL).cuda()
    if args.consistency_type == 'mse':
        consistency_criterion = losses.softmax_mse_loss
    elif args.consistency_type == 'kl':
        consistency_criterion = losses.softmax_kl_loss
    else:
        assert False, args.consistency_type

    # residual_logit_criterion = losses.symmetric_mse_loss

    meters = AverageMeterSet()

    # calculate epoch initial ema loss values
    if epoch != 0 and args.epoch_init_ema_loss:
        l_ema_loss, r_ema_loss = calculate_train_ema_loss(train_loader, l_model, r_model)

    l_model.train()
    r_model.train()
    c_model.train()

    end = time.time()
    for i, ((l_input, r_input), target) in enumerate(train_loader):
        meters.update('data_time', time.time() - end)

        # adjust learning rate, just for ramp-down now
        adjust_learning_rate(l_optimizer, epoch, i, len(train_loader))
        adjust_learning_rate(r_optimizer, epoch, i, len(train_loader))
        meters.update('l_lr', l_optimizer.param_groups[0]['lr'])
        meters.update('r_lr', r_optimizer.param_groups[0]['lr'])

        l_input_var = torch.autograd.Variable(l_input)
        r_input_var = torch.autograd.Variable(r_input)
        target_var = torch.autograd.Variable(target.cuda(async=True))

        minibatch_size = len(target_var)
        labeled_minibatch_size = target_var.data.ne(NO_LABEL).sum()
        assert labeled_minibatch_size > 0
        meters.update('labeled_minibatch_size', labeled_minibatch_size)

        # scales = [0.001, 0.005, 0.01, 0.999, 0.99, 0.995]
        # alpha = float(np.random.choice(scales, 1))
        # beta = float(np.random.choice(scales, 1))

        alpha = np.random.uniform(low=0.0, high=1.0)
        beta = np.random.uniform(low=0.0, high=1.0)
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

        alpha_classify_out = c_model(alpha_hyper_out)
        beta_classify_out = c_model(beta_hyper_out)

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

        # cons_loss ema_loss-----------------------------------------------
        # # update ema loss values
        # l_ema_loss = (1 - args.ema_loss) * l_class_loss.data[0] + args.ema_loss * l_ema_loss
        # r_ema_loss = (1 - args.ema_loss) * r_class_loss.data[0] + args.ema_loss * r_ema_loss
        # meters.update('l_ema_loss', l_ema_loss)
        # meters.update('r_ema_loss', r_ema_loss)

        # consistency_loss = 0
        # if args.consistency:
        #     consistency_weight = calculate_consistency_scale(epoch)
        #     meters.update('cons_weight', consistency_weight)

        #     # left model is better
        #     if l_ema_loss < r_ema_loss:
        #         # if l_class_loss.data[0] < r_class_loss.data[0]:
        #         l_cons_logit = Variable(l_cons_logit.detach().data, requires_grad=False)
        #         consistency_loss = consistency_weight * consistency_criterion(r_cons_logit, l_cons_logit) / minibatch_size
        #         r_loss += consistency_loss
        #         meters.update('better_model', -1.)  # -1 == left model
        #         meters.update('cons_loss', consistency_loss.data[0])

        #     # right model is better
        #     elif l_ema_loss > r_ema_loss:
        #         # elif l_class_loss.data[0] > r_class_loss.data[0]:
        #         r_cons_logit = Variable(r_cons_logit.detach().data, requires_grad=False)
        #         consistency_loss = consistency_weight * consistency_criterion(l_cons_logit, r_cons_logit) / minibatch_size
        #         l_loss += consistency_loss
        #         meters.update('better_model', 1.)    # 1 == right model
        #         meters.update('cons_loss', consistency_loss.data[0])

        #     else:
        #         consistency_loss = 0
        #         meters.update('cons_loss', 0)

        # else:
        #     consistency_loss = 0
        #     meters.update('cons_loss', 0)
        # -----------------------------------------------

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

        co_js_loss = 0
        if args.js_scale:
            js_weight = calculate_js_scale(epoch)
            co_js_loss = js_weight * losses.js_loss(alpha_cons_logit, beta_cons_logit) / minibatch_size
            meters.update('js_loss', co_js_loss.data[0])
        else:
            meters.update('js_loss', 0)
        # meters.update('js_loss', 0)

        l_optimizer.zero_grad()
        r_optimizer.zero_grad()
        c_optimizer.zero_grad()

        loss = alpha_loss + beta_loss
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
                    #  'Data-T {meters[data_time]:.3f}\t'
                    #  'L-EMA {meters[l_ema_loss]:.4f}\t'
                    #  'R-EMA {meters[r_ema_loss]:.4f}\t'
                     'L-Class {meters[a_class_loss]:.4f}\t'
                     'R-Class {meters[b_class_loss]:.4f}\t'
                    #  'Cons {meters[cons_loss]:.4f}\t'
                     'JS {meters[js_loss]:.4f}\n'
                    #  'Better-M {better.sum:.1f}\n'
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
    global l_ema_loss
    global r_ema_loss

    checkpoint_path = context.transient_dir
    training_log = context.create_train_log('training')
    l_validation_log = context.create_train_log('l_validation')
    r_validation_log = context.create_train_log('r_validation')

    dataset_config = datasets.__dict__[args.dataset]()
    num_classes = dataset_config.pop('num_classes')
    train_loader, eval_loader = create_data_loaders(**dataset_config, args=args)
    l_model = create_compite_model(side='l', num_classes=num_classes)
    r_model = create_compite_model(side='r', num_classes=num_classes)

    LOG.info(parameters_string(l_model))
    LOG.info(parameters_string(r_model))
    if args.same_net_init:
        LOG.info('same net init.')
        for l_param, r_param in zip(l_model.parameters(), r_model.parameters()):
            r_param.data.mul_(0.0).add_(l_param.data)

    # create c-model -----------------------------------
    LOG.info('=> creating {pretrained} c-model: {arch}'.format(
             pretrained='pre-trained' if args.pretrained else 'non-pre-trained', arch=args.arch + '_classify'))

    model_factory = architectures.__dict__[args.arch + '_classify']
    model_params = dict(pretrained=args.pretrained)
    c_model_module = model_factory(**model_params)
    c_model = nn.DataParallel(c_model_module).cuda()
    LOG.info(parameters_string(c_model))
    # --------------------------------------------------

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
        l_optimizer.load_state_dict(checkpoint['l_optimizer'])
        r_optimizer.load_state_dict(checkpoint['r_optimizer'])

        LOG.info('=> loaded checkpoint {} (epoch {})'.format(args.resume, checkpoint['epoch']))

    cudnn.benchmark = True

    if args.evaluate:
        LOG.info('Evaluating the left model: ')
        validate(eval_loader, l_model, c_model, l_validation_log, global_step, args.start_epoch)
        LOG.info('Evaluating the right model: ')
        validate(eval_loader, r_model, c_model, r_validation_log, global_step, args.start_epoch)
        return

    # l_ema_loss, r_ema_loss = calculate_train_ema_loss(train_loader, l_model, r_model)

    for epoch in range(args.start_epoch, args.epochs):
        start_time = time.time()
        # train for one epoch
        train_epoch(train_loader, l_model, r_model, c_model, l_optimizer, r_optimizer, c_optimizer, epoch, training_log)
        LOG.info('--- training epoch in {} seconds ---'.format(time.time()-start_time))

        is_best = False
        l_better = False
        if args.evaluation_epochs and (epoch + 1) % args.evaluation_epochs == 0:
            start_time = time.time()
            LOG.info('Evaluating the left model: ')
            l_prec1 = validate(eval_loader, l_model, c_model, l_validation_log, global_step, epoch + 1)
            LOG.info('Evaluating the right model: ')
            r_prec1 = validate(eval_loader, r_model, c_model, r_validation_log, global_step, epoch + 1)

            LOG.info('Evaluating the random hyper model: ')
            r_prec1 = validate(eval_loader, l_model, c_model, r_validation_log, global_step, epoch + 1, co_test=True, b_model=r_model)

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
