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

l_centers = {}
r_centers = {}
tmp_path = ''



def pca_drawer(x, y, epoch, name):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    global tmp_path
    global l_centers

    color_map = {
        -1: 'indianred',
        0: 'orange',
        1: 'khaki',
        2: 'lightgreen',
        3: 'paleturquoise',
        4: 'dodgerblue',
        5: 'lightsteelblue',
        6: 'slategray',
        7: 'mediumpurple',
        8: 'hotpink',
        9: 'silver',
    }

    plt.clf()

    plt.title(name)
    plt.grid(True)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)

    center_num = len(l_centers.keys())

    data_0 = x[..., 0]
    data_1 = x[..., 1]

    # plt.axis('tight')
    x_side = 0.3 * (np.max(data_0) - np.min(data_0))
    y_side = 0.3 * (np.max(data_1) - np.min(data_1))
    plt.axis([
        np.min(data_0) - x_side,
        np.max(data_0) + x_side,
        np.min(data_1) - y_side,
        np.max(data_1) + y_side
    ])

    # split data
    data = {}
    center = {}
    for i in range(-1, 10):
        data[i] = [[], []]
        center[i] = [[], []]

    for idx, label in enumerate(y):
        if len(y) - idx > center_num:
            data[label][0].append(x[idx][0])
            data[label][1].append(x[idx][1])
        else:
            center[label][0].append(x[idx][0])
            center[label][1].append(x[idx][1])

    for i in range(-1, 10):
        plt.scatter(data[i][0], data[i][1], label=i, marker='.', s=3, c=color_map[i])
    # for i in range(-1, 10):
        # plt.scatter(center[i][0], center[i][1], label=i, marker='v', s=30, edgecolor='black', c=color_map[i])

    plt.legend(loc='upper right')
    filename = name + '_{}.jpg'.format(epoch)
    file_path = os.path.join(tmp_path, filename)
    plt.savefig(file_path, dpi=300)
    plt.close('all')


def calculate_initial_center(l_model, r_model, label_num, feature_dim, transformation, datadir, args):
    global l_centers
    global r_centers
    
    total_samples = 0

    def create_loader(transformation, datadir, args):
        traindir = os.path.join(datadir, args.train_subdir)
        dataset = torchvision.datasets.ImageFolder(traindir, transformation)

        if args.labels:
            with open(args.labels) as f:
                labels = dict(line.split(' ') for line in f.read().splitlines())
            labeled_idxs, unlabeled_idxs = data.relabel_dataset(dataset, labels)

        batch_sampler = data.LabeledBatchSampler(labeled_idxs, args.batch_size)
        
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=args.workers,
            pin_memory=True)

        return loader

    for idx in range(0, label_num):
        l_centers[idx] = torch.zeros(128).cuda()
        r_centers[idx] = torch.zeros(128).cuda()

    l_model.eval()
    r_model.eval()

    data_loader = create_loader(transformation, datadir, args)

    end = time.time()
    for i, ((l_input, r_input), target) in enumerate(data_loader):
        l_input_var = torch.autograd.Variable(l_input, volatile=True)
        r_input_var = torch.autograd.Variable(r_input, volatile=True)
        target_var = torch.autograd.Variable(target.cuda(async=True), volatile=True)

        minibatch_size = len(target_var)
        total_samples += minibatch_size

        (l_output1, l_output2), l_features = l_model(l_input_var, debug=True)
        (r_output1, r_output2), r_features = r_model(r_input_var, debug=True)

        for idx, label in enumerate(target_var):
            label_int = label.data[0]
            l_centers[label_int] += l_features[idx].data
            r_centers[label_int] += r_features[idx].data

    for idx, _ in enumerate(l_centers):
        l_centers[idx] /= (total_samples / label_num)
        r_centers[idx] /= (total_samples / label_num)
    

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def copy_model_variables(source_model, target_model):
    target_model.load_state_dict(source_model.state_dict())


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

    if args.as_co_train_lr:
        lr *= ramps.ct_lr_rampdown(epoch, args.epochs)
    else:
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

def validate(eval_loader, model, log, global_step, epoch, name):
    global l_centers
    global r_centers

    class_criterion = nn.CrossEntropyLoss(
        size_average=False, ignore_index=NO_LABEL).cuda()
    meters = AverageMeterSet()

    # switch to evaluate mode
    model.eval()

    pca_features = []
    pca_labeles = []

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
        (output1, output2), feature = model(input_var, debug=True)
        softmax1, softmax2 = F.softmax(
            output1, dim=1), F.softmax(
                output2, dim=1)

        if True and feature is not None:
            f_data = feature.data.cpu().numpy()
            t_data = target_var.data.cpu().numpy()
            for idx, f in enumerate(f_data):
                pca_features.append(f)
                pca_labeles.append(t_data[idx])

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

    centers = l_centers if 'l' in name else r_centers

    # for key, value in centers.items():
        # pca_features.append(value.cpu().numpy())
        # pca_labeles.append(key)

    if True:
        LOG.info('--------------- PCA FEATURE DRAWER ---------------')
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import normalize, scale

        pca_features = np.asarray(pca_features)
        pca_labeles = np.asarray(pca_labeles)
        pca = PCA(n_components=2)
        pca_results = pca.fit_transform(normalize(pca_features, axis=0))
        # pca_results = pca.fit_transform(scale(pca_features, axis=0))
        pca_drawer(pca_results, pca_labeles, epoch=epoch, name=name)
        LOG.info('--------------------------------------------------')
    
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

def train_epoch(train_loader, l_model, r_model, le_model, re_model, l_optimizer, r_optimizer, epoch, log):
    global global_step
    global l_ema_loss
    global r_ema_loss

    global l_centers
    global r_centers

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

    class_criterion = nn.CrossEntropyLoss(size_average=False, ignore_index=NO_LABEL).cuda()
    if args.consistency_type == 'mse':
        consistency_criterion = losses.softmax_mse_loss
    elif args.consistency_type == 'kl':
        consistency_criterion = losses.softmax_kl_loss
    else:
        assert False, args.consistency_type

    residual_logit_criterion = losses.symmetric_mse_loss
    feature_criterion = losses.feature_mse_loss
    


    meters = AverageMeterSet()

    # calculate epoch initial ema loss values
    # if epoch != 0 and args.epoch_init_ema_loss:
        # l_ema_loss, r_ema_loss = calculate_train_ema_loss(train_loader, l_model, r_model)

    l_model.train()
    r_model.train()
    le_model.train()
    re_model.train()

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
        unlabeled_minibatch_size = minibatch_size - labeled_minibatch_size
        meters.update('labeled_minibatch_size', labeled_minibatch_size)

        l_model_out, l_x = l_model(l_input_var, debug=True, bs=minibatch_size, lbs=labeled_minibatch_size)
        r_model_out, r_x = r_model(r_input_var, debug=True, bs=minibatch_size, lbs=labeled_minibatch_size)
        le_model_out = le_model(r_input_var, bs=minibatch_size, lbs=labeled_minibatch_size)
        re_model_out = re_model(l_input_var, bs=minibatch_size, lbs=labeled_minibatch_size)

        # now just use 2 output for cifar10 dataset
        if isinstance(l_model_out, Variable):
            assert args.logit_distance_cost < 0
            l_logit1 = l_model_out
            r_logit1 = r_model_out
            le_logit1 = le_model_out
            re_logit1 = re_model_out

        elif len(l_model_out) == 2:
            assert len(r_model_out) == 2
            l_logit1, l_logit2 = l_model_out
            r_logit1, r_logit2 = r_model_out
            le_logit1, le_logit2 = le_model_out
            re_logit1, re_logit2 = re_model_out

        elif len(l_model_out) == 3:
            assert len(r_model_out) == 3
            l_logit1, l_logit2, l_sn_logit = l_model_out
            r_logit1, r_logit2, r_sn_logit = r_model_out
            le_logit1, le_logit2, le_sn_logit = le_model_out
            re_logit1, re_logit2, re_sn_logit = re_model_out

# ----- cluster center 
        # edge = 1
        # unlabeled_scale = 1.0
        # feature_loss_scale = 50.0
        # center_cluster_scale = 0.01
        # center_faraway_scale = 0.005
        # l_feature_loss = 0
        # r_feature_loss = 0
        # l_distance_loss = 0
        # r_distance_loss = 0
        # if True:
        #     _, le_prec_labels = torch.max(F.softmax(le_logit1, dim=1), 1)
        #     _, re_prec_labels = torch.max(F.softmax(re_logit1, dim=1), 1)

        #     l_ulf_labels = le_prec_labels.split(unlabeled_minibatch_size)[0]
        #     r_ulf_labels = re_prec_labels.split(unlabeled_minibatch_size)[0]
        #     # lf_labels = target_var.split(unlabeled_minibatch_size)[1]

        #     l_centers_var = {}
        #     r_centers_var = {}
        #     for idx, key in enumerate(l_centers.keys()):
        #         l_centers_var[key] = torch.autograd.Variable(l_centers[key], requires_grad=False)
        #         r_centers_var[key] = torch.autograd.Variable(r_centers[key], requires_grad=False)

        #     for idx in range(0, unlabeled_minibatch_size):
        #         l_feature_loss += unlabeled_scale * feature_criterion(l_x[idx], l_centers_var[l_ulf_labels[idx].data[0]])
        #         r_feature_loss += unlabeled_scale * feature_criterion(r_x[idx], r_centers_var[r_ulf_labels[idx].data[0]])

        #     for idx in range(unlabeled_minibatch_size, minibatch_size):
        #         l_feature_loss += feature_criterion(l_x[idx], l_centers_var[target_var[idx].data[0]])
        #         r_feature_loss += feature_criterion(r_x[idx], r_centers_var[target_var[idx].data[0]])

        #     l_feature_loss = l_feature_loss * feature_loss_scale * calculate_consistency_scale(epoch) / minibatch_size / args.consistency
        #     r_feature_loss = r_feature_loss * feature_loss_scale * calculate_consistency_scale(epoch) / minibatch_size / args.consistency
        #     meters.update('l_feature_loss', l_feature_loss.data[0])
        #     meters.update('r_feature_loss', r_feature_loss.data[0])

        #     tmp_l_centers = {}
        #     tmp_r_centers = {}
        #     sample_nums = {}
        #     for idx in range(0, 10):
        #         # tmp_l_centers[idx] = torch.zeros(128).cuda()
        #         # tmp_r_centers[idx] = torch.zeros(128).cuda()
        #         tmp_l_centers[idx] = 0
        #         tmp_r_centers[idx] = 0
        #         sample_nums[idx] = 0

        #     for idx in range(unlabeled_minibatch_size, minibatch_size):
        #         label_int = target_var[idx].data[0]
        #         tmp_l_centers[label_int] += l_x[idx]
        #         tmp_r_centers[label_int] += r_x[idx]
        #         sample_nums[label_int] += 1


        #     for idx, key in enumerate(tmp_l_centers.keys()):
        #         if sample_nums[key] == 0:
        #             continue
        #         tmp_l_centers[key].div_(sample_nums[key])                
        #         tmp_r_centers[key].div_(sample_nums[key])
            
        #     keys = list(l_centers.keys())
        #     for idx, key in enumerate(keys):
        #         if sample_nums[keys[idx]] == 0:
        #             continue
        #         for jdx in range(idx+1, len(keys)):
        #             if sample_nums[keys[jdx]] == 0:
        #                 continue
    
        #             l_distance = torch.abs(feature_criterion(tmp_l_centers[keys[idx]], tmp_l_centers[keys[jdx]]))
        #             r_distance = torch.abs(feature_criterion(tmp_r_centers[keys[idx]], tmp_r_centers[keys[jdx]]))
        #             l_value = l_distance.mul_(-1).add_(edge)
        #             r_value = r_distance.mul_(-1).add_(edge)
                    
        #             if i % args.print_freq == 0:
        #                 LOG.info('{0}, {1} distance: {2}'.format(keys[idx], keys[jdx], l_distance.data[0]))

        #             if l_value.data[0] < 0:
        #                 l_value.mul_(0.0)
        #             if r_value.data[0] < 0:
        #                 r_value.mul_(0.0)
                    
        #             l_distance_loss += l_value
        #             r_distance_loss += r_value
            
        #     l_distance_loss = l_distance_loss  * calculate_consistency_scale(epoch) / 10 / 10 / args.consistency
        #     r_distance_loss = r_distance_loss  * calculate_consistency_scale(epoch) / 10 / 10 / args.consistency
        #     meters.update('l_distance_loss', l_distance_loss.data[0])
        #     meters.update('r_distance_loss', r_distance_loss.data[0])

        #     for idx, key in enumerate(tmp_l_centers.keys()):
        #         if sample_nums[key] == 0:
        #             continue
        #         l_centers[key] -= center_cluster_scale * (l_centers[key] - tmp_l_centers[key].data)
        #         r_centers[key] -= center_cluster_scale * (r_centers[key] - tmp_r_centers[key].data)
            

        #     l_cluster_loss = l_feature_loss + l_distance_loss
        #     r_cluster_loss = r_feature_loss + r_distance_loss
        #     # l_cluster_loss = l_feature_loss
        #     # r_cluster_loss = r_feature_loss

        #     # keys = l_centers.keys()
        #     # for idx, key in enumerate(keys):
        #     #     for jdx in range(idx+1, len(keys)):
        #     #         distance = feature_criterion(l_centers[keys[idx]], l_centers[keys[jdx]])
        #     #         if distance < edge:
        #     #             l_centers[keys[idx]]


        #     # if i % args.print_freq == 0:
        #         # print(l_centers_var[0].data.cpu().numpy().tolist()[:5])
        # else:
        #     meters.update('l_feature_loss', 0)
        #     meters.update('r_feature_loss', 0)
        #     meters.update('l_distance_loss', 0)
        #     meters.update('r_distance_loss', 0)

# ----------------------------------------------------

        edge = 1
        l_feature_loss = 0
        r_feature_loss = 0
        same_l = 0
        same_r = 0
        if args.smooth_neighbor_scale is not None:
            _, le_prec_labels = torch.max(F.softmax(le_logit1, dim=1), 1)
            _, re_prec_labels = torch.max(F.softmax(re_logit1, dim=1), 1)
            # print(list(le_prec_labels.data.cpu().numpy()))
            # print(list(target_labels.data.cpu().numpy()))

            l_mix_labels = torch.cat((le_prec_labels[:unlabeled_minibatch_size], target_var[unlabeled_minibatch_size:]))
            r_mix_labels = torch.cat((re_prec_labels[:unlabeled_minibatch_size], target_var[unlabeled_minibatch_size:]))
            # print(list(l_mix_labels.data.cpu().numpy()))

            # l_pairs_index = np.array([_ for _ in range(0, minibatch_size)])
            # r_pairs_index = np.array([_ for _ in range(0, minibatch_size)])
            # np.random.shuffle(l_pairs_index)
            # np.random.shuffle(r_pairs_index)
    
            # left model
            for idx in range(0, unlabeled_minibatch_size):
                label1 = l_mix_labels[idx]
                label2 = l_mix_labels[idx + unlabeled_minibatch_size]
                feature1 = l_x[idx]
                feature2 = l_x[idx + unlabeled_minibatch_size]

                l_feature_loss += feature_criterion(feature1, feature2, edge, label1.data[0] == label2.data[0])
                if label1.data[0] == label2.data[0]:
                    same_l += 1

            # right model
            for idx in range(0, unlabeled_minibatch_size):
                label1 = r_mix_labels[idx]
                label2 = r_mix_labels[idx + unlabeled_minibatch_size]
                feature1 = r_x[idx]
                feature2 = r_x[idx + unlabeled_minibatch_size]

                r_feature_loss += feature_criterion(feature1, feature2, edge, label1.data[0] == label2.data[0])
                if label1.data[0] == label2.data[0]:
                    same_r += 1

            # if i % args.print_freq == 0:
                # print('same_l: {0}   same_r:{1}'.format(same_l, same_r))

            l_feature_loss = l_feature_loss * args.smooth_neighbor_scale * calculate_consistency_scale(epoch) / labeled_minibatch_size
            r_feature_loss = r_feature_loss * args.smooth_neighbor_scale * calculate_consistency_scale(epoch) / labeled_minibatch_size
            meters.update('l_feature_loss', l_feature_loss.data[0])
            meters.update('r_feature_loss', r_feature_loss.data[0])
        else:
            meters.update('l_feature_loss', 0)
            meters.update('r_feature_loss', 0)


        if args.logit_distance_cost >= 0:
            l_class_logit, l_cons_logit = l_logit1, l_logit2
            r_class_logit, r_cons_logit = r_logit1, r_logit2
            le_class_logit, le_cons_logit = le_logit1, le_logit2
            re_class_logit, re_cons_logit = re_logit1, re_logit2

            l_res_loss = args.logit_distance_cost * residual_logit_criterion(l_class_logit, l_cons_logit) / minibatch_size
            r_res_loss = args.logit_distance_cost * residual_logit_criterion(r_class_logit, r_cons_logit) / minibatch_size

            meters.update('l_res_loss', l_res_loss.data[0])
            meters.update('r_res_loss', r_res_loss.data[0])
        else:
            l_class_logit, l_cons_logit = l_logit1, l_logit1
            r_class_logit, r_cons_logit = r_logit1, r_logit1
            le_class_logit, le_cons_logit = le_logit1, le_logit1
            re_class_logit, re_cons_logit = re_logit1, re_logit1

            l_res_loss = 0.0
            r_res_loss = 0.0
            meters.update('l_res_loss', 0.0)
            meters.update('r_res_loss', 0.0)

        l_class_loss = class_criterion(l_class_logit, target_var) / minibatch_size
        r_class_loss = class_criterion(r_class_logit, target_var) / minibatch_size
        meters.update('l_class_loss', l_class_loss.data[0])
        meters.update('r_class_loss', r_class_loss.data[0])
        
        if args.ema_model_judge:
            le_class_loss = class_criterion(le_class_logit, target_var) / minibatch_size
            re_class_loss = class_criterion(re_class_logit, target_var) / minibatch_size
            meters.update('le_class_loss', le_class_loss.data[0])
            meters.update('re_class_loss', re_class_loss.data[0])

        l_loss, r_loss = l_class_loss, r_class_loss

        l_loss += l_res_loss
        r_loss += r_res_loss

        l_loss += l_feature_loss
        r_loss += r_feature_loss

        # update ema loss values
        if args.ema_model_judge:
            l_ema_loss = (1 - args.ema_loss) * le_class_loss.data[0] + args.ema_loss * l_ema_loss
            r_ema_loss = (1 - args.ema_loss) * re_class_loss.data[0] + args.ema_loss * r_ema_loss
        else:
            l_ema_loss = (1 - args.ema_loss) * l_class_loss.data[0] + args.ema_loss * l_ema_loss
            r_ema_loss = (1 - args.ema_loss) * r_class_loss.data[0] + args.ema_loss * r_ema_loss
        meters.update('l_ema_loss', l_ema_loss)
        meters.update('r_ema_loss', r_ema_loss)
        
        consistency_loss = 0
        if args.consistency:
            consistency_weight = calculate_consistency_scale(epoch)
            meters.update('cons_weight', consistency_weight)

            le_class_logit = Variable(le_class_logit.detach().data, requires_grad=False)
            l_consistency_loss = consistency_weight * consistency_criterion(l_cons_logit, le_class_logit) / minibatch_size
            l_loss += l_consistency_loss
            meters.update('l_cons_loss', l_consistency_loss.data[0])


            re_class_logit = Variable(re_class_logit.detach().data, requires_grad=False)
            r_consistency_loss = consistency_weight * consistency_criterion(r_cons_logit, re_class_logit) / minibatch_size
            r_loss += r_consistency_loss
            meters.update('r_cons_loss', r_consistency_loss.data[0])

            # left model is better
            if l_ema_loss < r_ema_loss:
                # if l_class_loss.data[0] < r_class_loss.data[0]:
                mask = torch.max(F.softmax(l_class_logit, dim=1), 1)[0]
                mask = (mask > args.threshold)

                # --- TODO: how to setting cons between competitive models ---
                in_r_cons_logit = Variable(r_cons_logit.detach().data, requires_grad=False)
                tar_l_class_logit = Variable(l_class_logit.detach().data, requires_grad=False)

                for idx, value in enumerate(mask.data.cpu().numpy()):
                    if value == 0:
                        tar_l_class_logit[idx, ...] = in_r_cons_logit[idx, ...]

                consistency_loss = consistency_weight * consistency_criterion(r_cons_logit, tar_l_class_logit) / minibatch_size
                # ------------------------------------------------------------

                r_loss += consistency_loss
                meters.update('better_model', -1.)  # -1 == left model
                meters.update('cons_loss', consistency_loss.data[0])

            # right model is better
            elif l_ema_loss > r_ema_loss:
                # elif l_class_loss.data[0] > r_class_loss.data[0]:

                mask = torch.max(F.softmax(r_class_logit, dim=1), 1)[0]
                mask = (mask > args.threshold)

                # --- TODO: how to setting cons between competitive models ---
                in_l_cons_logit = Variable(l_cons_logit.detach().data, requires_grad=False)
                tar_r_class_logit = Variable(r_class_logit.detach().data, requires_grad=False)

                for idx, value in enumerate(mask.data.cpu().numpy()):
                    if value == 0:
                        tar_r_class_logit[idx, ...] = in_l_cons_logit[idx, ...]

                consistency_loss = consistency_weight * consistency_criterion(l_cons_logit, tar_r_class_logit) / minibatch_size
                # ------------------------------------------------------------
                
                l_loss += consistency_loss
                meters.update('better_model', 1.)    # 1 == right model
                meters.update('cons_loss', consistency_loss.data[0])

            else:
                consistency_loss = 0
                meters.update('cons_loss', 0)

        else:
            consistency_loss = 0
            meters.update('cons_loss', 0)


        assert not (np.isnan(l_loss.data[0]) or l_loss.data[0] > 1e5), 'L-Loss explosion: {}'.format(l_loss.data[0])
        assert not (np.isnan(r_loss.data[0]) or r_loss.data[0] > 1e5), 'R-Loss explosion: {}'.format(r_loss.data[0])
        meters.update('l_loss', l_loss.data[0])
        meters.update('r_loss', r_loss.data[0])

        l_prec1, l_prec5 = accuracy(l_class_logit.data, target_var.data, topk=(1, 5))
        meters.update('l_top1', l_prec1[0], labeled_minibatch_size)
        meters.update('l_error1', 100. - l_prec1[0], labeled_minibatch_size)
        meters.update('l_top5', l_prec5[0], labeled_minibatch_size)
        meters.update('l_error5', 100. - l_prec5[0], labeled_minibatch_size)

        r_prec1, r_prec5 = accuracy(r_class_logit.data, target_var.data, topk=(1, 5))
        meters.update('r_top1', r_prec1[0], labeled_minibatch_size)
        meters.update('r_error1', 100. - r_prec1[0], labeled_minibatch_size)
        meters.update('r_top5', r_prec5[0], labeled_minibatch_size)
        meters.update('r_error5', 100. - r_prec5[0], labeled_minibatch_size)

        le_prec1, le_prec5 = accuracy(le_class_logit.data, target_var.data, topk=(1, 5))
        meters.update('le_top1', le_prec1[0], labeled_minibatch_size)
        meters.update('le_error1', 100. - le_prec1[0], labeled_minibatch_size)
        meters.update('le_top5', le_prec5[0], labeled_minibatch_size)
        meters.update('le_error5', 100. - le_prec5[0], labeled_minibatch_size)

        re_prec1, re_prec5 = accuracy(re_class_logit.data, target_var.data, topk=(1, 5))
        meters.update('re_top1', re_prec1[0], labeled_minibatch_size)
        meters.update('re_error1', 100. - re_prec1[0], labeled_minibatch_size)
        meters.update('re_top5', re_prec5[0], labeled_minibatch_size)
        meters.update('re_error5', 100. - re_prec5[0], labeled_minibatch_size)

        l_optimizer.zero_grad()
        # l_cluster_loss.backward(retain_graph=True)
        l_loss.backward()
        l_optimizer.step()

        r_optimizer.zero_grad()
        # r_cluster_loss.backward(retain_graph=True)
        r_loss.backward()
        r_optimizer.step()

        update_ema_variables(l_model, le_model, args.ema_decay, global_step)
        update_ema_variables(r_model, re_model, args.ema_decay, global_step)

        global_step += 1
        meters.update('batch_time', time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            LOG.info('Epoch: [{0}][{1}/{2}]\t'
                     'Batch-T {meters[batch_time]:.3f}\t'
                    #  'Data-T {meters[data_time]:.3f}\t'
                     'L-EMA {meters[l_ema_loss]:.4f}\t'
                     'R-EMA {meters[r_ema_loss]:.4f}\t'
                     'L-Class {meters[l_class_loss]:.4f}\t'
                     'R-Class {meters[r_class_loss]:.4f}\t'
                     'L-Cons {meters[l_cons_loss]:.4f}\t'
                     'R-Cons {meters[r_cons_loss]:.4f}\t'
                     'R-Res {meters[l_res_loss]:.4f}\t'
                     'R-Res {meters[r_res_loss]:.4f}\t'
                     'Cons {meters[cons_loss]:.4f}\t'
                     'L-FEA {meters[l_feature_loss]:.4f}\t'
                     'R-FEA {meters[r_feature_loss]:.4f}\t'
                    #  'L-DIS {meters[l_distance_loss]:.4f}\t'
                    #  'R-DIS {meters[r_distance_loss]:.4f}\t'
                     'Better-M {better.sum:.1f}\n'
                     'L-Prec@1 {meters[l_top1]:.3f}\t'
                     'R-Prec@1 {meters[r_top1]:.3f}\t'
                     'LE-Prec@1 {meters[le_top1]:.3f}\t'
                     'RE-Prec@1 {meters[re_top1]:.3f}\t'
                     'L-Prec@5 {meters[l_top5]:.3f}\t'
                     'R-Prec@5 {meters[r_top5]:.3f}'.format(
                     epoch, i, len(train_loader), meters=meters, better=meters['better_model']))

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

    global tmp_path

    tmp_path = context.tmp_dir
    checkpoint_path = context.transient_dir
    training_log = context.create_train_log('training')
    l_validation_log = context.create_train_log('l_validation')
    r_validation_log = context.create_train_log('r_validation')

    dataset_config = datasets.__dict__[args.dataset]()
    num_classes = dataset_config.pop('num_classes')
    train_loader, eval_loader = create_data_loaders(**dataset_config, args=args)
    l_model = create_compite_model(side='l', num_classes=num_classes)
    r_model = create_compite_model(side='r', num_classes=num_classes)
    le_model = create_compite_model(side='le', num_classes=num_classes)
    re_model = create_compite_model(side='re', num_classes=num_classes)

    copy_model_variables(l_model, le_model)
    copy_model_variables(r_model, re_model)

    LOG.info(parameters_string(l_model))
    LOG.info(parameters_string(r_model))
    LOG.info(parameters_string(le_model))
    LOG.info(parameters_string(re_model))

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

    # l_ema_loss, r_ema_loss = calculate_train_ema_loss(train_loader, l_model, r_model)
    # calculate_initial_center(l_model, r_model, 10, 128, dataset_config['train_transformation'], dataset_config['datadir'], args=args)

    if args.evaluate:
        LOG.info('Evaluating the left model: ')
        validate(eval_loader, l_model, l_validation_log, global_step, args.start_epoch, name='el')
        LOG.info('Evaluating the right model: ')
        validate(eval_loader, r_model, r_validation_log, global_step, args.start_epoch, name='er')
        return

    for epoch in range(args.start_epoch, args.epochs):
        start_time = time.time()
        # train for one epoch
        train_epoch(train_loader, l_model, r_model, le_model, re_model, l_optimizer, r_optimizer, epoch, training_log)
        LOG.info('--- training epoch in {} seconds ---'.format(time.time()-start_time))

        is_best = False
        l_better = False
        if args.evaluation_epochs and (epoch + 1) % args.evaluation_epochs == 0:
            start_time = time.time()
            LOG.info('Evaluating the left model: ')
            l_prec1 = validate(eval_loader, l_model, l_validation_log, global_step, epoch + 1, name='l')
            LOG.info('Evaluating the right model: ')
            r_prec1 = validate(eval_loader, r_model, r_validation_log, global_step, epoch + 1, name='r')
            LOG.info('Evaluating the left ema model: ')
            l_prec1 = validate(eval_loader, le_model, l_validation_log, global_step, epoch + 1, name='el')
            LOG.info('Evaluating the right ema model: ')
            r_prec1 = validate(eval_loader, re_model, r_validation_log, global_step, epoch + 1, name='er')
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
