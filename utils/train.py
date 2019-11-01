import os
import shutil
import time

import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import torch
import torchvision


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Err@1', ':6.2f')
    top5 = AverageMeter('Err@5', ':6.2f')
    progress = ProgressMeter(len(train_loader), batch_time, data_time, losses, top1,
                             top5, prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        # if args.gpu is None, the default GPU device is selected.
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(100.0-acc1.item(), images.size(0))
        top5.update(100.0-acc5.item(), images.size(0))

        # compute gradient and do updating step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.print(i)
            plt.pause(.01)

    return top1.avg, top5.avg, losses.avg


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Err@1', ':6.2f')
    top5 = AverageMeter('Err@5', ':6.2f')
    progress = ProgressMeter(len(val_loader), batch_time, losses, top1, top5,
                             prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(100.0-acc1.item(), images.size(0))
            top5.update(100.0-acc5.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.print(i)
                plt.pause(.01)

        print(' *val* Err@1 {top1.avg:.3f} Err@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg, top5.avg, losses.avg


def save_checkpoint(state, is_best, save_last, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(os.path.dirname(filename),'model_best.pth.tar'))
    if save_last:
        last_filename = filename.replace('epoch'+str(state['epoch']), 'epoch'+str(state['epoch']-1))
        if os.path.isfile(last_filename):
            os.remove(last_filename)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, *meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def print(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the one specified by a user"""
    lr = args.lrs[epoch]
    for param_group in optimizer.param_groups:
        if 'lr_scale' in param_group.keys():
            param_group['lr'] = lr * param_group['lr_scale']
        else:
            param_group['lr'] = lr
    return lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class ProgressPlotter(object):
    """Shows the learning status by graphs via matplotlib.pyplot"""
    def __init__(self, titles=('Err.'), legends=(('train','val')), ylims=((0,100)), yscales=None, vals=(([],[])), figsize=(9.6,4.8) ):
        # figure window
        self.fig = plt.figure(figsize=figsize)
        # number of subplot
        self.num = len(titles)
        # axies for subplots
        self.ax = [self.fig.add_subplot(1, self.num, i+1) for i in range(self.num)]
        # titles & legends for subplots
        for i in range(self.num):
            for j in range(len(legends[i])):
                self.ax[i].plot(range(1,1+len(vals[i][j])),vals[i][j])
            self.ax[i].legend(legends[i])
            self.ax[i].set_title(titles[i])
            self.ax[i].set_ylim(bottom=ylims[i][0], top=ylims[i][1])
            if yscales is not None:
                self.ax[i].set_yscale(yscales[i])

    def plot(self, vals=((0,0)) ):
        for i in range(self.num):
            # xymax = [0, 0]
            for j in range(len(vals[i])):
                bx, by = self.ax[i].lines[j].get_data()
                bx = np.append(bx, len(bx)+1)
                by = np.append(by, vals[i][j])
                self.ax[i].lines[j].set_data(bx, by)
                self.ax[i].set_xlim(left=0, right=max(bx))
                # xymax = np.max([xymax, [max(bx),max(by)]], axis=0)
            # self.ax[i].set_xlim(left=0, right=xymax[0])
            # self.ax[i].set_ylim(bottom=0, top=xymax[1])
        plt.pause(.01)

    def save(self, filename='plot.pdf' ):
        self.fig.savefig(filename)