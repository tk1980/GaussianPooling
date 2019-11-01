#%%
import argparse
import os
import random
import shutil
import time
import warnings

import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from utils.train import validate, train, adjust_learning_rate, save_checkpoint, ProgressPlotter
import models as mymodels

import imagenet_config as cf

#%%
mymodel_names = sorted(name for name in mymodels.__dict__
    if name.islower() and not name.startswith("__")
    and callable(mymodels.__dict__[name]))

pool_names = ['max','avg','skip','gauss_cuda_HWCN','gauss_cuda_CN','gauss_HWCN','gauss_CN']

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
#Data
parser.add_argument('--data', metavar='DIR',default='./datasets/imagenet12/images/', type=str,
                    help='path to dataset')
parser.add_argument('--dataset', metavar='DATASET',default='imagenet', type=str,
                    help='dataset name')

#Network
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=mymodel_names,
                    help='model architecture: ' +
                        ' | '.join(mymodel_names) +
                        ' (default: resnet50)')
parser.add_argument('--pool', metavar='POOL', default='gauss_cuda_CN',
                    choices=pool_names,
                    help='pooling type: ' +
                        ' | '.join(pool_names) +
                        ' (default: gauss_cuda_CN)')
parser.add_argument('--config-name', default='imagenet', type=str, metavar='CONFIG',
                    help='config name in config file (default: imagenet)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')

#Utility
parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-p', '--print-freq', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--out-dir', default='./', type=str,
                    help='path to output directory (default: ./)')
parser.add_argument('--pdf-filename', default='train_epochs.pdf', type=str,
                    help='path to output file saving training statistics (default: train_epochs.pdf)')
parser.add_argument('--save-last-checkpoint', dest='save_last_checkpoint', action='store_true',
                    help='save only the last checkpoint')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

#Mode
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

#Multi-GPUs/Processing
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://127.0.0.1:8080', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

#%%
stats = {'train_err1': [], 'train_err5': [], 'train_loss': [],
         'test_err1': [],  'test_err5': [],  'test_loss': []}

def main():
    # parameters
    args = parser.parse_args()

    # parameters specified by config file
    params = cf.__dict__[args.config_name]
    for name in ('batch_size', 'lrs', 'momentum', 'weight_decay', 'train_transform', 'test_transform'):
        if name not in params.keys():
            print('parameter \'{}\' is not specified in config file.'.format(name))
            return
        args.__dict__[name] = params[name]
        print(name+':', params[name])
    args.epochs = len(args.lrs)

    # output directory
    os.makedirs(args.out_dir, exist_ok=True)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global stats
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    
    # create model
    if args.dataset == 'imagenet':
        model = mymodels.__dict__[args.arch](num_classes=1000, ptype=args.pool)
    print(model)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            # only model
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=torch.device('cpu'))
            for old_key in list(checkpoint['state_dict'].keys()):
                if 'module' in old_key:
                    new_key = old_key.replace('module.','')
                    checkpoint['state_dict'][new_key] = checkpoint['state_dict'].pop(old_key, None)
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {}) for model"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            return


    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int(args.workers / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # Data loading code
    if args.dataset == 'imagenet': 
        # ImageNet
        traindir = os.path.join(args.data, 'train')
        valdir = os.path.join(args.data, 'val_dir')

        train_dataset = datasets.ImageFolder(
            traindir,
            args.train_transform
            )
        val_dataset = datasets.ImageFolder(
            valdir, 
            args.test_transform
            )

    # Data Sampling
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lrs[0],
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        # other state parameters
        if os.path.isfile(args.resume):
            args.start_epoch = checkpoint['epoch']
            stats = checkpoint['stats']
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {}) for the others"
                  .format(args.resume, checkpoint['epoch']))

    cudnn.benchmark = True

    # Do Train/Eval
    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    primary_worker = not args.multiprocessing_distributed or \
                    (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0)

    if primary_worker:
        progress = ProgressPlotter( titles=('LR', 'Loss', 'Top-1 Error.', 'Top-5 Error.'), 
            legends=(('learning rate',),('train','val'),('train','val'),('train','val')), ylims=((1e-6,1),(0,10),(0,100),(0,100)),
            yscales=('log','linear','linear','linear'), 
            vals=((args.lrs[:args.start_epoch],), (stats['train_loss'],stats['test_loss']), (stats['train_err1'],stats['test_err1']), (stats['train_err5'],stats['test_err5']) ) ) 

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        lr = adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        trnerr1, trnerr5, trnloss = train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        valerr1, valerr5, valloss = validate(val_loader, model, criterion, args)

        # statistics
        stats['train_err1'].append(trnerr1)
        stats['train_err5'].append(trnerr5)
        stats['train_loss'].append(trnloss)
        stats['test_err1'].append(valerr1)
        stats['test_err5'].append(valerr5)
        stats['test_loss'].append(valloss)

        # remember best err@1
        is_best = valerr1 <= min(stats['test_err1'])

        # @ Primary worker, show and save results
        if primary_worker:
            # progress.plot( ((trnloss,valloss), (trnerr1, valerr1), (trnerr5, valerr5)) )
            progress.plot( ((lr,), (trnloss,valloss), (trnerr1, valerr1), (trnerr5, valerr5)) )
            progress.save(filename=os.path.join(args.out_dir, args.pdf_filename))
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'stats': stats,
                'optimizer' : optimizer.state_dict(),
                'args' : args
            }, is_best, args.save_last_checkpoint, filename=os.path.join(args.out_dir, 'checkpoint-epoch{:d}.pth.tar'.format(epoch+1)))

    # @ Primary worker, show the final results
    if primary_worker:
        minind = stats['test_err1'].index(min(stats['test_err1']))
        print(' *BEST* Err@1 {:.3f} Err@5 {:.3f}'.format(stats['test_err1'][minind], stats['test_err5'][minind]))


if __name__ == '__main__':
    main()