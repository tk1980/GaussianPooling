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
import torchvision.models as models

from utils.train import validate
import utils.mydatasets as mydatasets
import models as mymodels

import imagenet_config as cf

#%%

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

mymodel_names = sorted(name for name in mymodels.__dict__
    if name.islower() and not name.startswith("__")
    and callable(mymodels.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Test')
#Data
parser.add_argument('--data', metavar='DIR',default='./datasets/imagenet12/images/', type=str,
                    help='path to dataset')
parser.add_argument('--dataset', metavar='DATASET',default='imagenet', type=str,
                    help='dataset name')

#Network
parser.add_argument('-a', '--arch', metavar='ARCH', default='vgg16bow_bn',
                    choices=model_names+mymodel_names,
                    help='model architecture: ' +
                        ' | '.join(model_names+mymodel_names) +
                        ' (default: vgg16bow_bn)')
parser.add_argument('--model-file', default='', type=str, metavar='PATH',
                    help='path to model file (default: none)')
parser.add_argument('--config-name', default='imagenet_largemargin', type=str, metavar='CONFIG',
                    help='config name in config file (default: imagenet_largemargin)')

#Utility
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-p', '--print-freq', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')

#Evaluation Mode
parser.add_argument('--nonblacklist', dest='nonblacklist', action='store_true',
                    help='exclude blacklisted validation image files')

#CPU/GPU
parser.add_argument('--cpu', dest='cpu', action='store_true',
                    help='do CPU mode')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')

#%%
def main():
    # parameters
    args = parser.parse_args()
    
    # parameters specified by config file
    params = cf.__dict__[args.config_name]
    args.test_transform = params['test_transform']

    args.distributed = False

    # Simply call main_worker function
    main_worker(args.gpu, args)


def main_worker(gpu, args):
    args.gpu = gpu

    if not args.cpu:
        if args.gpu is not None:
            print("Use GPU: {} for training".format(args.gpu))
    else:
        print("Use CPU")

    # create model
    if args.dataset == 'imagenet':
        if args.arch in mymodel_names:
            model = mymodels.__dict__[args.arch]()
        else:
            print("=> creating model '{}'".format(args.arch))
            model = models.__dict__[args.arch]()

    # load model
    if os.path.isfile(args.model_file):
        print("=> loading model '{}'".format(args.model_file))
        checkpoint = torch.load(args.model_file)
        d = checkpoint['state_dict']
        for old_key in list(d.keys()):
            if 'module.' in old_key:
                d[old_key.replace('module.','')] = d.pop(old_key,None) 
        model.load_state_dict(d)
        print("=> loaded model '{}'".format(args.model_file))
    else:
        print("=> no model found at '{}'".format(args.model_file))
        return

    if not args.cpu:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model = model.cuda(args.gpu)
        else:
            # DataParallel will divide and allocate batch_size to all available GPUs
            if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
                model.features = torch.nn.DataParallel(model.features)
                model.cuda()
            else:
                model = torch.nn.DataParallel(model).cuda()

        cudnn.benchmark = True

    # Data loading code
    if args.dataset == 'imagenet': 
        # ImageNet
        valdir = os.path.join(args.data, 'val_dir')

        if args.nonblacklist:
            val_dataset = mydatasets.ImageNetValFolder(
                valdir, 
                args.test_transform 
                )
            comment = 'non-blacklisted validation set'
        else:
            val_dataset = datasets.ImageFolder(
                valdir,
                args.test_transform 
                )
            comment = 'whole validation set'

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # define loss function (criterion)
    criterion = nn.CrossEntropyLoss()
    if not args.cpu:
        criterion = criterion.cuda(args.gpu)

    # evaluate on validation set
    validate(val_loader, model, criterion, args)

    # @ Primary worker, show the final results
    print('on {}'.format(comment))


if __name__ == '__main__':
    main()
