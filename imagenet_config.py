############### configuration file ###############
import numpy as np

import torch
import torchvision.transforms as transforms
import utils.mytransforms as mytransforms

#- Augmentation -#
Inception_Aug = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
    transforms.ToTensor(),
    mytransforms.Lighting(0.1, mytransforms.IMAGENET_PCA['eigval'], mytransforms.IMAGENET_PCA['eigvec']),
    transforms.Normalize(mytransforms.IMAGENET_STATS['mean'], mytransforms.IMAGENET_STATS['std'])
    ])

PyTorch_Aug = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mytransforms.IMAGENET_STATS['mean'], mytransforms.IMAGENET_STATS['std'])
    ])

ImageNet_Crop = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mytransforms.IMAGENET_STATS['mean'], mytransforms.IMAGENET_STATS['std'])
    ])

#----------Deep CNNs-------------#
imagenet = {
    'batch_size' : 256,
    'lrs' : [0.1]*30 + [0.01]*30 + [0.001]*30 + [0.0001]*30,
    'weight_decay' : 1e-4,
    'momentum': 0.9,
    'train_transform': Inception_Aug,
    'test_transform': ImageNet_Crop,
    'loss': {'name': 'Softmax'}
}