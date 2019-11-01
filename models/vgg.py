import numpy as np

import torch.nn as nn

from .modules.mylayers import GaussianPooling2d, GaussianPoolingCuda2d

__all__ = [
    'VGG', 'vgg11orig', 'vgg11orig_bn', 'vgg13orig', 'vgg13orig_bn', 'vgg16orig', 'vgg16orig_bn',
    'vgg19orig_bn', 'vgg19orig',
    'vgg11bow', 'vgg11bow_bn', 'vgg13bow', 'vgg13bow_bn', 'vgg16bow', 'vgg16bow_bn',
    'vgg19bow_bn', 'vgg19bow'
]

def _pooling(ptype, num_features, kernel_size, stride):
    if ptype == 'max':
        pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)
    elif ptype == 'avg':
        pool = nn.AvgPool2d(kernel_size=kernel_size, stride=stride)
    elif ptype == 'gauss_HWCN':
        pool = GaussianPooling2d(num_features=num_features, kernel_size=kernel_size, stride=stride)
    elif ptype == 'gauss_CN':
        pool = GaussianPooling2d(num_features=num_features, kernel_size=kernel_size, stride=stride, stochasticity='CN')
    elif ptype == 'gauss_cuda_HWCN':
        pool = GaussianPoolingCuda2d(num_features=num_features, kernel_size=kernel_size, stride=stride)
    elif ptype == 'gauss_cuda_CN':
        pool = GaussianPoolingCuda2d(num_features=num_features, kernel_size=kernel_size, stride=stride, stochasticity='CN')
    elif ptype == 'skip':
        pool = nn.Identity()
    else:
        raise ValueError("pooling type of {} is not supported.".format(ptype))
    return pool


class VGG(nn.Module):
    def __init__(self, features, num_classes=1000, init_weights=True, batchnorm=True):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        if batchnorm:
            self.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                nn.BatchNorm1d(4096, eps=1e-4),
                nn.ReLU(True),
                nn.Linear(4096, 4096),
                nn.BatchNorm1d(4096, eps=1e-4),
                nn.ReLU(True),
                nn.Linear(4096, num_classes),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(True),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Linear(4096, num_classes),
            )
    
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class VGGbow(VGG):
    def __init__(self, features, num_classes=1000, init_weights=True, batchnorm=True):
        super(VGGbow, self).__init__(features, num_classes=num_classes, init_weights=False)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        if batchnorm:
            self.classifier = nn.Sequential(
                nn.Linear(4096, 4096),
                nn.BatchNorm1d(4096, eps=1e-4), 
                nn.ReLU(True),
                nn.Linear(4096, num_classes),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Linear(4096, num_classes),
            )

        if init_weights:
            self._initialize_weights()


def make_layers(cfg, batch_norm=False, ptype='max'):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [_pooling(ptype=ptype, num_features=in_channels, kernel_size=2, stride=2)]
        elif v == 'W':
            NiN = nn.Conv2d(in_channels, 4096, kernel_size=1,  padding=0)
            if batch_norm:
                layers += [ NiN, nn.BatchNorm2d(4096, eps=1e-4), nn.ReLU(inplace=True)]
            else:
                layers += [ NiN, nn.ReLU(inplace=True)]
            in_channels = 4096
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v, eps=1e-4), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    'A': [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'], # 11
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'], # 13
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'], # 16
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'], # 19
}


def _vgg(arch, cfg, batch_norm, bow, ptype='max', **kwargs):
    seqs = list(cfgs[cfg])
    if not batch_norm:
        kwargs['batchnorm'] = False

    if bow:
        seqs[-1] = 'W'
        model = VGGbow(make_layers(seqs, batch_norm=batch_norm, ptype=ptype), **kwargs)
    else:
        model = VGG(make_layers(seqs, batch_norm=batch_norm, ptype=ptype), **kwargs)
    return model


#- VGG model -#

def vgg11orig(**kwargs):
    """VGG 11-layer model (configuration "A")
    """
    return _vgg('vgg11', 'A', False, False, **kwargs)


def vgg11orig_bn(**kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization
    """
    return _vgg('vgg11_bn', 'A', True, False, **kwargs)


def vgg13orig(**kwargs):
    """VGG 13-layer model (configuration "B")
    """
    return _vgg('vgg13', 'B', False, False, **kwargs)


def vgg13orig_bn(**kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization
    """
    return _vgg('vgg13_bn', 'B', True, False, **kwargs)


def vgg16orig(**kwargs):
    """VGG 16-layer model (configuration "D")
    """
    return _vgg('vgg16', 'D', False, False, **kwargs)


def vgg16orig_bn(**kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization
    """
    return _vgg('vgg16_bn', 'D', True, False, **kwargs)


def vgg19orig(**kwargs):
    """VGG 19-layer model (configuration "E")
    """
    return _vgg('vgg19', 'E', False, False, **kwargs)


def vgg19orig_bn(**kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization
    """
    return _vgg('vgg19_bn', 'E', True, False, **kwargs)


#- VGG-bow model -#

def vgg11bow(**kwargs):
    """VGG-bow 11-layer model (configuration "A")
    """
    return _vgg('vgg11', 'A', False, True, **kwargs)


def vgg11bow_bn(**kwargs):
    """VGG-bow 11-layer model (configuration "A") with batch normalization
    """
    return _vgg('vgg11_bn', 'A', True, True, **kwargs)


def vgg13bow(**kwargs):
    """VGG-bow 13-layer model (configuration "B")
    """
    return _vgg('vgg13', 'B', False, True, **kwargs)


def vgg13bow_bn(**kwargs):
    """VGG-bow 13-layer model (configuration "B") with batch normalization
    """
    return _vgg('vgg13_bn', 'B', True, True, **kwargs)


def vgg16bow(**kwargs):
    """VGG-bow 16-layer model (configuration "D")
    """
    return _vgg('vgg16', 'D', False, True, **kwargs)


def vgg16bow_bn(**kwargs):
    """VGG-bow 16-layer model (configuration "D") with batch normalization
    """
    return _vgg('vgg16_bn', 'D', True, True, **kwargs)


def vgg19bow(**kwargs):
    """VGG-bow 19-layer model (configuration "E")
    """
    return _vgg('vgg19', 'E', False, True, **kwargs)


def vgg19bow_bn(**kwargs):
    """VGG-bow 19-layer model (configuration 'E') with batch normalization
    """
    return _vgg('vgg19_bn', 'E', True, True, **kwargs)
