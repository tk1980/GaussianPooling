# Gaussian-Based Pooling for Convolutional Neural Networks

The Pytorch implementation for the NeurIPS2019 paper of "[Gaussian-Based Pooling for Convolutional Neural Networks](https://staff.aist.go.jp/takumi.kobayashi/publication/2019/NeurIPS2019.pdf)" by [Takumi Kobayashi](https://staff.aist.go.jp/takumi.kobayashi/).

### Citation

If you find our project useful in your research, please cite it as follows:

```
@inproceedings{kobayashi2019neurips,
  title={Gaussian-Based Pooling for Convolutional Neural Networks},
  author={Takumi Kobayashi},
  booktitle={Proceedings of the Thirty-third Conference on Neural Information Processing Systems (NeurIPS)},
  year={2019}
}
```

## Contents

1. [Introduction](#introduction)
2. [Install](#install)
3. [Usage](#usage)
4. [Results](#results)

## Introduction

This work proposes a local pooling method based on Gaussian-based probabilistic model.
The prior knowledge about the local pooling functionality enables us to define the inverse-softplus Gaussian distribution as a prior probabilistic model of local pooling.
During end-to-end training, the pooling output is stochastically drawn from the local probabilistic prior distribution whose parameters are adaptively estimated by GFGP [1].
At inference, we can leverage the "averaged" model of the prior to effectively compute the forward pass.
By simply replacing the existing pooling layer with the proposed one, we can enjoy performance improvement.
For the more detail, please refer to our [paper](https://staff.aist.go.jp/takumi.kobayashi/publication/2019/NeurIPS2019.pdf).

<!-- <img width=400 src="https://user-images.githubusercontent.com/53114307/67915023-26daf380-fbd5-11e9-8152-9089b910234d.png"> -->
<img width=400 src="https://user-images.githubusercontent.com/53114307/67915097-5f7acd00-fbd5-11e9-8c00-15768e310260.png">

Figure: inverse-softplus (iSP) Gaussian pooling

## Install

### Dependencies

- [Python3](https://www.python.org/downloads/)
- [PyTorch(>=1.0.0)](http://pytorch.org)

### Compile
Compile the CUDA-enabled pooling function as follows.
```bash
cd models/modules/gausspool_cuda
python setup.py build
cp build/lib.linux-<LINUX_ARCH>-<PYTHON_VER>/* build/
```

Note that, if you fail to compile it or don't like the compilation, you can also use the naive version of the pooling layer implemented by using simple Pytorch functions without compiling the CUDA codes; see [Training](#training).

## Usage

### Training
The iSP-Gaussian pooling layer is simply incorporated as in the other pooling layer by

```python
(CUDA)
from modules.mylayers import GaussianPoolingCuda2d
pool = GaussianPoolingCuda2d(num_features=num_features, kernel_size=kernel_size, stride=stride, padding=padding, stochasticity='CN')

(naive-Pytorch)
from modules.mylayers import GaussianPooling2d
pool = GaussianPooling2d(num_features=num_features, kernel_size=kernel_size, stride=stride, padding=padding, stochasticity='CN')
```

where `stochasticity` indicates whether we perform fully stochastic pooling (`stochasticity='HWCN'`) or partially stochastic one (`stochasticity='CN'`); see Section 2.4 in the [paper](https://staff.aist.go.jp/takumi.kobayashi/publication/2019/NeurIPS2019.pdf)).

For example, the ResNet-50 equipped with the iSP-Gaussian pooling is trained on ImageNet by

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python imagenet_train.py  --dataset imagenet  --data ./datasets/imagenet12/images/  --arch resnet50 --pool gauss_CN  --config-name imagenet  --out-dir ./results/imagenet/resnet50/max_ent/  --dist-url 'tcp://127.0.0.1:8080'  --dist-backend 'nccl'  --multiprocessing-distributed  --world-size 1  --rank 0
```

Note that the ImageNet dataset must be downloaded at `./datasets/imagenet12/` before the training.

## Results
These performance results are not the same as those reported in the paper because the methods were implemented by MatConvNet in the paper and accordingly trained in a (slightly) different training procedure.

#### ImageNet

| Network  | Pooling | Top-1 Err. |
|---|---|---|
| ResNet-50 [4]|  Skip | 23.45 |
| ResNet-50 [4]|  Gauss | 21.10 |

<!--
| VGG-16 mod [2]|  Max | 22.99 |
| VGG-16 mod [2]|  Gauss |  |
| VGG-16 [3]|  Max | 25.04 |
| VGG-16 [3]|  Gauss | |
| ResNeXt-50 [5]|  Skip | 22.42 |
| ResNeXt-50 [5]|  Gauss |  |
| DenseNet-169 [6]|  Avg. | 23.03 |
| DenseNet-169 [6]|  Gauss |  |
  -->

## References

[1] T. Kobayashi. "Global Feature Guided Local Pooling." In ICCV, pages 3365-3374, 2019. [pdf](https://staff.aist.go.jp/takumi.kobayashi/publication/2019/ICCV2019.pdf)

<!-- [2] T. Kobayashi. "Analyzing Filters Toward Efficient ConvNets." In CVPR, pages 5619-5628, 2018. [pdf](https://staff.aist.go.jp/takumi.kobayashi/publication/2018/CVPR2018.pdf) -->

<!-- [3] K. Simonyan and A. Zisserman. "Very Deep Convolutional Networks For Large-Scale Image Recognition." CoRR, abs/1409.1556, 2014. -->

[4] K. He, X. Zhang, S. Ren, and J. Sun. "Deep Residual Learning For Image Recognition." In CVPR, pages 770–778, 2016.

<!-- [5] S. Xie, R. Girshick, P. Dollar, Z. Tu, and K. He. "Aggregated Residual Transformations For Deep Neural Networks." In CVPR, pages 5987–5995, 2017. -->

<!-- [6] G. Huang, Z. Liu, L. Maaten and K.Q. Weinberger. "Densely Connected Convolutional Networks." In CVPR, pages 2261-2269, 2017. -->