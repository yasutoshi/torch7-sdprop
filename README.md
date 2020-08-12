Torch7 implementation of SDProp
===============================================

This is a Torch7 implementation of SDProp described in the paper of [Adaptive Learning Rate via Covariance Matrix Based Preconditioning for Deep Neural Networks](https://www.ijcai.org/proceedings/2017/0267.pdf)(Y. Ida et al., IJCAI, 2017).

SDProp is an adaptive learning rate algorithm that effectively trains deep neural networks.

## Usage

Import:

```
require 'sdprop.lua'
```

You can use SDProp as a part of optim package as follows:

```
sdpropState = sdpropState or {}
optim.sdprop(feval, parameters, sdpropState)
```

You can also set hyper-parameters as follows:

```
sdpropState = sdpropState or {
  learningRate = 0.0001,
  gamma = 0.999
}
optim.sdprop(feval, parameters, sdpropState)
```

## Demos

You can try demos in folders of demos-mnist and demos-cifar.

These demos are created based on [Demos and Tutorials for Torch7](https://github.com/torch/demos).

Usage in demos-mnist:

```
th train-on-mnist.lua -optimization SDProp
```

Results:

- training accuracies

![mnist_train](https://user-images.githubusercontent.com/18375631/29749040-9771171e-8b66-11e7-9422-90d5792148d2.png)

- test accuracies

![mnist_test](https://user-images.githubusercontent.com/18375631/29749045-d30c98f2-8b66-11e7-926c-78daa1df11e2.png)

Usage in demos-cifar:

```
th train-on-cifar.lua -optimization SDProp
```

Results:

- training accuracies

![cifar_train](https://user-images.githubusercontent.com/18375631/29749046-debab3a0-8b66-11e7-98c0-6ee801693e59.png)

- test accuracies

![cifar_test](https://user-images.githubusercontent.com/18375631/29749048-e326c06e-8b66-11e7-8820-391426d09a62.png)

## Citation

```
@inproceedings{ijcai2017-267,
  author    = {Yasutoshi Ida, Yasuhiro Fujiwara, Sotetsu Iwamura},
  title     = {Adaptive Learning Rate via Covariance Matrix Based Preconditioning for Deep Neural Networks},
  booktitle = {Proceedings of the Twenty-Sixth International Joint Conference on
               Artificial Intelligence, {IJCAI-17}},
  pages     = {1923--1929},
  year      = {2017},
  doi       = {10.24963/ijcai.2017/267},
  url       = {https://doi.org/10.24963/ijcai.2017/267},
}
```
