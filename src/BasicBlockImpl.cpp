#include "BasicBlockImpl.h"

BasicBlockImpl::BasicBlockImpl(int ichannels, int ochannels) {
    conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(ichannels, ochannels, 3).bias(false).padding(1)));
    norm1 = register_module("norm1", torch::nn::BatchNorm2d(ochannels));
    conv2 = register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(ochannels, ochannels, 3).bias(false).padding(1)));
    norm2 = register_module("norm2", torch::nn::BatchNorm2d(ochannels));
}

torch::Tensor BasicBlockImpl::forward(torch::Tensor x) {
    torch::Tensor identity = x;
    x = torch::relu(norm1->forward(conv1->forward(x)));
    x = torch::relu(norm2->forward(conv2->forward(x)));
    return x;
}