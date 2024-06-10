#include "ResNetImpl.h"

ResNetImpl::ResNetImpl() {
    conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 64, 3).bias(false).padding(1)));
    norm1 = register_module("norm1", torch::nn::BatchNorm2d(64));
    for (int i = 1; i <= 5; i++) {
        blocks.push_back(register_module("blck" + std::to_string(i), BasicBlock(64, 64)));
    }
    fltn1 = register_module("fltn1", torch::nn::Flatten());
    dnse1 = register_module("dnse1", torch::nn::Linear(3 * 3 * 64, 128));
    dnse2 = register_module("dnse2", torch::nn::Linear(128, 64));
    dnse3 = register_module("dnse3", torch::nn::Linear(64, 1));
}

torch::Tensor ResNetImpl::forward(torch::Tensor x) {
    x = torch::relu(norm1->forward(conv1->forward(x)));
    for (BasicBlock& block: blocks) {
        x = block->forward(x);
    }
    x = fltn1->forward(x);
    x = torch::relu(dnse1->forward(x));
    x = torch::relu(dnse2->forward(x));
    x = torch::tanh(dnse3->forward(x));
    return x;
}