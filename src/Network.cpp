#include "Network.h"

Network::Network() {
    conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 128, 3).bias(false).padding(1)));
    norm1 = register_module("norm1", torch::nn::BatchNorm2d(128));
    for (int i = 1; i <= 10; i++) {
        blocks.push_back(register_module("blck" + std::to_string(i), BasicBlock(128, 128)));
    }
    fltn1 = register_module("fltn1", torch::nn::Flatten());
    dnse1 = register_module("dnse1", torch::nn::Linear(3 * 3 * 128, 256));
    dnse2 = register_module("dnse2", torch::nn::Linear(256, 128));
    dnse3 = register_module("dnse3", torch::nn::Linear(128, 1));
}

torch::Tensor Network::forward(torch::Tensor x) {
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