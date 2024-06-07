#include <torch/torch.h>
#include "BasicBlockImpl.h"

struct Network: torch::nn::Module {
    torch::nn::Conv2d conv1{nullptr};
    torch::nn::BatchNorm2d norm1{nullptr};
    std::vector<BasicBlock> blocks;
    torch::nn::Flatten fltn1{nullptr};
    torch::nn::Linear dnse1{nullptr}, dnse2{nullptr}, dnse3{nullptr};
    
    Network();
    torch::Tensor forward(torch::Tensor x);
};