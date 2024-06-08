#include <torch/torch.h>
#include "BasicBlockImpl.h"

struct ResNetImpl: torch::nn::Module {
    torch::nn::Conv2d conv1{nullptr};
    torch::nn::BatchNorm2d norm1{nullptr};
    std::vector<BasicBlock> blocks;
    torch::nn::Flatten fltn1{nullptr};
    torch::nn::Linear dnse1{nullptr}, dnse2{nullptr};
    
    ResNetImpl();
    torch::Tensor forward(torch::Tensor x);
};

TORCH_MODULE(ResNet);