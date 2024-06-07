#include <torch/torch.h>

struct BasicBlockImpl: torch::nn::Module {
    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
    torch::nn::BatchNorm2d norm1{nullptr}, norm2{nullptr};
    
    BasicBlockImpl(int ichannels, int ochannels);
    torch::Tensor forward(torch::Tensor x);
};

TORCH_MODULE(BasicBlock);