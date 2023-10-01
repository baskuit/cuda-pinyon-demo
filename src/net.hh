#pragma once

#include <torch/torch.h>

class Net : public torch::nn::Module
{
public:
    torch::Device device = torch::kCUDA;
    torch::nn::Linear fc{nullptr};

    Net()
    {
        // input
        fc = register_module("fc_input", torch::nn::Linear(376, 1));
    }

    void to(const torch::Device &device)
    {
        fc->to(device);
        // static_cast<torch::nn::Module *>(this)->to(device);
        this->device = device;
    }

    torch::Tensor
    forward(torch::Tensor x)
    {
        torch::Tensor value = torch::sigmoid(fc(x));
        return value;
    }
};
