#pragma once

#include <torch/torch.h>

namespace Options
{
    const int hidden_size = 1 << 7;
    const int input_size = 376;
    const int outer_size = 1 << 5;
    const int policy_size = 151 + 165;
    const int n_res_blocks = 4;
};

class ResBlock : public torch::nn::Module
{
public:
    torch::nn::Linear fc1{nullptr};
    torch::nn::Linear fc2{nullptr};

    ResBlock()
    {
        fc1 = register_module("fc1", torch::nn::Linear(Options::hidden_size, Options::hidden_size));
        fc2 = register_module("fc2", torch::nn::Linear(Options::hidden_size, Options::hidden_size));
    }

    torch::Tensor forward(torch::Tensor input)
    {
        torch::Tensor residual = input.clone();
        input = torch::relu(fc2(torch::relu(fc1(input)))) + residual;
        return input;
    }
};

struct NetOutput
{
    torch::Tensor value, row_policy, col_policy;
};

void pt(torch::Tensor tensor)
{
    int a = tensor.size(0);
    int b = tensor.size(1);
    std::cout << "size: " << a << ' ' << b << std::endl; 
    for (int i = 0; i < a; ++i)
    {
        for (int j = 0; j < b; ++j)
        {
            std::cout << tensor.index({i, j}).item().toFloat() << " ";
        }
        std::cout << std::endl;
    }
}

// underlying libtorch model. no game logic or anything yet
class Net : public torch::nn::Module
{
public:
    torch::Device device = torch::kCUDA;
    torch::nn::Linear fc{nullptr};
    torch::nn::Sequential tower{};
    torch::nn::Linear fc_value_pre{nullptr};
    torch::nn::Linear fc_value{nullptr};
    torch::nn::Linear fc_row_logits_pre{nullptr};
    torch::nn::Linear fc_row_logits{nullptr};
    torch::nn::Linear fc_col_logits_pre{nullptr};
    torch::nn::Linear fc_col_logits{nullptr};
    Net()
    {
        // input
        fc = register_module("fc_input", torch::nn::Linear(Options::input_size, Options::hidden_size));
        // tower
        for (int i = 0; i < Options::n_res_blocks; ++i)
        {
            auto block = std::make_shared<ResBlock>();
            tower->push_back(register_module("b" + std::to_string(i), block));
        }
        // register_module("tower", tower);
        // value
        fc_value_pre = register_module("fc_value_pre", torch::nn::Linear(Options::hidden_size, Options::outer_size));
        fc_value = register_module("fc_value", torch::nn::Linear(Options::outer_size, 1));
        // policy
        fc_row_logits_pre = register_module("fc_row_logits_pre", torch::nn::Linear(Options::hidden_size, Options::outer_size));
        fc_row_logits = register_module("fc_row_logits", torch::nn::Linear(Options::outer_size, policy_size));
        fc_col_logits_pre = register_module("fc_col_logits_pre", torch::nn::Linear(Options::hidden_size, Options::outer_size));
        fc_col_logits = register_module("fc_col_logits", torch::nn::Linear(Options::outer_size, policy_size));
    }

    void to(const torch::Device &device)
    {
        static_cast<torch::nn::Module *>(this)->to(device);
        this->device = device;
    }

    NetOutput forward(torch::Tensor input, torch::Tensor joined_policy_indices)
    {
        // pt(input.index({"...", torch::indexing::Slice{0, 24, 1}}));
        torch::Tensor tower_ = tower->forward(torch::relu(fc(input)));
        // torch::Tensor a = torch::randint(1, 100, {64, 19}).to(this->device);
        torch::Tensor row_logits = fc_row_logits(torch::relu(fc_row_logits_pre(tower_)));
        torch::Tensor col_logits = fc_col_logits(torch::relu(fc_col_logits_pre(tower_)));
        torch::Tensor row_policy_indices = joined_policy_indices.index({"...", torch::indexing::Slice{0, 9, 1}});
        torch::Tensor col_policy_indices = joined_policy_indices.index({"...", torch::indexing::Slice{9, 18, 1}});
        torch::Tensor row_logits_picked = torch::gather(row_logits, 1, row_policy_indices);
        torch::Tensor col_logits_picked = torch::gather(col_logits, 1, col_policy_indices);
        torch::Tensor r = torch::log_softmax(row_logits_picked, 1);
        torch::Tensor c = torch::log_softmax(col_logits_picked, 1);
        // pt(joined_policy_indices);
        // pt(row_policy_indices);
        // pt(col_policy_indices);
        // pt(row_logits_picked);
        // pt(col_logits_picked);
        // pt(r);
        // pt(c);
        torch::Tensor value = torch::sigmoid(fc_value(torch::relu(fc_value_pre(tower_))));
        return {value, r, c};
    }
};
