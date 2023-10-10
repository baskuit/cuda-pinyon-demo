#pragma once

#include <torch/torch.h>

namespace Options
{
    const int hidden_size = 1 << 7;
    const int input_size = 376;
    const int outer_size = 1 << 5;
    const int n_res_blocks = 4;
};

struct NetOutput
{
    torch::Tensor value, row_log_policy, col_log_policy;
};

class InputLayer : public torch::nn::Module {
public:

    auto row_1_slice = torch::indexing::Slice(0, 24, 1);
    auto row_2_slice = torch::indexing::Slice(24, 48, 1);
    auto row_3_slice = torch::indexing::Slice(48, 72, 1);
    auto row_4_slice = torch::indexing::Slice(72, 96, 1);
    auto row_5_slice = torch::indexing::Slice(96, 120, 1);
    auto row_6_slice = torch::indexing::Slice(96, 120, 1);

    torch::Tensor forward (torch::Tensor input) {
        torch::Tensor row_1 = input.index({"...", row_1_slice});
    }

};

class MHANetImpl : public torch::nn::Module
{
public:
    torch::nn::Linear fc{nullptr};
    torch::nn::Sequential tower{};
    torch::nn::Linear fc_value_pre{nullptr};
    torch::nn::Linear fc_value{nullptr};
    torch::nn::Linear fc_row_logits_pre{nullptr};
    torch::nn::Linear fc_row_logits{nullptr};
    torch::nn::Linear fc_col_logits_pre{nullptr};
    torch::nn::Linear fc_col_logits{nullptr};

    MHANetImpl()
    {
        fc = register_module("fc_input", torch::nn::Linear(Options::input_size, Options::hidden_size));

        fc_value_pre = register_module("fc_value_pre", torch::nn::Linear(Options::hidden_size, Options::outer_size));
        fc_value = register_module("fc_value", torch::nn::Linear(Options::outer_size, 1));
        fc_row_logits_pre = register_module("fc_row_logits_pre", torch::nn::Linear(Options::hidden_size, Options::outer_size));
        fc_row_logits = register_module("fc_row_logits", torch::nn::Linear(Options::outer_size, policy_size - 1));
        fc_col_logits_pre = register_module("fc_col_logits_pre", torch::nn::Linear(Options::hidden_size, Options::outer_size));
        fc_col_logits = register_module("fc_col_logits", torch::nn::Linear(Options::outer_size, policy_size - 1));
    }

    void to(const torch::Device &device)
    {
        static_cast<torch::nn::Module *>(this)->to(device);
        this->device = device;
    }

    NetOutput forward(torch::Tensor input, torch::Tensor joined_policy_indices)
    {
        // torch::Tensor neg_inf = -1 * (1 << 10) * torch::ones({input.size(0), 1}, torch::kInt64).to(this->device);
        // torch::Tensor row_logits_picked = torch::gather(torch::cat({neg_inf, row_logits}, 1), 1, row_policy_indices);
        // torch::Tensor col_logits_picked = torch::gather(torch::cat({neg_inf, col_logits}, 1), 1, col_policy_indices);
        // torch::Tensor r = torch::log_softmax(row_logits_picked, 1);
        // torch::Tensor c = torch::log_softmax(col_logits_picked, 1);
        // torch::Tensor value = torch::sigmoid(fc_value(torch::relu(fc_value_pre(tower_))));
        // return {value, r, c};
    }
};

TORCH_MODULE(MHANet);
