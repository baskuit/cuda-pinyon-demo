#pragma once

#include <torch/torch.h>

template <typename A>
void copy_state_dict(A &target_model, const A &source_model)
{
    torch::autograd::GradMode::set_enabled(false);      // make parameters copying possible
    auto new_params = target_model->named_parameters(); // implement this
    auto params = source_model->named_parameters(true /*recurse*/);
    auto buffers = source_model->named_buffers(true /*recurse*/);
    for (auto &val : new_params)
    {
        auto name = val.key();
        auto *t = params.find(name);
        if (t != nullptr)
        {
            t->copy_(val.value());
            // std::cout << name << std::endl;
        }
        else
        {
            t = buffers.find(name);
            if (t != nullptr)
            {
                // std::cout << name << std::endl;
                t->copy_(val.value());
            }
        }
    }
    torch::autograd::GradMode::set_enabled(true);
}

// silly little helper function
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

namespace Options
{
    const int hidden_size = 1 << 7;
    const int input_size = 376;
    const int outer_size = 1 << 5;
    const int n_res_blocks = 4;
    const float dropout_rate = .5;
};

/*

ResNet of fully connected layers

*/

class ResBlockImpl : public torch::nn::Module
{
public:
    torch::nn::Linear fc1{nullptr};
    torch::nn::Linear fc2{nullptr};
    torch::nn::Dropout dropout1{nullptr};
    torch::nn::Dropout dropout2{nullptr};

    ResBlockImpl(
        const int hidden_size = Options::hidden_size,
        const double dropout_rate = Options::dropout_rate)
    {
        fc1 = register_module("fc1", torch::nn::Linear(hidden_size, hidden_size));
        fc2 = register_module("fc2", torch::nn::Linear(hidden_size, hidden_size));
        dropout1 = register_module("dropout1", torch::nn::Dropout(dropout_rate));
        dropout2 = register_module("dropout2", torch::nn::Dropout(dropout_rate));
    }

    torch::Tensor forward(torch::Tensor input)
    {
        torch::Tensor residual = input.clone();
        input = torch::relu(dropout2(fc2(torch::relu(dropout1(fc1(input)))))) + residual;
        return input;
    }
};

TORCH_MODULE(ResBlock);

struct NetOutput
{
    torch::Tensor value, row_log_policy, col_log_policy;
};

class FCResNetImpl : public torch::nn::Module
{
public:
    torch::nn::Linear fc{nullptr};
    torch::nn::Sequential tower{nullptr};
    torch::nn::Linear fc_value_pre{nullptr};
    torch::nn::Linear fc_value{nullptr};
    torch::nn::Linear fc_row_logits_pre{nullptr};
    torch::nn::Linear fc_row_logits{nullptr};
    torch::nn::Linear fc_col_logits_pre{nullptr};
    torch::nn::Linear fc_col_logits{nullptr};

    const int hidden_size;
    const int outer_size;
    const int n_res_blocks;

    FCResNetImpl(
        const int hidden_size = Options::hidden_size,
        const int outer_size = Options::outer_size,
        const int n_res_blocks = Options::n_res_blocks)
        : hidden_size{hidden_size}, outer_size{outer_size}, n_res_blocks{n_res_blocks}
    {
        fc = register_module("fc_input", torch::nn::Linear(Options::input_size, hidden_size));
        tower = register_module("tower", torch::nn::Sequential());
        for (int i = 0; i < n_res_blocks; ++i)
        {
            tower->push_back(register_module("b" + std::to_string(i), ResBlock(hidden_size)));
        }

        fc_value_pre = register_module("fc_value_pre", torch::nn::Linear(hidden_size, outer_size));
        fc_value = register_module("fc_value", torch::nn::Linear(outer_size, 1));
        fc_row_logits_pre = register_module("fc_row_logits_pre", torch::nn::Linear(hidden_size, outer_size));
        fc_row_logits = register_module("fc_row_logits", torch::nn::Linear(outer_size, policy_size - 1));
        fc_col_logits_pre = register_module("fc_col_logits_pre", torch::nn::Linear(hidden_size, outer_size));
        fc_col_logits = register_module("fc_col_logits", torch::nn::Linear(outer_size, policy_size - 1));
    }

    NetOutput forward(torch::Tensor input, torch::Tensor joined_policy_indices)
    {

        torch::Tensor tower_ = tower->forward(torch::relu(fc(input)));
        torch::Tensor row_logits = fc_row_logits(torch::relu(fc_row_logits_pre(tower_)));
        torch::Tensor col_logits = fc_col_logits(torch::relu(fc_col_logits_pre(tower_)));
        torch::Tensor row_policy_indices = joined_policy_indices.index({"...", torch::indexing::Slice{0, 9, 1}});
        torch::Tensor col_policy_indices = joined_policy_indices.index({"...", torch::indexing::Slice{9, 18, 1}});
        torch::Tensor neg_inf = -1 * (1 << 10) * torch::ones({input.size(0), 1}, torch::kInt64).to(input.device());
        torch::Tensor row_logits_picked = torch::gather(torch::cat({neg_inf, row_logits}, 1), 1, row_policy_indices);
        torch::Tensor col_logits_picked = torch::gather(torch::cat({neg_inf, col_logits}, 1), 1, col_policy_indices);
        torch::Tensor r = torch::log_softmax(row_logits_picked, 1);
        torch::Tensor c = torch::log_softmax(col_logits_picked, 1);
        torch::Tensor value = torch::sigmoid(fc_value(torch::relu(fc_value_pre(tower_))));
        return {value, r, c};
    }
};

TORCH_MODULE(FCResNet);

/*

MHA

*/

class InputLayer : public torch::nn::Module
{
public:
    // auto row_1_slice = torch::indexing::Slice(0, 24, 1);
    // auto row_2_slice = torch::indexing::Slice(24, 48, 1);
    // auto row_3_slice = torch::indexing::Slice(48, 72, 1);
    // auto row_4_slice = torch::indexing::Slice(72, 96, 1);
    // auto row_5_slice = torch::indexing::Slice(96, 120, 1);
    // auto row_6_slice = torch::indexing::Slice(96, 120, 1);

    // torch::Tensor forward (torch::Tensor input) {
    //     torch::Tensor row_1 = input.index({"...", row_1_slice});
    // }
};

class MHANetImpl : public torch::nn::Module
{
public:
    // torch::nn::Linear fc{nullptr};
    // torch::nn::Sequential tower{};
    // torch::nn::Linear fc_value_pre{nullptr};
    // torch::nn::Linear fc_value{nullptr};
    // torch::nn::Linear fc_row_logits_pre{nullptr};
    // torch::nn::Linear fc_row_logits{nullptr};
    // torch::nn::Linear fc_col_logits_pre{nullptr};
    // torch::nn::Linear fc_col_logits{nullptr};

    // MHANetImpl()
    // {
    //     fc = register_module("fc_input", torch::nn::Linear(Options::input_size, Options::hidden_size));

    //     fc_value_pre = register_module("fc_value_pre", torch::nn::Linear(Options::hidden_size, Options::outer_size));
    //     fc_value = register_module("fc_value", torch::nn::Linear(Options::outer_size, 1));
    //     fc_row_logits_pre = register_module("fc_row_logits_pre", torch::nn::Linear(Options::hidden_size, Options::outer_size));
    //     fc_row_logits = register_module("fc_row_logits", torch::nn::Linear(Options::outer_size, policy_size - 1));
    //     fc_col_logits_pre = register_module("fc_col_logits_pre", torch::nn::Linear(Options::hidden_size, Options::outer_size));
    //     fc_col_logits = register_module("fc_col_logits", torch::nn::Linear(Options::outer_size, policy_size - 1));
    // }

    // void to(const torch::Device &device)
    // {
    //     static_cast<torch::nn::Module *>(this)->to(device);
    //     this->device = device;
    // }

    // NetOutput forward(torch::Tensor input, torch::Tensor joined_policy_indices)
    // {
    //     // torch::Tensor neg_inf = -1 * (1 << 10) * torch::ones({input.size(0), 1}, torch::kInt64).to(this->device);
    //     // torch::Tensor row_logits_picked = torch::gather(torch::cat({neg_inf, row_logits}, 1), 1, row_policy_indices);
    //     // torch::Tensor col_logits_picked = torch::gather(torch::cat({neg_inf, col_logits}, 1), 1, col_policy_indices);
    //     // torch::Tensor r = torch::log_softmax(row_logits_picked, 1);
    //     // torch::Tensor c = torch::log_softmax(col_logits_picked, 1);
    //     // torch::Tensor value = torch::sigmoid(fc_value(torch::relu(fc_value_pre(tower_))));
    //     // return {value, r, c};
    // }
};

TORCH_MODULE(MHANet);