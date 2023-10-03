#pragma once

namespace Options {
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

    torch::Tensor forward(torch::Tensor x)
    {
        torch::Tensor residual = x.clone();
        x = torch::relu(fc2(torch::relu(fc1(x)))) + residual;
        return x;
    }
};

struct NetOutput {
    torch::Tensor value, row_logits, col_logits;
};

// underlying libtorch model. no game logic or anything yet
class Net : public torch::nn::Module
{
public:
    torch::Device device = torch::CUDA;
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

        torch::Tensor forward(torch::Tensor x)
    {
        x = torch::relu(fc(x));
        x = tower->forward(x);
        torch::Tensor value = torch::sigmoid(fc_value(torch::relu(fc_value_pre(x))));
        torch::Tensor row_logits = fc_row_logits(torch::relu(fc_row_logits_pre(x)));
        torch::Tensor col_logits = fc_col_logits(torch::relu(fc_col_logits_pre(x)));
        return value;
    }
};
