#pragma once

#include "./battle.hh"

template <typename Net>
struct CPUModel : BattleTypes
{

    struct ModelOutput
    {
        BattleTypes::VectorReal row_policy, col_policy;
        BattleTypes::Value value;
    };

    class Model
    {
    public:
        Net net;

        torch::Tensor input = torch::empty({1, 376});
        torch::Tensor joined_policy_indices = torch::empty({1, 18}, torch::kInt64);

        Model(std::string path)
        {
            torch::load(net, path);
            net->to(torch::kCPU);
            net->eval();
        }

        Model(const Net &other)
        {
            for (auto &targetParam : net.parameters())
            {
                for (const auto &sourceParam : other.parameters())
                {
                    if (targetParam.name() == sourceParam.name())
                    {
                        targetParam.data() = sourceParam.data().clone();
                        break;
                    }
                }
            }
            net->to(torch::kCPU);
            net->eval();
        }

        void inference(
            BattleTypes::State &&state,
            ModelOutput &output)
        {
            const size_t rows = state.row_actions.size();
            const size_t cols = state.col_actions.size();
            for (int byte = 0; byte < 376; ++byte)
            {
                input.index({0, byte}) = static_cast<float>(state.battle.bytes[byte]);
            }
            for (int row_idx = 0; row_idx < rows; ++row_idx)
            {
                joined_policy_indices.index({0, row_idx}) = get_index_from_action(state.battle.bytes, state.row_actions[row_idx].get(), 0);
            }
            for (int col_idx = 0; col_idx < cols; ++col_idx)
            {
                joined_policy_indices.index({0, 9 + col_idx}) = get_index_from_action(state.battle.bytes, state.col_actions[col_idx].get(), 184);
            }
            auto nn_output = net->forward(input, joined_policy_indices);
            torch::Tensor row_policy_tensor = torch::softmax(nn_output.row_log_policy, 1);
            torch::Tensor col_policy_tensor = torch::softmax(nn_output.col_log_policy, 1);
            output.row_policy.resize(rows);
            output.col_policy.resize(cols);
            for (int row_idx = 0; row_idx < rows; ++row_idx)
            {
                output.row_policy[row_idx] = BattleTypes::Real{row_policy_tensor.index({0, row_idx}).item().toFloat()};
            }
            for (int col_idx = 0; col_idx < cols; ++col_idx)
            {
                output.col_policy[col_idx] = BattleTypes::Real{col_policy_tensor.index({0, col_idx}).item().toFloat()};
            }
            output.value.row_value = BattleTypes::Real{nn_output.value.item().toFloat()};
        }
    };
};
