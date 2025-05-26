// file: main.cpp
// (Content is identical to the last version provided in the prompt, as it's stable for current debugging needs)
#include "models.h"
#include "gradient_checker.h"
#include <vector>
#include <string>
#include <memory>
#include <random>
#include <iostream>
#include <cmath>
#include <map>
#include <iomanip>
#include <cstdio>

// Forward declare functions defined later in this file
void demonstrate_broadcasting();
void training_example();
// AdamOptimizer class is defined below

class AdamOptimizer {
private:
    float lr_, beta1_, beta2_, eps_;
    int t_;
    std::map<Layer*, std::map<std::string, Tensor>> m_states_;
    std::map<Layer*, std::map<std::string, Tensor>> v_states_;

public:
    AdamOptimizer(float lr = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f)
            : lr_(lr), beta1_(beta1), beta2_(beta2), eps_(eps), t_(0) {}

    void step(const std::vector<Layer*>& layers_with_grads) {
        t_++;
        for (auto* layer : layers_with_grads) {
            if(layer->parameters().empty()) continue;

            for (auto& [param_name, param_tensor_ref] : layer->get_parameters_mut()) {
                if (layer->gradients().find(param_name) == layer->gradients().end() ||
                    layer->gradients().at(param_name).empty()) {
                    std::cout << "Warning: No gradient or empty gradient for parameter " << param_name
                              << " in layer " << layer->name() << ". Skipping update." << std::endl;
                    continue;
                }
                const Tensor& grad = layer->gradients().at(param_name);
                if (grad.rows() != param_tensor_ref.rows() || grad.cols() != param_tensor_ref.cols()) {
                    std::cerr << "Error: Mismatch between param and grad shape for " << param_name << " in " << layer->name() << std::endl;
                    std::cerr << "Param shape: " << param_tensor_ref.rows() << "x" << param_tensor_ref.cols() << std::endl;
                    std::cerr << "Grad shape: " << grad.rows() << "x" << grad.cols() << std::endl;
                    continue;
                }

                if (m_states_[layer].find(param_name) == m_states_[layer].end()) {
                    m_states_[layer][param_name] = Tensor::zeros(param_tensor_ref.rows(), param_tensor_ref.cols());
                    v_states_[layer][param_name] = Tensor::zeros(param_tensor_ref.rows(), param_tensor_ref.cols());
                }

                Tensor& m = m_states_[layer][param_name];
                Tensor& v = v_states_[layer][param_name];

                m = (m * beta1_) + (grad * (1.0f - beta1_));
                Tensor grad_squared = Tensor(grad * grad);
                v = (v * beta2_) + (grad_squared * (1.0f - beta2_));
                Tensor m_hat = m * (1.0f / (1.0f - std::pow(beta1_, t_)));
                Tensor v_hat = v * (1.0f / (1.0f - std::pow(beta2_, t_)));
                Tensor v_hat_sqrt = v_hat.sqrt();
                Tensor denominator = v_hat_sqrt + Tensor(eps_);
                Tensor update_val = m_hat / denominator;

                param_tensor_ref -= (update_val * lr_);
            }
        }
    }
};

void demonstrate_broadcasting() { // Definition moved here or ensure it's before main()
    std::mt19937 gen(42);
    auto batch_data = Tensor::randn(32, 128, gen);
    auto bias = Tensor::randn(1, 128, gen);
    auto scale = Tensor::randn(32, 1, gen);

    Tensor result1 = batch_data + bias;
    Tensor result2 = batch_data * scale;
    Tensor result3 = result1 * result2;

    std::cout << "Broadcasting demo completed successfully!" << std::endl;
    std::cout << "Result1 shape: " << result1.rows() << " x " << result1.cols() << std::endl;
    std::cout << "Result2 shape: " << result2.rows() << " x " << result2.cols() << std::endl;
    std::cout << "Result3 shape: " << result3.rows() << " x " << result3.cols() << std::endl;
}


void training_example() {
    TemporalConfig config;
    config.batch_size = 4;
    config.input_size = 3;
    config.hidden_size = 8;
    config.num_layers = 1;
    config.learning_rate = 0.005f;
    config.dropout_rate = 0.0f; // Ensure dropout is off for this example if not specifically testing it
    config.use_layer_norm = false;
    config.seed = 42;
    config.tbptt_steps = 5;

    std::string model_prefix = "bptt_predictor";
    TemporalPredictor model(config, 1, model_prefix);
    AdamOptimizer optimizer(config.learning_rate);
    MSELoss loss_fn;

    std::mt19937 data_gen(config.seed + 1);
    std::vector<Tensor> current_hidden_states(config.num_layers);
    for (int i = 0; i < config.num_layers; ++i) {
        current_hidden_states[i] = Tensor::zeros(config.batch_size, config.hidden_size);
    }

    int num_total_sequence_steps = 50;
    for (int step_idx = 0; step_idx < num_total_sequence_steps; step_idx += config.tbptt_steps) {
        GradientCache bptt_cache;
        model.zero_grad_all_layers();
        float total_bptt_loss = 0.0f;

        std::vector<Tensor> h_for_bptt_window = current_hidden_states;
        std::vector<std::string> h_prev_names_for_bptt_window_grad(config.num_layers);
        for(int i=0; i<config.num_layers; ++i) {
            h_prev_names_for_bptt_window_grad[i] = make_name(model_prefix, "h_prev_win_L" + std::to_string(i), step_idx);
        }
        std::string dt_name_for_window_grad = make_name(model_prefix, "dt_input_win", step_idx);

        for (int t = 0; t < config.tbptt_steps; ++t) {
            int current_seq_step = step_idx + t;
            if (current_seq_step >= num_total_sequence_steps) break;

            auto x_t_val = Tensor::randn(config.batch_size, config.input_size, data_gen);
            auto dt_t_val = Tensor::ones(config.batch_size, 1);
            auto target_t_val = Tensor::randn(config.batch_size, 1, data_gen);
            std::string x_t_name_for_grad_step = make_name(model_prefix, "x_input", current_seq_step);

            model.set_training_mode_all_layers(true);
            auto step_result = model.process_step(
                    x_t_val, x_t_name_for_grad_step,
                    dt_t_val, dt_name_for_window_grad,
                    nullptr,
                    target_t_val,
                    h_for_bptt_window,
                    h_prev_names_for_bptt_window_grad,
                    bptt_cache, loss_fn, true, current_seq_step
            );
            total_bptt_loss += step_result.loss_value;
        }

        if (!bptt_cache.backward_fns.empty()) {
            bptt_cache.backward();
            optimizer.step(model.get_all_trainable_layers());
        }
        current_hidden_states = h_for_bptt_window; // Update hidden states for next BPTT window

        if (config.tbptt_steps > 0) {
            int steps_in_window = std::min(config.tbptt_steps, num_total_sequence_steps - step_idx);
            if (steps_in_window > 0) {
                std::cout << "BPTT Window starting at step " << step_idx
                          << ", Average Loss in window: " << (total_bptt_loss / steps_in_window) << std::endl;
            }
        }
    }
    std::cout << "BPTT training loop example completed." << std::endl;
}


int main() {
    try {
        std::cout << "=== IMPROVED TENSOR FRAMEWORK DEMO ===" << std::endl;

        demonstrate_broadcasting();
        std::cout << std::endl;

        std::cout << "=== SIMPLE MLP TEST ===" << std::endl;
        std::mt19937 mlp_gen(123);
        SimpleMLP mlp({10, 20, 5, 1}, mlp_gen, "mlp1");

        Tensor mlp_input = Tensor::randn(4, 10, mlp_gen);
        std::string mlp_input_name = "mlp1_input";
        std::string mlp_output_name = "mlp1_output";
        Tensor mlp_target = Tensor::ones(4, 1);
        MSELoss mlp_loss_fn_for_check;

        GradientChecker::check_gradients_for_model(
                mlp, mlp_input, mlp_target, mlp_loss_fn_for_check,
                mlp_input_name, mlp_output_name,
                1e-4, 2e-2 // Adjusted tolerance for MLP
        );

        GradientCache mlp_train_cache;
        mlp.zero_grad();
        mlp.set_training_mode(true);
        Tensor mlp_train_output = mlp.forward(mlp_input, mlp_train_cache, mlp_input_name, mlp_output_name);
        std::cout << "MLP Output shape: " << mlp_train_output.rows() << "x" << mlp_train_output.cols() << std::endl;
        MSELoss mlp_train_loss_fn;
        float mlp_train_loss_val = mlp_train_loss_fn.forward(mlp_train_output, mlp_target);
        std::cout << "MLP Training Loss: " << mlp_train_loss_val << std::endl;
        mlp_train_loss_fn.backward(mlp_train_output, mlp_target, mlp_train_cache, mlp_output_name);
        mlp_train_cache.backward();
        std::cout << "MLP training backward pass completed." << std::endl;
        AdamOptimizer mlp_optimizer(0.01f);
        mlp_optimizer.step(mlp.get_all_layers_for_optimizer());
        std::cout << "MLP optimizer step completed." << std::endl;


        std::cout << "\n=== GRADIENT CHECK FOR SINGLE LINEAR LAYER ===" << std::endl;
        std::mt19937 single_gen(777);
        Linear single_linear_layer(5, 2, true, &single_gen, "single_linear_gc");
        Tensor single_layer_input = Tensor::randn(3, 5, single_gen);
        Tensor single_layer_target = Tensor::randn(3, 2, single_gen);
        MSELoss single_layer_loss_fn_gc;
        std::string sll_input_name_gc = "sll_input_gc";
        std::string sll_output_name_gc = "sll_output_gc";

        GradientChecker::check_gradients_for_model(
                single_linear_layer,
                single_layer_input,
                single_layer_target,
                single_layer_loss_fn_gc,
                sll_input_name_gc,
                sll_output_name_gc,
                1e-4, 1e-3
        );

        std::cout << "\n=== GRADIENT CHECK FOR TEMPORAL PREDICTOR (SINGLE STEP) ===" << std::endl;
        TemporalConfig tp_config_gc;
        tp_config_gc.batch_size = 2;
        tp_config_gc.input_size = 3;
        tp_config_gc.hidden_size = 4;
        tp_config_gc.num_layers = 1;
        tp_config_gc.use_exponential_decay = true;
        tp_config_gc.seed = 789;
        tp_config_gc.dropout_rate = 0.0f;
        tp_config_gc.use_layer_norm = true;

        std::string tp_model_prefix_gc = "tp_grad_check_model";
        TemporalPredictor tp_model_gc(tp_config_gc, 2, tp_model_prefix_gc);

        MSELoss tp_loss_fn_gc;
        std::mt19937 tp_data_gen_gc(tp_config_gc.seed + 10);

        Tensor x_t_gc = Tensor::randn(tp_config_gc.batch_size, tp_config_gc.input_size, tp_data_gen_gc);
        Tensor dt_t_gc = Tensor::ones(tp_config_gc.batch_size, 1);
        Tensor target_t_gc = Tensor::randn(tp_config_gc.batch_size, 2, tp_data_gen_gc);

        std::vector<Tensor> h_prev_gc_initial(tp_config_gc.num_layers);
        std::vector<std::string> h_prev_names_gc_initial(tp_config_gc.num_layers);
        for(int l=0; l<tp_config_gc.num_layers; ++l) {
            h_prev_gc_initial[l] = Tensor::zeros(tp_config_gc.batch_size, tp_config_gc.hidden_size);
            h_prev_names_gc_initial[l] = make_name(tp_model_prefix_gc, "h_prev_L" + std::to_string(l) + "_gc_step", 0);
        }
        std::string x_t_name_gc_step = make_name(tp_model_prefix_gc, "x_t_gc_step", 0);
        std::string dt_t_name_gc_step = make_name(tp_model_prefix_gc, "dt_t_gc_step", 0);
        int time_step_idx_for_grad_check = 0;

        GradientCache tp_analytical_cache_gc;
        tp_model_gc.zero_grad_all_layers();
        tp_model_gc.set_training_mode_all_layers(true);

        std::vector<Tensor> h_prev_for_analytical_call = h_prev_gc_initial;
        std::vector<std::string> h_prev_names_for_analytical_call = h_prev_names_gc_initial;

        TemporalPredictor::StepResult analytical_step_result = tp_model_gc.process_step(
                x_t_gc, x_t_name_gc_step,
                dt_t_gc, dt_t_name_gc_step,
                nullptr,
                target_t_gc,
                h_prev_for_analytical_call,
                h_prev_names_for_analytical_call,
                tp_analytical_cache_gc,
                tp_loss_fn_gc,
                true,
                time_step_idx_for_grad_check
        );
        tp_analytical_cache_gc.backward();

        GradientChecker::check_gradients_for_model(
                tp_model_gc,
                Tensor(),
                target_t_gc,
                tp_loss_fn_gc,
                "",
                "",
                1e-5, // Epsilon for numerical gradient
                6e-2, // Tolerance for TP
                &x_t_gc, &x_t_name_gc_step,
                &dt_t_gc, &dt_t_name_gc_step,
                nullptr,
                &h_prev_gc_initial,
                &h_prev_names_gc_initial,
                time_step_idx_for_grad_check,
                &tp_analytical_cache_gc,
                &analytical_step_result
        );

        std::cout << "\n=== TEMPORAL PREDICTOR TRAINING EXAMPLE (BPTT) ===" << std::endl;
        training_example();

        std::cout << "\n=== ALL DEMOS COMPLETED SUCCESSFULLY (Compilation & Basic Run) ===" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}