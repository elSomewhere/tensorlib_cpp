// file: gradient_checker.h
// (Content is identical to the last version provided in the prompt, as it's stable for current debugging needs)
#ifndef TENSOREIGEN_GRADIENT_CHECKER_H
#define TENSOREIGEN_GRADIENT_CHECKER_H

#include "models.h" // Includes tensor.h and model definitions (SimpleMLP, TemporalPredictor)
#include <vector>
#include <string>
#include <memory>
#include <cmath>
#include <random>
#include <iostream>
#include <iomanip>
#include <functional>
#include <type_traits>
#include <algorithm>
#include <map>
#include <cstdio> // For printf


namespace GradientChecker {

    // Forward declaration for calculate_loss_for_temporal_step
    template<typename LossFnType>
    float calculate_loss_for_temporal_step(
            TemporalPredictor& model_runner,
            Layer* layer_to_perturb,
            const std::string* param_name_to_perturb,
            int R_perturb, int C_perturb, float perturbation_val,
            const Tensor& x_t_gc, const std::string& x_t_name_gc,
            const Tensor& dt_t_gc, const std::string& dt_t_name_gc,
            const Tensor* mask_t_gc,
            const Tensor& target_t_gc,
            std::vector<Tensor>& h_prev_gc,
            std::vector<std::string>& h_prev_names_gc,
            LossFnType& loss_fn,
            int time_step_idx_gc);


    template<typename ModelToRunForward, typename LossFnType>
    double check_gradient_for_param_in_model(
            ModelToRunForward& model_runner,
            Layer* layer_to_perturb,
            const std::string& param_name_in_layer, // This is const std::string&
            int R, int C,
            const Tensor& model_input_tensor,
            const Tensor& target_tensor,
            LossFnType& loss_fn,
            const std::string& grad_check_model_input_name,
            const std::string& grad_check_model_output_name,
            double epsilon = 1e-5,
            const Tensor* x_t_tp = nullptr, const std::string* x_t_name_tp = nullptr,
            const Tensor* dt_t_tp = nullptr, const std::string* dt_t_name_tp = nullptr,
            const Tensor* mask_t_tp = nullptr,
            std::vector<Tensor>* h_prev_tp_original = nullptr,
            std::vector<std::string>* h_prev_names_tp_original = nullptr,
            int time_step_idx_tp = 0
    )
    {
        if (!layer_to_perturb) throw std::runtime_error("layer_to_perturb cannot be null");
        if (layer_to_perturb->parameters().find(param_name_in_layer) == layer_to_perturb->parameters().end()) {
            throw std::runtime_error("Parameter " + param_name_in_layer + " not found in layer " + layer_to_perturb->name());
        }

        Tensor& param_ref = layer_to_perturb->get_parameters_mut().at(param_name_in_layer);
        if (R >= param_ref.rows() || C >= param_ref.cols()) {
            throw std::out_of_range("GradientChecker: R,C out of bounds for parameter " + param_name_in_layer);
        }

        float original_value = param_ref.data()(R, C);
        double numerical_grad = 0.0;
        float loss_plus, loss_minus;

        if constexpr (std::is_same_v<ModelToRunForward, TemporalPredictor>) {
            if (!x_t_tp || !dt_t_tp || !h_prev_tp_original || !h_prev_names_tp_original || !x_t_name_tp || !dt_t_name_tp) {
                throw std::runtime_error("TemporalPredictor gradient check requires x_t, dt_t, h_prev, and their names.");
            }
            std::vector<Tensor> h_prev_copy_plus = *h_prev_tp_original;
            std::vector<std::string> h_prev_names_copy_plus = *h_prev_names_tp_original;
            loss_plus = calculate_loss_for_temporal_step(
                    model_runner, layer_to_perturb, &param_name_in_layer, R, C, original_value + static_cast<float>(epsilon),
                    *x_t_tp, *x_t_name_tp, *dt_t_tp, *dt_t_name_tp, mask_t_tp, target_tensor,
                    h_prev_copy_plus, h_prev_names_copy_plus, loss_fn, time_step_idx_tp);

            // calculate_loss_for_temporal_step restores the parameter.
            // So, param_ref now holds original_value.

            std::vector<Tensor> h_prev_copy_minus = *h_prev_tp_original;
            std::vector<std::string> h_prev_names_copy_minus = *h_prev_names_tp_original;
            loss_minus = calculate_loss_for_temporal_step(
                    model_runner, layer_to_perturb, &param_name_in_layer, R, C, original_value - static_cast<float>(epsilon),
                    *x_t_tp, *x_t_name_tp, *dt_t_tp, *dt_t_name_tp, mask_t_tp, target_tensor,
                    h_prev_copy_minus, h_prev_names_copy_minus, loss_fn, time_step_idx_tp);
        } else {
            auto calculate_total_loss_for_model = [&](ModelToRunForward& m_ref_lambda) -> float {
                GradientCache temp_cache_lambda;
                Tensor temp_output_lambda;
                if constexpr (std::is_base_of_v<Layer, std::remove_reference_t<ModelToRunForward>>) {
                    if (Layer* l_ptr_lambda = dynamic_cast<Layer*>(&m_ref_lambda)) {
                        temp_output_lambda = l_ptr_lambda->forward(model_input_tensor, temp_cache_lambda, grad_check_model_output_name);
                    } else {
                        throw std::runtime_error("Failed to cast ModelToRunForward to Layer* in loss lambda (is_base_of_v case)");
                    }
                } else if constexpr (std::is_same_v<ModelToRunForward, SimpleMLP>) {
                    temp_output_lambda = m_ref_lambda.forward(model_input_tensor, temp_cache_lambda, grad_check_model_input_name, grad_check_model_output_name);
                } else {
                    throw std::runtime_error("Unsupported ModelToRunForward type in loss calculation lambda.");
                }
                return loss_fn.forward(temp_output_lambda, target_tensor);
            };
            param_ref.data()(R, C) = original_value + static_cast<float>(epsilon);
            loss_plus = calculate_total_loss_for_model(model_runner);

            param_ref.data()(R, C) = original_value - static_cast<float>(epsilon);
            loss_minus = calculate_total_loss_for_model(model_runner);

            param_ref.data()(R, C) = original_value;
        }

        numerical_grad = (loss_plus - loss_minus) / (2.0 * epsilon);
        return numerical_grad;
    }

    template<typename LossFnType>
    float calculate_loss_for_temporal_step(
            TemporalPredictor& model_runner,
            Layer* layer_to_perturb,
            const std::string* param_name_to_perturb,
            int R_perturb, int C_perturb, float new_param_value,
            const Tensor& x_t_gc, const std::string& x_t_name_gc,
            const Tensor& dt_t_gc, const std::string& dt_t_name_gc,
            const Tensor* mask_t_gc,
            const Tensor& target_t_gc,
            std::vector<Tensor>& h_prev_gc, // Note: passed by ref, but it's a copy from caller
            std::vector<std::string>& h_prev_names_gc, // Note: passed by ref, but it's a copy from caller
            LossFnType& loss_fn,
            int time_step_idx_gc)
    {
        GradientCache temp_cache_gc; // Fresh cache for this specific +/- epsilon evaluation
        // Make a copy of h_prev_gc for this specific call to process_step,
        // because process_step modifies its hidden_states_io argument.
        std::vector<Tensor> h_prev_gc_call_copy = h_prev_gc;
        std::vector<std::string> h_prev_names_gc_call_copy = h_prev_names_gc;


        float original_param_scalar_val_local = 0.f;
        Tensor* param_ref_to_restore_local = nullptr;

        if (layer_to_perturb && param_name_to_perturb) {
            param_ref_to_restore_local = &layer_to_perturb->get_parameters_mut().at(*param_name_to_perturb);
            original_param_scalar_val_local = param_ref_to_restore_local->data()(R_perturb, C_perturb);
            param_ref_to_restore_local->data()(R_perturb, C_perturb) = new_param_value;
        }

        TemporalPredictor::StepResult result = model_runner.process_step(
                x_t_gc, x_t_name_gc, dt_t_gc, dt_t_name_gc, mask_t_gc, target_t_gc,
                h_prev_gc_call_copy, // Pass the copy that can be modified
                h_prev_names_gc_call_copy,
                temp_cache_gc, loss_fn,
                true, // is_training = true for numerical gradient calculation to match analytical
                time_step_idx_gc
        );

        if (layer_to_perturb && param_name_to_perturb && param_ref_to_restore_local) {
            param_ref_to_restore_local->data()(R_perturb, C_perturb) = original_param_scalar_val_local;
        }
        return result.loss_value;
    }


    template<typename ModelType, typename LossFnType>
    void check_gradients_for_model(
            ModelType& model,
            const Tensor& model_input,
            const Tensor& target,
            LossFnType& loss_fn,
            const std::string& model_input_name_for_grad,
            const std::string& model_output_name_for_grad,
            double epsilon = 1e-4,
            double tolerance = 1e-5,
            const Tensor* x_t_tp_param = nullptr, const std::string* x_t_name_tp_param = nullptr,
            const Tensor* dt_t_tp_param = nullptr, const std::string* dt_t_name_tp_param = nullptr,
            const Tensor* mask_t_tp_param = nullptr,
            std::vector<Tensor>* h_prev_tp_param = nullptr,
            std::vector<std::string>* h_prev_names_tp_param = nullptr,
            int time_step_idx_tp_param = 0,
            GradientCache* analytical_cache_for_tp = nullptr,
            const TemporalPredictor::StepResult* tp_analytical_step_result = nullptr
    )
    {
        std::string current_model_name_for_seed;
        if constexpr (std::is_same_v<ModelType, SimpleMLP> || std::is_same_v<ModelType, TemporalPredictor> || std::is_base_of_v<Layer, ModelType>) {
            current_model_name_for_seed = model.name();
        } else {
            current_model_name_for_seed = typeid(model).name();
        }
        std::cout << "\n--- Starting Gradient Check for Model: " << current_model_name_for_seed << " (" << typeid(model).name() << ") ---" << std::endl;


        GradientCache local_analytical_cache_storage;
        GradientCache& analytical_cache_ref = (std::is_same_v<ModelType, TemporalPredictor> && analytical_cache_for_tp)
                                              ? *analytical_cache_for_tp
                                              : local_analytical_cache_storage;

        if constexpr (!std::is_same_v<ModelType, TemporalPredictor>) {
            // For non-TP models, zero_grad and set_training_mode, then compute analytical grads here.
            if constexpr (std::is_same_v<ModelType, SimpleMLP>) {
                model.zero_grad();
                model.set_training_mode(true);
            } else if constexpr (std::is_base_of_v<Layer, std::remove_reference_t<ModelType>>) {
                static_cast<Layer&>(model).zero_grad();
                static_cast<Layer&>(model).set_training_mode(true);
            }
        } else {
            // For TemporalPredictor, analytical grads are pre-computed.
            // Ensure training mode was set before pre-computation.
            // zero_grad was also done before pre-computation.
            model.set_training_mode_all_layers(true);
        }


        Tensor analytical_output_tensor_val;
        std::string analytical_output_name_for_loss_bwd_val;

        if constexpr (std::is_same_v<ModelType, SimpleMLP>) {
            analytical_output_tensor_val = model.forward(model_input, analytical_cache_ref, model_input_name_for_grad, model_output_name_for_grad);
            analytical_output_name_for_loss_bwd_val = model_output_name_for_grad;
            loss_fn.forward(analytical_output_tensor_val, target);
            loss_fn.backward(analytical_output_tensor_val, target, analytical_cache_ref, analytical_output_name_for_loss_bwd_val);
            analytical_cache_ref.backward();
        } else if constexpr (std::is_base_of_v<Layer, std::remove_reference_t<ModelType>>) {
            if constexpr (!std::is_same_v<ModelType, TemporalPredictor>) {
                Layer& layer_model = static_cast<Layer&>(model);
                analytical_output_tensor_val = layer_model.forward(model_input, analytical_cache_ref, model_output_name_for_grad);
                analytical_output_name_for_loss_bwd_val = model_output_name_for_grad;

                loss_fn.forward(analytical_output_tensor_val, target);
                loss_fn.backward(analytical_output_tensor_val, target, analytical_cache_ref, analytical_output_name_for_loss_bwd_val);

                analytical_cache_ref.add_backward_fn([&layer_model, &analytical_cache_ref, analytical_output_name_for_loss_bwd_val]() {
                    if (!analytical_cache_ref.has_gradient(analytical_output_name_for_loss_bwd_val)) {
                        std::cerr << "Warning (GradCheck Single Layer): No gradient for " << analytical_output_name_for_loss_bwd_val
                                  << " (output of layer " << layer_model.name() << "). Skipping backward for this layer." << std::endl;
                        return;
                    }
                    Tensor dL_dOutput_tensor = analytical_cache_ref.get_gradient(analytical_output_name_for_loss_bwd_val);
                    layer_model.backward_calc_and_store_grads(dL_dOutput_tensor, analytical_cache_ref);
                });
                analytical_cache_ref.backward();
            }
        }


        if constexpr (std::is_same_v<ModelType, TemporalPredictor>) {
            if (!tp_analytical_step_result || !analytical_cache_for_tp ) {
                throw std::runtime_error("TemporalPredictor grad check requires pre-computed analytical_step_result and its cache.");
            }
            analytical_output_tensor_val = tp_analytical_step_result->prediction;
            analytical_output_name_for_loss_bwd_val = tp_analytical_step_result->prediction_name_for_grad_step;
        } else if (!std::is_base_of_v<Layer, std::remove_reference_t<ModelType>> && !std::is_same_v<ModelType, SimpleMLP>) {
            throw std::runtime_error("Grad check: Unsupported model type for analytical grad calculation path.");
        }

        std::vector<Layer*> layers_to_check;
        if constexpr (std::is_same_v<ModelType, SimpleMLP>) {
            layers_to_check = model.get_all_layers_for_optimizer();
        } else if constexpr (std::is_base_of_v<Layer, std::remove_reference_t<ModelType>>) {
            if constexpr (!std::is_same_v<ModelType, TemporalPredictor>) {
                layers_to_check.push_back(static_cast<Layer*>(&model));
            }
        }

        if constexpr (std::is_same_v<ModelType, TemporalPredictor>) {
            layers_to_check = model.get_all_trainable_layers();
        }


        bool all_ok = true;
        for (Layer* layer_ptr : layers_to_check) {
            if (!layer_ptr) {
                std::cerr << "Warning: Encountered null layer_ptr in layers_to_check. Skipping." << std::endl;
                all_ok = false;
                continue;
            }
            if (layer_ptr->parameters().empty()) continue;
            std::cout << "Checking Layer: " << layer_ptr->name() << std::endl;
            for (const auto& [param_name, param_tensor] : layer_ptr->parameters()) {
                std::cout << "  Parameter: " << param_name << " (" << param_tensor.rows() << "x" << param_tensor.cols() << ")" << std::endl;
                if (layer_ptr->gradients().find(param_name) == layer_ptr->gradients().end()) {
                    std::cout << "    Analytical gradient not found for " << param_name << ". Skipping." << std::endl;
                    all_ok = false; continue;
                }
                const Tensor& analytical_grad_tensor = layer_ptr->gradients().at(param_name);
                if (analytical_grad_tensor.empty()) {
                    std::cout << "    Analytical gradient is empty for " << param_name << ". Skipping." << std::endl;
                    all_ok = false; continue;
                }
                if (param_tensor.rows() == 0 || param_tensor.cols() == 0) {
                    std::cout << "    Parameter tensor " << param_name << " is empty. Skipping." << std::endl;
                    continue;
                }
                if (analytical_grad_tensor.rows() != param_tensor.rows() || analytical_grad_tensor.cols() != param_tensor.cols()) {
                    std::cout << "    Analytical gradient shape mismatch for " << param_name << ". Param: "
                              << param_tensor.rows() << "x" << param_tensor.cols() << ", Grad: "
                              << analytical_grad_tensor.rows() << "x" << analytical_grad_tensor.cols()
                              << ". Skipping." << std::endl;
                    all_ok = false; continue;
                }


                int num_checks_per_param = std::min(3, static_cast<int>(param_tensor.rows() * param_tensor.cols()));
                if (num_checks_per_param == 0 && param_tensor.rows() * param_tensor.cols() > 0) num_checks_per_param = 1;
                else if (param_tensor.rows() * param_tensor.cols() == 0) continue;

                std::vector<std::pair<int, int>> indices_to_check;
                if (param_tensor.rows() * param_tensor.cols() <= static_cast<size_t>(num_checks_per_param)) {
                    for (int r_idx = 0; r_idx < param_tensor.rows(); ++r_idx) for (int c_idx = 0; c_idx < param_tensor.cols(); ++c_idx) indices_to_check.push_back({r_idx, c_idx});
                } else {
                    std::mt19937 gen(std::hash<std::string>{}(layer_ptr->name() + param_name + current_model_name_for_seed));
                    std::uniform_int_distribution<> r_dist(0, param_tensor.rows() - 1);
                    std::uniform_int_distribution<> c_dist(0, param_tensor.cols() - 1);
                    std::vector<std::pair<int,int>> temp_indices;
                    for(int k=0; k<num_checks_per_param*5 && temp_indices.size() < static_cast<size_t>(num_checks_per_param) ; ++k) {
                        temp_indices.push_back({r_dist(gen), c_dist(gen)});
                    }
                    std::sort(temp_indices.begin(), temp_indices.end());
                    temp_indices.erase(std::unique(temp_indices.begin(), temp_indices.end()), temp_indices.end());
                    indices_to_check.assign(temp_indices.begin(), temp_indices.begin() + std::min(temp_indices.size(), static_cast<size_t>(num_checks_per_param)));
                    if (indices_to_check.empty() && param_tensor.rows() * param_tensor.cols() > 0) {
                        indices_to_check.push_back({r_dist(gen), c_dist(gen)});
                    }
                }


                for(const auto& rc_pair : indices_to_check) {
                    int r_val = rc_pair.first;
                    int c_val = rc_pair.second;

                    double num_grad = check_gradient_for_param_in_model(
                            model, layer_ptr, param_name, r_val, c_val,
                            model_input, target, loss_fn,
                            model_input_name_for_grad, model_output_name_for_grad, epsilon,
                            x_t_tp_param, x_t_name_tp_param, dt_t_tp_param, dt_t_name_tp_param, mask_t_tp_param,
                            h_prev_tp_param, h_prev_names_tp_param, time_step_idx_tp_param
                    );

                    float an_grad = analytical_grad_tensor.data()(r_val, c_val);
                    double denominator = std::max({static_cast<double>(std::abs(an_grad)), std::abs(num_grad), 1e-8});
                    double rel_error = denominator > 1e-12 ? (std::abs(an_grad - num_grad) / denominator) : std::abs(an_grad-num_grad);
                    double abs_error = std::abs(an_grad - num_grad);

                    bool current_param_ok = (rel_error < tolerance || abs_error < (tolerance * 1e-2));

                    if (std::abs(an_grad) < 1e-9 && std::abs(num_grad) < 1e-7) { // If both are very small, it's okay
                        current_param_ok = true;
                    }


                    printf("      [%d,%d]: An: %.5e, Num: %.5e, Abs.Err: %.2e, Rel.Err: %.2e ", r_val, c_val, an_grad, num_grad, abs_error, rel_error);
                    if (current_param_ok) {
                        std::cout << "\033[32mPASSED\033[0m" << std::endl;
                    } else {
                        std::cout << "\033[31mFAILED\033[0m" << std::endl;
                        all_ok = false;
                    }
                }
            }
        }
        if (all_ok) {
            std::cout << "--- Gradient Check for Model " << current_model_name_for_seed << " \033[32mPASSED globally\033[0m ---" << std::endl;
        } else {
            std::cout << "--- Gradient Check for Model " << current_model_name_for_seed << " \033[31mFAILED globally\033[0m ---" << std::endl;
        }
    }
} // namespace GradientChecker

#endif //TENSOREIGEN_GRADIENT_CHECKER_H