// file: models.h
// (Content is identical to the last version provided in the prompt, as it's stable for current debugging needs)
#ifndef TENSOREIGEN_MODELS_H
#define TENSOREIGEN_MODELS_H

#include "tensor.h" // For Layer, Tensor, GRUDCell, etc.
#include <vector>
#include <string>
#include <memory>
#include <random>
#include <iostream> // For warnings/errors
#include <algorithm> // For std::sort, std::unique

// Utility functions from main.cpp (needed by models)
inline std::string make_name(const std::string& prefix, int layer_idx, const std::string& suffix_type) {
    return prefix + "_L" + std::to_string(layer_idx) + "_" + suffix_type;
}
inline std::string make_name(const std::string& prefix, const std::string& suffix) {
    return prefix + "_" + suffix;
}
inline std::string make_name(const std::string& component_prefix, const std::string& type, int time_step_idx) {
    return component_prefix + "_" + type + "_t" + std::to_string(time_step_idx);
}


class SimpleMLP {
private:
    std::vector<std::unique_ptr<Layer>> layers_;
    std::string mlp_name_prefix_;
public:
    SimpleMLP(const std::vector<int>& layer_sizes, std::mt19937& gen, const std::string& name_prefix)
            : mlp_name_prefix_(name_prefix) {
        for (size_t i = 0; i < layer_sizes.size() - 1; ++i) {
            layers_.push_back(std::make_unique<Linear>(
                    layer_sizes[i], layer_sizes[i + 1], true, &gen,
                    make_name(mlp_name_prefix_, static_cast<int>(i), "linear")
            ));
            if (i < layer_sizes.size() - 2) { // Add activation for all but last linear layer
                layers_.push_back(std::make_unique<TanhLayer>(make_name(mlp_name_prefix_, static_cast<int>(i), "tanh")));
            }
        }
    }

    Tensor forward(const Tensor& input, GradientCache& cache,
                   const std::string& input_tensor_name, const std::string& output_tensor_name) {
        Tensor current_x = input;
        std::string current_x_name = input_tensor_name;

        // Add backward functions in FORWARD computational order (for reverse execution)
        for (size_t i = 0; i < layers_.size(); ++i) {
            Layer* layer = layers_[i].get();
            std::string layer_output_name = make_name(mlp_name_prefix_, "layer" + std::to_string(i) + "_out");
            if (i == layers_.size() - 1) {
                layer_output_name = output_tensor_name;
            }

            current_x = layer->forward(current_x, cache, layer_output_name);

            std::string dL_dOutput_key_for_this_layer = layer_output_name;
            std::string dL_dInput_key_for_this_layer = current_x_name;

            // Add backward function for THIS layer (forward order)
            cache.add_backward_fn([layer, dL_dOutput_key_for_this_layer, dL_dInput_key_for_this_layer, &cache]() {
                if (!cache.has_gradient(dL_dOutput_key_for_this_layer)) {
                    std::cerr << "Warning: No gradient for " << dL_dOutput_key_for_this_layer
                              << " (output of layer " << layer->name() << "). Skipping backward for this layer." << std::endl;
                    return;
                }
                Tensor dL_dOutput_tensor = cache.get_gradient(dL_dOutput_key_for_this_layer);
                Tensor dL_dInput_tensor = layer->backward_calc_and_store_grads(dL_dOutput_tensor, cache);

                if (!dL_dInput_key_for_this_layer.empty()) {
                    cache.accumulate_gradient(dL_dInput_key_for_this_layer, dL_dInput_tensor);
                }
            });

            current_x_name = layer_output_name;
        }
        return current_x;
    }

    void zero_grad() {
        for (auto& layer : layers_) {
            layer->zero_grad();
        }
    }
    void set_training_mode(bool training) {
        for (auto& layer : layers_) {
            layer->set_training_mode(training);
        }
    }
    std::vector<Layer*> get_all_layers_for_optimizer() {
        std::vector<Layer*> all_layers;
        for (const auto& layer_ptr : layers_) {
            all_layers.push_back(layer_ptr.get());
        }
        return all_layers;
    }
    const std::string& name() const { return mlp_name_prefix_; } // Added for gradient checker
};


class TemporalPredictor {
private:
    TemporalConfig config_;
    std::vector<std::unique_ptr<GRUDCell>> gru_layers_;
    std::unique_ptr<LayerNorm> output_layernorm_;
    std::unique_ptr<Dropout> output_dropout_;
    std::unique_ptr<Linear> readout_layer_;
    std::mt19937 gen_;
    std::string model_name_prefix_;

public:
    TemporalPredictor(const TemporalConfig& config, int output_dim, const std::string& model_name_prefix = "temporal_predictor")
            : config_(config), gen_(config.seed), model_name_prefix_(model_name_prefix) {

        int current_gru_input_size = config_.input_size;
        for (int i = 0; i < config_.num_layers; ++i) {
            std::string gru_cell_base_name = make_name(model_name_prefix_, i, "gru_cell");
            gru_layers_.push_back(std::make_unique<GRUDCell>(
                    config_, current_gru_input_size, gen_, gru_cell_base_name
            ));
            current_gru_input_size = config_.hidden_size;
        }

        if (config_.use_layer_norm) {
            output_layernorm_ = std::make_unique<LayerNorm>(
                    config_.hidden_size, 1e-5f, true, make_name(model_name_prefix_, "output_ln")
            );
        }
        if (config_.dropout_rate > 0) {
            output_dropout_ = std::make_unique<Dropout>(config_.dropout_rate, make_name(model_name_prefix_, "output_dropout"));
            output_dropout_->set_seed(config_.seed + 1000);
        }
        readout_layer_ = std::make_unique<Linear>(
                config_.hidden_size, output_dim, true, &gen_, make_name(model_name_prefix_, "readout")
        );
    }

    struct StepResult {
        Tensor prediction;
        float loss_value;
        // Store names for gradient checking a single step
        std::string x_t_name_for_grad_step;
        std::string dt_t_name_for_grad_step;
        std::vector<std::string> h_prev_names_for_grad_step;
        std::string prediction_name_for_grad_step;
    };

    StepResult process_step(
            const Tensor& x_t, const std::string& x_t_name_for_grad,
            const Tensor& dt_t, const std::string& dt_t_name_for_grad,
            const Tensor* mask_t,
            const Tensor& target_t,
            std::vector<Tensor>& hidden_states_io,
            std::vector<std::string>& h_prev_names_for_grad,
            GradientCache& cache,
            Loss& loss_fn,
            bool is_training,
            int time_step_idx
    ) {
        for(auto& gru_cell : gru_layers_) gru_cell->set_training_mode_all_sub_layers(is_training);
        if (output_layernorm_) output_layernorm_->set_training_mode(is_training);
        if (output_dropout_) output_dropout_->set_training_mode(is_training);
        readout_layer_->set_training_mode(is_training);

        Tensor current_input_to_gru_sequence = x_t;
        std::string current_input_name_for_grad_path = x_t_name_for_grad;

        std::vector<std::string> h_new_names_this_step(config_.num_layers);
        std::vector<std::string> h_prev_names_for_this_step_grad_check = h_prev_names_for_grad;


        for (int i = 0; i < config_.num_layers; ++i) {
            GRUDCell::GRUDInputData grud_step_input = {
                    (i == 0) ? x_t : current_input_to_gru_sequence,
                    hidden_states_io[i],
                    dt_t,
                    (i == 0) ? mask_t : nullptr
            };

            h_new_names_this_step[i] = make_name(gru_layers_[i]->name(), "h_new", time_step_idx);
            Tensor h_new_for_this_layer = gru_layers_[i]->forward_step(grud_step_input, cache, h_new_names_this_step[i]);

            std::string dL_dH_new_key = h_new_names_this_step[i];
            std::string dL_dH_prev_key = h_prev_names_for_grad[i]; // This is the name of h_prev_t for THIS layer i.
            std::string dL_dX_or_prev_gru_out_key = (i == 0) ? x_t_name_for_grad : h_new_names_this_step[i-1]; // If i > 0, this is h_new from layer i-1.
            std::string dL_dDt_key = dt_t_name_for_grad;

            GRUDCell* current_gru_cell_ptr = gru_layers_[i].get();
            cache.add_backward_fn([current_gru_cell_ptr, dL_dH_new_key, dL_dH_prev_key, dL_dX_or_prev_gru_out_key, dL_dDt_key, &cache]() {
                if (!cache.has_gradient(dL_dH_new_key)) {
                    std::cerr << "Warning: No gradient for " << dL_dH_new_key << " for GRU cell " << current_gru_cell_ptr->name() << std::endl; return;
                }
                Tensor dL_dH_new_tensor = cache.get_gradient(dL_dH_new_key);
                GRUDCell::GRUDBackwardOutput grud_grads = current_gru_cell_ptr->grud_backward_step(dL_dH_new_tensor, cache);

                if (!dL_dH_prev_key.empty()) cache.accumulate_gradient(dL_dH_prev_key, grud_grads.dL_dH_prev_t);
                if (!dL_dX_or_prev_gru_out_key.empty()) cache.accumulate_gradient(dL_dX_or_prev_gru_out_key, grud_grads.dL_dX_t_input);
                if (!dL_dDt_key.empty()) cache.accumulate_gradient(dL_dDt_key, grud_grads.dL_dDt_input);
            });

            current_input_to_gru_sequence = h_new_for_this_layer;
            current_input_name_for_grad_path = h_new_names_this_step[i];
            hidden_states_io[i] = h_new_for_this_layer; // Update hidden state for *next* step or next layer in this step
        }

        Tensor final_rnn_output = current_input_to_gru_sequence; // This is h_new from the last GRU layer
        std::string final_rnn_output_name_for_grad = current_input_name_for_grad_path;

        if (output_layernorm_) {
            std::string ln_input_name = final_rnn_output_name_for_grad;
            std::string ln_out_name = make_name(output_layernorm_->name(), "out", time_step_idx);
            final_rnn_output = output_layernorm_->forward(final_rnn_output, cache, ln_out_name);
            Layer* ln_ptr = output_layernorm_.get();
            cache.add_backward_fn([ln_ptr, ln_out_name, ln_input_name, &cache]() {
                if (!cache.has_gradient(ln_out_name)) { std::cerr << "Warning: No grad for " << ln_out_name << " (LN)" << std::endl; return; }
                Tensor dL_dOutput = cache.get_gradient(ln_out_name);
                Tensor dL_dInput = ln_ptr->backward_calc_and_store_grads(dL_dOutput, cache);
                if (!ln_input_name.empty()) cache.accumulate_gradient(ln_input_name, dL_dInput);
            });
            final_rnn_output_name_for_grad = ln_out_name;
        }

        if (output_dropout_) {
            std::string do_input_name = final_rnn_output_name_for_grad;
            std::string do_out_name = make_name(output_dropout_->name(), "out", time_step_idx);
            final_rnn_output = output_dropout_->forward(final_rnn_output, cache, do_out_name);
            Layer* do_ptr = output_dropout_.get();
            cache.add_backward_fn([do_ptr, do_out_name, do_input_name, &cache]() {
                if (!cache.has_gradient(do_out_name)) { std::cerr << "Warning: No grad for " << do_out_name << " (Dropout)" << std::endl; return; }
                Tensor dL_dOutput = cache.get_gradient(do_out_name);
                Tensor dL_dInput = do_ptr->backward_calc_and_store_grads(dL_dOutput, cache);
                if (!do_input_name.empty()) cache.accumulate_gradient(do_input_name, dL_dInput);
            });
            final_rnn_output_name_for_grad = do_out_name;
        }

        std::string prediction_input_name = final_rnn_output_name_for_grad;
        std::string prediction_output_name_for_grad_step = make_name(readout_layer_->name(), "out", time_step_idx);
        Tensor prediction = readout_layer_->forward(final_rnn_output, cache, prediction_output_name_for_grad_step);
        Layer* readout_ptr = readout_layer_.get();
        cache.add_backward_fn([readout_ptr, prediction_output_name_for_grad_step, prediction_input_name, &cache]() {
            if (!cache.has_gradient(prediction_output_name_for_grad_step)) { std::cerr << "Warning: No grad for " << prediction_output_name_for_grad_step << " (Readout)" << std::endl; return; }
            Tensor dL_dOutput = cache.get_gradient(prediction_output_name_for_grad_step);
            Tensor dL_dInput = readout_ptr->backward_calc_and_store_grads(dL_dOutput, cache);
            if (!prediction_input_name.empty()) cache.accumulate_gradient(prediction_input_name, dL_dInput);
        });

        float loss_val = loss_fn.forward(prediction, target_t);
        if (is_training) {
            loss_fn.backward(prediction, target_t, cache, prediction_output_name_for_grad_step);
        }

        // After processing all layers for this step, update the names for h_prev_for_grad
        // to reflect the *new* hidden state names (outputs of this step)
        // This is for BPTT where the output of this step becomes the input for the next.
        // For single step grad check, h_prev_names_for_this_step_grad_check holds the *input* names.
        for(int l=0; l<config_.num_layers; ++l) {
            h_prev_names_for_grad[l] = h_new_names_this_step[l];
        }

        return {prediction, loss_val,
                x_t_name_for_grad, dt_t_name_for_grad,
                h_prev_names_for_this_step_grad_check,
                prediction_output_name_for_grad_step};
    }

    std::vector<Layer*> get_all_trainable_layers() {
        std::vector<Layer*> all_layers_flat;
        for(auto& cell_ptr : gru_layers_) {
            if (!cell_ptr->parameters().empty()) {
                all_layers_flat.push_back(cell_ptr.get());
            }
            std::vector<Layer*> cell_sublayers = cell_ptr->get_all_sub_layers_for_optimizer();
            for(Layer* sub_layer : cell_sublayers) {
                if (!sub_layer->parameters().empty()){
                    all_layers_flat.push_back(sub_layer);
                }
            }
        }
        if(output_layernorm_ && !output_layernorm_->parameters().empty()) all_layers_flat.push_back(output_layernorm_.get());
        if(output_dropout_ && !output_dropout_->parameters().empty()) { // Should only add if it has params. Dropout usually doesn't.
            // all_layers_flat.push_back(output_dropout_.get()); // Typically dropout has no params.
        }
        if (readout_layer_ && !readout_layer_->parameters().empty()) all_layers_flat.push_back(readout_layer_.get());

        // Remove duplicates that might arise if a GRUDCell itself is added AND its sublayers are also added
        // (though the current logic adds GRUDCell if it has params, then its sublayers if they have params).
        // Sorting and unique is a good safeguard.
        std::sort(all_layers_flat.begin(), all_layers_flat.end());
        all_layers_flat.erase(std::unique(all_layers_flat.begin(), all_layers_flat.end()), all_layers_flat.end());
        return all_layers_flat;
    }

    void zero_grad_all_layers() {
        for(auto* layer : get_all_trainable_layers()){
            layer->zero_grad();
        }
    }
    void set_training_mode_all_layers(bool training) {
        for(auto* layer : get_all_trainable_layers()){
            layer->set_training_mode(training);
        }
    }
    const std::string& name() const { return model_name_prefix_; }
    const TemporalConfig& get_config() const { return config_; }
};


#endif //TENSOREIGEN_MODELS_H