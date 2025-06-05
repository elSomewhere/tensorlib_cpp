//
// grud/layers/temporal.h - Temporal neural network layers (GRU-D)
//

#ifndef GRUD_LAYERS_TEMPORAL_H
#define GRUD_LAYERS_TEMPORAL_H

#include "../core/module.h"
#include "basic.h"
#include <memory>
#include <vector>
#include <optional>

namespace grud {
namespace layers {

// ============================================================================
// TEMPORAL CONFIGURATION
// ============================================================================

struct TemporalConfig {
    // Architecture
    int input_size = 4;
    int hidden_size = 64;
    int num_layers = 2;

    // Decay / imputation
    bool use_exponential_decay = true;
    float softclip_threshold = 3.0f;
    float min_log_gamma = -10.0f;

    // Regularization
    float dropout = 0.1f;
    bool layer_norm = true;

    // Random seed for initialization
    int seed = 0;
};

// ============================================================================
// GAMMA COMPUTATION MODULE (for exponential decay)
// ============================================================================

/**
 * Computes gamma values for exponential decay in GRU-D
 */
class GammaComputation : public Module {
private:
    int hidden_size_;
    float softclip_threshold_;
    float min_log_gamma_;

public:
    Param decay_param;  // Learnable decay parameters

    GammaComputation(int hidden_size, float softclip_threshold = 3.0f, float min_log_gamma = -10.0f,
                    std::mt19937* gen = nullptr)
        : hidden_size_(hidden_size), softclip_threshold_(softclip_threshold), min_log_gamma_(min_log_gamma),
          decay_param(hidden_size, 1, "decay_param") {

        // Initialize decay parameters
        if (gen) {
            decay_param.init_normal(*gen, 0.0f, 0.1f);
        } else {
            decay_param.value.setZero();
        }
    }

    /**
     * Forward pass: compute gamma from decay parameters and dt
     * @param dt Time differences (batch_size, 1)
     */
    Eigen::MatrixXf forward(const Eigen::MatrixXf& dt, Context& ctx) {
        // Softclip decay parameters
        Eigen::VectorXf decay_clipped = softclip_vector(decay_param.value.col(0), softclip_threshold_);

        // Apply softplus
        Eigen::VectorXf softplus_val = softplus_vector(decay_clipped);

        // Compute log gamma: -dt * softplus_val (broadcast)
        Eigen::MatrixXf log_gamma_unclamped = -dt * softplus_val.transpose();

        // Clamp log gamma
        Eigen::MatrixXf log_gamma = log_gamma_unclamped.array()
            .cwiseMax(min_log_gamma_)
            .cwiseMin(-1e-4f);

        // Compute gamma = exp(log_gamma)
        Eigen::MatrixXf gamma = log_gamma.array().exp();

        // Save for backward pass
        ctx.save_for_backward(dt);
        ctx.save_for_backward(decay_clipped);
        ctx.save_for_backward(softplus_val);
        ctx.save_for_backward(log_gamma_unclamped);
        ctx.save_for_backward(log_gamma);
        ctx.save_for_backward(gamma);

        return gamma;
    }

    Eigen::MatrixXf backward(const Eigen::MatrixXf& grad_gamma, const Context& ctx) override {
        const Eigen::MatrixXf& dt = ctx.get_saved(0);
        const Eigen::VectorXf& decay_clipped = ctx.get_saved(1);
        const Eigen::VectorXf& softplus_val = ctx.get_saved(2);
        const Eigen::MatrixXf& log_gamma_unclamped = ctx.get_saved(3);
        const Eigen::MatrixXf& log_gamma = ctx.get_saved(4);
        const Eigen::MatrixXf& gamma = ctx.get_saved(5);

        // Backward through exp
        Eigen::MatrixXf grad_log_gamma = grad_gamma.array() * gamma.array();

        // Backward through clamp
        Eigen::MatrixXf grad_mask = ((log_gamma_unclamped.array() >= min_log_gamma_) &&
                                    (log_gamma_unclamped.array() <= -1e-4f)).cast<float>();
        Eigen::MatrixXf grad_log_gamma_unclamped = grad_log_gamma.array() * grad_mask.array();

        // Backward through broadcasting: -dt * softplus_val
        Eigen::VectorXf grad_softplus = -(grad_log_gamma_unclamped.array().colwise() * dt.col(0).array())
            .colwise().sum().transpose();

        // Backward through softplus
        Eigen::VectorXf grad_decay_clipped = grad_softplus.array() *
            math::sigmoid(decay_clipped).array();

        // Backward through softclip (derivative is 1)
        decay_param.grad.col(0) += grad_decay_clipped;

        // Return gradient w.r.t. dt (for completeness, though usually not needed)
        return Eigen::MatrixXf::Zero(dt.rows(), dt.cols());
    }

    std::vector<Param*> params() override {
        return {&decay_param};
    }

    std::string name() const override {
        return "GammaComputation(" + std::to_string(hidden_size_) + ")";
    }

private:
    Eigen::VectorXf softclip_vector(const Eigen::VectorXf& x, float threshold) {
        Eigen::VectorXf result(x.size());
        for (int i = 0; i < x.size(); ++i) {
            float xi = x(i);
            float abs_xi = std::abs(xi);
            if (abs_xi <= threshold) {
                result(i) = xi;
            } else {
                float y = abs_xi - threshold;
                float log1p_exp_y = math::stable_log1p_exp(y);
                float sign_xi = (xi >= 0.0f) ? 1.0f : -1.0f;
                result(i) = threshold + log1p_exp_y * sign_xi;
            }
        }
        return result;
    }

    Eigen::VectorXf softplus_vector(const Eigen::VectorXf& x) {
        Eigen::VectorXf result(x.size());
        for (int i = 0; i < x.size(); ++i) {
            result(i) = std::log(1.0f + std::exp(std::max(-80.0f, std::min(80.0f, x(i)))));
        }
        return result;
    }
};

// ============================================================================
// IMPUTATION MODULE
// ============================================================================

/**
 * Handles input imputation for missing values
 */
class ImputationModule : public Module {
public:
    std::unique_ptr<Linear> impute_linear;

private:
    int input_size_;
    int hidden_size_;

public:
    ImputationModule(int input_size, int hidden_size, std::mt19937* gen = nullptr)
        : input_size_(input_size), hidden_size_(hidden_size) {
        impute_linear = std::make_unique<Linear>(hidden_size, input_size, true, gen);
    }

    /**
     * Forward pass: impute missing values
     * @param input Raw input data
     * @param hidden Previous hidden state for imputation
     * @param mask Optional mask (1 = observed, 0 = missing)
     */
    std::tuple<Eigen::MatrixXf, Eigen::MatrixXf> forward_with_mask(
        const Eigen::MatrixXf& input,
        const Eigen::MatrixXf& hidden,
        const std::optional<Eigen::MatrixXf>& mask = std::nullopt) {

        // Create context for the linear layer
        Context linear_ctx(impute_linear.get());

        // Get imputed values from hidden state
        Eigen::MatrixXf imputed = impute_linear->forward(hidden, linear_ctx);

        // Apply mask if provided
        Eigen::MatrixXf current_mask = mask.value_or(Eigen::MatrixXf::Ones(input.rows(), input.cols()));

        // Combine observed and imputed values
        Eigen::MatrixXf output = current_mask.array() * input.array() +
                                (1.0f - current_mask.array()) * imputed.array();

        return {output, current_mask};
    }

    // This module doesn't follow the standard forward interface since it needs additional inputs
    Eigen::MatrixXf forward(const Eigen::MatrixXf& input, Context& ctx) override {
        throw std::runtime_error("ImputationModule requires forward_with_mask method");
    }

    Eigen::MatrixXf backward(const Eigen::MatrixXf& grad_output, const Context& ctx) override {
        throw std::runtime_error("ImputationModule requires manual backward pass");
    }

    std::vector<Module*> children() override {
        return {impute_linear.get()};
    }

    std::string name() const override {
        return "ImputationModule(" + std::to_string(input_size_) + ", " + std::to_string(hidden_size_) + ")";
    }
};

// ============================================================================
// GRU-D CELL
// ============================================================================

/**
 * GRU-D cell with decay and imputation
 */
class GRUDCell : public Module {
public:
    // Linear layers for gates
    std::unique_ptr<Linear> W_r, U_r, V_r;  // Reset gate
    std::unique_ptr<Linear> W_z, U_z, V_z;  // Update gate
    std::unique_ptr<Linear> W_h, U_h, V_h;  // Candidate hidden state

    // Imputation and decay
    std::unique_ptr<ImputationModule> imputation;
    std::unique_ptr<GammaComputation> gamma_computation;

private:
    int input_size_;
    int hidden_size_;
    bool use_exponential_decay_;

public:
    GRUDCell(int input_size, int hidden_size, bool use_exponential_decay = true, std::mt19937* gen = nullptr)
        : input_size_(input_size), hidden_size_(hidden_size), use_exponential_decay_(use_exponential_decay) {

        // Initialize gate layers
        W_r = std::make_unique<Linear>(input_size, hidden_size, true, gen);
        U_r = std::make_unique<Linear>(hidden_size, hidden_size, false, gen);
        V_r = std::make_unique<Linear>(1, hidden_size, false, gen);  // dt input

        W_z = std::make_unique<Linear>(input_size, hidden_size, true, gen);
        U_z = std::make_unique<Linear>(hidden_size, hidden_size, false, gen);
        V_z = std::make_unique<Linear>(1, hidden_size, false, gen);

        W_h = std::make_unique<Linear>(input_size, hidden_size, true, gen);
        U_h = std::make_unique<Linear>(hidden_size, hidden_size, false, gen);
        V_h = std::make_unique<Linear>(1, hidden_size, false, gen);

        // Initialize imputation
        imputation = std::make_unique<ImputationModule>(input_size, hidden_size, gen);

        // Initialize gamma computation for decay
        if (use_exponential_decay_) {
            gamma_computation = std::make_unique<GammaComputation>(hidden_size, 3.0f, -10.0f, gen);
        }

        // Initialize update gate bias to -1 (common practice)
        if (W_z->has_bias()) {
            W_z->bias.value.setConstant(-1.0f);
        }
    }

    /**
     * Forward pass through GRU-D cell
     * @param input Input at current timestep
     * @param hidden Previous hidden state
     * @param dt Time difference
     * @param mask Optional mask for missing values
     */
    std::tuple<Eigen::MatrixXf, Context> forward_cell(
        const Eigen::MatrixXf& input,
        const Eigen::MatrixXf& hidden,
        const Eigen::MatrixXf& dt,
        const std::optional<Eigen::MatrixXf>& mask = std::nullopt) {

        Context ctx(this);

        // 1. Imputation
        auto [x_tilde, current_mask] = imputation->forward_with_mask(input, hidden, mask);

        // 2. Exponential decay (if enabled)
        Eigen::MatrixXf h_decay = hidden;
        if (use_exponential_decay_ && gamma_computation) {
            Context gamma_ctx(gamma_computation.get());
            Eigen::MatrixXf gamma = gamma_computation->forward(dt, gamma_ctx);
            h_decay = hidden.array() * gamma.array();
        }

        // 3. Reset gate
        Context r_W_ctx(W_r.get()), r_U_ctx(U_r.get()), r_V_ctx(V_r.get());
        Eigen::MatrixXf r_W = W_r->forward(x_tilde, r_W_ctx);
        Eigen::MatrixXf r_U = U_r->forward(h_decay, r_U_ctx);
        Eigen::MatrixXf r_V = V_r->forward(dt, r_V_ctx);
        Eigen::MatrixXf r = math::sigmoid(r_W + r_U + r_V);

        // 4. Update gate
        Context z_W_ctx(W_z.get()), z_U_ctx(U_z.get()), z_V_ctx(V_z.get());
        Eigen::MatrixXf z_W = W_z->forward(x_tilde, z_W_ctx);
        Eigen::MatrixXf z_U = U_z->forward(h_decay, z_U_ctx);
        Eigen::MatrixXf z_V = V_z->forward(dt, z_V_ctx);
        Eigen::MatrixXf z = math::sigmoid(z_W + z_U + z_V);

        // 5. Candidate hidden state
        Eigen::MatrixXf rh = r.array() * h_decay.array();
        Context h_W_ctx(W_h.get()), h_U_ctx(U_h.get()), h_V_ctx(V_h.get());
        Eigen::MatrixXf h_W = W_h->forward(x_tilde, h_W_ctx);
        Eigen::MatrixXf h_U = U_h->forward(rh, h_U_ctx);
        Eigen::MatrixXf h_V = V_h->forward(dt, h_V_ctx);
        Eigen::MatrixXf h_candidate = math::tanh_activation(h_W + h_U + h_V);

        // 6. Final hidden state
        Eigen::MatrixXf h_new = (1.0f - z.array()) * h_decay.array() + z.array() * h_candidate.array();

        // Save intermediate values for backward pass
        ctx.save_for_backward(input);
        ctx.save_for_backward(hidden);
        ctx.save_for_backward(dt);
        ctx.save_for_backward(current_mask);
        ctx.save_for_backward(x_tilde);
        ctx.save_for_backward(h_decay);
        ctx.save_for_backward(r);
        ctx.save_for_backward(z);
        ctx.save_for_backward(h_candidate);
        ctx.save_for_backward(h_new);

        return {h_new, ctx};
    }
    layers::Linear* get_cell_for_testing() {
        return W_r.get();  // Return one of the linear layers for testing
    }
    // Standard Module interface (simplified version)
    Eigen::MatrixXf forward(const Eigen::MatrixXf& input, Context& ctx) override {
        // This is a simplified interface - the full cell requires additional inputs
        // In practice, this would be called by a higher-level RNN module
        throw std::runtime_error("GRUDCell requires forward_cell method with hidden state and dt");
    }

    Eigen::MatrixXf backward(const Eigen::MatrixXf& grad_output, const Context& ctx) override {
        // The backward pass would be quite complex and is typically handled
        // by the automatic differentiation system using the saved intermediate values
        // This is where the autograd system really shines - it handles this automatically
        throw std::runtime_error("GRUDCell backward pass handled by autograd system");
    }

    std::vector<Module*> children() override {
        std::vector<Module*> child_modules = {
            W_r.get(), U_r.get(), V_r.get(),
            W_z.get(), U_z.get(), V_z.get(),
            W_h.get(), U_h.get(), V_h.get(),
            imputation.get()
        };

        if (gamma_computation) {
            child_modules.push_back(gamma_computation.get());
        }

        return child_modules;
    }

    std::string name() const override {
        return "GRUDCell(" + std::to_string(input_size_) + ", " + std::to_string(hidden_size_) + ")";
    }
};

// ============================================================================
// TEMPORAL RNN LAYER
// ============================================================================

/**
 * RNN layer that wraps GRU-D cell with optional LayerNorm and Dropout
 */
class TemporalRNNLayer : public Module {
public:
    std::unique_ptr<GRUDCell> cell;
    std::unique_ptr<LayerNorm> layer_norm;
    std::unique_ptr<Dropout> dropout;

private:
    int layer_index_;
    bool use_layer_norm_;
    bool use_dropout_;

public:
    TemporalRNNLayer(int input_size, int hidden_size, int layer_index,
                    bool use_layer_norm = true, float dropout_p = 0.0f,
                    bool use_exponential_decay = true, std::mt19937* gen = nullptr)
        : layer_index_(layer_index), use_layer_norm_(use_layer_norm),
          use_dropout_(dropout_p > 0.0f && gen != nullptr) {

        // Create GRU-D cell
        cell = std::make_unique<GRUDCell>(input_size, hidden_size, use_exponential_decay, gen);

        // Create layer norm if requested
        if (use_layer_norm_) {
            layer_norm = std::make_unique<LayerNorm>(hidden_size);
        }

        // Create dropout if requested
        if (use_dropout_ && gen) {
            dropout = std::make_unique<Dropout>(dropout_p, *gen);
        }
    }

    /**
     * Forward pass through the temporal layer
     */
    std::tuple<Eigen::MatrixXf, Context> forward_temporal(
        const Eigen::MatrixXf& input,
        const Eigen::MatrixXf& hidden,
        const Eigen::MatrixXf& dt,
        const std::optional<Eigen::MatrixXf>& mask = std::nullopt) {

        Context layer_ctx(this);

        // Forward through GRU-D cell
        auto [h_cell, cell_ctx] = cell->forward_cell(input, hidden, dt, mask);

        // Apply layer normalization if enabled
        Eigen::MatrixXf h_norm = h_cell;
        if (use_layer_norm_ && layer_norm) {
            Context norm_ctx(layer_norm.get());
            h_norm = layer_norm->forward(h_cell, norm_ctx);
        }

        // Apply dropout if enabled and in training mode
        Eigen::MatrixXf h_final = h_norm;
        if (use_dropout_ && dropout && training_mode) {
            Context dropout_ctx(dropout.get());
            h_final = dropout->forward(h_norm, dropout_ctx);
        }

        return {h_final, layer_ctx};
    }
    // ADDED: Method for testing access
    GRUDCell* get_cell_for_testing() {
        return cell.get();  // Assuming cell is a std::unique_ptr<GRUDCell>
    }
    Eigen::MatrixXf forward(const Eigen::MatrixXf& input, Context& ctx) override {
        throw std::runtime_error("TemporalRNNLayer requires forward_temporal method");
    }

    Eigen::MatrixXf backward(const Eigen::MatrixXf& grad_output, const Context& ctx) override {
        throw std::runtime_error("TemporalRNNLayer backward pass handled by autograd system");
    }

    std::vector<Module*> children() override {
        std::vector<Module*> child_modules = {cell.get()};
        if (layer_norm) child_modules.push_back(layer_norm.get());
        if (dropout) child_modules.push_back(dropout.get());
        return child_modules;
    }

    std::string name() const override {
        return "TemporalRNNLayer[" + std::to_string(layer_index_) + "]";
    }
};

} // namespace layers
} // namespace grud

#endif // GRUD_LAYERS_TEMPORAL_H