//
// Created by Esteban Lanter on 03.06.2025.
//

#ifndef TENSOREIGEN_GRUD_EIGEN_H
#define TENSOREIGEN_GRUD_EIGEN_H

// Enhanced GRU-D implementation with optional mask projection learning
#include <Eigen/Dense>
#include <vector>
#include <deque>
#include <memory>
#include <cmath>
#include <random>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <string>
#include <optional>
#include <chrono>
#include <fstream>

using namespace Eigen;

// Type aliases for cleaner code
using MatrixXf = Eigen::MatrixXf;
using VectorXf = Eigen::VectorXf;
using ArrayXf = Eigen::ArrayXf;
using ArrayXXf = Eigen::ArrayXXf;

// ENHANCED: Configuration structure with mask learning control
struct NpTemporalConfig
{
    // Architecture
    int batch_size = 1;
    int in_size = 4;
    int hid_size = 64;
    int num_layers = 2;

    // Decay / imputation
    bool use_exponential_decay = true;
    float softclip_threshold = 3.0f;
    float min_log_gamma = -10.0f;

    // NEW: Mask learning control
    bool enable_mask_learning = false;  // When false, behaves exactly as before

    // Weighting ramp
    float ramp_start = 0.5f;
    float ramp_end = 1.0f;

    // Regularizers
    float dropout = 0.1f;
    float final_dropout = 0.1f;
    bool layer_norm = true;
    std::optional<float> clip_grad_norm = 5.0f;
    float weight_decay = 0.0f;

    // Optimization
    int tbptt_steps = 20;
    float lr = 2e-3f;
    std::string optimizer = "sgd"; // "sgd" or "adamw"
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float eps_adam = 1e-8f;

    // Losses
    std::string loss = "huber"; // "mse", "mae", or "huber"
    int loss_horizon = -1;      // -1 means use all steps in window

    // Random seed
    int seed = 0;
};

// Forward declaration of cache structures (unchanged)
struct LayerNormCache
{
    MatrixXf x;                       // (B, H)
    MatrixXf mu;                      // (B, 1) - mean per sample
    MatrixXf x_normalized_pre_affine; // (B, H)
    MatrixXf std_dev_w_eps;           // (B, 1) - std_dev per sample
    float N;                          // H (number of features)
};

struct LinearCache
{
    MatrixXf input;
};

struct GammaCache
{
    VectorXf decay_param;
    MatrixXf dt_val;
    VectorXf log_gamma_arg_softclipped;
    VectorXf softplus_val;
    MatrixXf log_gamma_unclamped;
    MatrixXf log_gamma;
    MatrixXf gamma;
};

struct DecayInfoCache
{
    bool decay_active = false;
    MatrixXf h_prev_for_decay;
    std::optional<GammaCache> gamma_h_internals_cache;
    MatrixXf gamma_h_value;
};

struct ImputationOpCache
{
    MatrixXf impute_x_raw;
    MatrixXf impute_mask;
    MatrixXf impute_x_hat;
};

struct CellFwdCache
{
    ImputationOpCache imputation_op_cache;
    LinearCache impute_linear_fwd_input_cache;
    DecayInfoCache decay_info_cache;
    MatrixXf x_tilde_val;
    MatrixXf h_decay_val;
    MatrixXf r_arg_val;
    MatrixXf r_val;
    MatrixXf z_arg_val;
    MatrixXf z_val;
    MatrixXf rh_decay_prod_val;
    MatrixXf h_candidate_arg_val;
    MatrixXf h_candidate_val;
    LinearCache r_arg_Wx_ic, r_arg_Uh_ic, r_arg_Vdt_ic;
    LinearCache z_arg_Wx_ic, z_arg_Uh_ic, z_arg_Vdt_ic;
    LinearCache hc_arg_Wx_ic, hc_arg_Uh_ic, hc_arg_Vdt_ic;
};

struct LayerFwdCache
{
    CellFwdCache cell_fwd_cache;
    bool ln_active = false;
    std::optional<LayerNormCache> ln_fwd_cache;
    bool dropout_active = false;
    std::optional<MatrixXf> dropout_mask_cache; // This mask is for (B,H)
};

// Math helper functions (unchanged)
inline VectorXf sigmoid(const VectorXf &x)
{
    return (1.0f / (1.0f + (-x.array().cwiseMax(-80.0f).cwiseMin(80.0f)).exp())).matrix();
}

inline MatrixXf sigmoid(const MatrixXf &x)
{
    return (1.0f / (1.0f + (-x.array().cwiseMax(-80.0f).cwiseMin(80.0f)).exp())).matrix();
}

inline VectorXf d_sigmoid(const VectorXf &x)
{
    VectorXf s = sigmoid(x);
    return s.array() * (1.0f - s.array());
}

inline MatrixXf d_sigmoid(const MatrixXf &x)
{
    MatrixXf s = sigmoid(x);
    return s.array() * (1.0f - s.array());
}

inline MatrixXf tanh_activation(const MatrixXf &x)
{
    return x.array().tanh().matrix();
}

inline MatrixXf d_tanh(const MatrixXf &x)
{
    ArrayXXf tanh_x = x.array().tanh();
    return (1.0f - tanh_x * tanh_x).matrix();
}

inline VectorXf softplus(const VectorXf &x)
{
    VectorXf x_clipped = x.array().cwiseMax(-80.0f).cwiseMin(80.0f).matrix();
    return (1.0f + x_clipped.array().exp()).log().matrix();
}

inline MatrixXf softplus(const MatrixXf &x)
{
    MatrixXf x_clipped = x.array().cwiseMax(-80.0f).cwiseMin(80.0f).matrix();
    return (1.0f + x_clipped.array().exp()).log().matrix();
}

inline VectorXf d_softplus(const VectorXf &x)
{
    return sigmoid(x);
}

inline MatrixXf d_softplus(const MatrixXf &x)
{
    return sigmoid(x);
}

VectorXf softclip(const VectorXf &x, float threshold)
{
    if (threshold <= 0)
    {
        throw std::invalid_argument("Threshold must be a positive scalar for logarithmic softclip.");
    }
    ArrayXf abs_x = x.array().abs();
    return (abs_x <= threshold).select(x.array(), x.array().sign() * (threshold + (abs_x - threshold + 1.0f).log())).matrix();
}

MatrixXf softclip(const MatrixXf &x, float threshold)
{
    if (threshold <= 0)
    {
        throw std::invalid_argument("Threshold must be a positive scalar for logarithmic softclip.");
    }
    ArrayXXf abs_x = x.array().abs();
    return (abs_x <= threshold).select(x.array(), x.array().sign() * (threshold + (abs_x - threshold + 1.0f).log())).matrix();
}

VectorXf d_softclip(const VectorXf &x, float threshold)
{
    if (threshold <= 0)
    {
        throw std::invalid_argument("Threshold must be a positive scalar for logarithmic softclip.");
    }
    ArrayXf abs_x = x.array().abs();
    ArrayXf is_linear = (abs_x > threshold).cast<float>();
    ArrayXf sigmoid_arg = abs_x - threshold;
    ArrayXf sigmoid_val = 1.0f / (1.0f + (-sigmoid_arg.cwiseMax(-80.0f).cwiseMin(80.0f)).exp());
    return (1.0f - is_linear + is_linear * sigmoid_val * x.array().sign()).matrix();
}

MatrixXf d_softclip(const MatrixXf &x, float threshold)
{
    if (threshold <= 0)
    {
        throw std::invalid_argument("Threshold must be a positive scalar for logarithmic softclip.");
    }
    ArrayXXf abs_x = x.array().abs();
    ArrayXXf is_linear = (abs_x > threshold).cast<float>();
    ArrayXXf sigmoid_arg = abs_x - threshold;
    ArrayXXf sigmoid_val = 1.0f / (1.0f + (-sigmoid_arg.cwiseMax(-80.0f).cwiseMin(80.0f)).exp());
    return (1.0f - is_linear + is_linear * sigmoid_val * x.array().sign()).matrix();
}

// Random initialization (unchanged)
MatrixXf glorot_uniform_initializer(int out_features, int in_features, std::mt19937 &gen)
{
    float limit = std::sqrt(6.0f / (in_features + out_features));
    std::uniform_real_distribution<float> dist(-limit, limit);
    MatrixXf result(out_features, in_features);
    for (int i = 0; i < result.rows(); ++i)
    {
        for (int j = 0; j < result.cols(); ++j)
        {
            result(i, j) = dist(gen);
        }
    }
    return result;
}

// LayerNorm, Dropout, and Linear implementations (unchanged from your original code)
class NpLayerNorm
{
private:
    int normalized_shape_len; // H (number of features to normalize over)
    float eps;
    bool elementwise_affine;
    bool training_mode;

public:
    VectorXf gamma; // (H,)
    VectorXf beta;  // (H,)
    VectorXf grad_gamma;
    VectorXf grad_beta;

    NpLayerNorm(int normalized_shape_len, float eps_val = 1e-5f, bool affine = true)
            : normalized_shape_len(normalized_shape_len), eps(eps_val),
              elementwise_affine(affine), training_mode(true)
    {
        if (elementwise_affine)
        {
            gamma = VectorXf::Ones(normalized_shape_len);
            beta = VectorXf::Zero(normalized_shape_len);
            grad_gamma = VectorXf::Zero(normalized_shape_len);
            grad_beta = VectorXf::Zero(normalized_shape_len);
        }
    }

    std::pair<MatrixXf, LayerNormCache> forward(const MatrixXf &x)
    { // x: (B, H)
        if (x.rows() == 0 || x.cols() == 0)
        {
            // std::cerr << "LayerNorm forward: Empty input!" << std::endl;
        }

        MatrixXf mu = x.rowwise().mean();                                     // Shape (B, 1), mean for each sample
        MatrixXf centered_x = x.colwise() - mu.col(0);                        // Broadcast mu (B,1) to (B,H)
        MatrixXf var = centered_x.array().square().rowwise().mean().matrix(); // Shape (B, 1), var for each sample
        MatrixXf std_dev_w_eps = (var.array() + eps).sqrt().matrix();         // Shape (B, 1)

        // Normalize each row: (B, H).array() / (B, 1).array() broadcasted
        MatrixXf x_normalized = centered_x.array().colwise() / std_dev_w_eps.col(0).array();

        LayerNormCache cache;
        cache.x = x;
        cache.mu = mu;
        cache.x_normalized_pre_affine = x_normalized;
        cache.std_dev_w_eps = std_dev_w_eps;
        cache.N = static_cast<float>(x.cols()); // H (number of features)

        MatrixXf output;
        if (elementwise_affine)
        {
            // Apply affine transformation using element-wise operations to avoid broadcasting issues
            output = MatrixXf(x.rows(), x.cols());
            for (int i = 0; i < x.rows(); ++i)
            {
                output.row(i) = x_normalized.row(i).array() * gamma.transpose().array() + beta.transpose().array();
            }
        }
        else
        {
            output = x_normalized;
        }

        return {output, cache};
    }

    MatrixXf backward(const MatrixXf &d_output, const LayerNormCache &cache)
    { // d_output: (B, H)
        MatrixXf d_x_normalized;
        const MatrixXf &x_pre_affine = cache.x_normalized_pre_affine; // (B, H)

        if (elementwise_affine)
        {
            // Sum over batch dimension (axis 0) for parameter gradients
            // (B,H) .* (B,H) -> sum over B -> (1,H) -> transpose -> (H,1)
            grad_gamma.noalias() += (d_output.array() * x_pre_affine.array()).matrix().colwise().sum().transpose();
            grad_beta.noalias() += d_output.colwise().sum().transpose();

            // Use element-wise operations to avoid broadcasting issues
            d_x_normalized = MatrixXf(d_output.rows(), d_output.cols());
            for (int i = 0; i < d_output.rows(); ++i)
            {
                d_x_normalized.row(i) = d_output.row(i).array() * gamma.transpose().array();
            }
        }
        else
        {
            d_x_normalized = d_output;
        }

        float N_feat = cache.N;                                // H (number of features)
        const MatrixXf &x_orig = cache.x;                      // (B, H)
        const MatrixXf &mu_b = cache.mu;                       // (B, 1), mu per batch sample
        const MatrixXf &std_dev_w_eps_b = cache.std_dev_w_eps; // (B, 1), std per batch sample

        // Replicate mu_b and std_dev_w_eps_b to (B, H) for element-wise operations
        MatrixXf mu_bh = mu_b.replicate(1, x_orig.cols());                       // (B,H)
        MatrixXf std_dev_w_eps_bh = std_dev_w_eps_b.replicate(1, x_orig.cols()); // (B,H)
        MatrixXf x_centered = x_orig - mu_bh;                                    // (B,H)

        // dL/dvar = sum_k dL/dx_norm_k * (x_k-mu) * (-1/2) * (std_w_eps)^(-3)
        // All are (B,H), sum over H (features) -> (B,1)
        MatrixXf d_L_dvar = (d_x_normalized.array() * x_centered.array() * -0.5f * std_dev_w_eps_bh.array().pow(-3.0f))
                .matrix()
                .rowwise()
                .sum(); // (B,1)

        // dL/dmu = sum_k dL/dx_norm_k * (-1/std_w_eps) + dL/dvar * sum_k (-2*(x_k-mu)/N)
        // First term: (B,H) .* (B,H) -> sum over H -> (B,1)
        MatrixXf d_L_dmu_term1 = (d_x_normalized.array() * (-1.0f / std_dev_w_eps_bh.array()))
                .matrix()
                .rowwise()
                .sum(); // (B,1)
        // Second term: (B,1).array() * ( (B,H).rowwise().sum() * (-2/N_feat) -> (B,1).array() )
        MatrixXf d_L_dmu_term2 = d_L_dvar.array() *
                                 (x_centered.array().rowwise().sum() * (-2.0f / N_feat)); // (B,1)
        MatrixXf d_L_dmu = d_L_dmu_term1 + d_L_dmu_term2;                                 // (B,1)

        // dL/dx_i = dL/dx_norm_i * (1/std_w_eps) + dL/dvar * (2*(x_i-mu)/N_feat) + dL/dmu * (1/N_feat)
        // All terms will be (B,H) after broadcasting d_L_dvar and d_L_dmu
        MatrixXf d_input = (d_x_normalized.array() / std_dev_w_eps_bh.array()).matrix() +
                           (d_L_dvar.replicate(1, x_orig.cols()).array() * (2.0f * x_centered.array() / N_feat)).matrix() +
                           (d_L_dmu.replicate(1, x_orig.cols()).array() / N_feat).matrix(); // (B,H)

        return d_input;
    }

    void zero_grad()
    {
        if (elementwise_affine)
        {
            grad_gamma.setZero();
            grad_beta.setZero();
        }
    }

    void set_training_mode(bool is_training)
    {
        training_mode = is_training;
    }
    bool get_elementwise_affine() const
    {
        return elementwise_affine;
    }
};

class NpDropout
{
private:
    float p;
    bool training_mode;
    std::mt19937 &rng; // Use reference to shared generator
    std::uniform_real_distribution<float> dist;

public:
    NpDropout(float p_val, std::mt19937 &generator) : p(p_val), training_mode(true), rng(generator), dist(0.0f, 1.0f)
    {
        if (p < 0.0f || p > 1.0f)
        {
            throw std::invalid_argument("Dropout probability has to be between 0 and 1");
        }
    }

    // Input x: (AnyShape), e.g., (B,H) or (T*B, H)
    std::pair<MatrixXf, std::optional<MatrixXf>> forward(const MatrixXf &x)
    {
        if (!training_mode || p == 0.0f)
        {
            return {x, std::nullopt};
        }

        float scale_factor = 1.0f / (1.0f - p);
        MatrixXf mask(x.rows(), x.cols());

        for (int r = 0; r < mask.rows(); ++r)
        {
            for (int c = 0; c < mask.cols(); ++c)
            {
                mask(r, c) = (dist(rng) > p) ? 1.0f : 0.0f;
            }
        }
        return {(x.array() * mask.array() * scale_factor).matrix(), mask};
    }

    MatrixXf backward(const MatrixXf &d_output, const std::optional<MatrixXf> &mask)
    {
        if (!training_mode || p == 0.0f || !mask.has_value())
        {
            return d_output;
        }

        float scale_factor = 1.0f / (1.0f - p);
        return (d_output.array() * mask.value().array() * scale_factor).matrix();
    }

    void set_training_mode(bool is_training)
    {
        training_mode = is_training;
    }
};

class NpLinear
{
private:
    int in_features;
    int out_features;
    bool use_bias;

public:
    MatrixXf weights; // (out_features, in_features)
    VectorXf bias;    // (out_features)
    MatrixXf grad_weights;
    VectorXf grad_bias;

    NpLinear(int in_feat, int out_feat, bool bias_flag = true, std::mt19937 *gen_ptr = nullptr) // Pass generator by pointer
            : in_features(in_feat), out_features(out_feat), use_bias(bias_flag)
    {
        // If gen_ptr is null, create a local one for this layer (not ideal for global seed consistency)
        std::mt19937 local_gen_for_init(gen_ptr ? (*gen_ptr)() : std::random_device{}());
        std::mt19937 &gen_ref = gen_ptr ? *gen_ptr : local_gen_for_init;

        weights = glorot_uniform_initializer(out_features, in_features, gen_ref);
        grad_weights = MatrixXf::Zero(out_features, in_features);

        if (use_bias)
        {
            bias = VectorXf::Zero(out_features); // Eigen initializes to zero by default, but explicit is fine
            grad_bias = VectorXf::Zero(out_features);
        }
    }

    // Input x: (BatchDim, in_features)
    std::pair<MatrixXf, LinearCache> forward(const MatrixXf &x)
    {
        LinearCache cache;
        cache.input = x; // Copy for backward pass

        MatrixXf output = x * weights.transpose(); // (B, I) * (I, O) -> (B, O)
        if (use_bias)
        {
            output.rowwise() += bias.transpose(); // Broadcast (1,O) bias to each row of (B,O)
        }

        return {output, cache};
    }

    // d_output: (BatchDim, out_features)
    MatrixXf backward(const MatrixXf &d_output, const LinearCache &cache)
    {
        // grad_weights: (O,I) += (O,B) * (B,I)
        grad_weights.noalias() += d_output.transpose() * cache.input;

        if (use_bias)
        {
            // grad_bias: (O,) += sum over B of (B,O) -> (O,)
            grad_bias.noalias() += d_output.colwise().sum().transpose();
        }

        // d_input: (B,I) = (B,O) * (O,I)
        return d_output * weights;
    }

    void zero_grad()
    {
        grad_weights.setZero();
        if (use_bias)
        {
            grad_bias.setZero();
        }
    }
};

// ENHANCED: Base temporal cell with optional mask learning
class NpBaseTemporalCell
{
protected:
    NpTemporalConfig cfg;
    int hid_size;
    int in_size;
    std::mt19937 &gen; // Use reference to shared generator

public:
    NpLinear impute_linear;
    VectorXf decay_h;
    VectorXf grad_decay_h;

    NpBaseTemporalCell(const NpTemporalConfig &config, int current_in_size, std::mt19937 &generator)
            : cfg(config), hid_size(cfg.hid_size), in_size(current_in_size), gen(generator),
              impute_linear(hid_size, in_size, true, &gen)
    { // Pass generator pointer
        if (cfg.use_exponential_decay)
        {
            decay_h = VectorXf::Zero(hid_size);
            grad_decay_h = VectorXf::Zero(hid_size);
        }
    }

    std::pair<MatrixXf, GammaCache> gamma(const VectorXf &decay_param, const MatrixXf &dt)
    {
        VectorXf log_gamma_arg_softclipped = softclip(decay_param, cfg.softclip_threshold);
        VectorXf softplus_val = softplus(log_gamma_arg_softclipped); // (H,)

        // Broadcasting: dt (B,1) * softplus_val.transpose() (1,H) -> (B,H)
        MatrixXf log_gamma_unclamped = -dt * softplus_val.transpose();

        MatrixXf log_gamma = log_gamma_unclamped.array().cwiseMax(cfg.min_log_gamma).cwiseMin(-1e-4f).matrix();
        MatrixXf gamma_val = log_gamma.array().exp().matrix();

        GammaCache cache;
        cache.decay_param = decay_param;                             // Copy
        cache.dt_val = dt;                                           // Copy
        cache.log_gamma_arg_softclipped = log_gamma_arg_softclipped; // Copy
        cache.softplus_val = softplus_val;                           // Copy
        cache.log_gamma_unclamped = log_gamma_unclamped;             // Copy
        cache.log_gamma = log_gamma;                                 // Copy
        cache.gamma = gamma_val;                                     // Copy

        return {gamma_val, cache};
    }

    VectorXf backward_gamma(const MatrixXf &d_gamma, const GammaCache &cache)
    {
        MatrixXf d_log_gamma = d_gamma.array() * cache.gamma.array();

        // Fix: Replace && operator with proper Eigen operations using element-wise multiplication
        ArrayXXf cond1 = (cache.log_gamma_unclamped.array() >= cfg.min_log_gamma).cast<float>();
        ArrayXXf cond2 = (cache.log_gamma_unclamped.array() <= -1e-4f).cast<float>();
        MatrixXf grad_clip_mask = (cond1 * cond2).matrix();
        MatrixXf d_log_gamma_unclamped = d_log_gamma.array() * grad_clip_mask.array();

        // d_softplus_val_each_b: (B,H).array() * (-(B,1).array() broadcasted)
        MatrixXf d_softplus_val_each_b = d_log_gamma_unclamped.array().colwise() * (-cache.dt_val.col(0).array());
        VectorXf d_softplus_val = d_softplus_val_each_b.colwise().sum().transpose(); // (H,)

        VectorXf d_log_gamma_arg_softclipped = d_softplus_val.array() *
                                               d_softplus(cache.log_gamma_arg_softclipped).array();
        VectorXf d_decay_param = d_log_gamma_arg_softclipped.array() *
                                 d_softclip(cache.decay_param, cfg.softclip_threshold).array();

        return d_decay_param;
    }

    std::pair<MatrixXf, DecayInfoCache> apply_decay_h_op(const MatrixXf &h_prev, const MatrixXf &dt)
    {
        DecayInfoCache decay_info_cache;
        decay_info_cache.decay_active = false;
        decay_info_cache.h_prev_for_decay = h_prev; // Copy

        if (!cfg.use_exponential_decay)
        {
            return {h_prev, decay_info_cache};
        }

        decay_info_cache.decay_active = true;
        auto [gamma_h, gamma_h_internals_cache_val] = gamma(decay_h, dt);
        decay_info_cache.gamma_h_internals_cache = gamma_h_internals_cache_val;
        decay_info_cache.gamma_h_value = gamma_h; // Copy

        return {(h_prev.array() * gamma_h.array()).matrix(), decay_info_cache};
    }

    std::pair<MatrixXf, std::optional<VectorXf>> backward_apply_decay_h_op(
            const MatrixXf &d_h_decayed, const DecayInfoCache &decay_info_cache)
    {

        if (!decay_info_cache.decay_active || !cfg.use_exponential_decay)
        {
            return {d_h_decayed, std::nullopt};
        }

        MatrixXf h_prev_this_step = decay_info_cache.h_prev_for_decay;
        MatrixXf gamma_h_val = decay_info_cache.gamma_h_value;

        MatrixXf d_h_prev = d_h_decayed.array() * gamma_h_val.array();
        MatrixXf d_gamma_h = d_h_decayed.array() * h_prev_this_step.array();

        VectorXf grad_decay_h_update = backward_gamma(d_gamma_h,
                                                      decay_info_cache.gamma_h_internals_cache.value());

        return {d_h_prev, grad_decay_h_update};
    }

    std::tuple<MatrixXf, ImputationOpCache, LinearCache> impute_x_op(
            const MatrixXf &x, const MatrixXf &h_prev, const std::optional<MatrixXf> &mask)
    {

        MatrixXf current_mask = mask.has_value() ? mask.value() : MatrixXf::Ones(x.rows(), x.cols());

        auto [x_hat, impute_linear_cache] = impute_linear.forward(h_prev);

        MatrixXf x_tilde = current_mask.array() * x.array() +
                           (1.0f - current_mask.array()) * x_hat.array();

        ImputationOpCache imputation_op_cache_val;
        imputation_op_cache_val.impute_x_raw = x;           // Copy
        imputation_op_cache_val.impute_mask = current_mask; // Copy
        imputation_op_cache_val.impute_x_hat = x_hat;       // Copy

        return {x_tilde, imputation_op_cache_val, impute_linear_cache};
    }

    // ENHANCED: Two versions of backward_impute_x_op - with and without mask gradients

    // Original version (used when mask learning is disabled)
    std::pair<MatrixXf, MatrixXf> backward_impute_x_op(
            const MatrixXf &d_x_tilde, const ImputationOpCache &cache, const LinearCache &linear_cache)
    {
        MatrixXf mask_val = cache.impute_mask;
        MatrixXf d_x_raw = d_x_tilde.array() * mask_val.array();
        MatrixXf d_x_hat = d_x_tilde.array() * (1.0f - mask_val.array());
        MatrixXf d_h_prev_from_imputation = impute_linear.backward(d_x_hat, linear_cache);

        return {d_x_raw, d_h_prev_from_imputation};
    }

    // NEW: Enhanced version (used when mask learning is enabled)
    std::tuple<MatrixXf, MatrixXf, MatrixXf> backward_impute_x_op_with_mask_grad(
            const MatrixXf &d_x_tilde, const ImputationOpCache &cache, const LinearCache &linear_cache)
    {
        const MatrixXf& mask_val = cache.impute_mask;       // (B, F)
        const MatrixXf& x_raw = cache.impute_x_raw;         // (B, F)
        const MatrixXf& x_hat = cache.impute_x_hat;         // (B, F)

        // Original gradients
        MatrixXf d_x_raw = d_x_tilde.array() * mask_val.array();
        MatrixXf d_x_hat = d_x_tilde.array() * (1.0f - mask_val.array());
        MatrixXf d_h_prev_from_imputation = impute_linear.backward(d_x_hat, linear_cache);

        // NEW: Gradient w.r.t. mask
        // From x_tilde = mask * x + (1 - mask) * x_hat
        // ∂L/∂mask = ∂L/∂x_tilde * ∂x_tilde/∂mask = ∂L/∂x_tilde * (x - x_hat)
        MatrixXf d_mask = d_x_tilde.array() * (x_raw.array() - x_hat.array());

        return {d_x_raw, d_h_prev_from_imputation, d_mask};
    }
};

// ENHANCED: GRU-D Cell with optional mask learning
class NpGRUDCell : public NpBaseTemporalCell
{
public:
    NpLinear W_r, U_r, V_r;
    NpLinear W_z, U_z, V_z;
    NpLinear W_h_candidate, U_h_candidate, V_h_candidate;

    NpGRUDCell(const NpTemporalConfig &config, int current_in_size, std::mt19937 &generator)
            : NpBaseTemporalCell(config, current_in_size, generator),
              W_r(current_in_size, cfg.hid_size, true, &gen),
              U_r(cfg.hid_size, cfg.hid_size, false, &gen),
              V_r(1, cfg.hid_size, false, &gen), // dt is (B,1)
              W_z(current_in_size, cfg.hid_size, true, &gen),
              U_z(cfg.hid_size, cfg.hid_size, false, &gen),
              V_z(1, cfg.hid_size, false, &gen), // dt is (B,1)
              W_h_candidate(current_in_size, cfg.hid_size, true, &gen),
              U_h_candidate(cfg.hid_size, cfg.hid_size, false, &gen),
              V_h_candidate(1, cfg.hid_size, false, &gen)
    { // dt is (B,1)
        if (W_z.bias.size() > 0)
        {
            W_z.bias.setConstant(-1.0f);
        }
    }

    std::pair<MatrixXf, CellFwdCache> forward(const MatrixXf &x, const MatrixXf &h_prev,
                                              const MatrixXf &dt, const std::optional<MatrixXf> &mask)
    {
        MatrixXf x_tilde;
        ImputationOpCache imputation_op_cache_val;
        LinearCache ilic;

        std::tie(x_tilde, imputation_op_cache_val, ilic) = impute_x_op(x, h_prev, mask);

        auto [h_decay, decay_info_cache_val] = apply_decay_h_op(h_prev, dt);

        auto [r_arg_Wx_out, r_arg_Wx_ic_val] = W_r.forward(x_tilde);
        auto [r_arg_Uh_out, r_arg_Uh_ic_val] = U_r.forward(h_decay);
        auto [r_arg_Vdt_out, r_arg_Vdt_ic_val] = V_r.forward(dt);
        MatrixXf r_arg_val_ = r_arg_Wx_out + r_arg_Uh_out + r_arg_Vdt_out;
        MatrixXf r_val_ = sigmoid(r_arg_val_);

        auto [z_arg_Wx_out, z_arg_Wx_ic_val] = W_z.forward(x_tilde);
        auto [z_arg_Uh_out, z_arg_Uh_ic_val] = U_z.forward(h_decay);
        auto [z_arg_Vdt_out, z_arg_Vdt_ic_val] = V_z.forward(dt);
        MatrixXf z_arg_val_ = z_arg_Wx_out + z_arg_Uh_out + z_arg_Vdt_out;
        MatrixXf z_val_ = sigmoid(z_arg_val_);

        MatrixXf rh_decay_prod_val_ = r_val_.array() * h_decay.array();
        auto [hc_arg_Wx_out, hc_arg_Wx_ic_val] = W_h_candidate.forward(x_tilde);
        auto [hc_arg_Uh_out, hc_arg_Uh_ic_val] = U_h_candidate.forward(rh_decay_prod_val_);
        auto [hc_arg_Vdt_out, hc_arg_Vdt_ic_val] = V_h_candidate.forward(dt);
        MatrixXf h_candidate_arg_val_ = hc_arg_Wx_out + hc_arg_Uh_out + hc_arg_Vdt_out;
        MatrixXf h_candidate_val_ = tanh_activation(h_candidate_arg_val_);

        MatrixXf h_next = (1.0f - z_val_.array()) * h_decay.array() + z_val_.array() * h_candidate_val_.array();

        CellFwdCache cell_cache_val;
        cell_cache_val.imputation_op_cache = imputation_op_cache_val;
        cell_cache_val.impute_linear_fwd_input_cache = ilic;
        cell_cache_val.decay_info_cache = decay_info_cache_val;
        cell_cache_val.x_tilde_val = x_tilde;                      // Copy
        cell_cache_val.h_decay_val = h_decay;                      // Copy
        cell_cache_val.r_arg_val = r_arg_val_;                     // Copy
        cell_cache_val.r_val = r_val_;                             // Copy
        cell_cache_val.z_arg_val = z_arg_val_;                     // Copy
        cell_cache_val.z_val = z_val_;                             // Copy
        cell_cache_val.rh_decay_prod_val = rh_decay_prod_val_;     // Copy
        cell_cache_val.h_candidate_arg_val = h_candidate_arg_val_; // Copy
        cell_cache_val.h_candidate_val = h_candidate_val_;         // Copy
        cell_cache_val.r_arg_Wx_ic = r_arg_Wx_ic_val;
        cell_cache_val.r_arg_Uh_ic = r_arg_Uh_ic_val;
        cell_cache_val.r_arg_Vdt_ic = r_arg_Vdt_ic_val;
        cell_cache_val.z_arg_Wx_ic = z_arg_Wx_ic_val;
        cell_cache_val.z_arg_Uh_ic = z_arg_Uh_ic_val;
        cell_cache_val.z_arg_Vdt_ic = z_arg_Vdt_ic_val;
        cell_cache_val.hc_arg_Wx_ic = hc_arg_Wx_ic_val;
        cell_cache_val.hc_arg_Uh_ic = hc_arg_Uh_ic_val;
        cell_cache_val.hc_arg_Vdt_ic = hc_arg_Vdt_ic_val;

        return {h_next, cell_cache_val};
    }

    // ENHANCED: Backward pass with optional mask gradient computation
    // When mask learning is disabled, behaves exactly as before
    // When mask learning is enabled, computes and returns mask gradients
    std::tuple<MatrixXf, MatrixXf, MatrixXf> backward(const MatrixXf &d_h_next, const CellFwdCache &cache)
    {
        // Common backward pass logic
        const auto &imputation_op_cache = cache.imputation_op_cache;
        const auto &ilic = cache.impute_linear_fwd_input_cache;
        const auto &decay_info_cache = cache.decay_info_cache;

        const MatrixXf &h_decay = cache.h_decay_val;
        const MatrixXf &r = cache.r_val;
        const MatrixXf &z = cache.z_val;
        const MatrixXf &h_cand = cache.h_candidate_val;

        MatrixXf d_z_elementwise = d_h_next.array() * (h_cand.array() - h_decay.array());
        MatrixXf d_h_decay_from_h_next_eq = d_h_next.array() * (1.0f - z.array());
        MatrixXf d_h_candidate_from_h_next_eq = d_h_next.array() * z.array();

        MatrixXf d_h_cand_arg = d_h_candidate_from_h_next_eq.array() * d_tanh(cache.h_candidate_arg_val).array();

        MatrixXf d_x_tilde_from_Wh = W_h_candidate.backward(d_h_cand_arg, cache.hc_arg_Wx_ic);
        MatrixXf d_rh_decay_prod_from_Uh = U_h_candidate.backward(d_h_cand_arg, cache.hc_arg_Uh_ic);
        MatrixXf d_dt_from_Vh = V_h_candidate.backward(d_h_cand_arg, cache.hc_arg_Vdt_ic);

        MatrixXf d_r_from_rh_prod = d_rh_decay_prod_from_Uh.array() * h_decay.array();
        MatrixXf d_h_decay_from_rh_prod = d_rh_decay_prod_from_Uh.array() * r.array();

        MatrixXf d_z_arg = d_z_elementwise.array() * d_sigmoid(cache.z_arg_val).array();

        MatrixXf d_x_tilde_from_Wz = W_z.backward(d_z_arg, cache.z_arg_Wx_ic);
        MatrixXf d_h_decay_from_Uz = U_z.backward(d_z_arg, cache.z_arg_Uh_ic);
        MatrixXf d_dt_from_Vz = V_z.backward(d_z_arg, cache.z_arg_Vdt_ic);

        MatrixXf d_r_arg = d_r_from_rh_prod.array() * d_sigmoid(cache.r_arg_val).array();

        MatrixXf d_x_tilde_from_Wr = W_r.backward(d_r_arg, cache.r_arg_Wx_ic);
        MatrixXf d_h_decay_from_Ur = U_r.backward(d_r_arg, cache.r_arg_Uh_ic);
        MatrixXf d_dt_from_Vr = V_r.backward(d_r_arg, cache.r_arg_Vdt_ic);

        // Accumulate dt gradients
        MatrixXf d_dt_total = d_dt_from_Vh + d_dt_from_Vz + d_dt_from_Vr;

        MatrixXf d_x_tilde = d_x_tilde_from_Wh + d_x_tilde_from_Wz + d_x_tilde_from_Wr;
        MatrixXf d_h_decay_total = d_h_decay_from_h_next_eq + d_h_decay_from_rh_prod +
                                   d_h_decay_from_Uz + d_h_decay_from_Ur;

        // ENHANCED: Use appropriate imputation backward based on configuration
        MatrixXf d_x_raw, d_h_prev_from_impute;

        if (cfg.enable_mask_learning) {
            // When mask learning is enabled, compute mask gradients but don't return them
            // (they would be handled by the higher-level mask projector)
            auto [d_x_raw_temp, d_h_prev_temp, d_mask_temp] = backward_impute_x_op_with_mask_grad(d_x_tilde, imputation_op_cache, ilic);
            d_x_raw = d_x_raw_temp;
            d_h_prev_from_impute = d_h_prev_temp;
            // d_mask_temp contains the mask gradients but we don't expose them at this level
            // They would be used by external mask projectors if needed
        } else {
            // When mask learning is disabled, use the efficient original version
            auto [d_x_raw_temp, d_h_prev_temp] = backward_impute_x_op(d_x_tilde, imputation_op_cache, ilic);
            d_x_raw = d_x_raw_temp;
            d_h_prev_from_impute = d_h_prev_temp;
        }

        auto [d_h_prev_from_decay, grad_decay_h_update] = backward_apply_decay_h_op(d_h_decay_total, decay_info_cache);

        MatrixXf d_h_prev_total = d_h_prev_from_impute + d_h_prev_from_decay;

        if (grad_decay_h_update.has_value() && cfg.use_exponential_decay)
        {
            grad_decay_h.noalias() += grad_decay_h_update.value();
        }

        return {d_x_raw, d_h_prev_total, d_dt_total};
    }

    // NEW: Special backward method that exposes mask gradients (for use by mask projectors)
    std::tuple<MatrixXf, MatrixXf, MatrixXf, MatrixXf> backward_with_mask_grad(const MatrixXf &d_h_next, const CellFwdCache &cache)
    {
        if (!cfg.enable_mask_learning) {
            // If mask learning is disabled, return zeros for mask gradient
            auto [d_x_raw, d_h_prev, d_dt] = backward(d_h_next, cache);
            MatrixXf d_mask_zero = MatrixXf::Zero(d_x_raw.rows(), d_x_raw.cols());
            return {d_x_raw, d_h_prev, d_dt, d_mask_zero};
        }

        // Same logic as backward() but expose the mask gradient
        const auto &imputation_op_cache = cache.imputation_op_cache;
        const auto &ilic = cache.impute_linear_fwd_input_cache;
        const auto &decay_info_cache = cache.decay_info_cache;

        const MatrixXf &h_decay = cache.h_decay_val;
        const MatrixXf &r = cache.r_val;
        const MatrixXf &z = cache.z_val;
        const MatrixXf &h_cand = cache.h_candidate_val;

        MatrixXf d_z_elementwise = d_h_next.array() * (h_cand.array() - h_decay.array());
        MatrixXf d_h_decay_from_h_next_eq = d_h_next.array() * (1.0f - z.array());
        MatrixXf d_h_candidate_from_h_next_eq = d_h_next.array() * z.array();

        MatrixXf d_h_cand_arg = d_h_candidate_from_h_next_eq.array() * d_tanh(cache.h_candidate_arg_val).array();

        MatrixXf d_x_tilde_from_Wh = W_h_candidate.backward(d_h_cand_arg, cache.hc_arg_Wx_ic);
        MatrixXf d_rh_decay_prod_from_Uh = U_h_candidate.backward(d_h_cand_arg, cache.hc_arg_Uh_ic);
        MatrixXf d_dt_from_Vh = V_h_candidate.backward(d_h_cand_arg, cache.hc_arg_Vdt_ic);

        MatrixXf d_r_from_rh_prod = d_rh_decay_prod_from_Uh.array() * h_decay.array();
        MatrixXf d_h_decay_from_rh_prod = d_rh_decay_prod_from_Uh.array() * r.array();

        MatrixXf d_z_arg = d_z_elementwise.array() * d_sigmoid(cache.z_arg_val).array();

        MatrixXf d_x_tilde_from_Wz = W_z.backward(d_z_arg, cache.z_arg_Wx_ic);
        MatrixXf d_h_decay_from_Uz = U_z.backward(d_z_arg, cache.z_arg_Uh_ic);
        MatrixXf d_dt_from_Vz = V_z.backward(d_z_arg, cache.z_arg_Vdt_ic);

        MatrixXf d_r_arg = d_r_from_rh_prod.array() * d_sigmoid(cache.r_arg_val).array();

        MatrixXf d_x_tilde_from_Wr = W_r.backward(d_r_arg, cache.r_arg_Wx_ic);
        MatrixXf d_h_decay_from_Ur = U_r.backward(d_r_arg, cache.r_arg_Uh_ic);
        MatrixXf d_dt_from_Vr = V_r.backward(d_r_arg, cache.r_arg_Vdt_ic);

        // Accumulate dt gradients
        MatrixXf d_dt_total = d_dt_from_Vh + d_dt_from_Vz + d_dt_from_Vr;

        MatrixXf d_x_tilde = d_x_tilde_from_Wh + d_x_tilde_from_Wz + d_x_tilde_from_Wr;
        MatrixXf d_h_decay_total = d_h_decay_from_h_next_eq + d_h_decay_from_rh_prod +
                                   d_h_decay_from_Uz + d_h_decay_from_Ur;

        // Get mask gradient from imputation operation
        auto [d_x_raw, d_h_prev_from_impute, d_mask] = backward_impute_x_op_with_mask_grad(d_x_tilde, imputation_op_cache, ilic);

        auto [d_h_prev_from_decay, grad_decay_h_update] = backward_apply_decay_h_op(d_h_decay_total, decay_info_cache);

        MatrixXf d_h_prev_total = d_h_prev_from_impute + d_h_prev_from_decay;

        if (grad_decay_h_update.has_value() && cfg.use_exponential_decay) {
            grad_decay_h.noalias() += grad_decay_h_update.value();
        }

        return {d_x_raw, d_h_prev_total, d_dt_total, d_mask};
    }

    void zero_grad_cell_params()
    {
        impute_linear.zero_grad();
        W_r.zero_grad();
        U_r.zero_grad();
        V_r.zero_grad();
        W_z.zero_grad();
        U_z.zero_grad();
        V_z.zero_grad();
        W_h_candidate.zero_grad();
        U_h_candidate.zero_grad();
        V_h_candidate.zero_grad();
        if (cfg.use_exponential_decay)
        {
            grad_decay_h.setZero();
        }
    }

    std::vector<std::pair<MatrixXf *, MatrixXf *>> get_cell_params_grads()
    {
        std::vector<std::pair<MatrixXf *, MatrixXf *>> pg_list;
        auto add_linear_params = [&](NpLinear &layer)
        {
            pg_list.push_back({&layer.weights, &layer.grad_weights});
        };
        add_linear_params(impute_linear);
        add_linear_params(W_r);
        add_linear_params(U_r);
        add_linear_params(V_r);
        add_linear_params(W_z);
        add_linear_params(U_z);
        add_linear_params(V_z);
        add_linear_params(W_h_candidate);
        add_linear_params(U_h_candidate);
        add_linear_params(V_h_candidate);
        return pg_list;
    }

    std::vector<std::pair<VectorXf *, VectorXf *>> get_cell_bias_params_grads()
    {
        std::vector<std::pair<VectorXf *, VectorXf *>> pg_list;
        auto add_bias = [&](NpLinear &layer)
        {
            if (layer.bias.size() > 0)
                pg_list.push_back({&layer.bias, &layer.grad_bias});
        };
        add_bias(impute_linear);
        add_bias(W_r);
        add_bias(W_z);
        add_bias(W_h_candidate);
        return pg_list;
    }

    std::pair<VectorXf *, VectorXf *> get_decay_params_grads()
    {
        if (cfg.use_exponential_decay)
            return {&decay_h, &grad_decay_h};
        return {nullptr, nullptr};
    }
};

// ENHANCED: RNN Layer with optional mask gradient support
class NpRNNLayer
{
private:
    NpTemporalConfig cfg;
    int layer_idx;
    std::unique_ptr<NpGRUDCell> cell;
    std::unique_ptr<NpLayerNorm> layer_norm_module;
    std::unique_ptr<NpDropout> dropout_module;

public:
    NpRNNLayer(const NpTemporalConfig &config, int l_idx, int current_in_size, std::mt19937 &gen)
            : cfg(config), layer_idx(l_idx)
    {
        cell = std::make_unique<NpGRUDCell>(cfg, current_in_size, gen);

        if (cfg.layer_norm)
        {
            layer_norm_module = std::make_unique<NpLayerNorm>(cfg.hid_size, 1e-5f, true);
        }

        if (cfg.dropout > 0 && layer_idx < cfg.num_layers - 1)
        {
            dropout_module = std::make_unique<NpDropout>(cfg.dropout, gen);
        }
    }

    std::pair<MatrixXf, LayerFwdCache> forward(const MatrixXf &x_t_in, const MatrixXf &h_prev,
                                               const MatrixXf &dt_t, const std::optional<MatrixXf> &mask_t)
    {
        LayerFwdCache layer_cache_val;

        std::optional<MatrixXf> effective_mask = (layer_idx == 0) ? mask_t : std::nullopt;
        auto [h_from_cell, cell_cache_val] = cell->forward(x_t_in, h_prev, dt_t, effective_mask);
        layer_cache_val.cell_fwd_cache = cell_cache_val;

        MatrixXf h_after_ln = h_from_cell;
        layer_cache_val.ln_active = false;
        if (layer_norm_module)
        {
            auto [h_ln, ln_cache_val] = layer_norm_module->forward(h_from_cell);
            h_after_ln = h_ln;
            layer_cache_val.ln_fwd_cache = ln_cache_val;
            layer_cache_val.ln_active = true;
        }

        MatrixXf h_after_dropout = h_after_ln;
        layer_cache_val.dropout_active = false;
        if (dropout_module)
        {
            // Dropout takes (B,H) and its mask will be (B,H)
            auto [h_drop, mask_val] = dropout_module->forward(h_after_ln);
            h_after_dropout = h_drop;
            layer_cache_val.dropout_mask_cache = mask_val;
            layer_cache_val.dropout_active = true;
        }

        return {h_after_dropout, layer_cache_val};
    }

    // Original backward method (no mask gradients)
    std::tuple<MatrixXf, MatrixXf, MatrixXf> backward(const MatrixXf &d_h_output, const LayerFwdCache &cache)
    {
        MatrixXf d_h_before_dropout = d_h_output;
        if (dropout_module && cache.dropout_active)
        {
            d_h_before_dropout = dropout_module->backward(d_h_output, cache.dropout_mask_cache);
        }

        MatrixXf d_h_before_ln = d_h_before_dropout;
        if (layer_norm_module && cache.ln_active)
        {
            d_h_before_ln = layer_norm_module->backward(d_h_before_dropout, cache.ln_fwd_cache.value());
        }

        return cell->backward(d_h_before_ln, cache.cell_fwd_cache);
    }

    // NEW: Enhanced backward method that can return mask gradients (for layer 0 only)
    std::tuple<MatrixXf, MatrixXf, MatrixXf, std::optional<MatrixXf>> backward_with_mask_grad(
            const MatrixXf &d_h_output, const LayerFwdCache &cache)
    {
        MatrixXf d_h_before_dropout = d_h_output;
        if (dropout_module && cache.dropout_active) {
            d_h_before_dropout = dropout_module->backward(d_h_output, cache.dropout_mask_cache);
        }

        MatrixXf d_h_before_ln = d_h_before_dropout;
        if (layer_norm_module && cache.ln_active) {
            d_h_before_ln = layer_norm_module->backward(d_h_before_dropout, cache.ln_fwd_cache.value());
        }

        // Get gradients from cell (including mask gradient if layer 0 and mask learning enabled)
        if (layer_idx == 0 && cfg.enable_mask_learning) {
            auto [d_input, d_h_prev, d_dt, d_mask] = cell->backward_with_mask_grad(d_h_before_ln, cache.cell_fwd_cache);
            return {d_input, d_h_prev, d_dt, std::make_optional(d_mask)};
        } else {
            auto [d_input, d_h_prev, d_dt] = cell->backward(d_h_before_ln, cache.cell_fwd_cache);
            return {d_input, d_h_prev, d_dt, std::nullopt};
        }
    }

    void zero_grad_layer_params()
    {
        cell->zero_grad_cell_params();
        if (layer_norm_module)
            layer_norm_module->zero_grad();
    }

    std::vector<std::pair<MatrixXf *, MatrixXf *>> get_layer_params_grads()
    {
        return cell->get_cell_params_grads();
    }

    std::vector<std::pair<VectorXf *, VectorXf *>> get_layer_bias_params_grads()
    {
        return cell->get_cell_bias_params_grads();
    }

    std::pair<VectorXf *, VectorXf *> get_layer_decay_params_grads()
    {
        return cell->get_decay_params_grads();
    }

    std::pair<VectorXf *, VectorXf *> get_layer_norm_params_grads()
    {
        if (layer_norm_module && layer_norm_module->get_elementwise_affine())
            return {&layer_norm_module->gamma, &layer_norm_module->grad_gamma};
        return {nullptr, nullptr};
    }

    std::pair<VectorXf *, VectorXf *> get_layer_norm_bias_params_grads()
    {
        if (layer_norm_module && layer_norm_module->get_elementwise_affine())
            return {&layer_norm_module->beta, &layer_norm_module->grad_beta};
        return {nullptr, nullptr};
    }

    void set_training_mode(bool is_training)
    {
        if (layer_norm_module)
            layer_norm_module->set_training_mode(is_training);
        if (dropout_module)
            dropout_module->set_training_mode(is_training);
        // Cell itself doesn't have training mode.
    }

    NpGRUDCell *get_cell_for_testing() { return cell.get(); }
    NpLayerNorm *get_layernorm_for_testing() { return layer_norm_module.get(); }
};

// ENHANCED: Temporal RNN with optional mask gradient support
class NpTemporalRNN
{
public:
    NpTemporalConfig cfg;
    std::vector<std::unique_ptr<NpRNNLayer>> rnn_layers_list;
    std::unique_ptr<NpDropout> final_dropout_module;

    // Caches for the most recent forward pass
    std::vector<std::vector<LayerFwdCache>> forward_pass_rnn_layer_caches;
    std::optional<MatrixXf> forward_pass_final_dropout_cache;

public:
    NpTemporalRNN(const NpTemporalConfig &config, std::mt19937 &gen) : cfg(config)
    {
        int current_feat_size = cfg.in_size;
        for (int i = 0; i < cfg.num_layers; ++i)
        {
            rnn_layers_list.push_back(std::make_unique<NpRNNLayer>(cfg, i, current_feat_size, gen));
            current_feat_size = cfg.hid_size;
        }

        if (cfg.final_dropout > 0)
        {
            final_dropout_module = std::make_unique<NpDropout>(cfg.final_dropout, gen);
        }
    }

    std::pair<MatrixXf, std::vector<MatrixXf>> forward(
            const std::vector<MatrixXf> &X_seq,
            const std::vector<MatrixXf> &dt_seq,
            const std::vector<std::optional<MatrixXf>> &mask_seq,
            const std::vector<MatrixXf> &initial_h_per_layer)
    {
        if (static_cast<int>(initial_h_per_layer.size()) != cfg.num_layers) {
            throw std::runtime_error("NpTemporalRNN::forward: Mismatch in initial_h_per_layer size and num_layers.");
        }
        for (const auto& h_init_l : initial_h_per_layer) {
            if (h_init_l.rows() != cfg.batch_size || h_init_l.cols() != cfg.hid_size) {
                throw std::runtime_error("NpTemporalRNN::forward: Mismatch in initial_h_per_layer dimensions.");
            }
        }

        int T_win = X_seq.size();
        if (T_win == 0) {
            return {MatrixXf(0, cfg.hid_size), initial_h_per_layer};
        }
        if (X_seq[0].rows() != cfg.batch_size) {
            throw std::runtime_error("NpTemporalRNN::forward: Mismatch in X_seq batch_size and cfg.batch_size.");
        }

        std::vector<MatrixXf> current_h_states_per_layer = initial_h_per_layer;
        std::vector<MatrixXf> outputs_final_layer_seq;
        outputs_final_layer_seq.reserve(T_win);

        forward_pass_rnn_layer_caches.assign(T_win, std::vector<LayerFwdCache>(cfg.num_layers));
        forward_pass_final_dropout_cache = std::nullopt;

        for (int t = 0; t < T_win; ++t)
        {
            MatrixXf input_to_current_layer = X_seq[t];

            for (int l_idx = 0; l_idx < cfg.num_layers; ++l_idx)
            {
                std::optional<MatrixXf> mask_for_layer_t = (l_idx == 0) ? mask_seq[t] : std::nullopt;

                auto [h_new_l_t, layer_cache_l_t] = rnn_layers_list[l_idx]->forward(
                        input_to_current_layer,
                        current_h_states_per_layer[l_idx],
                        dt_seq[t],
                        mask_for_layer_t);

                forward_pass_rnn_layer_caches[t][l_idx] = layer_cache_l_t;
                current_h_states_per_layer[l_idx] = h_new_l_t;
                input_to_current_layer = h_new_l_t;
            }
            outputs_final_layer_seq.push_back(input_to_current_layer);
        }

        MatrixXf H_out_seq_raw_stacked(T_win * cfg.batch_size, cfg.hid_size);
        if (T_win > 0) {
            for (int t = 0; t < T_win; ++t)
            {
                H_out_seq_raw_stacked.block(t * cfg.batch_size, 0, cfg.batch_size, cfg.hid_size) = outputs_final_layer_seq[t];
            }
        }

        MatrixXf H_out_seq_final_stacked = H_out_seq_raw_stacked;
        if (final_dropout_module)
        {
            auto [dropped, mask] = final_dropout_module->forward(H_out_seq_raw_stacked);
            H_out_seq_final_stacked = dropped;
            forward_pass_final_dropout_cache = mask;
        }

        return {H_out_seq_final_stacked, current_h_states_per_layer};
    }

    // ENHANCED: Backward output structure with optional mask gradients
    struct RNNBackwardOutput {
        std::vector<MatrixXf> d_X_seq_window;
        std::vector<MatrixXf> d_initial_h_window;
        std::vector<MatrixXf> d_mask_seq_window;     // NEW: Only populated if mask learning is enabled
    };

    RNNBackwardOutput backward(const MatrixXf &d_H_out_final_stacked)
    {
        int T_win = forward_pass_rnn_layer_caches.size();
        if (T_win == 0) {
            std::vector<MatrixXf> d_initial_h_window_empty(cfg.num_layers);
            for(int l=0; l < cfg.num_layers; ++l) {
                d_initial_h_window_empty[l] = MatrixXf::Zero(cfg.batch_size, cfg.hid_size);
            }
            return {{}, d_initial_h_window_empty, {}};
        }

        int num_layers = rnn_layers_list.size();

        RNNBackwardOutput result;
        result.d_X_seq_window.resize(T_win);
        result.d_initial_h_window.assign(num_layers, MatrixXf::Zero(cfg.batch_size, cfg.hid_size));

        // Only allocate mask gradients if mask learning is enabled
        if (cfg.enable_mask_learning) {
            result.d_mask_seq_window.resize(T_win);
        }

        MatrixXf d_H_out_raw_stacked = d_H_out_final_stacked;
        if (final_dropout_module && forward_pass_final_dropout_cache.has_value())
        {
            d_H_out_raw_stacked = final_dropout_module->backward(d_H_out_final_stacked,
                                                                 forward_pass_final_dropout_cache);
        }

        std::vector<MatrixXf> d_h_from_next_t_or_above(num_layers);
        for (int l = 0; l < num_layers; ++l)
        {
            d_h_from_next_t_or_above[l] = MatrixXf::Zero(cfg.batch_size, cfg.hid_size);
        }

        for (int t = T_win - 1; t >= 0; --t)
        {
            MatrixXf d_output_of_layer_L_at_t = d_H_out_raw_stacked.block(t * cfg.batch_size, 0, cfg.batch_size, cfg.hid_size);

            MatrixXf d_h_for_current_layer = d_output_of_layer_L_at_t;
            if (num_layers > 0) {
                d_h_for_current_layer += d_h_from_next_t_or_above[num_layers-1];
            }

            for (int l_idx = num_layers - 1; l_idx >= 0; --l_idx)
            {
                if (cfg.enable_mask_learning && l_idx == 0) {
                    // Use mask-aware backward for layer 0 when mask learning is enabled
                    auto [d_input_for_layer_l_at_t, d_h_prev_for_layer_l_at_t, d_dt_for_layer_l_at_t, d_mask_for_layer_l_at_t] =
                            rnn_layers_list[l_idx]->backward_with_mask_grad(d_h_for_current_layer,
                                                                            forward_pass_rnn_layer_caches[t][l_idx]);

                    d_h_from_next_t_or_above[l_idx] = d_h_prev_for_layer_l_at_t;
                    result.d_X_seq_window[t] = d_input_for_layer_l_at_t;

                    if (d_mask_for_layer_l_at_t.has_value()) {
                        result.d_mask_seq_window[t] = d_mask_for_layer_l_at_t.value();
                    } else {
                        result.d_mask_seq_window[t] = MatrixXf::Zero(cfg.batch_size, cfg.in_size);
                    }
                } else {
                    // Use standard backward for all other cases
                    auto [d_input_for_layer_l_at_t, d_h_prev_for_layer_l_at_t, d_dt_for_layer_l_at_t] =
                            rnn_layers_list[l_idx]->backward(d_h_for_current_layer,
                                                             forward_pass_rnn_layer_caches[t][l_idx]);

                    d_h_from_next_t_or_above[l_idx] = d_h_prev_for_layer_l_at_t;

                    if (l_idx > 0) {
                        d_h_for_current_layer = d_input_for_layer_l_at_t;
                        d_h_for_current_layer += d_h_from_next_t_or_above[l_idx-1];
                    } else { // l_idx == 0
                        result.d_X_seq_window[t] = d_input_for_layer_l_at_t;
                    }
                }
            }
        }

        result.d_initial_h_window = d_h_from_next_t_or_above;
        return result;
    }

    void zero_grad_rnn_params()
    {
        for (auto &layer : rnn_layers_list)
            layer->zero_grad_layer_params();
    }

    std::vector<std::pair<MatrixXf *, MatrixXf *>> get_rnn_params_grads()
    {
        std::vector<std::pair<MatrixXf *, MatrixXf *>> all_pg;
        for (auto &layer : rnn_layers_list)
        {
            auto layer_pg = layer->get_layer_params_grads();
            all_pg.insert(all_pg.end(), layer_pg.begin(), layer_pg.end());
        }
        return all_pg;
    }

    std::vector<std::pair<VectorXf *, VectorXf *>> get_rnn_bias_params_grads()
    {
        std::vector<std::pair<VectorXf *, VectorXf *>> all_pg;
        for (auto &layer : rnn_layers_list)
        {
            auto layer_pg = layer->get_layer_bias_params_grads();
            all_pg.insert(all_pg.end(), layer_pg.begin(), layer_pg.end());
        }
        return all_pg;
    }

    std::vector<std::pair<VectorXf *, VectorXf *>> get_rnn_vector_params_grads()
    {
        std::vector<std::pair<VectorXf *, VectorXf *>> all_pg;
        for (auto &layer : rnn_layers_list)
        {
            auto [p_decay, g_decay] = layer->get_layer_decay_params_grads();
            if (p_decay) all_pg.push_back({p_decay, g_decay});

            auto [p_ln_gamma, g_ln_gamma] = layer->get_layer_norm_params_grads();
            if (p_ln_gamma) all_pg.push_back({p_ln_gamma, g_ln_gamma});

            auto [p_ln_beta, g_ln_beta] = layer->get_layer_norm_bias_params_grads();
            if (p_ln_beta) all_pg.push_back({p_ln_beta, g_ln_beta});
        }
        return all_pg;
    }

    void set_training_mode(bool is_training)
    {
        for (auto &layer : rnn_layers_list)
            layer->set_training_mode(is_training);
        if (final_dropout_module)
            final_dropout_module->set_training_mode(is_training);
    }

    void clear_forward_caches()
    {
        forward_pass_rnn_layer_caches.clear();
        forward_pass_final_dropout_cache = std::nullopt;
    }

    NpRNNLayer *get_layer_for_testing(int layer_idx)
    {
        if (layer_idx >= 0 && layer_idx < static_cast<int>(rnn_layers_list.size()))
        {
            return rnn_layers_list[layer_idx].get();
        }
        return nullptr;
    }
};

// Usage example showing how to enable/disable mask learning:
/*
void example_usage() {
    std::mt19937 gen(42);

    // Configuration WITHOUT mask learning (behaves exactly as before)
    NpTemporalConfig config_no_mask_learning;
    config_no_mask_learning.enable_mask_learning = false;  // Default
    config_no_mask_learning.batch_size = 2;
    config_no_mask_learning.in_size = 4;
    config_no_mask_learning.hid_size = 8;
    config_no_mask_learning.num_layers = 1;

    // Configuration WITH mask learning (enables mask gradient computation)
    NpTemporalConfig config_with_mask_learning = config_no_mask_learning;
    config_with_mask_learning.enable_mask_learning = true;

    // Create RNNs with different configurations
    NpTemporalRNN rnn_standard(config_no_mask_learning, gen);    // No mask learning
    NpTemporalRNN rnn_mask_aware(config_with_mask_learning, gen); // With mask learning

    // Both RNNs can be used identically for forward pass
    // The difference is in the backward pass behavior
}
*/

#endif //TENSOREIGEN_GRUD_EIGEN_H
