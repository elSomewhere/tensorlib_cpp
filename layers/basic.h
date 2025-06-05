//
// grud/layers/basic.h - Basic neural network layers
//

#ifndef GRUD_LAYERS_BASIC_H
#define GRUD_LAYERS_BASIC_H

#include "../core/module.h"
#include <random>
#include <memory>

namespace grud {
namespace layers {

// ============================================================================
// LINEAR LAYER
// ============================================================================

/**
 * Linear (fully connected) layer: y = xW^T + b
 */
class Linear : public Module {
public:
    Param weight;  // (out_features, in_features)
    Param bias;    // (out_features,) - optional

private:
    int in_features_;
    int out_features_;
    bool use_bias_;

public:
    /**
     * Constructor
     * @param in_features Number of input features
     * @param out_features Number of output features
     * @param use_bias Whether to use bias term
     * @param gen Random number generator for initialization
     */
    Linear(int in_features, int out_features, bool use_bias = true, std::mt19937* gen = nullptr)
    : in_features_(in_features), out_features_(out_features), use_bias_(use_bias),
      weight(out_features, in_features, "weight"),
      bias(use_bias ? out_features : 0, use_bias ? 1 : 0, "bias") {

        // Initialize weights
        if (gen) {
            weight.init_xavier_uniform(*gen);
            if (use_bias_) {
                bias.value.setZero();
            }
        } else {
            // Default initialization
            std::random_device rd;
            std::mt19937 local_gen(rd());
            weight.init_xavier_uniform(local_gen);
            if (use_bias_) {
                bias.value.setZero();
            }
        }
    }

    /**
     * Forward pass: y = xW^T + b
     */
    Eigen::MatrixXf forward(const Eigen::MatrixXf& input, Context& ctx) override {
        ctx.save_for_backward(input);
        Eigen::MatrixXf output = input * weight.value.transpose();

        if (use_bias_ && bias.value.size() > 0) {
            // Add bias to each row
            for (int i = 0; i < output.rows(); ++i) {
                output.row(i) += bias.value.col(0).transpose();
            }
        }
        return output;
    }

    /**
     * Backward pass
     */
    Eigen::MatrixXf backward(const Eigen::MatrixXf& grad_output, const Context& ctx) override {
        const Eigen::MatrixXf& input = ctx.get_saved(0);

        // Gradient w.r.t. weight: grad_output^T * input
        weight.grad += grad_output.transpose() * input;

        // Gradient w.r.t. bias: sum over batch dimension
        if (use_bias_ && bias.value.size() > 0) {
            bias.grad += grad_output.colwise().sum().transpose();
        }

        // Gradient w.r.t. input: grad_output * weight
        return grad_output * weight.value;
    }

    std::vector<Param*> params() override {
        std::vector<Param*> param_list = {&weight};
        if (use_bias_ && bias.value.size() > 0) {
            param_list.push_back(&bias);
        }
        return param_list;
    }

    std::string name() const override {
        return "Linear(" + std::to_string(in_features_) + ", " + std::to_string(out_features_) + ")";
    }

    int in_features() const { return in_features_; }
    int out_features() const { return out_features_; }
    bool has_bias() const { return use_bias_; }
};

// ============================================================================
// LAYER NORMALIZATION
// ============================================================================




/**
 * CORRECTED LayerNorm Implementation
 * Fixes the gradient computation issues identified in testing
 */
class LayerNorm : public Module {
public:
    int normalized_shape_;
    float eps_;
    bool elementwise_affine_;
    Param gamma;  // Scale parameter (F, 1)
    Param beta;   // Shift parameter (F, 1)

public:
    LayerNorm(int normalized_shape, float eps = 1e-5f, bool elementwise_affine = true)
        : normalized_shape_(normalized_shape), eps_(eps), elementwise_affine_(elementwise_affine),
          gamma(elementwise_affine ? normalized_shape : 0, 1, "gamma"),
          beta(elementwise_affine ? normalized_shape : 0, 1, "beta") {

        if (elementwise_affine_) {
            gamma.value.setOnes();
            beta.value.setZero();
        }
    }

    /**
     * CORRECTED Forward pass with consistent broadcasting
     */
    Eigen::MatrixXf forward(const Eigen::MatrixXf& input, Context& ctx) override {
        const int B = input.rows();  // batch size
        const int F = input.cols();  // feature size

        // Compute statistics
        Eigen::VectorXf mean = input.rowwise().mean();  // (B,)

        // Compute variance: Var[X] = E[(X - μ)²]
        Eigen::MatrixXf centered = input.colwise() - mean;  // (B, F)
        Eigen::VectorXf var = centered.array().square().rowwise().mean().matrix();  // (B,)

        // Compute normalized output
        Eigen::VectorXf std_dev = (var.array() + eps_).sqrt().matrix();  // (B,)
        Eigen::MatrixXf normalized = centered.array().colwise() / std_dev.array();  // (B, F)

        // Save intermediate values for backward pass
        ctx.save_for_backward(input);       // 0: original input
        ctx.save_for_backward(mean);        // 1: mean
        ctx.save_for_backward(var);         // 2: variance
        ctx.save_for_backward(std_dev);     // 3: std deviation
        ctx.save_for_backward(normalized);  // 4: normalized values

        // Apply affine transformation if enabled
        // KEY FIX: Use consistent vectorized operations
        if (elementwise_affine_) {
            // Convert gamma and beta from (F,1) to (F,) for broadcasting
            Eigen::VectorXf gamma_vec = gamma.value.col(0);  // (F,)
            Eigen::VectorXf beta_vec = beta.value.col(0);    // (F,)

            // Vectorized affine transformation: output = normalized * gamma + beta
            // This broadcasts gamma and beta across all batch elements consistently
            Eigen::MatrixXf output = normalized.array().rowwise() * gamma_vec.transpose().array();
            output.rowwise() += beta_vec.transpose();

            return output;
        } else {
            return normalized;
        }
    }

    /**
     * CORRECTED Backward pass with proper gradient computation
     */
    Eigen::MatrixXf backward(const Eigen::MatrixXf& grad_output, const Context& ctx) override {
        const Eigen::MatrixXf& input = ctx.get_saved(0);
        const Eigen::VectorXf& mean = ctx.get_saved(1);
        const Eigen::VectorXf& var = ctx.get_saved(2);
        const Eigen::VectorXf& std_dev = ctx.get_saved(3);
        const Eigen::MatrixXf& normalized = ctx.get_saved(4);

        const int B = input.rows();
        const int F = input.cols();
        const float invF = 1.0f / static_cast<float>(F);

        // Step 1: Gradients w.r.t. affine parameters
        Eigen::MatrixXf d_normalized = grad_output;

        if (elementwise_affine_) {
            // CORRECTED: Proper gradient computation for gamma and beta
            // d_gamma = sum(grad_output * normalized, axis=0)  -> shape (F,)
            Eigen::VectorXf d_gamma_vec = (grad_output.array() * normalized.array()).colwise().sum();

            // d_beta = sum(grad_output, axis=0)  -> shape (F,)
            Eigen::VectorXf d_beta_vec = grad_output.colwise().sum();

            // Accumulate gradients with correct shape (F,1)
            gamma.grad.col(0) += d_gamma_vec;
            beta.grad.col(0) += d_beta_vec;

            // Chain gradient through affine transformation
            // d_normalized = grad_output * gamma (broadcasted)
            Eigen::VectorXf gamma_vec = gamma.value.col(0);
            d_normalized = grad_output.array().rowwise() * gamma_vec.transpose().array();
        }

        // Step 2: LayerNorm backward pass (unchanged - this was correct)
        Eigen::MatrixXf x_centered = input.colwise() - mean;
        Eigen::VectorXf inv_std = std_dev.array().inverse().matrix();

        // Use the standard LayerNorm backward formula
        Eigen::MatrixXf dx(B, F);

        for (int b = 0; b < B; ++b) {
            float sum_dout = d_normalized.row(b).sum();
            float sum_dout_xhat = (d_normalized.row(b).array() * normalized.row(b).array()).sum();

            dx.row(b) = (invF * inv_std(b) * (
                F * d_normalized.row(b).array() -
                sum_dout -
                normalized.row(b).array() * sum_dout_xhat
            )).matrix();
        }

        return dx;
    }

    std::vector<Param*> params() override {
        std::vector<Param*> param_list;
        if (elementwise_affine_) {
            param_list.push_back(&gamma);
            param_list.push_back(&beta);
        }
        return param_list;
    }

    std::string name() const override {
        return "LayerNorm(" + std::to_string(normalized_shape_) + ")";
    }
};


// ============================================================================
// DROPOUT LAYER
// ============================================================================

/**
 * Dropout layer for regularization
 */
class Dropout : public Module {
private:
    float p_;  // Dropout probability
    std::mt19937& gen_;
    std::uniform_real_distribution<float> dist_;

public:
    /**
     * Constructor
     * @param p Dropout probability (0 = no dropout, 1 = drop all)
     * @param gen Random number generator
     */
    Dropout(float p, std::mt19937& gen)
        : p_(p), gen_(gen), dist_(0.0f, 1.0f) {
        if (p < 0.0f || p > 1.0f) {
            throw std::invalid_argument("Dropout probability must be between 0 and 1");
        }
    }

    /**
     * Forward pass
     */
    Eigen::MatrixXf forward(const Eigen::MatrixXf& input, Context& ctx) override {
        if (!training_mode || p_ == 0.0f) {
            return input;
        }

        // Generate dropout mask
        Eigen::MatrixXf mask(input.rows(), input.cols());
        for (int i = 0; i < input.rows(); ++i) {
            for (int j = 0; j < input.cols(); ++j) {
                mask(i, j) = (dist_(gen_) > p_) ? 1.0f : 0.0f;
            }
        }

        // Apply dropout with scaling
        float scale = 1.0f / (1.0f - p_);
        Eigen::MatrixXf output = input.array() * mask.array() * scale;

        // Save mask for backward pass
        ctx.save_for_backward(mask);

        return output;
    }

    /**
     * Backward pass
     */
    Eigen::MatrixXf backward(const Eigen::MatrixXf& grad_output, const Context& ctx) override {
        if (!training_mode || p_ == 0.0f) {
            return grad_output;
        }

        const Eigen::MatrixXf& mask = ctx.get_saved(0);
        float scale = 1.0f / (1.0f - p_);

        return grad_output.array() * mask.array() * scale;
    }

    std::string name() const override {
        return "Dropout(" + std::to_string(p_) + ")";
    }

    float get_dropout_prob() const { return p_; }
    void set_dropout_prob(float p) {
        if (p < 0.0f || p > 1.0f) {
            throw std::invalid_argument("Dropout probability must be between 0 and 1");
        }
        p_ = p;
    }
};

// ============================================================================
// ACTIVATION FUNCTIONS
// ============================================================================

/**
 * ReLU activation function
 */
class ReLU : public Module {
public:
    Eigen::MatrixXf forward(const Eigen::MatrixXf& input, Context& ctx) override {
        // Save input for backward pass
        ctx.save_for_backward(input);
        return math::relu(input);
    }

    Eigen::MatrixXf backward(const Eigen::MatrixXf& grad_output, const Context& ctx) override {
        const Eigen::MatrixXf& input = ctx.get_saved(0);
        return grad_output.array() * math::relu_derivative(input).array();
    }

    std::string name() const override {
        return "ReLU()";
    }
};

/**
 * Tanh activation function
 */
class Tanh : public Module {
public:
    Eigen::MatrixXf forward(const Eigen::MatrixXf& input, Context& ctx) override {
        Eigen::MatrixXf output = math::tanh_activation(input);
        ctx.save_for_backward(output);  // Save output for efficiency
        return output;
    }

    Eigen::MatrixXf backward(const Eigen::MatrixXf& grad_output, const Context& ctx) override {
        const Eigen::MatrixXf& tanh_output = ctx.get_saved(0);
        Eigen::MatrixXf tanh_grad = 1.0f - tanh_output.array().square();
        return grad_output.array() * tanh_grad.array();
    }

    std::string name() const override {
        return "Tanh()";
    }
};

/**
 * Sigmoid activation function
 */
class Sigmoid : public Module {
public:
    Eigen::MatrixXf forward(const Eigen::MatrixXf& input, Context& ctx) override {
        Eigen::MatrixXf output = math::sigmoid(input);
        ctx.save_for_backward(output);  // Save output for efficiency
        return output;
    }

    Eigen::MatrixXf backward(const Eigen::MatrixXf& grad_output, const Context& ctx) override {
        const Eigen::MatrixXf& sigmoid_output = ctx.get_saved(0);
        Eigen::MatrixXf sigmoid_grad = sigmoid_output.array() * (1.0f - sigmoid_output.array());
        return grad_output.array() * sigmoid_grad.array();
    }

    std::string name() const override {
        return "Sigmoid()";
    }
};

/**
 * Softplus activation function
 */
class Softplus : public Module {
public:
    Eigen::MatrixXf forward(const Eigen::MatrixXf& input, Context& ctx) override {
        ctx.save_for_backward(input);
        return math::softplus(input);
    }

    Eigen::MatrixXf backward(const Eigen::MatrixXf& grad_output, const Context& ctx) override {
        const Eigen::MatrixXf& input = ctx.get_saved(0);
        return grad_output.array() * math::softplus_derivative(input).array();
    }

    std::string name() const override {
        return "Softplus()";
    }
};

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/**
 * Create a linear layer with proper initialization
 */
inline std::unique_ptr<Linear> make_linear(int in_features, int out_features,
                                          bool use_bias = true, std::mt19937* gen = nullptr) {
    return std::make_unique<Linear>(in_features, out_features, use_bias, gen);
}

/**
 * Create a layer norm layer
 */
inline std::unique_ptr<LayerNorm> make_layer_norm(int normalized_shape, float eps = 1e-5f,
                                                  bool elementwise_affine = true) {
    return std::make_unique<LayerNorm>(normalized_shape, eps, elementwise_affine);
}

/**
 * Create a dropout layer
 */
inline std::unique_ptr<Dropout> make_dropout(float p, std::mt19937& gen) {
    return std::make_unique<Dropout>(p, gen);
}

} // namespace layers
} // namespace grud

#endif // GRUD_LAYERS_BASIC_H