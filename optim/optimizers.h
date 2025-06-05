//
// grud/optim/optimizers.h - Optimizers for training neural networks
//

#ifndef GRUD_OPTIM_OPTIMIZERS_H
#define GRUD_OPTIM_OPTIMIZERS_H

#include "../core/module.h"
#include <vector>
#include <memory>
#include <cmath>
#include <algorithm>

namespace grud {
namespace optim {

// ============================================================================
// BASE OPTIMIZER
// ============================================================================

/**
 * Base class for all optimizers
 */
class Optimizer {
public:
    virtual ~Optimizer() = default;

    /**
     * Perform one optimization step
     * @param parameters List of parameters to optimize
     */
    virtual void step(const std::vector<Param*>& parameters) = 0;

    /**
     * Zero gradients for all parameters
     * @param parameters List of parameters
     */
    virtual void zero_grad(const std::vector<Param*>& parameters) {
        for (auto* param : parameters) {
            param->zero_grad();
        }
    }

    /**
     * Get current learning rate
     */
    virtual float get_learning_rate() const = 0;

    /**
     * Set learning rate
     */
    virtual void set_learning_rate(float lr) = 0;

    /**
     * Get current step count
     */
    virtual int get_step_count() const { return 0; }

    /**
     * Reset optimizer state
     */
    virtual void reset() {}
};

// ============================================================================
// SGD OPTIMIZER
// ============================================================================

/**
 * Stochastic Gradient Descent optimizer with momentum
 */
class SGD : public Optimizer {
private:
    float lr_;
    float momentum_;
    float weight_decay_;
    bool nesterov_;

    // Momentum buffers
    std::vector<Eigen::MatrixXf> momentum_buffers_;
    bool buffers_initialized_;

public:
    /**
     * Constructor
     * @param lr Learning rate
     * @param momentum Momentum factor (0 = no momentum)
     * @param weight_decay Weight decay (L2 regularization)
     * @param nesterov Use Nesterov momentum
     */
    SGD(float lr = 1e-3f, float momentum = 0.0f, float weight_decay = 0.0f, bool nesterov = false)
        : lr_(lr), momentum_(momentum), weight_decay_(weight_decay), nesterov_(nesterov),
          buffers_initialized_(false) {

        if (lr_ <= 0.0f) {
            throw std::invalid_argument("Learning rate must be positive");
        }
        if (momentum_ < 0.0f || momentum_ >= 1.0f) {
            throw std::invalid_argument("Momentum must be in [0, 1)");
        }
        if (weight_decay_ < 0.0f) {
            throw std::invalid_argument("Weight decay must be non-negative");
        }
    }

    void step(const std::vector<Param*>& parameters) override {
        if (parameters.empty()) {
            return;
        }

        // Initialize momentum buffers if needed
        if (!buffers_initialized_) {
            momentum_buffers_.clear();
            momentum_buffers_.reserve(parameters.size());

            for (const auto* param : parameters) {
                momentum_buffers_.emplace_back(Eigen::MatrixXf::Zero(param->value.rows(), param->value.cols()));
            }
            buffers_initialized_ = true;
        }

        // Update parameters
        for (size_t i = 0; i < parameters.size(); ++i) {
            Param* param = parameters[i];
            Eigen::MatrixXf& momentum_buf = momentum_buffers_[i];

            // Compute gradient with weight decay
            Eigen::MatrixXf grad = param->grad;
            if (weight_decay_ > 0.0f) {
                grad += weight_decay_ * param->value;
            }

            if (momentum_ > 0.0f) {
                // Update momentum buffer
                momentum_buf = momentum_ * momentum_buf + grad;

                if (nesterov_) {
                    // Nesterov momentum: param -= lr * (momentum * momentum_buf + grad)
                    grad = momentum_ * momentum_buf + grad;
                } else {
                    // Standard momentum: use momentum buffer
                    grad = momentum_buf;
                }
            }

            // Update parameters
            param->value -= lr_ * grad;
        }
    }

    float get_learning_rate() const override {
        return lr_;
    }

    void set_learning_rate(float lr) override {
        if (lr <= 0.0f) {
            throw std::invalid_argument("Learning rate must be positive");
        }
        lr_ = lr;
    }

    void reset() override {
        momentum_buffers_.clear();
        buffers_initialized_ = false;
    }

    float get_momentum() const { return momentum_; }
    float get_weight_decay() const { return weight_decay_; }
    bool is_nesterov() const { return nesterov_; }
};

// ============================================================================
// ADAMW OPTIMIZER
// ============================================================================

/**
 * AdamW optimizer (Adam with decoupled weight decay)
 */
class AdamW : public Optimizer {
private:
    float lr_;
    float beta1_;
    float beta2_;
    float eps_;
    float weight_decay_;
    bool amsgrad_;

    // Adam state
    std::vector<Eigen::MatrixXf> m_buffers_;  // First moment
    std::vector<Eigen::MatrixXf> v_buffers_;  // Second moment
    std::vector<Eigen::MatrixXf> max_v_buffers_;  // Max second moment (for AMSGrad)
    int step_count_;
    bool buffers_initialized_;

public:
    /**
     * Constructor
     * @param lr Learning rate
     * @param beta1 Coefficient for computing running averages of gradient
     * @param beta2 Coefficient for computing running averages of squared gradient
     * @param eps Term added to denominator for numerical stability
     * @param weight_decay Weight decay coefficient
     * @param amsgrad Use AMSGrad variant
     */
    AdamW(float lr = 1e-3f, float beta1 = 0.9f, float beta2 = 0.999f,
          float eps = 1e-8f, float weight_decay = 1e-2f, bool amsgrad = false)
        : lr_(lr), beta1_(beta1), beta2_(beta2), eps_(eps), weight_decay_(weight_decay),
          amsgrad_(amsgrad), step_count_(0), buffers_initialized_(false) {

        if (lr_ <= 0.0f) {
            throw std::invalid_argument("Learning rate must be positive");
        }
        if (beta1_ < 0.0f || beta1_ >= 1.0f) {
            throw std::invalid_argument("Beta1 must be in [0, 1)");
        }
        if (beta2_ < 0.0f || beta2_ >= 1.0f) {
            throw std::invalid_argument("Beta2 must be in [0, 1)");
        }
        if (eps_ <= 0.0f) {
            throw std::invalid_argument("Epsilon must be positive");
        }
        if (weight_decay_ < 0.0f) {
            throw std::invalid_argument("Weight decay must be non-negative");
        }
    }

    void step(const std::vector<Param*>& parameters) override {
        if (parameters.empty()) {
            return;
        }

        step_count_++;

        // Initialize buffers if needed
        if (!buffers_initialized_) {
            m_buffers_.clear();
            v_buffers_.clear();
            if (amsgrad_) {
                max_v_buffers_.clear();
            }

            m_buffers_.reserve(parameters.size());
            v_buffers_.reserve(parameters.size());
            if (amsgrad_) {
                max_v_buffers_.reserve(parameters.size());
            }

            for (const auto* param : parameters) {
                int rows = param->value.rows();
                int cols = param->value.cols();
                m_buffers_.emplace_back(Eigen::MatrixXf::Zero(rows, cols));
                v_buffers_.emplace_back(Eigen::MatrixXf::Zero(rows, cols));
                if (amsgrad_) {
                    max_v_buffers_.emplace_back(Eigen::MatrixXf::Zero(rows, cols));
                }
            }
            buffers_initialized_ = true;
        }

        // Bias correction terms
        float bias_correction1 = 1.0f - std::pow(beta1_, step_count_);
        float bias_correction2 = 1.0f - std::pow(beta2_, step_count_);
        float corrected_lr = lr_ * std::sqrt(bias_correction2) / bias_correction1;

        // Update parameters
        for (size_t i = 0; i < parameters.size(); ++i) {
            Param* param = parameters[i];
            Eigen::MatrixXf& m = m_buffers_[i];
            Eigen::MatrixXf& v = v_buffers_[i];

            const Eigen::MatrixXf& grad = param->grad;

            // Update biased first moment estimate
            m = beta1_ * m + (1.0f - beta1_) * grad;

            // Update biased second moment estimate
            v = beta2_ * v + (1.0f - beta2_) * grad.array().square().matrix();

            // Compute update direction
            Eigen::MatrixXf denom;
            if (amsgrad_) {
                Eigen::MatrixXf& max_v = max_v_buffers_[i];
                max_v = v.cwiseMax(max_v);
                denom = (max_v.array().sqrt() + eps_).matrix();
            } else {
                denom = (v.array().sqrt() + eps_).matrix();
            }

            Eigen::MatrixXf update = corrected_lr * m.array() / denom.array();

            // Apply weight decay (decoupled from gradient)
            if (weight_decay_ > 0.0f) {
                param->value -= lr_ * weight_decay_ * param->value;
            }

            // Apply gradient update
            param->value -= update;
        }
    }

    float get_learning_rate() const override {
        return lr_;
    }

    void set_learning_rate(float lr) override {
        if (lr <= 0.0f) {
            throw std::invalid_argument("Learning rate must be positive");
        }
        lr_ = lr;
    }

    int get_step_count() const override {
        return step_count_;
    }

    void reset() override {
        m_buffers_.clear();
        v_buffers_.clear();
        max_v_buffers_.clear();
        step_count_ = 0;
        buffers_initialized_ = false;
    }

    float get_beta1() const { return beta1_; }
    float get_beta2() const { return beta2_; }
    float get_eps() const { return eps_; }
    float get_weight_decay() const { return weight_decay_; }
    bool is_amsgrad() const { return amsgrad_; }
};

// ============================================================================
// LEARNING RATE SCHEDULERS
// ============================================================================

/**
 * Base class for learning rate schedulers
 */
class LRScheduler {
public:
    virtual ~LRScheduler() = default;

    /**
     * Update learning rate based on scheduler logic
     * @param optimizer Optimizer to update
     * @param epoch Current epoch (optional)
     * @param metric Current metric value (optional)
     */
    virtual void step(Optimizer& optimizer, int epoch = -1, float metric = 0.0f) = 0;

    /**
     * Get current learning rate
     */
    virtual float get_lr() const = 0;
};

/**
 * Step learning rate scheduler
 */
class StepLR : public LRScheduler {
private:
    float initial_lr_;
    float gamma_;
    int step_size_;
    int last_epoch_;

public:
    StepLR(float initial_lr, int step_size, float gamma = 0.1f)
        : initial_lr_(initial_lr), gamma_(gamma), step_size_(step_size), last_epoch_(0) {

        if (step_size <= 0) {
            throw std::invalid_argument("Step size must be positive");
        }
        if (gamma <= 0.0f || gamma > 1.0f) {
            throw std::invalid_argument("Gamma must be in (0, 1]");
        }
    }

    void step(Optimizer& optimizer, int epoch = -1, float metric = 0.0f) override {
        if (epoch >= 0) {
            last_epoch_ = epoch;
        } else {
            last_epoch_++;
        }

        float new_lr = initial_lr_ * std::pow(gamma_, last_epoch_ / step_size_);
        optimizer.set_learning_rate(new_lr);
    }

    float get_lr() const override {
        return initial_lr_ * std::pow(gamma_, last_epoch_ / step_size_);
    }
};

/**
 * Exponential learning rate scheduler
 */
class ExponentialLR : public LRScheduler {
private:
    float initial_lr_;
    float gamma_;
    int last_epoch_;

public:
    ExponentialLR(float initial_lr, float gamma)
        : initial_lr_(initial_lr), gamma_(gamma), last_epoch_(0) {

        if (gamma <= 0.0f || gamma > 1.0f) {
            throw std::invalid_argument("Gamma must be in (0, 1]");
        }
    }

    void step(Optimizer& optimizer, int epoch = -1, float metric = 0.0f) override {
        if (epoch >= 0) {
            last_epoch_ = epoch;
        } else {
            last_epoch_++;
        }

        float new_lr = initial_lr_ * std::pow(gamma_, last_epoch_);
        optimizer.set_learning_rate(new_lr);
    }

    float get_lr() const override {
        return initial_lr_ * std::pow(gamma_, last_epoch_);
    }
};

/**
 * Cosine annealing learning rate scheduler
 */
class CosineAnnealingLR : public LRScheduler {
private:
    float initial_lr_;
    float min_lr_;
    int T_max_;
    int last_epoch_;

public:
    CosineAnnealingLR(float initial_lr, int T_max, float min_lr = 0.0f)
        : initial_lr_(initial_lr), min_lr_(min_lr), T_max_(T_max), last_epoch_(0) {

        if (T_max <= 0) {
            throw std::invalid_argument("T_max must be positive");
        }
        if (min_lr < 0.0f || min_lr > initial_lr) {
            throw std::invalid_argument("min_lr must be in [0, initial_lr]");
        }
    }

    void step(Optimizer& optimizer, int epoch = -1, float metric = 0.0f) override {
        if (epoch >= 0) {
            last_epoch_ = epoch;
        } else {
            last_epoch_++;
        }

        float cosine_term = std::cos(M_PI * (last_epoch_ % T_max_) / T_max_);
        float new_lr = min_lr_ + (initial_lr_ - min_lr_) * (1.0f + cosine_term) / 2.0f;
        optimizer.set_learning_rate(new_lr);
    }

    float get_lr() const override {
        float cosine_term = std::cos(M_PI * (last_epoch_ % T_max_) / T_max_);
        return min_lr_ + (initial_lr_ - min_lr_) * (1.0f + cosine_term) / 2.0f;
    }
};

} // namespace optim

// ============================================================================
// LOSS FUNCTIONS
// ============================================================================

namespace loss {

/**
 * Base class for loss functions
 */
class Loss {
public:
    virtual ~Loss() = default;

    /**
     * Compute loss value
     * @param predictions Model predictions
     * @param targets Ground truth targets
     * @param mask Optional mask for valid elements
     * @return Loss value
     */
    virtual float forward(const Eigen::MatrixXf& predictions,
                         const Eigen::MatrixXf& targets,
                         const std::optional<Eigen::MatrixXf>& mask = std::nullopt) = 0;

    /**
     * Compute gradient of loss w.r.t. predictions
     * @param predictions Model predictions
     * @param targets Ground truth targets
     * @param mask Optional mask for valid elements
     * @return Gradient w.r.t. predictions
     */
    virtual Eigen::MatrixXf backward(const Eigen::MatrixXf& predictions,
                                    const Eigen::MatrixXf& targets,
                                    const std::optional<Eigen::MatrixXf>& mask = std::nullopt) = 0;

    /**
     * Compute both loss and gradient in one call
     */
    virtual std::pair<float, Eigen::MatrixXf> forward_backward(
        const Eigen::MatrixXf& predictions,
        const Eigen::MatrixXf& targets,
        const std::optional<Eigen::MatrixXf>& mask = std::nullopt) {

        float loss_val = forward(predictions, targets, mask);
        Eigen::MatrixXf grad = backward(predictions, targets, mask);
        return {loss_val, grad};
    }

protected:
    /**
     * Apply mask to computation and return normalization factor
     */
    std::pair<Eigen::MatrixXf, float> apply_mask(
        const Eigen::MatrixXf& values,
        const std::optional<Eigen::MatrixXf>& mask) {

        if (!mask.has_value()) {
            return {values, static_cast<float>(values.size())};
        }

        Eigen::MatrixXf masked_values = values.array() * mask.value().array();
        float normalizer = std::max(1.0f, mask.value().sum());
        return {masked_values, normalizer};
    }
};

/**
 * Mean Squared Error loss
 */
class MSELoss : public Loss {
public:
    float forward(const Eigen::MatrixXf& predictions,
                 const Eigen::MatrixXf& targets,
                 const std::optional<Eigen::MatrixXf>& mask = std::nullopt) override {

        Eigen::MatrixXf diff = predictions - targets;
        Eigen::MatrixXf squared_diff = diff.array().square();

        auto [masked_loss, normalizer] = apply_mask(squared_diff, mask);
        return masked_loss.sum() / normalizer;
    }

    Eigen::MatrixXf backward(const Eigen::MatrixXf& predictions,
                            const Eigen::MatrixXf& targets,
                            const std::optional<Eigen::MatrixXf>& mask = std::nullopt) override {

        Eigen::MatrixXf grad = 2.0f * (predictions - targets);

        if (mask.has_value()) {
            grad.array() *= mask.value().array();
            float normalizer = std::max(1.0f, mask.value().sum());
            grad /= normalizer;
        } else {
            grad /= static_cast<float>(grad.size());
        }

        return grad;
    }
};

/**
 * Mean Absolute Error loss
 */
class MAELoss : public Loss {
public:
    float forward(const Eigen::MatrixXf& predictions,
                 const Eigen::MatrixXf& targets,
                 const std::optional<Eigen::MatrixXf>& mask = std::nullopt) override {

        Eigen::MatrixXf abs_diff = (predictions - targets).array().abs();

        auto [masked_loss, normalizer] = apply_mask(abs_diff, mask);
        return masked_loss.sum() / normalizer;
    }

    Eigen::MatrixXf backward(const Eigen::MatrixXf& predictions,
                            const Eigen::MatrixXf& targets,
                            const std::optional<Eigen::MatrixXf>& mask = std::nullopt) override {

        Eigen::MatrixXf diff = predictions - targets;
        Eigen::MatrixXf grad = diff.array().sign();  // Sign function

        if (mask.has_value()) {
            grad.array() *= mask.value().array();
            float normalizer = std::max(1.0f, mask.value().sum());
            grad /= normalizer;
        } else {
            grad /= static_cast<float>(grad.size());
        }

        return grad;
    }
};

/**
 * Huber loss (smooth L1 loss)
 */
class HuberLoss : public Loss {
private:
    float delta_;

public:
    HuberLoss(float delta = 1.0f) : delta_(delta) {
        if (delta <= 0.0f) {
            throw std::invalid_argument("Delta must be positive");
        }
    }

    float forward(const Eigen::MatrixXf& predictions,
                 const Eigen::MatrixXf& targets,
                 const std::optional<Eigen::MatrixXf>& mask = std::nullopt) override {

        Eigen::MatrixXf abs_diff = (predictions - targets).array().abs();

        // Huber loss: 0.5 * x^2 if |x| <= delta, delta * (|x| - 0.5 * delta) otherwise
        Eigen::MatrixXf loss = Eigen::MatrixXf::Zero(abs_diff.rows(), abs_diff.cols());

        for (int i = 0; i < abs_diff.rows(); ++i) {
            for (int j = 0; j < abs_diff.cols(); ++j) {
                float abs_val = abs_diff(i, j);
                if (abs_val <= delta_) {
                    loss(i, j) = 0.5f * abs_val * abs_val;
                } else {
                    loss(i, j) = delta_ * (abs_val - 0.5f * delta_);
                }
            }
        }

        auto [masked_loss, normalizer] = apply_mask(loss, mask);
        return masked_loss.sum() / normalizer;
    }

    Eigen::MatrixXf backward(const Eigen::MatrixXf& predictions,
                            const Eigen::MatrixXf& targets,
                            const std::optional<Eigen::MatrixXf>& mask = std::nullopt) override {

        Eigen::MatrixXf diff = predictions - targets;
        Eigen::MatrixXf grad = Eigen::MatrixXf::Zero(diff.rows(), diff.cols());

        for (int i = 0; i < diff.rows(); ++i) {
            for (int j = 0; j < diff.cols(); ++j) {
                float val = diff(i, j);
                float abs_val = std::abs(val);

                if (abs_val <= delta_) {
                    grad(i, j) = val;  // x if |x| <= delta
                } else {
                    grad(i, j) = delta_ * (val >= 0.0f ? 1.0f : -1.0f);  // delta * sign(x)
                }
            }
        }

        if (mask.has_value()) {
            grad.array() *= mask.value().array();
            float normalizer = std::max(1.0f, mask.value().sum());
            grad /= normalizer;
        } else {
            grad /= static_cast<float>(grad.size());
        }

        return grad;
    }

    float get_delta() const { return delta_; }
    void set_delta(float delta) {
        if (delta <= 0.0f) {
            throw std::invalid_argument("Delta must be positive");
        }
        delta_ = delta;
    }
};

/**
 * Temporal loss with time-dependent weighting
 */
class TemporalLoss : public Loss {
private:
    std::unique_ptr<Loss> base_loss_;
    float ramp_start_;
    float ramp_end_;

public:
    TemporalLoss(std::unique_ptr<Loss> base_loss, float ramp_start = 1.0f, float ramp_end = 1.0f)
        : base_loss_(std::move(base_loss)), ramp_start_(ramp_start), ramp_end_(ramp_end) {}

    float forward(const Eigen::MatrixXf& predictions,
                 const Eigen::MatrixXf& targets,
                 const std::optional<Eigen::MatrixXf>& mask = std::nullopt) override {

        // Apply temporal weighting
        auto [weighted_pred, weighted_targ, temporal_mask] = apply_temporal_weighting(predictions, targets, mask);

        return base_loss_->forward(weighted_pred, weighted_targ, temporal_mask);
    }

    Eigen::MatrixXf backward(const Eigen::MatrixXf& predictions,
                            const Eigen::MatrixXf& targets,
                            const std::optional<Eigen::MatrixXf>& mask = std::nullopt) override {

        auto [weighted_pred, weighted_targ, temporal_mask] = apply_temporal_weighting(predictions, targets, mask);

        return base_loss_->backward(weighted_pred, weighted_targ, temporal_mask);
    }

private:
    std::tuple<Eigen::MatrixXf, Eigen::MatrixXf, std::optional<Eigen::MatrixXf>>
    apply_temporal_weighting(const Eigen::MatrixXf& predictions,
                            const Eigen::MatrixXf& targets,
                            const std::optional<Eigen::MatrixXf>& mask) {

        // Assume predictions/targets are stacked as (seq_len * batch_size, features)
        // We need to determine seq_len from the context or make it a parameter
        // For now, assume equal weighting (this would need to be extended)

        if (ramp_start_ == 1.0f && ramp_end_ == 1.0f) {
            // No temporal weighting
            return {predictions, targets, mask};
        }

        // This is a simplified version - in practice, you'd need seq_len information
        return {predictions, targets, mask};
    }
};

} // namespace loss
} // namespace grud

#endif // GRUD_OPTIM_OPTIMIZERS_H