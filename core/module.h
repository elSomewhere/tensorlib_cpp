//
// grud/core/module.h - Core module system for autograd
//

#ifndef GRUD_CORE_MODULE_H
#define GRUD_CORE_MODULE_H

#include <Eigen/Dense>
#include <vector>
#include <string>
#include <memory>
#include <random>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iostream>

namespace grud {

// Forward declarations
class Module;
struct Context;

// ============================================================================
// PARAMETER SYSTEM
// ============================================================================

/**
 * Represents a trainable parameter with value and gradient
 */
struct Param {
    Eigen::MatrixXf value;     // Parameter values
    Eigen::MatrixXf grad;      // Accumulated gradients
    std::string name;          // Human-readable name for debugging

    Param() = default;

    Param(const Eigen::MatrixXf& val, const std::string& param_name = "")
        : value(val), grad(Eigen::MatrixXf::Zero(val.rows(), val.cols())), name(param_name) {}

    Param(int rows, int cols, const std::string& param_name = "")
        : value(Eigen::MatrixXf::Zero(rows, cols)),
          grad(Eigen::MatrixXf::Zero(rows, cols)),
          name(param_name) {}

    void zero_grad() {
        grad.setZero();
    }

    // Initialize with Xavier/Glorot uniform
    void init_xavier_uniform(std::mt19937& gen) {
        float limit = std::sqrt(6.0f / (value.rows() + value.cols()));
        std::uniform_real_distribution<float> dist(-limit, limit);
        for (int i = 0; i < value.rows(); ++i) {
            for (int j = 0; j < value.cols(); ++j) {
                value(i, j) = dist(gen);
            }
        }
    }

    // Initialize with normal distribution
    void init_normal(std::mt19937& gen, float mean = 0.0f, float std = 0.01f) {
        std::normal_distribution<float> dist(mean, std);
        for (int i = 0; i < value.rows(); ++i) {
            for (int j = 0; j < value.cols(); ++j) {
                value(i, j) = dist(gen);
            }
        }
    }
};

// ============================================================================
// CONTEXT SYSTEM
// ============================================================================

/**
 * Context stores intermediate values from forward pass for use in backward pass
 */
struct Context {
    Module* op;                              // Pointer to the module that created this context
    std::vector<Eigen::MatrixXf> saved;      // Saved tensors for backward pass
    std::vector<size_t> child_indices;       // Indices of child contexts (for graph mode)

    Context(Module* owner = nullptr) : op(owner) {}

    // Save a tensor for use in backward pass
    void save_for_backward(const Eigen::MatrixXf& tensor) {
        saved.push_back(tensor);
    }

    // Get saved tensor by index
    const Eigen::MatrixXf& get_saved(size_t idx) const {
        return saved.at(idx);
    }

    // Clear all saved data
    void clear() {
        saved.clear();
        child_indices.clear();
    }
};

// ============================================================================
// MODULE BASE CLASS
// ============================================================================

/**
 * Base class for all neural network modules
 */
class Module {
public:
    virtual ~Module() = default;

    /**
     * Forward pass - must be implemented by all modules
     * @param input Input tensor
     * @param ctx Context to store intermediate values
     * @return Output tensor
     */
    virtual Eigen::MatrixXf forward(const Eigen::MatrixXf& input, Context& ctx) = 0;

    /**
     * Backward pass - must be implemented by all modules
     * @param grad_output Gradient w.r.t. output
     * @param ctx Context from forward pass
     * @return Gradient w.r.t. input
     */
    virtual Eigen::MatrixXf backward(const Eigen::MatrixXf& grad_output, const Context& ctx) = 0;

    /**
     * Get all trainable parameters in this module
     * @return Vector of parameter pointers
     */
    virtual std::vector<Param*> params() {
        return {};
    }

    /**
     * Get all child modules
     * @return Vector of child module pointers
     */
    virtual std::vector<Module*> children() {
        return {};
    }

    /**
     * Zero all gradients in this module and its children
     */
    void zero_grad() {
        // Zero gradients in this module
        for (auto* param : params()) {
            param->zero_grad();
        }

        // Recursively zero gradients in children
        for (auto* child : children()) {
            child->zero_grad();
        }
    }

    /**
     * Get total number of parameters in this module and its children
     */
    size_t num_parameters() const {
        size_t count = 0;

        // Count parameters in this module
        for (const auto* param : const_cast<Module*>(this)->params()) {
            count += param->value.size();
        }

        // Recursively count parameters in children
        for (const auto* child : const_cast<Module*>(this)->children()) {
            count += child->num_parameters();
        }

        return count;
    }

    /**
     * Collect all parameters from this module and its children
     */
    std::vector<Param*> all_parameters() {
        std::vector<Param*> all_params;
        collect_parameters_recursive(all_params);
        return all_params;
    }

    /**
     * Set training mode for this module and its children
     */
    virtual void set_training(bool training = true) {
        training_mode = training;
        for (auto* child : children()) {
            child->set_training(training);
        }
    }

    /**
     * Set evaluation mode for this module and its children
     */
    void eval() {
        set_training(false);
    }

    /**
     * Check if module is in training mode
     */
    bool is_training() const {
        return training_mode;
    }

    /**
     * Print module structure (for debugging)
     */
    virtual std::string name() const {
        return "Module";
    }

    void print_structure(int indent = 0) const {
        std::string indent_str(indent * 2, ' ');
        std::cout << indent_str << name() << " (" << num_parameters() << " params)" << std::endl;

        for (const auto* child : const_cast<Module*>(this)->children()) {
            child->print_structure(indent + 1);
        }
    }

protected:
    bool training_mode = true;

private:
    void collect_parameters_recursive(std::vector<Param*>& all_params) {
        // Add parameters from this module
        auto my_params = params();
        all_params.insert(all_params.end(), my_params.begin(), my_params.end());

        // Recursively add parameters from children
        for (auto* child : children()) {
            child->collect_parameters_recursive(all_params);
        }
    }
};

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/**
 * Math utility functions for neural networks
 */
namespace math {

    inline Eigen::MatrixXf sigmoid(const Eigen::MatrixXf& x) {
        return (1.0f / (1.0f + (-x.array().cwiseMax(-80.0f).cwiseMin(80.0f)).exp())).matrix();
    }

    inline Eigen::MatrixXf sigmoid_derivative(const Eigen::MatrixXf& x) {
        Eigen::MatrixXf s = sigmoid(x);
        return s.array() * (1.0f - s.array());
    }

    inline Eigen::MatrixXf tanh_activation(const Eigen::MatrixXf& x) {
        return x.array().tanh().matrix();
    }

    inline Eigen::MatrixXf tanh_derivative(const Eigen::MatrixXf& x) {
        Eigen::MatrixXf tanh_x = tanh_activation(x);
        return (1.0f - tanh_x.array().square()).matrix();
    }

    inline Eigen::MatrixXf relu(const Eigen::MatrixXf& x) {
        return x.array().cwiseMax(0.0f).matrix();
    }

    inline Eigen::MatrixXf relu_derivative(const Eigen::MatrixXf& x) {
        return (x.array() > 0.0f).cast<float>();
    }

    inline Eigen::MatrixXf softplus(const Eigen::MatrixXf& x) {
        Eigen::MatrixXf x_clipped = x.array().cwiseMax(-80.0f).cwiseMin(80.0f).matrix();
        return (1.0f + x_clipped.array().exp()).log().matrix();
    }

    inline Eigen::MatrixXf softplus_derivative(const Eigen::MatrixXf& x) {
        return sigmoid(x);
    }

    // Numerically stable log1p(exp(x)) for large x
    inline float stable_log1p_exp(float x) {
        if (x < 1e-5f) {
            return x;
        } else if (x > 20.0f) {
            return x;
        } else {
            return std::log(1.0f + std::exp(x));
        }
    }

    // Softclip function (from original code)
    inline Eigen::MatrixXf softclip(const Eigen::MatrixXf& x, float threshold = 3.0f) {
        if (threshold <= 0.0f) {
            throw std::invalid_argument("Threshold must be positive for softclip");
        }

        Eigen::MatrixXf result(x.rows(), x.cols());

        for (int i = 0; i < x.rows(); ++i) {
            for (int j = 0; j < x.cols(); ++j) {
                float xi = x(i, j);
                float abs_xi = std::abs(xi);

                if (abs_xi <= threshold) {
                    result(i, j) = xi;
                } else {
                    float y = abs_xi - threshold;
                    float log1p_exp_y = stable_log1p_exp(y);
                    float sign_xi = (xi >= 0.0f) ? 1.0f : -1.0f;
                    result(i, j) = threshold + log1p_exp_y * sign_xi;
                }
            }
        }

        return result;
    }

    inline Eigen::MatrixXf softclip_derivative(const Eigen::MatrixXf& x, float threshold = 3.0f) {
        (void)threshold;  // Suppress unused parameter warning
        return Eigen::MatrixXf::Ones(x.rows(), x.cols());
    }

} // namespace math

} // namespace grud

#endif // GRUD_CORE_MODULE_H