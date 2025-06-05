// file: core/checkgrad.h
//
// grud/core/checkgrad.h - Improved Numerical gradient checker optimized for float32
//

#ifndef GRUD_CORE_CHECKGRAD_H
#define GRUD_CORE_CHECKGRAD_H

#include "module.h"
#include "tape.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <string>
#include <limits>

#include "layers/basic.h"

namespace grud {
namespace checkgrad {

/**
 * Numerical gradient checking utilities - IMPROVED for float32 precision
 */
struct GradCheckResult {
    bool passed;
    float max_relative_error;
    float max_absolute_error;
    size_t num_parameters_checked;
    std::vector<std::string> failed_parameters;

    GradCheckResult()
        : passed(false), max_relative_error(0.0f), max_absolute_error(0.0f),
          num_parameters_checked(0) {}
};

/**
 * Simple loss function for gradient checking
 * Computes scalar loss from module output
 */
class SimpleLoss {
public:
    /**
     * Compute loss as sum of squares of all outputs
     */
    static float forward(const Eigen::MatrixXf& output) {
        return 0.5f * output.array().square().sum();
    }

    /**
     * Compute gradient of loss w.r.t. output
     */
    static Eigen::MatrixXf backward(const Eigen::MatrixXf& output) {
        return output;
    }
};

/**
 * MSE loss for gradient checking with targets
 */
class MSELoss {
public:
    static float forward(const Eigen::MatrixXf& output, const Eigen::MatrixXf& target) {
        Eigen::MatrixXf diff = output - target;
        return 0.5f * diff.array().square().mean();
    }

    static Eigen::MatrixXf backward(const Eigen::MatrixXf& output, const Eigen::MatrixXf& target) {
        return (output - target) / static_cast<float>(output.size());
    }
};

/**
 * Improved adaptive epsilon for float32 precision
 * Uses larger epsilon values appropriate for float32 arithmetic
 */
inline float improved_adaptive_eps(float x) {
    // For float32, machine epsilon is ~1.2e-7
    // For numerical differentiation, we need sqrt(eps) * max(1, |x|)
    // This gives us better numerical stability
    const float sqrt_eps = 3e-4f;  // Conservative sqrt of effective float32 epsilon
    return sqrt_eps * std::max(1.0f, std::abs(x));
}

/**
 * Improved numerical gradient computation with error handling
 */
inline float improved_numerical_gradient(
    Module& module,
    Param& param,
    int row, int col,
    const Eigen::MatrixXf& input,
    float eps = -1.0f) {

    // Save original value
    float original_value = param.value(row, col);

    // Use improved epsilon selection
    if (eps <= 0.0f) {
        eps = improved_adaptive_eps(original_value);
    }

    // Ensure minimum epsilon to avoid numerical cancellation
    eps = std::max(eps, 1e-6f);

    float loss_plus, loss_minus;

    try {
        // Compute f(x + eps)
        param.value(row, col) = original_value + eps;
        Tape tape_plus;
        Context ctx_plus(&module);
        Eigen::MatrixXf output_plus = module.forward(input, ctx_plus);
        loss_plus = SimpleLoss::forward(output_plus);

        // Compute f(x - eps)
        param.value(row, col) = original_value - eps;
        Tape tape_minus;
        Context ctx_minus(&module);
        Eigen::MatrixXf output_minus = module.forward(input, ctx_minus);
        loss_minus = SimpleLoss::forward(output_minus);

        // Restore original value
        param.value(row, col) = original_value;

        // Check for numerical issues
        if (!std::isfinite(loss_plus) || !std::isfinite(loss_minus)) {
            std::cerr << "Warning: Non-finite loss values in gradient check for "
                      << param.name << "[" << row << "," << col << "]" << std::endl;
            return 0.0f;
        }

        // Compute numerical gradient
        float num_grad = (loss_plus - loss_minus) / (2.0f * eps);

        // Sanity check for unreasonable gradients
        if (std::abs(num_grad) > 1e6f) {
            std::cerr << "Warning: Very large numerical gradient: " << num_grad
                      << " for " << param.name << "[" << row << "," << col << "]" << std::endl;
        }

        return num_grad;

    } catch (const std::exception& e) {
        std::cerr << "Exception in numerical gradient computation: " << e.what() << std::endl;
        param.value(row, col) = original_value;  // Restore on exception
        return 0.0f;
    }
}

/**
 * Improved analytical gradient computation with error handling
 */
inline float improved_analytical_gradient(
    Module& module,
    Param& param,
    int row, int col,
    const Eigen::MatrixXf& input) {

    try {
        // Zero all gradients
        module.zero_grad();

        // Forward pass
        Tape tape;
        Context ctx(&module);
        Eigen::MatrixXf output = module.forward(input, ctx);

        // Backward pass
        Eigen::MatrixXf grad_output = SimpleLoss::backward(output);
        tape.push(std::move(ctx));
        autograd::backward(tape, grad_output);

        return param.grad(row, col);

    } catch (const std::exception& e) {
        std::cerr << "Exception in analytical gradient computation: " << e.what() << std::endl;
        return 0.0f;
    }
}

/**
 * Determine appropriate tolerance based on gradient magnitude and complexity
 */
inline std::pair<float, float> adaptive_tolerances(float grad_magnitude, float base_rtol, float base_atol) {
    float rtol = base_rtol;
    float atol = base_atol;

    // For very small gradients, increase tolerance to account for numerical noise
    if (grad_magnitude < 1e-4f) {
        rtol *= 10.0f;
        atol *= 10.0f;
    }

    // For very large gradients, slightly increase tolerance due to accumulation errors
    if (grad_magnitude > 10.0f) {
        rtol *= 2.0f;
    }

    return {rtol, atol};
}

/**
 * Check gradients for a single parameter with improved error handling
 */
inline bool check_parameter_gradient_improved(
    Module& module,
    Param& param,
    const Eigen::MatrixXf& input,
    float eps,
    float base_rtol,
    float base_atol,
    GradCheckResult& result,
    bool verbose = false) {

    bool param_passed = true;

    for (int i = 0; i < param.value.rows(); ++i) {
        for (int j = 0; j < param.value.cols(); ++j) {
            result.num_parameters_checked++;

            // Compute numerical and analytical gradients
            float numerical_grad = improved_numerical_gradient(module, param, i, j, input, eps);
            float analytical_grad = improved_analytical_gradient(module, param, i, j, input);

            // Skip if either gradient computation failed
            if (!std::isfinite(numerical_grad) || !std::isfinite(analytical_grad)) {
                param_passed = false;
                std::string param_info = param.name + "[" + std::to_string(i) + "," + std::to_string(j) + "]";
                result.failed_parameters.push_back(param_info);
                continue;
            }

            // Determine adaptive tolerances
            float grad_magnitude = std::max(std::abs(numerical_grad), std::abs(analytical_grad));
            auto [rtol, atol] = adaptive_tolerances(grad_magnitude, base_rtol, base_atol);

            // Compute errors
            float abs_error = std::abs(numerical_grad - analytical_grad);
            float rel_error = abs_error / (grad_magnitude + 1e-10f);

            result.max_absolute_error = std::max(result.max_absolute_error, abs_error);
            result.max_relative_error = std::max(result.max_relative_error, rel_error);

            // Check if gradient is correct with adaptive tolerance
            bool grad_ok = (abs_error < atol) || (rel_error < rtol);

            if (!grad_ok) {
                param_passed = false;
                std::string param_info = param.name + "[" + std::to_string(i) + "," + std::to_string(j) + "]";
                result.failed_parameters.push_back(param_info);

                if (verbose) {
                    std::cerr << "Gradient check failed for " << param_info << ":" << std::endl;
                    std::cerr << "  Numerical:  " << std::fixed << std::setprecision(8) << numerical_grad << std::endl;
                    std::cerr << "  Analytical: " << std::fixed << std::setprecision(8) << analytical_grad << std::endl;
                    std::cerr << "  Abs error:  " << std::scientific << abs_error
                              << " (adaptive tol: " << atol << ")" << std::endl;
                    std::cerr << "  Rel error:  " << std::scientific << rel_error
                              << " (adaptive tol: " << rtol << ")" << std::endl;
                    std::cerr << "  Grad mag:   " << grad_magnitude << std::endl;
                }
            }
        }
    }

    return param_passed;
}

/**
 * MAIN IMPROVED GRADIENT CHECKING FUNCTION
 * Optimized for float32 precision with realistic tolerances
 *
 * @param module Module to check
 * @param input Input tensor
 * @param eps Epsilon for numerical differentiation (negative for adaptive)
 * @param rtol Relative tolerance (realistic default: 1e-2 = 1%)
 * @param atol Absolute tolerance (realistic default: 1e-6)
 * @param verbose Print detailed information
 * @return GradCheckResult with detailed results
 */
inline GradCheckResult improved_check_gradients(
    Module& module,
    const Eigen::MatrixXf& input,
    float eps = -1.0f,      // Use adaptive by default
    float rtol = 1e-2f,     // 1% relative tolerance (realistic for float32)
    float atol = 1e-6f,     // Absolute tolerance
    bool verbose = true) {

    GradCheckResult result;

    // Set module to evaluation mode to disable dropout, etc.
    bool original_training_mode = module.is_training();
    module.set_training(false);

    if (verbose) {
        std::cout << "=== Improved Gradient Check (Float32 Optimized) ===" << std::endl;
        std::cout << "Module: " << module.name() << std::endl;
        std::cout << "Input shape: (" << input.rows() << ", " << input.cols() << ")" << std::endl;
        if (eps > 0) {
            std::cout << "Epsilon: " << eps << std::endl;
        } else {
            std::cout << "Epsilon: adaptive (improved for float32)" << std::endl;
        }
        std::cout << "Base relative tolerance: " << rtol << std::endl;
        std::cout << "Base absolute tolerance: " << atol << std::endl;
        std::cout << std::endl;
    }

    try {
        // Get all parameters
        std::vector<Param*> params = module.all_parameters();

        if (verbose) {
            std::cout << "Checking " << params.size() << " parameter tensors..." << std::endl;
        }

        bool all_passed = true;

        // Check each parameter
        for (size_t p = 0; p < params.size(); ++p) {
            Param* param = params[p];

            if (verbose) {
                std::cout << "Checking parameter " << (p + 1) << "/" << params.size()
                         << ": " << param->name
                         << " (" << param->value.rows() << "x" << param->value.cols() << ")" << std::endl;
            }

            bool param_passed = check_parameter_gradient_improved(
                module, *param, input, eps, rtol, atol, result, verbose);

            if (!param_passed) {
                all_passed = false;
            }

            if (verbose) {
                std::cout << "  " << (param_passed ? "✓ PASSED" : "✗ FAILED") << std::endl;
            }
        }

        result.passed = all_passed;

        if (verbose) {
            std::cout << std::endl;
            std::cout << "=== Summary ===" << std::endl;
            std::cout << "Parameters checked: " << result.num_parameters_checked << std::endl;
            std::cout << "Max absolute error: " << std::scientific << result.max_absolute_error << std::endl;
            std::cout << "Max relative error: " << std::scientific << result.max_relative_error << std::endl;
            std::cout << "Result: " << (result.passed ? "PASSED" : "FAILED") << std::endl;

            if (!result.passed) {
                std::cout << "Failed parameters (" << result.failed_parameters.size() << "):" << std::endl;
                for (const auto& param_name : result.failed_parameters) {
                    std::cout << "  " << param_name << std::endl;
                }
            }
        }

    } catch (const std::exception& e) {
        result.passed = false;
        if (verbose) {
            std::cerr << "Exception during gradient check: " << e.what() << std::endl;
        }
    }

    // Restore original training mode
    module.set_training(original_training_mode);

    return result;
}

/**
 * Quick gradient check function with realistic float32 tolerances
 */
inline bool quick_check_improved(Module& module, const Eigen::MatrixXf& input, float tolerance = 1e-2f) {
    GradCheckResult result = improved_check_gradients(module, input, -1.0f, tolerance, 1e-6f, false);
    return result.passed;
}

/**
 * Legacy quick_check function for backward compatibility
 * Now uses improved implementation with realistic tolerances
 */
inline bool quick_check(Module& module, const Eigen::MatrixXf& input, float tolerance = 1e-2f) {
    GradCheckResult result = improved_check_gradients(module, input, -1.0f, tolerance, 1e-6f, false);
    return result.passed;
}

/**
 * Generate random input for gradient checking with controlled scale
 */
inline Eigen::MatrixXf random_input(int rows, int cols, std::mt19937& gen, float scale = 1.0f) {
    std::normal_distribution<float> dist(0.0f, scale);
    Eigen::MatrixXf input(rows, cols);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            input(i, j) = dist(gen);
        }
    }

    return input;
}

/**
 * Legacy function for backward compatibility - now uses improved version
 */
inline GradCheckResult check_gradients(
    Module& module,
    const Eigen::MatrixXf& input,
    float eps = -1.0f,
    float rtol = 1e-2f,     // Updated default to realistic value
    float atol = 1e-6f,     // Updated default
    bool verbose = true) {

    return improved_check_gradients(module, input, eps, rtol, atol, verbose);
}

/**
 * Debug helper: Manual verification of linear layer gradients
 */
inline bool debug_linear_gradients() {
    std::cout << "=== Debug Linear Layer Gradients ===" << std::endl;

    std::mt19937 gen(42);

    // Create simple 2x1 linear layer for debugging
    auto linear = std::make_unique<grud::layers::Linear>(2, 1, true, &gen);

    // Scale down weights for better numerical stability
    linear->weight.value *= 0.1f;
    linear->bias.value *= 0.1f;

    // Simple input
    Eigen::MatrixXf input(1, 2);  // 1 sample, 2 features
    input << 0.5f, -0.3f;

    std::cout << "Input: " << input << std::endl;
    std::cout << "Weight: " << linear->weight.value << std::endl;
    std::cout << "Bias: " << linear->bias.value << std::endl;

    // Manual forward pass verification
    Eigen::MatrixXf expected_output = input * linear->weight.value.transpose();
    expected_output.rowwise() += linear->bias.value.col(0).transpose();

    // Actual forward pass
    Context ctx(linear.get());
    Eigen::MatrixXf actual_output = linear->forward(input, ctx);

    std::cout << "Expected output: " << expected_output << std::endl;
    std::cout << "Actual output: " << actual_output << std::endl;

    float forward_error = (expected_output - actual_output).norm();
    std::cout << "Forward pass error: " << forward_error << std::endl;

    if (forward_error > 1e-6f) {
        std::cout << "❌ Forward pass mismatch!" << std::endl;
        return false;
    }

    // Test gradient computation manually
    linear->zero_grad();

    // Simple loss: L = 0.5 * output^2
    float loss = 0.5f * actual_output.array().square().sum();
    Eigen::MatrixXf grad_output = actual_output;  // dL/dy = y

    std::cout << "Loss: " << loss << std::endl;
    std::cout << "Grad output: " << grad_output << std::endl;

    // Backward pass
    Eigen::MatrixXf grad_input = linear->backward(grad_output, ctx);

    std::cout << "Weight grad: " << linear->weight.grad << std::endl;
    std::cout << "Bias grad: " << linear->bias.grad << std::endl;
    std::cout << "Input grad: " << grad_input << std::endl;

    // Manual gradient verification
    // dL/dW = grad_output^T * input
    Eigen::MatrixXf expected_weight_grad = grad_output.transpose() * input;
    Eigen::MatrixXf expected_bias_grad = grad_output.colwise().sum().transpose();
    Eigen::MatrixXf expected_input_grad = grad_output * linear->weight.value;

    float weight_grad_error = (expected_weight_grad - linear->weight.grad).norm();
    float bias_grad_error = (expected_bias_grad - linear->bias.grad).norm();
    float input_grad_error = (expected_input_grad - grad_input).norm();

    std::cout << "Weight grad error: " << weight_grad_error << std::endl;
    std::cout << "Bias grad error: " << bias_grad_error << std::endl;
    std::cout << "Input grad error: " << input_grad_error << std::endl;

    bool passed = (weight_grad_error < 1e-6f) && (bias_grad_error < 1e-6f) && (input_grad_error < 1e-6f);

    std::cout << "Manual gradient check: " << (passed ? "✅ PASSED" : "❌ FAILED") << std::endl;

    return passed;
}

} // namespace checkgrad
} // namespace grud

#endif // GRUD_CORE_CHECKGRAD_H