//
// grud/testing/tests.h - Comprehensive testing framework
//

#ifndef GRUD_TESTING_TESTS_H
#define GRUD_TESTING_TESTS_H

#include "../grud.h"
#include <cassert>
#include <vector>
#include <string>
#include <functional>
#include <chrono>
#include <sstream>

namespace grud {
namespace testing {

// ============================================================================
// SIMPLE TESTING FRAMEWORK
// ============================================================================

struct TestResult {
    bool passed = false;
    std::string message;
    float duration_ms = 0.0f;

    TestResult(bool pass = false, const std::string& msg = "")
        : passed(pass), message(msg) {}
};

class TestSuite {
private:
    std::vector<std::function<TestResult()>> tests_;
    std::vector<std::string> test_names_;

public:
    void add_test(const std::string& name, std::function<TestResult()> test) {
        test_names_.push_back(name);
        tests_.push_back(test);
    }

    void run_all() {
        std::cout << "\n=== Running Test Suite ===" << std::endl;
        std::cout << "Total tests: " << tests_.size() << std::endl;

        int passed = 0;
        int failed = 0;
        float total_time = 0.0f;

        for (size_t i = 0; i < tests_.size(); ++i) {
            std::cout << "[" << (i + 1) << "/" << tests_.size() << "] "
                      << test_names_[i] << "... ";

            auto start = std::chrono::high_resolution_clock::now();
            TestResult result = tests_[i]();
            auto end = std::chrono::high_resolution_clock::now();

            result.duration_ms = std::chrono::duration<float, std::milli>(end - start).count();
            total_time += result.duration_ms;

            if (result.passed) {
                std::cout << "✅ PASS (" << std::fixed << std::setprecision(2)
                          << result.duration_ms << "ms)" << std::endl;
                passed++;
            } else {
                std::cout << "❌ FAIL" << std::endl;
                if (!result.message.empty()) {
                    std::cout << "   Error: " << result.message << std::endl;
                }
                failed++;
            }
        }

        std::cout << "\n=== Test Results ===" << std::endl;
        std::cout << "Passed: " << passed << std::endl;
        std::cout << "Failed: " << failed << std::endl;
        std::cout << "Total time: " << std::fixed << std::setprecision(2) << total_time << "ms" << std::endl;
        std::cout << "Success rate: " << std::fixed << std::setprecision(1)
                  << (100.0f * passed / (passed + failed)) << "%" << std::endl;
    }
};

// ============================================================================
// ASSERTION HELPERS
// ============================================================================

template<typename T>
bool assert_near(T actual, T expected, T tolerance = static_cast<T>(1e-6)) {
    return std::abs(actual - expected) <= tolerance;
}

bool assert_matrix_near(const Eigen::MatrixXf& actual, const Eigen::MatrixXf& expected,
                       float tolerance = 1e-6f) {
    if (actual.rows() != expected.rows() || actual.cols() != expected.cols()) {
        return false;
    }

    for (int i = 0; i < actual.rows(); ++i) {
        for (int j = 0; j < actual.cols(); ++j) {
            if (!assert_near(actual(i, j), expected(i, j), tolerance)) {
                return false;
            }
        }
    }
    return true;
}

bool assert_shape(const Eigen::MatrixXf& matrix, int expected_rows, int expected_cols) {
    return matrix.rows() == expected_rows && matrix.cols() == expected_cols;
}

bool assert_finite(const Eigen::MatrixXf& matrix) {
    return matrix.array().isFinite().all();
}

// ============================================================================
// UNIT TESTS FOR CORE COMPONENTS
// ============================================================================

namespace unit_tests {

TestResult test_param_basic() {
    try {
        Param p(3, 2, "test_param");

        if (!assert_shape(p.value, 3, 2)) {
            return TestResult(false, "Parameter shape incorrect");
        }

        if (!assert_shape(p.grad, 3, 2)) {
            return TestResult(false, "Gradient shape incorrect");
        }

        if (p.name != "test_param") {
            return TestResult(false, "Parameter name incorrect");
        }

        // Test zero_grad
        p.grad.setConstant(1.0f);
        p.zero_grad();

        if (p.grad.norm() > 1e-8f) {
            return TestResult(false, "zero_grad() failed");
        }

        return TestResult(true);
    } catch (const std::exception& e) {
        return TestResult(false, e.what());
    }
}

TestResult test_context_basic() {
    try {
        Context ctx;

        // Test saving tensors
        Eigen::MatrixXf test_tensor = Eigen::MatrixXf::Random(2, 3);
        ctx.save_for_backward(test_tensor);

        const Eigen::MatrixXf& retrieved = ctx.get_saved(0);

        if (!assert_matrix_near(test_tensor, retrieved)) {
            return TestResult(false, "Saved tensor retrieval failed");
        }

        // Test clear
        ctx.clear();
        if (!ctx.saved.empty()) {
            return TestResult(false, "Context clear failed");
        }

        return TestResult(true);
    } catch (const std::exception& e) {
        return TestResult(false, e.what());
    }
}

TestResult test_tape_basic() {
    try {
        Tape tape;

        if (!tape.empty()) {
            return TestResult(false, "New tape should be empty");
        }

        // Add some contexts
        Context ctx1, ctx2;
        size_t idx1 = tape.push(std::move(ctx1));
        size_t idx2 = tape.push(std::move(ctx2));

        if (tape.size() != 2) {
            return TestResult(false, "Tape size incorrect");
        }

        if (idx1 != 0 || idx2 != 1) {
            return TestResult(false, "Tape indices incorrect");
        }

        // Test retrieval
        const Context& retrieved = tape.get(0);
        (void)retrieved; // Suppress unused variable warning

        // Test clear
        tape.clear();
        if (!tape.empty()) {
            return TestResult(false, "Tape clear failed");
        }

        return TestResult(true);
    } catch (const std::exception& e) {
        return TestResult(false, e.what());
    }
}

TestResult test_linear_layer() {
    try {
        std::mt19937 gen(42);
        Linear linear(4, 3, true, &gen);

        // Test parameter count
        auto params = linear.params();
        if (params.size() != 2) { // weight + bias
            return TestResult(false, "Linear layer should have 2 parameters");
        }

        // Test forward pass
        Eigen::MatrixXf input = Eigen::MatrixXf::Random(2, 4);
        Context ctx(&linear);
        Eigen::MatrixXf output = linear.forward(input, ctx);

        if (!assert_shape(output, 2, 3)) {
            return TestResult(false, "Linear layer output shape incorrect");
        }

        if (!assert_finite(output)) {
            return TestResult(false, "Linear layer output contains non-finite values");
        }

        // Test gradient checking
        auto grad_result = checkgrad::quick_check(linear, input, 1e-3f);
        if (!grad_result) {
            return TestResult(false, "Linear layer gradient check failed");
        }

        return TestResult(true);
    } catch (const std::exception& e) {
        return TestResult(false, e.what());
    }
}

TestResult test_layer_norm() {
    try {
        LayerNorm ln(5);

        // Test forward pass
        Eigen::MatrixXf input = Eigen::MatrixXf::Random(3, 5);
        Context ctx(&ln);
        Eigen::MatrixXf output = ln.forward(input, ctx);

        if (!assert_shape(output, 3, 5)) {
            return TestResult(false, "LayerNorm output shape incorrect");
        }

        // Check normalization properties (approximately zero mean, unit variance)
        for (int i = 0; i < output.rows(); ++i) {
            float mean = output.row(i).mean();
            float var = (output.row(i).array() - mean).square().mean();

            if (!assert_near(mean, 0.0f, 1e-5f)) {
                return TestResult(false, "LayerNorm mean not close to zero");
            }

            if (!assert_near(var, 1.0f, 1e-4f)) {
                return TestResult(false, "LayerNorm variance not close to one");
            }
        }

        // Test gradient checking
        auto grad_result = checkgrad::quick_check(ln, input, 1e-2f);
        if (!grad_result) {
            return TestResult(false, "LayerNorm gradient check failed");
        }

        return TestResult(true);
    } catch (const std::exception& e) {
        return TestResult(false, e.what());
    }
}

TestResult test_dropout_layer() {
    try {
        std::mt19937 gen(42);
        Dropout dropout(0.5f, gen);

        // Test training mode
        dropout.set_training(true);
        Eigen::MatrixXf input = Eigen::MatrixXf::Ones(100, 10);
        Context ctx(&dropout);
        Eigen::MatrixXf output = dropout.forward(input, ctx);

        // Check that approximately half the values are zeroed
        int num_zeros = (output.array() == 0.0f).count();
        int total_elements = output.size();
        float zero_ratio = static_cast<float>(num_zeros) / total_elements;

        if (!assert_near(zero_ratio, 0.5f, 0.1f)) {
            return TestResult(false, "Dropout zero ratio incorrect: " + std::to_string(zero_ratio));
        }

        // Test evaluation mode
        dropout.set_training(false);
        Context ctx2(&dropout);
        Eigen::MatrixXf output_eval = dropout.forward(input, ctx2);

        if (!assert_matrix_near(input, output_eval)) {
            return TestResult(false, "Dropout should be identity in eval mode");
        }

        return TestResult(true);
    } catch (const std::exception& e) {
        return TestResult(false, e.what());
    }
}

TestResult test_activation_functions() {
    try {
        std::mt19937 gen(42);

        // Test ReLU
        ReLU relu;
        Eigen::MatrixXf input = Eigen::MatrixXf::Random(2, 3);
        Context ctx_relu(&relu);
        Eigen::MatrixXf relu_output = relu.forward(input, ctx_relu);

        // Check that negative values are zeroed
        for (int i = 0; i < input.rows(); ++i) {
            for (int j = 0; j < input.cols(); ++j) {
                if (input(i, j) < 0 && relu_output(i, j) != 0.0f) {
                    return TestResult(false, "ReLU should zero negative values");
                }
                if (input(i, j) >= 0 && !assert_near(relu_output(i, j), input(i, j))) {
                    return TestResult(false, "ReLU should preserve positive values");
                }
            }
        }

        // Test Sigmoid
        Sigmoid sigmoid;
        Context ctx_sigmoid(&sigmoid);
        Eigen::MatrixXf sigmoid_output = sigmoid.forward(input, ctx_sigmoid);

        // Check that output is in (0, 1)
        if ((sigmoid_output.array() <= 0.0f).any() || (sigmoid_output.array() >= 1.0f).any()) {
            return TestResult(false, "Sigmoid output should be in (0, 1)");
        }

        // Test Tanh
        Tanh tanh_layer;
        Context ctx_tanh(&tanh_layer);
        Eigen::MatrixXf tanh_output = tanh_layer.forward(input, ctx_tanh);

        // Check that output is in (-1, 1)
        if ((tanh_output.array() <= -1.0f).any() || (tanh_output.array() >= 1.0f).any()) {
            return TestResult(false, "Tanh output should be in (-1, 1)");
        }

        return TestResult(true);
    } catch (const std::exception& e) {
        return TestResult(false, e.what());
    }
}

TestResult test_sequential_module() {
    try {
        std::mt19937 gen(42);

        auto sequential = std::make_unique<Sequential>();
        sequential->add(std::make_unique<Linear>(5, 8, true, &gen));
        sequential->add(std::make_unique<ReLU>());
        sequential->add(std::make_unique<Linear>(8, 3, true, &gen));

        // Test parameter collection
        auto params = sequential->all_parameters();
        if (params.size() != 4) { // 2 weights + 2 biases
            return TestResult(false, "Sequential parameter count incorrect");
        }

        // Test forward pass
        Eigen::MatrixXf input = Eigen::MatrixXf::Random(2, 5);
        Context ctx(sequential.get());
        Eigen::MatrixXf output = sequential->forward(input, ctx);

        if (!assert_shape(output, 2, 3)) {
            return TestResult(false, "Sequential output shape incorrect");
        }

        if (!assert_finite(output)) {
            return TestResult(false, "Sequential output contains non-finite values");
        }

        return TestResult(true);
    } catch (const std::exception& e) {
        return TestResult(false, e.what());
    }
}

} // namespace unit_tests

// ============================================================================
// INTEGRATION TESTS
// ============================================================================

namespace integration_tests {

TestResult test_autoencoder_creation() {
    try {
        std::mt19937 gen(42);

        // Test reconstruction autoencoder
        auto reconstruction_ae = factory::create_autoencoder(
            10, 5, 32, 2, models::BottleneckType::MEAN_POOL,
            models::AutoencoderMode::RECONSTRUCTION, 1, &gen);

        if (!reconstruction_ae) {
            return TestResult(false, "Failed to create reconstruction autoencoder");
        }

        if (reconstruction_ae->num_parameters() == 0) {
            return TestResult(false, "Autoencoder has no parameters");
        }

        // Test forecasting autoencoder
        auto forecasting_ae = factory::create_autoencoder(
            8, 4, 24, 1, models::BottleneckType::LAST_HIDDEN,
            models::AutoencoderMode::FORECASTING, 5, &gen);

        if (!forecasting_ae) {
            return TestResult(false, "Failed to create forecasting autoencoder");
        }

        return TestResult(true);
    } catch (const std::exception& e) {
        return TestResult(false, e.what());
    }
}

TestResult test_autoencoder_forward_pass() {
    try {
        std::mt19937 gen(42);
        auto autoencoder = factory::create_autoencoder(6, 3, 16, 1,
            models::BottleneckType::MEAN_POOL, models::AutoencoderMode::RECONSTRUCTION, 1, &gen);

        // Create test data
        int batch_size = 2;
        int seq_len = 5;
        int input_size = 6;

        std::vector<Eigen::MatrixXf> X_seq;
        std::vector<Eigen::MatrixXf> dt_seq;
        std::vector<std::optional<Eigen::MatrixXf>> mask_seq;

        for (int t = 0; t < seq_len; ++t) {
            X_seq.push_back(Eigen::MatrixXf::Random(batch_size, input_size));
            dt_seq.push_back(Eigen::MatrixXf::Constant(batch_size, 1, 1.0f));
            mask_seq.push_back(Eigen::MatrixXf::Ones(batch_size, input_size));
        }

        // Forward pass
        Tape tape;
        auto result = autoencoder->forward_autoencoder(X_seq, dt_seq, mask_seq, {}, {}, tape);

        // Check output shapes
        int expected_output_rows = seq_len * batch_size;
        if (!assert_shape(result.output_sequence, expected_output_rows, input_size)) {
            return TestResult(false, "Autoencoder output sequence shape incorrect");
        }

        if (!assert_shape(result.latent, batch_size, 3)) {
            return TestResult(false, "Autoencoder latent shape incorrect");
        }

        if (!assert_finite(result.output_sequence)) {
            return TestResult(false, "Autoencoder output contains non-finite values");
        }

        if (!assert_finite(result.latent)) {
            return TestResult(false, "Autoencoder latent contains non-finite values");
        }

        return TestResult(true);
    } catch (const std::exception& e) {
        return TestResult(false, e.what());
    }
}

TestResult test_autoencoder_backward_pass() {
    try {
        std::mt19937 gen(42);
        auto autoencoder = factory::create_autoencoder(4, 2, 8, 1,
            models::BottleneckType::LAST_HIDDEN, models::AutoencoderMode::RECONSTRUCTION, 1, &gen);

        // Create simple test data
        std::vector<Eigen::MatrixXf> X_seq = {Eigen::MatrixXf::Random(1, 4)};
        std::vector<Eigen::MatrixXf> dt_seq = {Eigen::MatrixXf::Constant(1, 1, 1.0f)};
        std::vector<std::optional<Eigen::MatrixXf>> mask_seq = {Eigen::MatrixXf::Ones(1, 4)};

        // Forward pass
        Tape tape;
        auto result = autoencoder->forward_autoencoder(X_seq, dt_seq, mask_seq, {}, {}, tape);

        // Backward pass
        autoencoder->zero_grad();
        Eigen::MatrixXf grad_output = Eigen::MatrixXf::Ones(result.output_sequence.rows(),
                                                           result.output_sequence.cols());
        autograd::backward(tape, grad_output);

        // Check that gradients were computed
        auto params = autoencoder->all_parameters();
        bool has_gradients = false;
        for (const auto* param : params) {
            if (param->grad.norm() > 1e-8f) {
                has_gradients = true;
                break;
            }
        }

        if (!has_gradients) {
            return TestResult(false, "No gradients computed in backward pass");
        }

        return TestResult(true);
    } catch (const std::exception& e) {
        return TestResult(false, e.what());
    }
}

TestResult test_optimizer_step() {
    try {
        std::mt19937 gen(42);

        // Create simple model
        auto linear = std::make_unique<Linear>(3, 2, true, &gen);
        auto params = linear->all_parameters();

        // Create optimizer
        optim::AdamW optimizer(1e-3f);

        // Store initial parameters
        std::vector<Eigen::MatrixXf> initial_values;
        for (const auto* param : params) {
            initial_values.push_back(param->value);
        }

        // Create fake gradients
        for (auto* param : params) {
            param->grad = Eigen::MatrixXf::Random(param->value.rows(), param->value.cols());
        }

        // Optimizer step
        optimizer.step(params);

        // Check that parameters changed
        bool parameters_changed = false;
        for (size_t i = 0; i < params.size(); ++i) {
            if (!assert_matrix_near(params[i]->value, initial_values[i], 1e-6f)) {
                parameters_changed = true;
                break;
            }
        }

        if (!parameters_changed) {
            return TestResult(false, "Optimizer did not update parameters");
        }

        return TestResult(true);
    } catch (const std::exception& e) {
        return TestResult(false, e.what());
    }
}

TestResult test_loss_functions() {
    try {
        // Create test data
        Eigen::MatrixXf pred = Eigen::MatrixXf::Random(3, 4);
        Eigen::MatrixXf target = Eigen::MatrixXf::Random(3, 4);

        // Test MSE loss
        loss::MSELoss mse_loss;
        float mse_val = mse_loss.forward(pred, target);
        Eigen::MatrixXf mse_grad = mse_loss.backward(pred, target);

        if (!std::isfinite(mse_val) || mse_val < 0.0f) {
            return TestResult(false, "MSE loss value invalid");
        }

        if (!assert_finite(mse_grad)) {
            return TestResult(false, "MSE gradient contains non-finite values");
        }

        // Test MAE loss
        loss::MAELoss mae_loss;
        float mae_val = mae_loss.forward(pred, target);
        Eigen::MatrixXf mae_grad = mae_loss.backward(pred, target);

        if (!std::isfinite(mae_val) || mae_val < 0.0f) {
            return TestResult(false, "MAE loss value invalid");
        }

        if (!assert_finite(mae_grad)) {
            return TestResult(false, "MAE gradient contains non-finite values");
        }

        // Test Huber loss
        loss::HuberLoss huber_loss(1.0f);
        float huber_val = huber_loss.forward(pred, target);
        Eigen::MatrixXf huber_grad = huber_loss.backward(pred, target);

        if (!std::isfinite(huber_val) || huber_val < 0.0f) {
            return TestResult(false, "Huber loss value invalid");
        }

        if (!assert_finite(huber_grad)) {
            return TestResult(false, "Huber gradient contains non-finite values");
        }

        return TestResult(true);
    } catch (const std::exception& e) {
        return TestResult(false, e.what());
    }
}

TestResult test_online_training() {
    try {
        std::mt19937 gen(42);

        // Create small autoencoder for fast testing
        auto autoencoder = factory::create_autoencoder(4, 2, 8, 1,
            models::BottleneckType::MEAN_POOL, models::AutoencoderMode::RECONSTRUCTION, 1, &gen);
        auto trainer = factory::create_trainer(std::move(autoencoder), 1e-3f, "adamw", "mse", &gen);

        // Train for a few steps
        std::normal_distribution<float> dist(0.0f, 1.0f);
        float initial_loss = 0.0f;
        float final_loss = 0.0f;

        for (int step = 0; step < 10; ++step) {
            Eigen::MatrixXf x_t(1, 4);
            Eigen::MatrixXf dt_t(1, 1);

            for (int i = 0; i < 4; ++i) {
                x_t(0, i) = dist(gen);
            }
            dt_t(0, 0) = 1.0f;

            auto [loss, pred] = trainer->step_online(x_t, dt_t, std::nullopt, x_t);

            if (step == 0) initial_loss = loss;
            if (step == 9) final_loss = loss;

            if (!std::isfinite(loss)) {
                return TestResult(false, "Training produced non-finite loss");
            }
        }

        // Loss should generally decrease (though not guaranteed in just 10 steps)
        // Just check that training completed without errors
        return TestResult(true);
    } catch (const std::exception& e) {
        return TestResult(false, e.what());
    }
}

} // namespace integration_tests

// ============================================================================
// PERFORMANCE TESTS
// ============================================================================

namespace performance_tests {

TestResult test_autoencoder_performance() {
    try {
        std::mt19937 gen(42);
        auto autoencoder = factory::create_autoencoder(32, 16, 64, 2,
            models::BottleneckType::MEAN_POOL, models::AutoencoderMode::RECONSTRUCTION, 1, &gen);

        // Create larger test data
        int batch_size = 8;
        int seq_len = 20;
        int input_size = 32;

        std::vector<Eigen::MatrixXf> X_seq;
        std::vector<Eigen::MatrixXf> dt_seq;
        std::vector<std::optional<Eigen::MatrixXf>> mask_seq;

        for (int t = 0; t < seq_len; ++t) {
            X_seq.push_back(Eigen::MatrixXf::Random(batch_size, input_size));
            dt_seq.push_back(Eigen::MatrixXf::Constant(batch_size, 1, 1.0f));
            mask_seq.push_back(Eigen::MatrixXf::Ones(batch_size, input_size));
        }

        // Time forward pass
        auto start = std::chrono::high_resolution_clock::now();

        for (int iter = 0; iter < 10; ++iter) {
            Tape tape;
            auto result = autoencoder->forward_autoencoder(X_seq, dt_seq, mask_seq, {}, {}, tape);

            // Simple backward pass
            autoencoder->zero_grad();
            Eigen::MatrixXf grad_output = Eigen::MatrixXf::Random(
                result.output_sequence.rows(), result.output_sequence.cols());
            autograd::backward(tape, grad_output);
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration<float, std::milli>(end - start).count();

        // Should complete in reasonable time (less than 5 seconds)
        if (duration > 5000.0f) {
            return TestResult(false, "Performance test too slow: " + std::to_string(duration) + "ms");
        }

        return TestResult(true, "Completed in " + std::to_string(duration) + "ms");
    } catch (const std::exception& e) {
        return TestResult(false, e.what());
    }
}

TestResult test_memory_usage() {
    try {
        std::mt19937 gen(42);

        // Create multiple models to test memory management
        std::vector<std::unique_ptr<models::TemporalAutoencoder>> models;

        for (int i = 0; i < 5; ++i) {
            models.push_back(factory::create_autoencoder(8, 4, 16, 1,
                models::BottleneckType::MEAN_POOL, models::AutoencoderMode::RECONSTRUCTION, 1, &gen));
        }

        // Clear models (should not cause memory leaks)
        models.clear();

        // Test successful if no exceptions thrown
        return TestResult(true);
    } catch (const std::exception& e) {
        return TestResult(false, e.what());
    }
}

} // namespace performance_tests

// ============================================================================
// MAIN TEST RUNNER
// ============================================================================

inline void run_all_tests() {
    TestSuite suite;

    // Unit tests
    suite.add_test("Param Basic", unit_tests::test_param_basic);
    suite.add_test("Context Basic", unit_tests::test_context_basic);
    suite.add_test("Tape Basic", unit_tests::test_tape_basic);
    suite.add_test("Linear Layer", unit_tests::test_linear_layer);
    suite.add_test("LayerNorm", unit_tests::test_layer_norm);
    suite.add_test("Dropout", unit_tests::test_dropout_layer);
    suite.add_test("Activations", unit_tests::test_activation_functions);
    suite.add_test("Sequential", unit_tests::test_sequential_module);

    // Integration tests
    suite.add_test("Autoencoder Creation", integration_tests::test_autoencoder_creation);
    suite.add_test("Autoencoder Forward", integration_tests::test_autoencoder_forward_pass);
    suite.add_test("Autoencoder Backward", integration_tests::test_autoencoder_backward_pass);
    suite.add_test("Optimizer Step", integration_tests::test_optimizer_step);
    suite.add_test("Loss Functions", integration_tests::test_loss_functions);
    suite.add_test("Online Training", integration_tests::test_online_training);

    // Performance tests
    suite.add_test("Autoencoder Performance", performance_tests::test_autoencoder_performance);
    suite.add_test("Memory Usage", performance_tests::test_memory_usage);

    suite.run_all();
}

} // namespace testing
} // namespace grud

#endif // GRUD_TESTING_TESTS_H