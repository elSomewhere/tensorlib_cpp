//
// grud.h - Complete GRU-D Autoencoder Framework
// Single header include for the entire framework
//

#ifndef GRUD_H
#define GRUD_H

/*
GRU-D Autoencoder Framework with Automatic Differentiation

This is a complete redesign of the GRU-D autoencoder following the autograd design
document. The framework provides:

1. Module-based architecture with automatic differentiation
2. Tape-based computational graph for backward pass
3. Gradient checking utilities
4. Complete GRU-D implementation with temporal dynamics
5. Flexible autoencoder models (reconstruction/forecasting)
6. Training utilities with streaming and batch support
7. Modern optimizers (SGD, AdamW) and loss functions

Key Features:
- Header-only design (no external dependencies except Eigen)
- Thread-safe with deterministic random number generation
- Numerical gradient checking for all modules
- Memory-efficient streaming training with TBPTT
- Flexible bottleneck aggregation (last hidden, mean/max pool, attention)
- Support for missing value imputation and exponential decay

Usage:
#include "grud.h"

auto autoencoder = grud::factory::create_autoencoder(12, 8, 64, 2);
auto trainer = grud::factory::create_trainer(std::move(autoencoder));
// ... training code

See example_usage.h for comprehensive examples.
*/

// Version information
#define GRUD_VERSION_MAJOR 2
#define GRUD_VERSION_MINOR 0
#define GRUD_VERSION_PATCH 0

// Core framework components
#include "core/module.h"
#include "core/tape.h"
#include "core/checkgrad.h"

// Layer implementations
#include "layers/basic.h"
#include "layers/temporal.h"

// Model implementations
#include "models/autoencoder.h"

// Optimization and training
#include "optim/optimizers.h"
#include "training/trainer.h"

// Examples and utilities
#include "example_usage.h"

// ============================================================================
// MAIN FRAMEWORK NAMESPACE
// ============================================================================

namespace grud {

/**
 * Framework version information
 */
struct Version {
    int major = GRUD_VERSION_MAJOR;
    int minor = GRUD_VERSION_MINOR;
    int patch = GRUD_VERSION_PATCH;

    std::string to_string() const {
        return std::to_string(major) + "." + std::to_string(minor) + "." + std::to_string(patch);
    }
};

/**
 * Get framework version
 */
inline Version get_version() {
    return Version{};
}

/**
 * Print framework information
 */
inline void print_info() {
    std::cout << "GRU-D Autoencoder Framework v" << get_version().to_string() << std::endl;
    std::cout << "Built with Eigen " << EIGEN_WORLD_VERSION << "."
              << EIGEN_MAJOR_VERSION << "." << EIGEN_MINOR_VERSION << std::endl;
    std::cout << "Features: Autograd, TBPTT, Streaming Training, Gradient Checking" << std::endl;
}

// ============================================================================
// CONVENIENCE ALIASES
// ============================================================================

// Core types
using Tensor = Eigen::MatrixXf;
using Parameter = Param;
using ModulePtr = std::unique_ptr<Module>;

// Layer types
using Linear = layers::Linear;
using LayerNorm = layers::LayerNorm;
using Dropout = layers::Dropout;
using ReLU = layers::ReLU;
using Tanh = layers::Tanh;
using Sigmoid = layers::Sigmoid;

// Temporal types
using GRUDCell = layers::GRUDCell;
using TemporalRNNLayer = layers::TemporalRNNLayer;

// Model types
using TemporalAutoencoder = models::TemporalAutoencoder;
using AutoencoderConfig = models::AutoencoderConfig;

// Training types
using OnlineTrainer = training::OnlineTrainer;
using BatchTrainer = training::BatchTrainer;
using TrainingConfig = training::TrainingConfig;
using Batch = training::Batch;

// Optimizer types
using SGD = optim::SGD;
using AdamW = optim::AdamW;

// Loss types
using MSELoss = loss::MSELoss;
using MAELoss = loss::MAELoss;
using HuberLoss = loss::HuberLoss;

// ============================================================================
// QUICK START UTILITIES
// ============================================================================

namespace quick {

/**
 * Create a reconstruction autoencoder with minimal configuration
 */
inline std::unique_ptr<TemporalAutoencoder> reconstruction_model(
    int input_size, int latent_size = -1, int hidden_size = -1, std::mt19937* gen = nullptr) {

    if (latent_size < 0) latent_size = std::max(2, input_size / 2);
    if (hidden_size < 0) hidden_size = std::max(latent_size * 2, 32);

    return factory::create_autoencoder(
        input_size, latent_size, hidden_size, 2,
        models::BottleneckType::MEAN_POOL,
        models::AutoencoderMode::RECONSTRUCTION,
        1, gen);
}

/**
 * Create a forecasting autoencoder with minimal configuration
 */
inline std::unique_ptr<TemporalAutoencoder> forecasting_model(
    int input_size, int forecast_horizon, int latent_size = -1,
    int hidden_size = -1, std::mt19937* gen = nullptr) {

    if (latent_size < 0) latent_size = std::max(2, input_size / 2);
    if (hidden_size < 0) hidden_size = std::max(latent_size * 2, 32);

    return factory::create_autoencoder(
        input_size, latent_size, hidden_size, 2,
        models::BottleneckType::LAST_HIDDEN,
        models::AutoencoderMode::FORECASTING,
        forecast_horizon, gen);
}

/**
 * Quick training on synthetic data (for testing/prototyping)
 */
inline training::Metrics quick_train(
    std::unique_ptr<TemporalAutoencoder> model,
    int num_epochs = 20,
    int batch_size = 4,
    int seq_len = 10,
    std::mt19937* gen = nullptr) {

    std::unique_ptr<std::mt19937> local_gen;
    if (!gen) {
        local_gen = std::make_unique<std::mt19937>(42);
        gen = local_gen.get();
    }

    // Create trainer
    auto trainer = factory::create_batch_trainer(std::move(model), num_epochs, 1e-3f, "adamw", gen);

    // Generate synthetic data
    int input_size = trainer->get_model().config.input_size;
    auto train_data = training::generate_synthetic_dataset(50, batch_size, seq_len, input_size, *gen);
    auto val_data = training::generate_synthetic_dataset(10, batch_size, seq_len, input_size, *gen);

    // Train
    trainer->train(train_data, val_data);

    // Return final metrics
    const auto& train_losses = trainer->get_train_losses();
    training::Metrics final_metrics;
    if (!train_losses.empty()) {
        final_metrics.loss = train_losses.back();
    }

    return final_metrics;
}

/**
 * Quick gradient check for any module - now uses improved implementation
 */
inline bool quick_grad_check(Module& module, const Tensor& input, float tolerance = 1e-2f) {
    return checkgrad::quick_check(module, input, tolerance);
}

/**
 * Quick performance benchmark
 */
inline void benchmark(Module& module, int batch_size = 32, int input_size = 64, int num_iterations = 100) {
    std::mt19937 gen(42);
    Tensor input = checkgrad::random_input(batch_size, input_size, gen);

    module.set_training(true);

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_iterations; ++i) {
        Tape tape;
        Context ctx(&module);

        // Forward pass
        Tensor output = module.forward(input, ctx);
        tape.push(std::move(ctx));

        // Backward pass
        Tensor grad_output = Tensor::Random(output.rows(), output.cols());
        module.zero_grad();
        autograd::backward(tape, grad_output);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "Benchmark Results:" << std::endl;
    std::cout << "  Module: " << module.name() << std::endl;
    std::cout << "  Input shape: (" << batch_size << ", " << input_size << ")" << std::endl;
    std::cout << "  Iterations: " << num_iterations << std::endl;
    std::cout << "  Total time: " << duration.count() << " ms" << std::endl;
    std::cout << "  Time per iteration: " << (duration.count() / static_cast<float>(num_iterations)) << " ms" << std::endl;
    std::cout << "  Parameters: " << module.num_parameters() << std::endl;
}

} // namespace quick

// ============================================================================
// FRAMEWORK DIAGNOSTICS
// ============================================================================

namespace diagnostics {

/**
 * Check framework installation and basic functionality
 */
inline bool self_test() {
    std::cout << "Running GRUD framework self-test..." << std::endl;

    try {
        std::mt19937 gen(42);

        // Test 1: Basic linear layer
        auto linear = std::make_unique<Linear>(4, 3, true, &gen);
        Tensor input = Tensor::Random(2, 4);
        Context ctx(linear.get());
        Tensor output = linear->forward(input, ctx);

        if (output.rows() != 2 || output.cols() != 3) {
            std::cout << "❌ Linear layer output shape incorrect" << std::endl;
            return false;
        }

        // Test 2: Gradient checking with realistic tolerance
        bool grad_result = checkgrad::quick_check(*linear, input, 1e-2f);  // 1% tolerance
        if (!grad_result) {
            std::cout << "❌ Gradient check failed with 1% tolerance" << std::endl;
            return false;
        }

        // Test 3: Autoencoder creation
        auto ae = quick::reconstruction_model(6, 3, 12, &gen);
        if (!ae || ae->num_parameters() == 0) {
            std::cout << "❌ Autoencoder creation failed" << std::endl;
            return false;
        }

        // Test 4: Training step
        auto trainer = factory::create_trainer(std::move(ae), 1e-3f, "adamw", "mse", &gen);
        Tensor x = Tensor::Random(2, 6);
        Tensor dt = Tensor::Constant(2, 1, 1.0f);
        auto [loss, pred] = trainer->step_online(x, dt, std::nullopt, x);

        if (!std::isfinite(loss)) {
            std::cout << "❌ Training step produced invalid loss" << std::endl;
            return false;
        }

        std::cout << "✅ All self-tests passed!" << std::endl;
        return true;

    } catch (const std::exception& e) {
        std::cout << "❌ Self-test failed with exception: " << e.what() << std::endl;
        return false;
    }
}

/**
 * Print detailed system information
 */
inline void system_info() {
    std::cout << "\n=== GRUD Framework System Information ===" << std::endl;

    // Framework info
    print_info();
    std::cout << std::endl;

    // Eigen info
    std::cout << "Eigen Configuration:" << std::endl;
//    std::cout << "  Vectorization: " <<
//#ifdef EIGEN_VECTORIZE
//        "Enabled"
//#else
//        "Disabled"
//#endif
//        << std::endl;
//
//    std::cout << "  SIMD: " <<
//#ifdef EIGEN_VECTORIZE_SSE2
//        "SSE2 "
//#endif
//#ifdef EIGEN_VECTORIZE_SSE3
//        "SSE3 "
//#endif
//#ifdef EIGEN_VECTORIZE_SSE4_1
//        "SSE4.1 "
//#endif
//#ifdef EIGEN_VECTORIZE_AVX
//        "AVX "
//#endif
//#ifdef EIGEN_VECTORIZE_AVX2
//        "AVX2 "
//#endif
//        << std::endl;

    // Memory layout
    std::cout << "\nMemory Layout:" << std::endl;
    std::cout << "  MatrixXf storage: " <<
        (Eigen::MatrixXf::IsRowMajor ? "Row-major" : "Column-major") << std::endl;

    // Type sizes
    std::cout << "\nType Sizes:" << std::endl;
    std::cout << "  Param: " << sizeof(Param) << " bytes" << std::endl;
    std::cout << "  Context: " << sizeof(Context) << " bytes" << std::endl;
    std::cout << "  Tape: " << sizeof(Tape) << " bytes" << std::endl;
    std::cout << "  Linear: " << sizeof(Linear) << " bytes" << std::endl;
    std::cout << "  GRUDCell: " << sizeof(GRUDCell) << " bytes" << std::endl;

    // Feature support
    std::cout << "\nFeature Support:" << std::endl;
    std::cout << "  Autograd: ✅" << std::endl;
    std::cout << "  Gradient Checking: ✅" << std::endl;
    std::cout << "  TBPTT: ✅" << std::endl;
    std::cout << "  Streaming Training: ✅" << std::endl;
    std::cout << "  Missing Value Imputation: ✅" << std::endl;
    std::cout << "  Exponential Decay: ✅" << std::endl;
    std::cout << "  Multiple Optimizers: ✅ (SGD, AdamW)" << std::endl;
    std::cout << "  Multiple Loss Functions: ✅ (MSE, MAE, Huber)" << std::endl;

    std::cout << "\n===========================================" << std::endl;
}

} // namespace diagnostics

} // namespace grud

// ============================================================================
// CONVENIENCE MACROS (OPTIONAL)
// ============================================================================

// Uncomment to enable convenience macros
/*
#define GRUD_TENSOR grud::Tensor
#define GRUD_PARAM grud::Parameter
#define GRUD_MODULE grud::ModulePtr
#define GRUD_LINEAR(...) std::make_unique<grud::Linear>(__VA_ARGS__)
#define GRUD_AUTOENCODER(...) grud::factory::create_autoencoder(__VA_ARGS__)
#define GRUD_TRAINER(...) grud::factory::create_trainer(__VA_ARGS__)
*/

namespace factory {

inline std::unique_ptr<grud::models::TemporalAutoencoder> create_autoencoder(
    int input_size, int latent_size, int hidden_size, int num_layers,
    grud::models::BottleneckType bottleneck_type = grud::models::BottleneckType::MEAN_POOL,
    grud::models::AutoencoderMode mode = grud::models::AutoencoderMode::RECONSTRUCTION,
    int forecast_horizon = 1, std::mt19937* gen = nullptr) {

    std::unique_ptr<std::mt19937> local_gen;
    if (!gen) {
        local_gen = std::make_unique<std::mt19937>(42);
        gen = local_gen.get();
    }

    grud::models::AutoencoderConfig config;
    config.input_size = input_size;
    config.latent_size = latent_size;
    config.hidden_size = hidden_size;
    config.num_layers = num_layers;
    config.bottleneck_type = bottleneck_type;
    config.mode = mode;
    config.forecast_horizon = forecast_horizon;

    return std::make_unique<grud::models::TemporalAutoencoder>(config, *gen);
}

inline std::unique_ptr<grud::training::OnlineTrainer> create_trainer(
    std::unique_ptr<grud::models::TemporalAutoencoder> model,
    float learning_rate = 1e-3f,
    const std::string& optimizer = "adamw",
    const std::string& loss_type = "mse",
    std::mt19937* gen = nullptr) {

    std::unique_ptr<std::mt19937> local_gen;
    if (!gen) {
        local_gen = std::make_unique<std::mt19937>(42);
        gen = local_gen.get();
    }

    grud::training::TrainingConfig config;
    config.learning_rate = learning_rate;
    config.optimizer = optimizer;
    config.loss_type = loss_type;

    return std::make_unique<grud::training::OnlineTrainer>(std::move(model), config, *gen);
}

inline std::unique_ptr<grud::training::BatchTrainer> create_batch_trainer(
    std::unique_ptr<grud::models::TemporalAutoencoder> model,
    int max_epochs = 100,
    float learning_rate = 1e-3f,
    const std::string& optimizer = "adamw",
    std::mt19937* gen = nullptr) {

    std::unique_ptr<std::mt19937> local_gen;
    if (!gen) {
        local_gen = std::make_unique<std::mt19937>(42);
        gen = local_gen.get();
    }

    grud::training::TrainingConfig config;
    config.max_epochs = max_epochs;
    config.learning_rate = learning_rate;
    config.optimizer = optimizer;

    return std::make_unique<grud::training::BatchTrainer>(std::move(model), config, *gen);
}

} // namespace factory

#endif // GRUD_H

/*
Example minimal usage:

#include "grud.h"

int main() {
    // Check framework
    if (!grud::diagnostics::self_test()) {
        return 1;
    }

    // Create and train a model
    std::mt19937 gen(42);
    auto model = grud::quick::reconstruction_model(12, 6, 32, &gen);
    auto metrics = grud::quick::quick_train(std::move(model), 10, 4, 8, &gen);

    std::cout << "Training completed with loss: " << metrics.loss << std::endl;
    return 0;
}
*/