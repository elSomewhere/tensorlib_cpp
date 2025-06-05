//
// grud/example_usage.h - Complete usage example and factory functions
//

#ifndef GRUD_EXAMPLE_USAGE_H
#define GRUD_EXAMPLE_USAGE_H

#include "core/module.h"
#include "core/tape.h"
#include "core/checkgrad.h"
#include "layers/basic.h"
#include "layers/temporal.h"
#include "models/autoencoder.h"
#include "optim/optimizers.h"
#include "training/trainer.h"

#include <memory>
#include <iostream>
#include <random>

namespace grud {
namespace factory {

// ============================================================================
// FACTORY FUNCTIONS FOR EASY MODEL CREATION
// ============================================================================

/**
 * Create a complete autoencoder model with sensible defaults
 */
inline std::unique_ptr<models::TemporalAutoencoder> create_autoencoder(
    int input_size = 12,
    int latent_size = 8,
    int hidden_size = 64,
    int num_layers = 2,
    models::BottleneckType bottleneck = models::BottleneckType::MEAN_POOL,
    models::AutoencoderMode mode = models::AutoencoderMode::RECONSTRUCTION,
    int forecast_horizon = 1,
    std::mt19937* gen = nullptr) {

    // Create default generator if none provided
    std::unique_ptr<std::mt19937> local_gen;
    if (!gen) {
        local_gen = std::make_unique<std::mt19937>(42);
        gen = local_gen.get();
    }

    models::AutoencoderConfig config;
    config.input_size = input_size;
    config.latent_size = latent_size;
    config.hidden_size = hidden_size;
    config.num_layers = num_layers;
    config.bottleneck_type = bottleneck;
    config.mode = mode;
    config.forecast_horizon = forecast_horizon;

    // Sensible defaults
    config.use_input_projection = true;
    config.internal_projection_size = std::min(32, std::max(input_size, 16));
    config.dropout = 0.1f;
    config.final_dropout = 0.1f;
    config.layer_norm = true;
    config.use_exponential_decay = true;

    return std::make_unique<models::TemporalAutoencoder>(config, *gen);
}

/**
 * Create a trainer with sensible defaults
 */
inline std::unique_ptr<training::OnlineTrainer> create_trainer(
    std::unique_ptr<models::TemporalAutoencoder> model,
    float learning_rate = 2e-3f,
    const std::string& optimizer = "adamw",
    const std::string& loss_type = "mse",
    std::mt19937* gen = nullptr) {

    std::unique_ptr<std::mt19937> local_gen;
    if (!gen) {
        local_gen = std::make_unique<std::mt19937>(42);
        gen = local_gen.get();
    }

    training::TrainingConfig config;
    config.learning_rate = learning_rate;
    config.optimizer = optimizer;
    config.loss_type = loss_type;
    config.grad_clip_norm = 5.0f;
    config.weight_decay = 1e-4f;
    config.tbptt_steps = 20;

    return std::make_unique<training::OnlineTrainer>(std::move(model), config, *gen);
}

/**
 * Create a batch trainer for offline training
 */
inline std::unique_ptr<training::BatchTrainer> create_batch_trainer(
    std::unique_ptr<models::TemporalAutoencoder> model,
    int max_epochs = 100,
    float learning_rate = 2e-3f,
    const std::string& optimizer = "adamw",
    std::mt19937* gen = nullptr) {

    std::unique_ptr<std::mt19937> local_gen;
    if (!gen) {
        local_gen = std::make_unique<std::mt19937>(42);
        gen = local_gen.get();
    }

    training::TrainingConfig config;
    config.max_epochs = max_epochs;
    config.learning_rate = learning_rate;
    config.optimizer = optimizer;
    config.loss_type = "mse";
    config.validate_every = 10;
    config.log_every = 10;
    config.use_early_stopping = true;
    config.patience = 20;

    return std::make_unique<training::BatchTrainer>(std::move(model), config, *gen);
}

} // namespace factory

namespace examples {

// ============================================================================
// BASIC LAYER EXAMPLES
// ============================================================================

/**
 * Example: Basic layer usage and gradient checking
 */
inline void example_basic_layers() {
    std::cout << "\n=== Basic Layer Examples ===" << std::endl;

    std::mt19937 gen(42);

    // Create a simple linear layer
    auto linear = std::make_unique<layers::Linear>(10, 5, true, &gen);
    std::cout << "Created " << linear->name() << std::endl;

    // Test forward pass
    Eigen::MatrixXf input = Eigen::MatrixXf::Random(3, 10);  // Batch size 3
    Context ctx(linear.get());
    Eigen::MatrixXf output = linear->forward(input, ctx);

    std::cout << "Input shape: (" << input.rows() << ", " << input.cols() << ")" << std::endl;
    std::cout << "Output shape: (" << output.rows() << ", " << output.cols() << ")" << std::endl;

    // Gradient checking
    std::cout << "Performing gradient check..." << std::endl;
    auto grad_result = checkgrad::check_gradients(*linear, input, 1e-5f, 1e-3f, 1e-8f, false);

    if (grad_result.passed) {
        std::cout << "✓ Gradient check PASSED" << std::endl;
    } else {
        std::cout << "✗ Gradient check FAILED" << std::endl;
        std::cout << "Max relative error: " << grad_result.max_relative_error << std::endl;
    }

    // Create a sequential model
    auto sequential = std::make_unique<Sequential>();
    sequential->add(std::make_unique<layers::Linear>(10, 20, true, &gen));
    sequential->add(std::make_unique<layers::ReLU>());
    sequential->add(std::make_unique<layers::LayerNorm>(20));
    sequential->add(std::make_unique<layers::Dropout>(0.1f, gen));
    sequential->add(std::make_unique<layers::Linear>(20, 5, true, &gen));

    std::cout << "\nCreated sequential model:" << std::endl;
    sequential->print_structure();
}

// ============================================================================
// TEMPORAL MODEL EXAMPLES
// ============================================================================

/**
 * Example: Temporal RNN usage
 */
inline void example_temporal_rnn() {
    std::cout << "\n=== Temporal RNN Examples ===" << std::endl;

    std::mt19937 gen(42);

    // Create GRU-D cell
    auto gru_cell = std::make_unique<layers::GRUDCell>(8, 16, true, &gen);
    std::cout << "Created " << gru_cell->name() << std::endl;

    // Test single step
    Eigen::MatrixXf input(2, 8);    // Batch size 2, input size 8
    Eigen::MatrixXf hidden(2, 16);  // Batch size 2, hidden size 16
    Eigen::MatrixXf dt(2, 1);       // Time differences

    input.setRandom();
    hidden.setZero();
    dt.setConstant(1.0f);

    auto [new_hidden, cell_ctx] = gru_cell->forward_cell(input, hidden, dt);

    std::cout << "Input shape: (" << input.rows() << ", " << input.cols() << ")" << std::endl;
    std::cout << "Hidden shape: (" << hidden.rows() << ", " << hidden.cols() << ")" << std::endl;
    std::cout << "New hidden shape: (" << new_hidden.rows() << ", " << new_hidden.cols() << ")" << std::endl;

    // Create temporal RNN layer
    auto temporal_layer = std::make_unique<layers::TemporalRNNLayer>(
        8, 16, 0, true, 0.1f, true, &gen);
    std::cout << "Created " << temporal_layer->name() << std::endl;
}

// ============================================================================
// AUTOENCODER EXAMPLES
// ============================================================================

/**
 * Example: Complete autoencoder usage
 */
inline void example_autoencoder() {
    std::cout << "\n=== Autoencoder Examples ===" << std::endl;

    std::mt19937 gen(42);

    // Create autoencoder for reconstruction
    auto autoencoder = factory::create_autoencoder(
        12,  // input_size
        8,   // latent_size
        32,  // hidden_size
        2,   // num_layers
        models::BottleneckType::MEAN_POOL,
        models::AutoencoderMode::RECONSTRUCTION,
        1,   // forecast_horizon (unused for reconstruction)
        &gen
    );

    std::cout << "Created autoencoder:" << std::endl;
    autoencoder->print_structure();
    std::cout << "Total parameters: " << autoencoder->num_parameters() << std::endl;

    // Create sample data
    int batch_size = 4;
    int seq_len = 10;
    int input_size = 12;

    std::vector<Eigen::MatrixXf> X_seq;
    std::vector<Eigen::MatrixXf> dt_seq;
    std::vector<std::optional<Eigen::MatrixXf>> mask_seq;

    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::uniform_real_distribution<float> dt_dist(0.5f, 2.0f);

    for (int t = 0; t < seq_len; ++t) {
        // Random input
        Eigen::MatrixXf x_t(batch_size, input_size);
        for (int i = 0; i < batch_size; ++i) {
            for (int j = 0; j < input_size; ++j) {
                x_t(i, j) = dist(gen);
            }
        }
        X_seq.push_back(x_t);

        // Random time differences
        Eigen::MatrixXf dt_t(batch_size, 1);
        for (int i = 0; i < batch_size; ++i) {
            dt_t(i, 0) = dt_dist(gen);
        }
        dt_seq.push_back(dt_t);

        // Random mask (80% observed)
        Eigen::MatrixXf mask_t(batch_size, input_size);
        std::bernoulli_distribution mask_dist(0.8);
        for (int i = 0; i < batch_size; ++i) {
            for (int j = 0; j < input_size; ++j) {
                mask_t(i, j) = mask_dist(gen) ? 1.0f : 0.0f;
            }
        }
        mask_seq.push_back(mask_t);
    }

    // Forward pass
    std::cout << "\nPerforming forward pass..." << std::endl;
    Tape tape;
    auto result = autoencoder->forward_autoencoder(
        X_seq, dt_seq, mask_seq, {}, {}, tape);

    std::cout << "Output sequence shape: (" << result.output_sequence.rows()
              << ", " << result.output_sequence.cols() << ")" << std::endl;
    std::cout << "Latent shape: (" << result.latent.rows()
              << ", " << result.latent.cols() << ")" << std::endl;
    std::cout << "Tape operations: " << tape.size() << std::endl;

    // Test autograd
    std::cout << "Testing autograd..." << std::endl;
    Eigen::MatrixXf grad_output = Eigen::MatrixXf::Random(
        result.output_sequence.rows(), result.output_sequence.cols());

    autoencoder->zero_grad();
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

    if (has_gradients) {
        std::cout << "✓ Gradients computed successfully" << std::endl;
    } else {
        std::cout << "✗ No gradients found" << std::endl;
    }
}

// ============================================================================
// TRAINING EXAMPLES
// ============================================================================

/**
 * Example: Online training
 */
inline void example_online_training() {
    std::cout << "\n=== Online Training Examples ===" << std::endl;

    std::mt19937 gen(42);

    // Create autoencoder
    auto autoencoder = factory::create_autoencoder(8, 4, 16, 1,
        models::BottleneckType::LAST_HIDDEN,
        models::AutoencoderMode::RECONSTRUCTION, 1, &gen);

    // Create trainer
    auto trainer = factory::create_trainer(std::move(autoencoder), 1e-3f, "adamw", "mse", &gen);

    std::cout << "Created online trainer" << std::endl;
    std::cout << "Model parameters: " << trainer->get_model().num_parameters() << std::endl;

    // Simulate streaming data
    int batch_size = 2;
    int input_size = 8;
    int num_steps = 100;

    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::uniform_real_distribution<float> dt_dist(0.5f, 1.5f);

    float running_loss = 0.0f;
    int log_every = 20;

    for (int step = 0; step < num_steps; ++step) {
        // Generate random data
        Eigen::MatrixXf x_t(batch_size, input_size);
        Eigen::MatrixXf dt_t(batch_size, 1);

        for (int i = 0; i < batch_size; ++i) {
            for (int j = 0; j < input_size; ++j) {
                x_t(i, j) = dist(gen);
            }
            dt_t(i, 0) = dt_dist(gen);
        }

        // Online training step
        auto [loss, prediction] = trainer->step_online(x_t, dt_t, std::nullopt, x_t);

        running_loss = 0.9f * running_loss + 0.1f * loss;

        if (step % log_every == 0) {
            std::cout << "Step " << std::setw(3) << step
                      << ", Loss: " << std::fixed << std::setprecision(6) << running_loss
                      << ", Pred norm: " << std::setprecision(3) << prediction.norm() << std::endl;
        }
    }

    // Print final metrics
    const auto& metrics = trainer->get_train_metrics();
    std::cout << "\nFinal training metrics:" << std::endl;
    metrics.print();
}

/**
 * Example: Batch training
 */
inline void example_batch_training() {
    std::cout << "\n=== Batch Training Examples ===" << std::endl;

    std::mt19937 gen(42);

    // Create autoencoder
    auto autoencoder = factory::create_autoencoder(6, 3, 12, 1,
        models::BottleneckType::MEAN_POOL,
        models::AutoencoderMode::RECONSTRUCTION, 1, &gen);

    // Create batch trainer
    auto trainer = factory::create_batch_trainer(std::move(autoencoder), 20, 1e-3f, "adamw", &gen);

    std::cout << "Created batch trainer" << std::endl;

    // Generate synthetic dataset
    int num_train_batches = 50;
    int num_val_batches = 10;
    int batch_size = 4;
    int seq_len = 8;
    int input_size = 6;

    auto train_dataset = training::generate_synthetic_dataset(
        num_train_batches, batch_size, seq_len, input_size, gen);
    auto val_dataset = training::generate_synthetic_dataset(
        num_val_batches, batch_size, seq_len, input_size, gen);

    std::cout << "Generated dataset: " << train_dataset.size() << " training batches, "
              << val_dataset.size() << " validation batches" << std::endl;

    // Train the model
    trainer->train(train_dataset, val_dataset);

    // Print training history
    const auto& train_losses = trainer->get_train_losses();
    const auto& val_losses = trainer->get_val_losses();

    std::cout << "\nTraining completed!" << std::endl;
    std::cout << "Final train loss: " << train_losses.back() << std::endl;
    if (!val_losses.empty()) {
        std::cout << "Final val loss: " << val_losses.back() << std::endl;
    }
}

// ============================================================================
// COMPREHENSIVE EXAMPLE
// ============================================================================

/**
 * Comprehensive example showing all components
 */
inline void comprehensive_example() {
    std::cout << "\n=== Comprehensive Example ===" << std::endl;

    std::mt19937 gen(42);

    // 1. Create different types of autoencoders
    std::cout << "1. Creating different autoencoder configurations..." << std::endl;

    auto reconstruction_ae = factory::create_autoencoder(
        10, 5, 32, 2, models::BottleneckType::MEAN_POOL,
        models::AutoencoderMode::RECONSTRUCTION, 1, &gen);

    auto forecasting_ae = factory::create_autoencoder(
        10, 5, 32, 2, models::BottleneckType::LAST_HIDDEN,
        models::AutoencoderMode::FORECASTING, 5, &gen);

    std::cout << "   Reconstruction AE: " << reconstruction_ae->num_parameters() << " params" << std::endl;
    std::cout << "   Forecasting AE: " << forecasting_ae->num_parameters() << " params" << std::endl;

    // 2. Gradient checking
    std::cout << "\n2. Performing gradient checks..." << std::endl;

    // Create a simple linear layer for testing
    auto test_linear = std::make_unique<layers::Linear>(5, 3, true, &gen);
    Eigen::MatrixXf test_input = Eigen::MatrixXf::Random(2, 5);

    auto grad_result = checkgrad::check_gradients(*test_linear, test_input, 1e-5f, 1e-2f, 1e-8f, false);
    std::cout << "   Linear layer gradient check: " << (grad_result.passed ? "PASSED" : "FAILED") << std::endl;

    // 3. Compare different optimizers
    std::cout << "\n3. Testing different optimizers..." << std::endl;

    auto sgd_trainer = factory::create_trainer(
        factory::create_autoencoder(6, 3, 16, 1, models::BottleneckType::MEAN_POOL,
                                   models::AutoencoderMode::RECONSTRUCTION, 1, &gen),
        1e-3f, "sgd", "mse", &gen);

    auto adamw_trainer = factory::create_trainer(
        factory::create_autoencoder(6, 3, 16, 1, models::BottleneckType::MEAN_POOL,
                                   models::AutoencoderMode::RECONSTRUCTION, 1, &gen),
        1e-3f, "adamw", "mse", &gen);

    // Quick training comparison
    Eigen::MatrixXf sample_input = Eigen::MatrixXf::Random(2, 6);
    Eigen::MatrixXf sample_dt = Eigen::MatrixXf::Constant(2, 1, 1.0f);

    auto [sgd_loss, sgd_pred] = sgd_trainer->step_online(sample_input, sample_dt, std::nullopt, sample_input);
    auto [adamw_loss, adamw_pred] = adamw_trainer->step_online(sample_input, sample_dt, std::nullopt, sample_input);

    std::cout << "   SGD loss: " << sgd_loss << std::endl;
    std::cout << "   AdamW loss: " << adamw_loss << std::endl;

    // 4. Test different loss functions
    std::cout << "\n4. Testing different loss functions..." << std::endl;

    Eigen::MatrixXf pred = Eigen::MatrixXf::Random(3, 4);
    Eigen::MatrixXf target = Eigen::MatrixXf::Random(3, 4);

    loss::MSELoss mse_loss;
    loss::MAELoss mae_loss;
    loss::HuberLoss huber_loss(1.0f);

    float mse_val = mse_loss.forward(pred, target);
    float mae_val = mae_loss.forward(pred, target);
    float huber_val = huber_loss.forward(pred, target);

    std::cout << "   MSE loss: " << mse_val << std::endl;
    std::cout << "   MAE loss: " << mae_val << std::endl;
    std::cout << "   Huber loss: " << huber_val << std::endl;

    // 5. Memory and performance info
    std::cout << "\n5. System information:" << std::endl;
    std::cout << "   sizeof(Param): " << sizeof(Param) << " bytes" << std::endl;
    std::cout << "   sizeof(Context): " << sizeof(Context) << " bytes" << std::endl;
    std::cout << "   sizeof(Linear): " << sizeof(layers::Linear) << " bytes" << std::endl;

    std::cout << "\n✓ Comprehensive example completed successfully!" << std::endl;
}

} // namespace examples

// ============================================================================
// MAIN EXAMPLE RUNNER
// ============================================================================

/**
 * Run all examples
 */
inline void run_all_examples() {
    std::cout << "GRUD Autoencoder Framework Examples" << std::endl;
    std::cout << "====================================" << std::endl;

    try {
        examples::example_basic_layers();
        examples::example_temporal_rnn();
        examples::example_autoencoder();
        examples::example_online_training();
        examples::example_batch_training();
        examples::comprehensive_example();

        std::cout << "\n🎉 All examples completed successfully!" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "\n❌ Error running examples: " << e.what() << std::endl;
    }
}

} // namespace grud

// ============================================================================
// SIMPLE MAIN FUNCTION FOR TESTING
// ============================================================================

/*
// Uncomment to create standalone executable
int main() {
    grud::run_all_examples();
    return 0;
}
*/

#endif // GRUD_EXAMPLE_USAGE_H