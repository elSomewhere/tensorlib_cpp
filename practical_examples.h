//
// examples/practical_examples.cpp - Complete working examples
//

#include "../include/grud.h"
#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>

// ============================================================================
// EXAMPLE 1: BASIC RECONSTRUCTION AUTOENCODER
// ============================================================================

void example_basic_reconstruction() {
    std::cout << "\n=== Example 1: Basic Reconstruction Autoencoder ===" << std::endl;

    std::mt19937 gen(42);

    // Create model
    auto model = grud::quick::reconstruction_model(6, 3, 16, &gen);
    std::cout << "Created model with " << model->num_parameters() << " parameters" << std::endl;

    // Generate synthetic sine wave data
    auto data = grud::utils::SequenceGenerator::generate_sinusoidal(4, 15, 6, 0.1f, gen);
    std::cout << "Generated " << data.sequence_length() << " timesteps of data" << std::endl;

    // Create trainer
    auto trainer = grud::factory::create_trainer(std::move(model), 1e-3f, "adamw", "mse", &gen);

    // Training loop
    std::cout << "Training..." << std::endl;
    for (int epoch = 0; epoch < 50; ++epoch) {
        auto [loss, predictions] = trainer->process_batch(data);

        if (epoch % 10 == 0) {
            std::cout << "Epoch " << std::setw(3) << epoch
                      << ", Loss: " << std::fixed << std::setprecision(6) << loss << std::endl;
        }
    }

    // Final evaluation
    const auto& metrics = trainer->get_train_metrics();
    std::cout << "\nFinal training metrics:" << std::endl;
    metrics.print("  ");

    std::cout << "✓ Basic reconstruction example completed" << std::endl;
}

// ============================================================================
// EXAMPLE 2: FORECASTING WITH MISSING VALUES
// ============================================================================

void example_forecasting_with_missing_values() {
    std::cout << "\n=== Example 2: Forecasting with Missing Values ===" << std::endl;

    std::mt19937 gen(42);

    // Create forecasting model (predict 3 steps ahead)
    auto model = grud::quick::forecasting_model(8, 3, 4, 24, &gen);
    std::cout << "Created forecasting model with " << model->num_parameters() << " parameters" << std::endl;

    // Generate autoregressive data
    std::vector<float> ar_coeffs = {0.8f, -0.3f, 0.1f};
    auto full_data = grud::utils::SequenceGenerator::generate_autoregressive(2, 20, 8, ar_coeffs, 0.05f, gen);

    // Split into history (first 15 steps) and target (next 3 steps)
    grud::training::Batch history_data;
    history_data.X_seq.assign(full_data.X_seq.begin(), full_data.X_seq.begin() + 15);
    history_data.dt_seq.assign(full_data.dt_seq.begin(), full_data.dt_seq.begin() + 15);
    history_data.mask_seq.assign(full_data.mask_seq.begin(), full_data.mask_seq.begin() + 15);

    // Add missing values to history (20% missing rate)
    for (auto& mask : history_data.mask_seq) {
        mask = grud::utils::MaskGenerator::random_mask(2, 8, 0.2f, gen);
    }

    // Create target (stacked future timesteps)
    Eigen::MatrixXf target(3 * 2, 8);  // 3 timesteps * 2 batch size
    for (int t = 0; t < 3; ++t) {
        target.block(t * 2, 0, 2, 8) = full_data.X_seq[15 + t];
    }
    history_data.target = target;

    std::cout << "Prepared data with " << history_data.sequence_length() << " history steps" << std::endl;

    // Create trainer
    auto trainer = grud::factory::create_trainer(std::move(model), 2e-3f, "adamw", "mse", &gen);

    // Training
    std::cout << "Training forecasting model..." << std::endl;
    for (int epoch = 0; epoch < 100; ++epoch) {
        auto [loss, predictions] = trainer->process_batch(history_data);

        if (epoch % 20 == 0) {
            std::cout << "Epoch " << std::setw(3) << epoch
                      << ", Loss: " << std::fixed << std::setprecision(6) << loss << std::endl;
        }
    }

    // Evaluate final forecast
    trainer->get_model().set_training(false);
    grud::Tape eval_tape;
    auto result = trainer->get_model().forward_autoencoder(
        history_data.X_seq, history_data.dt_seq, history_data.mask_seq, {}, {}, eval_tape);

    auto forecast_metrics = grud::utils::MetricsCalculator::calculate_metrics(
        result.output_sequence, target, std::nullopt, 2, 3);

    std::cout << "\nForecast evaluation:" << std::endl;
    grud::utils::MetricsCalculator::print_metrics(forecast_metrics, "  ");

    std::cout << "✓ Forecasting with missing values example completed" << std::endl;
}

// ============================================================================
// EXAMPLE 3: STREAMING ONLINE LEARNING
// ============================================================================

void example_streaming_online_learning() {
    std::cout << "\n=== Example 3: Streaming Online Learning ===" << std::endl;

    std::mt19937 gen(42);

    // Create small model for fast online learning
    auto model = grud::quick::reconstruction_model(4, 2, 8, &gen);
    auto trainer = grud::factory::create_trainer(std::move(model), 5e-3f, "adamw", "huber", &gen);

    std::cout << "Starting online learning simulation..." << std::endl;

    // Simulate streaming data
    std::normal_distribution<float> noise_dist(0.0f, 0.1f);
    std::uniform_real_distribution<float> dt_dist(0.8f, 1.2f);

    float running_loss = 0.0f;
    const int total_steps = 500;
    const int log_interval = 50;

    for (int step = 0; step < total_steps; ++step) {
        // Generate current timestep data
        Eigen::MatrixXf x_t(1, 4);
        Eigen::MatrixXf dt_t(1, 1);

        // Simple sinusoidal pattern with noise
        float time = step * 0.1f;
        for (int i = 0; i < 4; ++i) {
            float freq = 0.5f + i * 0.2f;
            x_t(0, i) = std::sin(freq * time) + noise_dist(gen);
        }
        dt_t(0, 0) = dt_dist(gen);

        // Simulate random missing values (10% probability)
        std::optional<Eigen::MatrixXf> mask_t = std::nullopt;
        if (std::bernoulli_distribution(0.1)(gen)) {
            mask_t = grud::utils::MaskGenerator::random_mask(1, 4, 0.3f, gen);
        }

        // Online learning step
        auto [loss, prediction] = trainer->step_online(x_t, dt_t, mask_t, x_t);

        // Update running average
        running_loss = 0.95f * running_loss + 0.05f * loss;

        if (step % log_interval == 0) {
            float mse = (prediction - x_t).array().square().mean();
            std::cout << "Step " << std::setw(4) << step
                      << ", Running Loss: " << std::fixed << std::setprecision(6) << running_loss
                      << ", MSE: " << std::setprecision(6) << mse << std::endl;
        }
    }

    // Final metrics
    const auto& final_metrics = trainer->get_train_metrics();
    std::cout << "\nFinal online learning metrics:" << std::endl;
    final_metrics.print("  ");

    std::cout << "✓ Streaming online learning example completed" << std::endl;
}

// ============================================================================
// EXAMPLE 4: MODEL PERSISTENCE AND CHECKPOINTING
// ============================================================================

void example_model_persistence() {
    std::cout << "\n=== Example 4: Model Persistence and Checkpointing ===" << std::endl;

    std::mt19937 gen(42);

    // Create and train a model
    auto model = grud::quick::reconstruction_model(5, 3, 12, &gen);
    std::cout << "Created model with " << model->num_parameters() << " parameters" << std::endl;

    // Quick training
    auto data = grud::utils::SequenceGenerator::generate_random_walk(3, 10, 5, 0.5f, gen);
    auto trainer = grud::factory::create_trainer(std::move(model), 1e-3f, "adamw", "mse", &gen);

    std::cout << "Training model..." << std::endl;
    for (int epoch = 0; epoch < 20; ++epoch) {
        trainer->process_batch(data);
    }

    float initial_loss = trainer->get_train_metrics().loss;
    std::cout << "Initial model loss: " << initial_loss << std::endl;

    // Save model
    std::string model_file = "temp_model.grud";
    grud::io::save_model(trainer->get_model(), model_file);
    std::cout << "Saved model to " << model_file << std::endl;

    // Create checkpoint manager and save complete checkpoint
    grud::io::CheckpointManager checkpoint_manager("./temp_checkpoints", 3);
    grud::io::TrainingState training_state;
    training_state.epoch = 20;
    training_state.step = trainer->get_step_count();
    training_state.best_loss = initial_loss;
    training_state.train_losses = {initial_loss};

    checkpoint_manager.save_checkpoint(trainer->get_model(), training_state, "test_checkpoint");
    std::cout << "Saved complete checkpoint" << std::endl;

    // Load model and verify
    auto loaded_model = grud::io::load_model(model_file, gen);
    std::cout << "Loaded model with " << loaded_model->num_parameters() << " parameters" << std::endl;

    // Test loaded model
    grud::Tape tape;
    auto original_result = trainer->get_model().forward_autoencoder(
        data.X_seq, data.dt_seq, data.mask_seq, {}, {}, tape);

    tape.clear();
    auto loaded_result = loaded_model->forward_autoencoder(
        data.X_seq, data.dt_seq, data.mask_seq, {}, {}, tape);

    // Verify outputs are identical
    float output_diff = (original_result.output_sequence - loaded_result.output_sequence).norm();
    float latent_diff = (original_result.latent - loaded_result.latent).norm();

    std::cout << "Output difference: " << std::scientific << output_diff << std::endl;
    std::cout << "Latent difference: " << std::scientific << latent_diff << std::endl;

    if (output_diff < 1e-6f && latent_diff < 1e-6f) {
        std::cout << "✓ Model serialization verified successfully" << std::endl;
    } else {
        std::cout << "✗ Model serialization verification failed" << std::endl;
    }

    // Export human-readable summary
    grud::io::ModelExporter::export_summary(*loaded_model, "temp_model_summary.txt");
    std::cout << "Exported model summary to temp_model_summary.txt" << std::endl;

    // Cleanup
    std::remove(model_file.c_str());
    std::remove("temp_model_summary.txt");

    std::cout << "✓ Model persistence example completed" << std::endl;
}

// ============================================================================
// EXAMPLE 5: DATA PREPROCESSING PIPELINE
// ============================================================================

void example_data_preprocessing() {
    std::cout << "\n=== Example 5: Data Preprocessing Pipeline ===" << std::endl;

    std::mt19937 gen(42);

    // Generate raw data with different scales and patterns
    std::cout << "Generating raw data..." << std::endl;
    auto raw_batch1 = grud::utils::SequenceGenerator::generate_sinusoidal(5, 20, 3, 0.2f, gen);
    auto raw_batch2 = grud::utils::SequenceGenerator::generate_random_walk(5, 20, 3, 2.0f, gen);
    auto raw_batch3 = grud::utils::SequenceGenerator::generate_autoregressive(5, 20, 3, {0.7f, -0.2f}, 0.1f, gen);

    // Combine into dataset
    std::vector<Eigen::MatrixXf> all_data;
    for (const auto& batch : {raw_batch1, raw_batch2, raw_batch3}) {
        for (const auto& x : batch.X_seq) {
            all_data.push_back(x);
        }
    }

    std::cout << "Combined dataset size: " << all_data.size() << " timesteps" << std::endl;

    // Analyze raw data statistics
    Eigen::VectorXf raw_means = Eigen::VectorXf::Zero(3);
    Eigen::VectorXf raw_stds = Eigen::VectorXf::Zero(3);
    int total_samples = 0;

    for (const auto& data : all_data) {
        raw_means += data.colwise().sum().transpose();
        total_samples += data.rows();
    }
    raw_means /= total_samples;

    for (const auto& data : all_data) {
        for (int i = 0; i < data.rows(); ++i) {
            Eigen::VectorXf diff = data.row(i).transpose() - raw_means;
            raw_stds += diff.array().square();
        }
    }
    raw_stds /= (total_samples - 1);
    raw_stds = raw_stds.array().sqrt();

    std::cout << "Raw data statistics:" << std::endl;
    for (int i = 0; i < 3; ++i) {
        std::cout << "  Feature " << i << ": mean=" << std::fixed << std::setprecision(3)
                  << raw_means(i) << ", std=" << raw_stds(i) << std::endl;
    }

    // Apply standardization
    std::cout << "\nApplying standardization..." << std::endl;
    grud::utils::StandardScaler scaler;
    auto normalized_data = scaler.fit_transform(all_data);

    // Verify normalization
    Eigen::VectorXf norm_means = Eigen::VectorXf::Zero(3);
    Eigen::VectorXf norm_stds = Eigen::VectorXf::Zero(3);
    total_samples = 0;

    for (const auto& data : normalized_data) {
        norm_means += data.colwise().sum().transpose();
        total_samples += data.rows();
    }
    norm_means /= total_samples;

    for (const auto& data : normalized_data) {
        for (int i = 0; i < data.rows(); ++i) {
            Eigen::VectorXf diff = data.row(i).transpose() - norm_means;
            norm_stds += diff.array().square();
        }
    }
    norm_stds /= (total_samples - 1);
    norm_stds = norm_stds.array().sqrt();

    std::cout << "Normalized data statistics:" << std::endl;
    for (int i = 0; i < 3; ++i) {
        std::cout << "  Feature " << i << ": mean=" << std::fixed << std::setprecision(6)
                  << norm_means(i) << ", std=" << norm_stds(i) << std::endl;
    }

    // Create batches with missing values
    std::cout << "\nCreating batches with missing value patterns..." << std::endl;
    std::vector<grud::training::Batch> processed_dataset;

    for (size_t start = 0; start < normalized_data.size() - 9; start += 10) {
        grud::training::Batch batch;

        for (int t = 0; t < 10; ++t) {
            batch.X_seq.push_back(normalized_data[start + t]);
            batch.dt_seq.push_back(Eigen::MatrixXf::Ones(5, 1));

            // Add different missing patterns
            if (t % 3 == 0) {
                // Random missing (20% rate)
                batch.mask_seq.push_back(grud::utils::MaskGenerator::random_mask(5, 3, 0.2f, gen));
            } else {
                // All observed
                batch.mask_seq.push_back(Eigen::MatrixXf::Ones(5, 3));
            }
        }

        processed_dataset.push_back(batch);
    }

    std::cout << "Created " << processed_dataset.size() << " batches" << std::endl;

    // Split dataset
    auto splits = grud::utils::DatasetSplitter::train_val_test_split(processed_dataset, 0.6f, 0.2f, gen);
    std::cout << "Dataset split: " << splits.train.size() << " train, "
              << splits.val.size() << " val, " << splits.test.size() << " test" << std::endl;

    // Train model on processed data
    std::cout << "\nTraining on preprocessed data..." << std::endl;
    auto model = grud::quick::reconstruction_model(3, 2, 8, &gen);
    auto trainer = grud::factory::create_trainer(std::move(model), 1e-3f, "adamw", "mse", &gen);

    for (int epoch = 0; epoch < 20; ++epoch) {
        float epoch_loss = 0.0f;
        for (const auto& batch : splits.train) {
            auto [loss, _] = trainer->process_batch(batch);
            epoch_loss += loss;
        }
        epoch_loss /= splits.train.size();

        if (epoch % 5 == 0) {
            std::cout << "Epoch " << epoch << ", Training Loss: "
                      << std::fixed << std::setprecision(6) << epoch_loss << std::endl;
        }
    }

    // Test inverse transformation
    std::cout << "\nTesting inverse transformation..." << std::endl;
    if (!splits.test.empty()) {
        grud::Tape tape;
        auto result = trainer->get_model().forward_autoencoder(
            splits.test[0].X_seq, splits.test[0].dt_seq, splits.test[0].mask_seq, {}, {}, tape);

        // Transform back to original scale
        auto original_scale_output = scaler.inverse_transform(result.output_sequence);
        std::cout << "Output shape after inverse transform: ("
                  << original_scale_output.rows() << ", " << original_scale_output.cols() << ")" << std::endl;
    }

    std::cout << "✓ Data preprocessing pipeline example completed" << std::endl;
}

// ============================================================================
// EXAMPLE 6: CUSTOM LAYER DEVELOPMENT
// ============================================================================

// Custom attention layer example
class SimpleAttention : public grud::Module {
private:
    grud::Param attention_weights;
    int feature_size_;

public:
    SimpleAttention(int feature_size, std::mt19937& gen)
        : feature_size_(feature_size), attention_weights(1, feature_size, "attention_weights") {
        attention_weights.init_normal(gen, 0.0f, 0.1f);
    }

    Eigen::MatrixXf forward(const Eigen::MatrixXf& input, grud::Context& ctx) override {
        // input: (batch_size, feature_size)
        // Compute attention scores
        Eigen::VectorXf scores = input * attention_weights.value.transpose();

        // Softmax
        float max_score = scores.maxCoeff();
        Eigen::VectorXf exp_scores = (scores.array() - max_score).exp();
        float sum_exp = exp_scores.sum();
        Eigen::VectorXf softmax_scores = exp_scores / sum_exp;

        // Weighted sum
        Eigen::MatrixXf output = Eigen::MatrixXf::Zero(1, feature_size_);
        for (int i = 0; i < input.rows(); ++i) {
            output += softmax_scores(i) * input.row(i);
        }

        // Save for backward
        ctx.save_for_backward(input);
        ctx.save_for_backward(softmax_scores);

        return output;
    }

    Eigen::MatrixXf backward(const Eigen::MatrixXf& grad_output, const grud::Context& ctx) override {
        const Eigen::MatrixXf& input = ctx.get_saved(0);
        const Eigen::VectorXf& softmax_scores = ctx.get_saved(1);

        // Gradient w.r.t. attention weights
        Eigen::VectorXf weighted_input = Eigen::VectorXf::Zero(feature_size_);
        for (int i = 0; i < input.rows(); ++i) {
            weighted_input += softmax_scores(i) * input.row(i).transpose();
        }
        attention_weights.grad += grad_output.transpose() * weighted_input.transpose();

        // Gradient w.r.t. input (simplified)
        Eigen::MatrixXf grad_input = Eigen::MatrixXf::Zero(input.rows(), input.cols());
        for (int i = 0; i < input.rows(); ++i) {
            grad_input.row(i) = softmax_scores(i) * grad_output;
        }

        return grad_input;
    }

    std::vector<grud::Param*> params() override {
        return {&attention_weights};
    }

    std::string name() const override {
        return "SimpleAttention(" + std::to_string(feature_size_) + ")";
    }
};

void example_custom_layer() {
    std::cout << "\n=== Example 6: Custom Layer Development ===" << std::endl;

    std::mt19937 gen(42);

    // Create custom attention layer
    auto attention = std::make_unique<SimpleAttention>(4, gen);
    std::cout << "Created custom attention layer: " << attention->name() << std::endl;

    // Test forward pass
    Eigen::MatrixXf input = Eigen::MatrixXf::Random(3, 4);  // 3 sequence elements, 4 features
    grud::Context ctx(attention.get());
    Eigen::MatrixXf output = attention->forward(input, ctx);

    std::cout << "Input shape: (" << input.rows() << ", " << input.cols() << ")" << std::endl;
    std::cout << "Output shape: (" << output.rows() << ", " << output.cols() << ")" << std::endl;

    // Gradient checking
    std::cout << "Performing gradient check..." << std::endl;
    auto grad_result = grud::checkgrad::check_gradients(*attention, input, 1e-5f, 1e-2f, 1e-8f, false);

    if (grad_result.passed) {
        std::cout << "✓ Custom layer gradient check passed" << std::endl;
    } else {
        std::cout << "✗ Custom layer gradient check failed" << std::endl;
        std::cout << "  Max relative error: " << grad_result.max_relative_error << std::endl;
    }

    // Use in a larger network
    auto sequential = std::make_unique<grud::Sequential>();
    sequential->add(std::make_unique<grud::Linear>(4, 8, true, &gen));
    sequential->add(std::make_unique<grud::ReLU>());
    sequential->add(std::make_unique<SimpleAttention>(8, gen));

    std::cout << "\nCreated network with custom layer:" << std::endl;
    sequential->print_structure();

    std::cout << "✓ Custom layer development example completed" << std::endl;
}

// ============================================================================
// MAIN FUNCTION - RUN ALL EXAMPLES
// ============================================================================

int main() {
    std::cout << "GRU-D Autoencoder Framework - Practical Examples" << std::endl;
    std::cout << "=================================================" << std::endl;

    // Framework self-test
    if (!grud::diagnostics::self_test()) {
        std::cerr << "❌ Framework self-test failed!" << std::endl;
        return 1;
    }

    try {
        // Run all examples
        example_basic_reconstruction();
        example_forecasting_with_missing_values();
        example_streaming_online_learning();
        example_model_persistence();
        example_data_preprocessing();
        example_custom_layer();

        std::cout << "\n🎉 All practical examples completed successfully!" << std::endl;
        std::cout << "\nFramework capabilities demonstrated:" << std::endl;
        std::cout << "  ✓ Basic reconstruction autoencoders" << std::endl;
        std::cout << "  ✓ Multi-step forecasting with missing values" << std::endl;
        std::cout << "  ✓ Streaming online learning" << std::endl;
        std::cout << "  ✓ Model serialization and checkpointing" << std::endl;
        std::cout << "  ✓ Data preprocessing pipelines" << std::endl;
        std::cout << "  ✓ Custom layer development" << std::endl;

        // Print final system info
        std::cout << "\n--- System Information ---" << std::endl;
        grud::diagnostics::system_info();

    } catch (const std::exception& e) {
        std::cerr << "❌ Error running examples: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}