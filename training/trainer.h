//
// grud/training/trainer.h - Training system and utilities
//

#ifndef GRUD_TRAINING_TRAINER_H
#define GRUD_TRAINING_TRAINER_H

#include "../core/module.h"
#include "../core/tape.h"
#include "../optim/optimizers.h"
#include "../models/autoencoder.h"
#include <memory>
#include <vector>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <deque>

namespace grud {
namespace training {

// ============================================================================
// TRAINING CONFIGURATION
// ============================================================================

struct TrainingConfig {
    // Training parameters
    int max_epochs = 100;
    int batch_size = 32;
    int tbptt_steps = 20;  // Truncated BPTT window size

    // Learning parameters
    float learning_rate = 2e-3f;
    float weight_decay = 1e-4f;
    float grad_clip_norm = 5.0f;

    // Optimizer settings
    std::string optimizer = "adamw";  // "sgd", "adamw"
    float momentum = 0.9f;  // For SGD
    float beta1 = 0.9f;     // For Adam/AdamW
    float beta2 = 0.999f;   // For Adam/AdamW
    float eps = 1e-8f;      // For Adam/AdamW

    // Loss settings
    std::string loss_type = "mse";  // "mse", "mae", "huber"
    float huber_delta = 1.0f;

    // Validation and logging
    int validate_every = 10;
    int log_every = 100;
    bool save_best_model = true;
    std::string checkpoint_dir = "./checkpoints";

    // Early stopping
    bool use_early_stopping = true;
    int patience = 20;
    float min_delta = 1e-6f;

    // Learning rate scheduling
    bool use_lr_scheduler = false;
    std::string scheduler_type = "step";  // "step", "exponential", "cosine"
    int step_size = 30;
    float gamma = 0.1f;
    int T_max = 100;  // For cosine annealing
};

// ============================================================================
// METRICS COMPUTATION
// ============================================================================

struct Metrics {
    float loss = 0.0f;
    float mse = 0.0f;
    float mae = 0.0f;
    float rmse = 0.0f;
    size_t num_samples = 0;

    void reset() {
        loss = mse = mae = rmse = 0.0f;
        num_samples = 0;
    }

    void update(float loss_val, const Eigen::MatrixXf& predictions,
               const Eigen::MatrixXf& targets,
               const std::optional<Eigen::MatrixXf>& mask = std::nullopt) {

        float valid_elements = mask.has_value() ? mask.value().sum() :
                              static_cast<float>(predictions.size());

        if (valid_elements == 0) return;

        Eigen::MatrixXf diff = predictions - targets;
        Eigen::MatrixXf squared_diff = diff.array().square();
        Eigen::MatrixXf abs_diff = diff.array().abs();

        if (mask.has_value()) {
            squared_diff.array() *= mask.value().array();
            abs_diff.array() *= mask.value().array();
        }

        float batch_mse = squared_diff.sum() / valid_elements;
        float batch_mae = abs_diff.sum() / valid_elements;

        // Running average
        float n = static_cast<float>(num_samples);
        loss = (loss * n + loss_val) / (n + 1);
        mse = (mse * n + batch_mse) / (n + 1);
        mae = (mae * n + batch_mae) / (n + 1);
        rmse = std::sqrt(mse);

        num_samples++;
    }

    void print(const std::string& prefix = "") const {
        std::cout << prefix << "Loss: " << std::fixed << std::setprecision(6) << loss
                  << ", MSE: " << mse
                  << ", MAE: " << mae
                  << ", RMSE: " << rmse
                  << " (samples: " << num_samples << ")" << std::endl;
    }
};

// ============================================================================
// BATCH DATA STRUCTURE
// ============================================================================

struct Batch {
    std::vector<Eigen::MatrixXf> X_seq;
    std::vector<Eigen::MatrixXf> dt_seq;
    std::vector<std::optional<Eigen::MatrixXf>> mask_seq;
    std::optional<Eigen::MatrixXf> target;

    bool empty() const {
        return X_seq.empty();
    }

    size_t sequence_length() const {
        return X_seq.size();
    }

    int batch_size() const {
        return X_seq.empty() ? 0 : X_seq[0].rows();
    }
};

// ============================================================================
// ONLINE TRAINER FOR STREAMING DATA
// ============================================================================

class OnlineTrainer {
private:
    std::unique_ptr<models::TemporalAutoencoder> model_;
    std::unique_ptr<optim::Optimizer> optimizer_;
    std::unique_ptr<loss::Loss> loss_fn_;
    std::unique_ptr<optim::LRScheduler> scheduler_;

    TrainingConfig config_;
    std::mt19937& gen_;

    // Streaming state
    std::vector<Eigen::MatrixXf> h_encoder_;
    std::vector<Eigen::MatrixXf> h_decoder_;

    // TBPTT window
    std::deque<Eigen::MatrixXf> window_X_;
    std::deque<Eigen::MatrixXf> window_dt_;
    std::deque<std::optional<Eigen::MatrixXf>> window_mask_;
    std::deque<Eigen::MatrixXf> window_targets_;

    // Training state
    int step_count_;
    Metrics train_metrics_;
    Metrics val_metrics_;



public:
    OnlineTrainer(std::unique_ptr<models::TemporalAutoencoder> model,
                  const TrainingConfig& config,
                  std::mt19937& gen)
        : model_(std::move(model)), config_(config), gen_(gen),
          step_count_(0), best_val_loss_(std::numeric_limits<float>::infinity()),
          epochs_without_improvement_(0) {

        setup_optimizer();
        setup_loss_function();
        setup_scheduler();
        reset_streaming_state();
    }

    /**
     * Process a single timestep of streaming data
     */
    std::pair<float, Eigen::MatrixXf> step_online(
        const Eigen::MatrixXf& x_t,
        const Eigen::MatrixXf& dt_t,
        const std::optional<Eigen::MatrixXf>& mask_t = std::nullopt,
        const std::optional<Eigen::MatrixXf>& target_t = std::nullopt) {

        // Update streaming state dimensions if needed
        int batch_size = x_t.rows();
        if (h_encoder_.empty() || h_encoder_[0].rows() != batch_size) {
            reset_streaming_state(batch_size);
        }

        // Add to TBPTT window
        window_X_.push_back(x_t);
        window_dt_.push_back(dt_t);
        window_mask_.push_back(mask_t);
        if (target_t.has_value()) {
            window_targets_.push_back(target_t.value());
        }

        // Maintain window size
        while (static_cast<int>(window_X_.size()) > config_.tbptt_steps) {
            window_X_.pop_front();
            window_dt_.pop_front();
            window_mask_.pop_front();
            if (!window_targets_.empty()) {
                window_targets_.pop_front();
            }
        }

        // Skip training if window too small
        if (static_cast<int>(window_X_.size()) < 2) {
            return {0.0f, x_t};  // Return input as prediction
        }

        // Convert to vectors
        std::vector<Eigen::MatrixXf> X_seq(window_X_.begin(), window_X_.end());
        std::vector<Eigen::MatrixXf> dt_seq(window_dt_.begin(), window_dt_.end());
        std::vector<std::optional<Eigen::MatrixXf>> mask_seq(window_mask_.begin(), window_mask_.end());

        // Prepare target
        std::optional<Eigen::MatrixXf> target = std::nullopt;
        if (!window_targets_.empty()) {
            if (model_->config.mode == models::AutoencoderMode::RECONSTRUCTION) {
                // Stack all targets for reconstruction
                int seq_len = window_targets_.size();
                target = Eigen::MatrixXf(seq_len * batch_size, model_->config.input_size);
                for (int t = 0; t < seq_len; ++t) {
                    target.value().block(t * batch_size, 0, batch_size, model_->config.input_size) =
                        window_targets_[t];
                }
            } else {
                // Use last target for forecasting
                target = window_targets_.back();
            }
        }

        // Forward pass
        model_->set_training(true);
        Tape tape;
        auto result = model_->forward_autoencoder(
            X_seq, dt_seq, mask_seq, h_encoder_, h_decoder_, tape);

        // Compute loss
        float loss_val = 0.0f;
        if (target.has_value()) {
            loss_val = loss_fn_->forward(result.output_sequence, target.value());

            // Backward pass
            auto all_params = model_->all_parameters();
            optimizer_->zero_grad(all_params);

            Eigen::MatrixXf grad_output = loss_fn_->backward(result.output_sequence, target.value());
            autograd::backward(tape, grad_output);

            // Gradient clipping
            if (config_.grad_clip_norm > 0.0f) {
                clip_gradients(all_params, config_.grad_clip_norm);
            }

            // Optimizer step
            optimizer_->step(all_params);
            step_count_++;

            // Update metrics
            train_metrics_.update(loss_val, result.output_sequence, target.value());
        }

        // Update streaming states (detach from gradients)
        h_encoder_ = result.encoder_final_hidden;
        h_decoder_ = result.decoder_final_hidden;

        // Extract prediction
        Eigen::MatrixXf prediction;
        if (model_->config.mode == models::AutoencoderMode::RECONSTRUCTION) {
            // Get last timestep reconstruction
            int seq_len = result.output_sequence.rows() / batch_size;
            if (seq_len > 0) {
                prediction = result.output_sequence.block(
                    (seq_len - 1) * batch_size, 0, batch_size, result.output_sequence.cols());
            } else {
                prediction = x_t;  // Fallback
            }
        } else {
            prediction = result.output_sequence;
        }

        return {loss_val, prediction};
    }

    /**
     * Process a batch of data
     */
    std::pair<float, std::vector<Eigen::MatrixXf>> process_batch(const Batch& batch) {
        if (batch.empty()) {
            return {0.0f, {}};
        }

        int batch_size = batch.batch_size();
        reset_streaming_state(batch_size);

        std::vector<Eigen::MatrixXf> predictions;
        float total_loss = 0.0f;
        int valid_steps = 0;

        for (size_t t = 0; t < batch.sequence_length(); ++t) {
            std::optional<Eigen::MatrixXf> target_t = std::nullopt;
            if (batch.target.has_value()) {
                // Extract target for this timestep
                target_t = batch.X_seq[t];  // For reconstruction mode
            }

            std::optional<Eigen::MatrixXf> mask_t = std::nullopt;
            if (t < batch.mask_seq.size()) {
                mask_t = batch.mask_seq[t];
            }

            auto [loss, pred] = step_online(batch.X_seq[t], batch.dt_seq[t], mask_t, target_t);
            predictions.push_back(pred);

            if (std::isfinite(loss)) {
                total_loss += loss;
                valid_steps++;
            }
        }

        float avg_loss = valid_steps > 0 ? total_loss / valid_steps : 0.0f;
        return {avg_loss, predictions};
    }

    /**
     * Validate the model
     */
    Metrics validate(const std::vector<Batch>& val_batches) {
        model_->set_training(false);
        Metrics val_metrics;
        val_metrics.reset();

        for (const auto& batch : val_batches) {
            if (batch.empty()) continue;

            // Forward pass without training
            Tape tape;
            auto result = model_->forward_autoencoder(
                batch.X_seq, batch.dt_seq, batch.mask_seq,
                {}, {}, tape);  // Empty initial states

            if (batch.target.has_value()) {
                float loss_val = loss_fn_->forward(result.output_sequence, batch.target.value());
                val_metrics.update(loss_val, result.output_sequence, batch.target.value());
            }
        }

        model_->set_training(true);
        return val_metrics;
    }

    /**
     * Get current metrics
     */
    const Metrics& get_train_metrics() const { return train_metrics_; }
    const Metrics& get_val_metrics() const { return val_metrics_; }

    /**
     * Get current step count
     */
    int get_step_count() const { return step_count_; }

    /**
     * Reset metrics
     */
    void reset_metrics() {
        train_metrics_.reset();
        val_metrics_.reset();
    }

    /**
     * Get model reference
     */
    models::TemporalAutoencoder& get_model() { return *model_; }
    const models::TemporalAutoencoder& get_model() const { return *model_; }
    // Best model tracking
    float best_val_loss_;
    int epochs_without_improvement_;

private:
    void setup_optimizer() {
        auto params = model_->all_parameters();

        if (config_.optimizer == "sgd") {
            optimizer_ = std::make_unique<optim::SGD>(
                config_.learning_rate, config_.momentum, config_.weight_decay);
        } else if (config_.optimizer == "adamw") {
            optimizer_ = std::make_unique<optim::AdamW>(
                config_.learning_rate, config_.beta1, config_.beta2,
                config_.eps, config_.weight_decay);
        } else {
            throw std::invalid_argument("Unknown optimizer: " + config_.optimizer);
        }
    }

    void setup_loss_function() {
        if (config_.loss_type == "mse") {
            loss_fn_ = std::make_unique<loss::MSELoss>();
        } else if (config_.loss_type == "mae") {
            loss_fn_ = std::make_unique<loss::MAELoss>();
        } else if (config_.loss_type == "huber") {
            loss_fn_ = std::make_unique<loss::HuberLoss>(config_.huber_delta);
        } else {
            throw std::invalid_argument("Unknown loss type: " + config_.loss_type);
        }
    }

    void setup_scheduler() {
        if (!config_.use_lr_scheduler) {
            return;
        }

        if (config_.scheduler_type == "step") {
            scheduler_ = std::make_unique<optim::StepLR>(
                config_.learning_rate, config_.step_size, config_.gamma);
        } else if (config_.scheduler_type == "exponential") {
            scheduler_ = std::make_unique<optim::ExponentialLR>(
                config_.learning_rate, config_.gamma);
        } else if (config_.scheduler_type == "cosine") {
            scheduler_ = std::make_unique<optim::CosineAnnealingLR>(
                config_.learning_rate, config_.T_max);
        }
    }

    void reset_streaming_state(int batch_size = 1) {
        h_encoder_.clear();
        h_decoder_.clear();

        for (int l = 0; l < model_->config.num_layers; ++l) {
            h_encoder_.push_back(Eigen::MatrixXf::Zero(batch_size, model_->config.hidden_size));
            h_decoder_.push_back(Eigen::MatrixXf::Zero(batch_size, model_->config.hidden_size));
        }

        window_X_.clear();
        window_dt_.clear();
        window_mask_.clear();
        window_targets_.clear();
    }

    void clip_gradients(const std::vector<Param*>& parameters, float max_norm) {
        float total_norm = 0.0f;

        // Compute total gradient norm
        for (const auto* param : parameters) {
            total_norm += param->grad.array().square().sum();
        }
        total_norm = std::sqrt(total_norm);

        // Apply clipping if needed
        if (total_norm > max_norm) {
            float scale = max_norm / total_norm;
            for (auto* param : parameters) {
                param->grad *= scale;
            }
        }
    }
};

// ============================================================================
// BATCH TRAINER FOR OFFLINE TRAINING
// ============================================================================

class BatchTrainer {
private:
    std::unique_ptr<models::TemporalAutoencoder> model_;
    std::unique_ptr<OnlineTrainer> online_trainer_;
    TrainingConfig config_;

    // Training history
    std::vector<float> train_losses_;
    std::vector<float> val_losses_;

public:
    BatchTrainer(std::unique_ptr<models::TemporalAutoencoder> model,
                 const TrainingConfig& config,
                 std::mt19937& gen)
        : model_(std::move(model)), config_(config) {

        online_trainer_ = std::make_unique<OnlineTrainer>(
            std::move(model_), config_, gen);
    }

    /**
     * Train for one epoch
     */
    Metrics train_epoch(const std::vector<Batch>& train_batches) {
        online_trainer_->reset_metrics();

        for (const auto& batch : train_batches) {
            online_trainer_->process_batch(batch);
        }

        return online_trainer_->get_train_metrics();
    }

    /**
     * Validate for one epoch
     */
    Metrics validate_epoch(const std::vector<Batch>& val_batches) {
        return online_trainer_->validate(val_batches);
    }

    /**
     * Full training loop
     */
    void train(const std::vector<Batch>& train_batches,
               const std::vector<Batch>& val_batches = {}) {

        std::cout << "Starting training..." << std::endl;
        std::cout << "Model parameters: " << online_trainer_->get_model().num_parameters() << std::endl;

        auto start_time = std::chrono::high_resolution_clock::now();

        for (int epoch = 0; epoch < config_.max_epochs; ++epoch) {
            // Training phase
            auto train_metrics = train_epoch(train_batches);
            train_losses_.push_back(train_metrics.loss);

            // Validation phase
            Metrics val_metrics;
            if (!val_batches.empty() && (epoch % config_.validate_every == 0)) {
                val_metrics = validate_epoch(val_batches);
                val_losses_.push_back(val_metrics.loss);
            }

            // Logging
            if (epoch % config_.log_every == 0 || epoch == config_.max_epochs - 1) {
                auto current_time = std::chrono::high_resolution_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                    current_time - start_time).count();

                std::cout << "Epoch " << std::setw(4) << epoch
                          << " [" << elapsed << "s] ";
                train_metrics.print("Train - ");

                if (!val_batches.empty() && !val_losses_.empty()) {
                    val_metrics.print("Val - ");
                }
                std::cout << std::endl;
            }

            // Early stopping check
            if (config_.use_early_stopping && !val_losses_.empty()) {
                if (val_metrics.loss < online_trainer_->best_val_loss_ - config_.min_delta) {
                    online_trainer_->best_val_loss_ = val_metrics.loss;
                    online_trainer_->epochs_without_improvement_ = 0;
                } else {
                    online_trainer_->epochs_without_improvement_++;

                    if (online_trainer_->epochs_without_improvement_ >= config_.patience) {
                        std::cout << "Early stopping at epoch " << epoch << std::endl;
                        break;
                    }
                }
            }
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto total_time = std::chrono::duration_cast<std::chrono::seconds>(
            end_time - start_time).count();

        std::cout << "Training completed in " << total_time << " seconds" << std::endl;
    }

    /**
     * Get training history
     */
    const std::vector<float>& get_train_losses() const { return train_losses_; }
    const std::vector<float>& get_val_losses() const { return val_losses_; }

    /**
     * Get model reference
     */
    models::TemporalAutoencoder& get_model() { return online_trainer_->get_model(); }
    const models::TemporalAutoencoder& get_model() const { return online_trainer_->get_model(); }
};

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/**
 * Create random batch for testing
 */
inline Batch create_random_batch(int batch_size, int seq_len, int input_size,
                                std::mt19937& gen) {
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::uniform_real_distribution<float> dt_dist(0.1f, 2.0f);
    std::bernoulli_distribution mask_dist(0.8);  // 80% observed

    Batch batch;
    batch.X_seq.reserve(seq_len);
    batch.dt_seq.reserve(seq_len);
    batch.mask_seq.reserve(seq_len);

    for (int t = 0; t < seq_len; ++t) {
        // Generate random input
        Eigen::MatrixXf x_t(batch_size, input_size);
        for (int i = 0; i < batch_size; ++i) {
            for (int j = 0; j < input_size; ++j) {
                x_t(i, j) = dist(gen);
            }
        }
        batch.X_seq.push_back(x_t);

        // Generate random dt
        Eigen::MatrixXf dt_t(batch_size, 1);
        for (int i = 0; i < batch_size; ++i) {
            dt_t(i, 0) = dt_dist(gen);
        }
        batch.dt_seq.push_back(dt_t);

        // Generate random mask
        Eigen::MatrixXf mask_t(batch_size, input_size);
        for (int i = 0; i < batch_size; ++i) {
            for (int j = 0; j < input_size; ++j) {
                mask_t(i, j) = mask_dist(gen) ? 1.0f : 0.0f;
            }
        }
        batch.mask_seq.push_back(mask_t);
    }

    return batch;
}

/**
 * Generate synthetic dataset for testing
 */
inline std::vector<Batch> generate_synthetic_dataset(
    int num_batches, int batch_size, int seq_len, int input_size, std::mt19937& gen) {

    std::vector<Batch> dataset;
    dataset.reserve(num_batches);

    for (int i = 0; i < num_batches; ++i) {
        dataset.push_back(create_random_batch(batch_size, seq_len, input_size, gen));
    }

    return dataset;
}

} // namespace training
} // namespace grud

#endif // GRUD_TRAINING_TRAINER_H