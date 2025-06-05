//
// grud/utils/data.h - Data preprocessing and utility functions
//

#ifndef GRUD_UTILS_DATA_H
#define GRUD_UTILS_DATA_H

#include "../grud.h"
#include <algorithm>
#include <numeric>
#include <random>
#include <cmath>

namespace grud {
namespace utils {

// ============================================================================
// DATA PREPROCESSING
// ============================================================================

/**
 * Standard scaler for normalizing data
 */
class StandardScaler {
private:
    Eigen::VectorXf mean_;
    Eigen::VectorXf std_;
    bool fitted_ = false;
    float eps_ = 1e-8f;

public:
    StandardScaler(float eps = 1e-8f) : eps_(eps) {}

    /**
     * Fit the scaler to data
     */
    void fit(const std::vector<Eigen::MatrixXf>& data_sequence) {
        if (data_sequence.empty()) {
            throw std::invalid_argument("Cannot fit scaler on empty data");
        }

        int num_features = data_sequence[0].cols();
        mean_ = Eigen::VectorXf::Zero(num_features);
        std_ = Eigen::VectorXf::Zero(num_features);

        // Compute mean
        int total_samples = 0;
        for (const auto& data : data_sequence) {
            mean_ += data.colwise().sum().transpose();
            total_samples += data.rows();
        }
        mean_ /= static_cast<float>(total_samples);

        // Compute standard deviation
        for (const auto& data : data_sequence) {
            for (int i = 0; i < data.rows(); ++i) {
                Eigen::VectorXf diff = data.row(i).transpose() - mean_;
                std_ += diff.array().square().matrix();
            }
        }
        std_ /= static_cast<float>(total_samples - 1);
        std_ = std_.array().sqrt();

        // Prevent division by zero
        for (int i = 0; i < std_.size(); ++i) {
            if (std_(i) < eps_) {
                std_(i) = 1.0f;
            }
        }

        fitted_ = true;
    }

    /**
     * Transform data using fitted parameters
     */
    Eigen::MatrixXf transform(const Eigen::MatrixXf& data) const {
        if (!fitted_) {
            throw std::runtime_error("Scaler must be fitted before transform");
        }

        Eigen::MatrixXf result(data.rows(), data.cols());
        for (int i = 0; i < data.rows(); ++i) {
            result.row(i) = (data.row(i).transpose() - mean_).array() / std_.array();
        }
        return result;
    }

    /**
     * Inverse transform (denormalize) data
     */
    Eigen::MatrixXf inverse_transform(const Eigen::MatrixXf& data) const {
        if (!fitted_) {
            throw std::runtime_error("Scaler must be fitted before inverse_transform");
        }

        Eigen::MatrixXf result(data.rows(), data.cols());
        for (int i = 0; i < data.rows(); ++i) {
            result.row(i) = (data.row(i).transpose().array() * std_.array() + mean_.array()).matrix();
        }
        return result;
    }

    /**
     * Fit and transform in one step
     */
    std::vector<Eigen::MatrixXf> fit_transform(const std::vector<Eigen::MatrixXf>& data_sequence) {
        fit(data_sequence);

        std::vector<Eigen::MatrixXf> transformed;
        transformed.reserve(data_sequence.size());
        for (const auto& data : data_sequence) {
            transformed.push_back(transform(data));
        }
        return transformed;
    }

    const Eigen::VectorXf& get_mean() const { return mean_; }
    const Eigen::VectorXf& get_std() const { return std_; }
    bool is_fitted() const { return fitted_; }
};

/**
 * Min-Max scaler for normalizing data to [0, 1] range
 */
class MinMaxScaler {
private:
    Eigen::VectorXf min_;
    Eigen::VectorXf max_;
    bool fitted_ = false;
    float eps_ = 1e-8f;

public:
    MinMaxScaler(float eps = 1e-8f) : eps_(eps) {}

    void fit(const std::vector<Eigen::MatrixXf>& data_sequence) {
        if (data_sequence.empty()) {
            throw std::invalid_argument("Cannot fit scaler on empty data");
        }

        int num_features = data_sequence[0].cols();
        min_ = Eigen::VectorXf::Constant(num_features, std::numeric_limits<float>::infinity());
        max_ = Eigen::VectorXf::Constant(num_features, -std::numeric_limits<float>::infinity());

        // Find min and max for each feature
        for (const auto& data : data_sequence) {
            for (int j = 0; j < num_features; ++j) {
                float col_min = data.col(j).minCoeff();
                float col_max = data.col(j).maxCoeff();
                min_(j) = std::min(min_(j), col_min);
                max_(j) = std::max(max_(j), col_max);
            }
        }

        // Prevent division by zero
        for (int i = 0; i < min_.size(); ++i) {
            if (max_(i) - min_(i) < eps_) {
                max_(i) = min_(i) + 1.0f;
            }
        }

        fitted_ = true;
    }

    Eigen::MatrixXf transform(const Eigen::MatrixXf& data) const {
        if (!fitted_) {
            throw std::runtime_error("Scaler must be fitted before transform");
        }

        Eigen::MatrixXf result(data.rows(), data.cols());
        for (int i = 0; i < data.rows(); ++i) {
            result.row(i) = (data.row(i).transpose() - min_).array() / (max_ - min_).array();
        }
        return result;
    }

    Eigen::MatrixXf inverse_transform(const Eigen::MatrixXf& data) const {
        if (!fitted_) {
            throw std::runtime_error("Scaler must be fitted before inverse_transform");
        }

        Eigen::MatrixXf result(data.rows(), data.cols());
        for (int i = 0; i < data.rows(); ++i) {
            result.row(i) = (data.row(i).transpose().array() * (max_ - min_).array() + min_.array()).matrix();
        }
        return result;
    }

    std::vector<Eigen::MatrixXf> fit_transform(const std::vector<Eigen::MatrixXf>& data_sequence) {
        fit(data_sequence);

        std::vector<Eigen::MatrixXf> transformed;
        transformed.reserve(data_sequence.size());
        for (const auto& data : data_sequence) {
            transformed.push_back(transform(data));
        }
        return transformed;
    }

    const Eigen::VectorXf& get_min() const { return min_; }
    const Eigen::VectorXf& get_max() const { return max_; }
    bool is_fitted() const { return fitted_; }
};

// ============================================================================
// MISSING VALUE HANDLING
// ============================================================================

/**
 * Generate random missing value masks
 */
class MaskGenerator {
public:
    /**
     * Generate random mask with specified missing rate
     */
    static Eigen::MatrixXf random_mask(int rows, int cols, float missing_rate, std::mt19937& gen) {
        std::bernoulli_distribution dist(1.0f - missing_rate);
        Eigen::MatrixXf mask(rows, cols);

        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                mask(i, j) = dist(gen) ? 1.0f : 0.0f;
            }
        }

        return mask;
    }

    /**
     * Generate block missing pattern (missing consecutive timesteps)
     */
    static std::vector<Eigen::MatrixXf> block_missing_sequence(
        int batch_size, int seq_len, int input_size,
        float block_prob, int max_block_size, std::mt19937& gen) {

        std::vector<Eigen::MatrixXf> masks;
        masks.reserve(seq_len);

        std::bernoulli_distribution block_start_dist(block_prob);
        std::uniform_int_distribution<int> block_size_dist(1, max_block_size);

        for (int b = 0; b < batch_size; ++b) {
            std::vector<bool> is_missing(seq_len, false);

            for (int t = 0; t < seq_len; ++t) {
                if (!is_missing[t] && block_start_dist(gen)) {
                    int block_size = block_size_dist(gen);
                    for (int k = 0; k < block_size && (t + k) < seq_len; ++k) {
                        is_missing[t + k] = true;
                    }
                }
            }

            // Create masks for this batch element
            for (int t = 0; t < seq_len; ++t) {
                if (masks.size() <= static_cast<size_t>(t)) {
                    masks.resize(t + 1, Eigen::MatrixXf::Ones(batch_size, input_size));
                }

                if (is_missing[t]) {
                    masks[t].row(b).setZero();
                }
            }
        }

        return masks;
    }

    /**
     * Generate feature-wise missing pattern (some features always missing)
     */
    static std::vector<Eigen::MatrixXf> feature_missing_sequence(
        int batch_size, int seq_len, int input_size,
        float feature_missing_rate, std::mt19937& gen) {

        // Determine which features are missing for each batch element
        std::bernoulli_distribution feature_dist(feature_missing_rate);
        Eigen::MatrixXf feature_mask(batch_size, input_size);

        for (int b = 0; b < batch_size; ++b) {
            for (int f = 0; f < input_size; ++f) {
                feature_mask(b, f) = feature_dist(gen) ? 0.0f : 1.0f;
            }
        }

        // Apply same feature mask to all timesteps
        std::vector<Eigen::MatrixXf> masks(seq_len, feature_mask);
        return masks;
    }
};

// ============================================================================
// SEQUENCE GENERATION
// ============================================================================

/**
 * Generate synthetic temporal sequences for testing
 */
class SequenceGenerator {
public:
    /**
     * Generate sinusoidal sequences with noise
     */
    static training::Batch generate_sinusoidal(
        int batch_size, int seq_len, int num_features,
        float noise_level = 0.1f, std::mt19937& gen = std::mt19937(42)) {

        training::Batch batch;

        std::normal_distribution<float> noise_dist(0.0f, noise_level);
        std::uniform_real_distribution<float> freq_dist(0.1f, 2.0f);
        std::uniform_real_distribution<float> phase_dist(0.0f, 2.0f * M_PI);
        std::uniform_real_distribution<float> dt_dist(0.8f, 1.2f);

        for (int t = 0; t < seq_len; ++t) {
            Eigen::MatrixXf x_t(batch_size, num_features);
            Eigen::MatrixXf dt_t(batch_size, 1);

            for (int b = 0; b < batch_size; ++b) {
                for (int f = 0; f < num_features; ++f) {
                    float freq = freq_dist(gen);
                    float phase = phase_dist(gen);
                    float signal = std::sin(freq * t + phase) + noise_dist(gen);
                    x_t(b, f) = signal;
                }
                dt_t(b, 0) = dt_dist(gen);
            }

            batch.X_seq.push_back(x_t);
            batch.dt_seq.push_back(dt_t);
            batch.mask_seq.push_back(Eigen::MatrixXf::Ones(batch_size, num_features));
        }

        return batch;
    }

    /**
     * Generate autoregressive sequences
     */
    static training::Batch generate_autoregressive(
        int batch_size, int seq_len, int num_features,
        const std::vector<float>& ar_coeffs, float noise_level = 0.1f,
        std::mt19937& gen = std::mt19937(42)) {

        training::Batch batch;

        std::normal_distribution<float> noise_dist(0.0f, noise_level);
        std::normal_distribution<float> init_dist(0.0f, 1.0f);
        std::uniform_real_distribution<float> dt_dist(0.9f, 1.1f);

        // Initialize with random values
        std::vector<Eigen::MatrixXf> history;
        for (int lag = 0; lag < static_cast<int>(ar_coeffs.size()); ++lag) {
            Eigen::MatrixXf x_init(batch_size, num_features);
            for (int b = 0; b < batch_size; ++b) {
                for (int f = 0; f < num_features; ++f) {
                    x_init(b, f) = init_dist(gen);
                }
            }
            history.push_back(x_init);
        }

        for (int t = 0; t < seq_len; ++t) {
            Eigen::MatrixXf x_t = Eigen::MatrixXf::Zero(batch_size, num_features);

            // Apply AR model
            for (size_t lag = 0; lag < ar_coeffs.size() && lag < history.size(); ++lag) {
                x_t += ar_coeffs[lag] * history[history.size() - 1 - lag];
            }

            // Add noise
            for (int b = 0; b < batch_size; ++b) {
                for (int f = 0; f < num_features; ++f) {
                    x_t(b, f) += noise_dist(gen);
                }
            }

            Eigen::MatrixXf dt_t(batch_size, 1);
            for (int b = 0; b < batch_size; ++b) {
                dt_t(b, 0) = dt_dist(gen);
            }

            batch.X_seq.push_back(x_t);
            batch.dt_seq.push_back(dt_t);
            batch.mask_seq.push_back(Eigen::MatrixXf::Ones(batch_size, num_features));

            // Update history
            history.push_back(x_t);
            if (history.size() > ar_coeffs.size()) {
                history.erase(history.begin());
            }
        }

        return batch;
    }

    /**
     * Generate random walk sequences
     */
    static training::Batch generate_random_walk(
        int batch_size, int seq_len, int num_features,
        float step_size = 1.0f, std::mt19937& gen = std::mt19937(42)) {

        training::Batch batch;

        std::normal_distribution<float> step_dist(0.0f, step_size);
        std::normal_distribution<float> init_dist(0.0f, 1.0f);
        std::uniform_real_distribution<float> dt_dist(0.9f, 1.1f);

        // Initialize starting positions
        Eigen::MatrixXf current_pos(batch_size, num_features);
        for (int b = 0; b < batch_size; ++b) {
            for (int f = 0; f < num_features; ++f) {
                current_pos(b, f) = init_dist(gen);
            }
        }

        for (int t = 0; t < seq_len; ++t) {
            // Take random steps
            for (int b = 0; b < batch_size; ++b) {
                for (int f = 0; f < num_features; ++f) {
                    current_pos(b, f) += step_dist(gen);
                }
            }

            Eigen::MatrixXf dt_t(batch_size, 1);
            for (int b = 0; b < batch_size; ++b) {
                dt_t(b, 0) = dt_dist(gen);
            }

            batch.X_seq.push_back(current_pos);
            batch.dt_seq.push_back(dt_t);
            batch.mask_seq.push_back(Eigen::MatrixXf::Ones(batch_size, num_features));
        }

        return batch;
    }
};

// ============================================================================
// DATASET UTILITIES
// ============================================================================

/**
 * Split dataset into train/validation/test sets
 */
class DatasetSplitter {
public:
    struct SplitResult {
        std::vector<training::Batch> train;
        std::vector<training::Batch> val;
        std::vector<training::Batch> test;
    };

    static SplitResult train_val_test_split(
        const std::vector<training::Batch>& dataset,
        float train_ratio = 0.7f, float val_ratio = 0.15f,
        std::mt19937& gen = std::mt19937(42)) {

        if (train_ratio + val_ratio >= 1.0f) {
            throw std::invalid_argument("train_ratio + val_ratio must be < 1.0");
        }

        // Shuffle dataset
        std::vector<size_t> indices(dataset.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), gen);

        // Calculate split points
        size_t train_end = static_cast<size_t>(dataset.size() * train_ratio);
        size_t val_end = static_cast<size_t>(dataset.size() * (train_ratio + val_ratio));

        SplitResult result;

        // Train set
        for (size_t i = 0; i < train_end; ++i) {
            result.train.push_back(dataset[indices[i]]);
        }

        // Validation set
        for (size_t i = train_end; i < val_end; ++i) {
            result.val.push_back(dataset[indices[i]]);
        }

        // Test set
        for (size_t i = val_end; i < indices.size(); ++i) {
            result.test.push_back(dataset[indices[i]]);
        }

        return result;
    }
};

// ============================================================================
// EVALUATION METRICS
// ============================================================================

/**
 * Comprehensive evaluation metrics for temporal models
 */
class MetricsCalculator {
public:
    struct DetailedMetrics {
        float mse = 0.0f;
        float mae = 0.0f;
        float rmse = 0.0f;
        float mape = 0.0f;  // Mean Absolute Percentage Error
        float smape = 0.0f; // Symmetric MAPE
        float r2_score = 0.0f;
        std::vector<float> per_feature_mse;
        std::vector<float> per_timestep_mse;
        size_t num_samples = 0;
    };

    static DetailedMetrics calculate_metrics(
        const Eigen::MatrixXf& predictions,
        const Eigen::MatrixXf& targets,
        const std::optional<Eigen::MatrixXf>& mask = std::nullopt,
        int batch_size = -1, int seq_len = -1) {

        DetailedMetrics metrics;

        Eigen::MatrixXf pred = predictions;
        Eigen::MatrixXf targ = targets;
        Eigen::MatrixXf effective_mask = mask.value_or(Eigen::MatrixXf::Ones(pred.rows(), pred.cols()));

        float total_weight = effective_mask.sum();
        if (total_weight == 0) {
            return metrics;
        }

        metrics.num_samples = static_cast<size_t>(total_weight);

        // Element-wise differences
        Eigen::MatrixXf diff = pred - targ;
        Eigen::MatrixXf abs_diff = diff.array().abs();
        Eigen::MatrixXf squared_diff = diff.array().square();

        // Apply mask
        abs_diff.array() *= effective_mask.array();
        squared_diff.array() *= effective_mask.array();

        // Basic metrics
        metrics.mse = squared_diff.sum() / total_weight;
        metrics.mae = abs_diff.sum() / total_weight;
        metrics.rmse = std::sqrt(metrics.mse);

        // MAPE and SMAPE
        float mape_sum = 0.0f;
        float smape_sum = 0.0f;
        float mape_count = 0.0f;

        for (int i = 0; i < targ.rows(); ++i) {
            for (int j = 0; j < targ.cols(); ++j) {
                if (effective_mask(i, j) > 0.5f) {
                    float target_val = targ(i, j);
                    float pred_val = pred(i, j);

                    if (std::abs(target_val) > 1e-8f) {
                        mape_sum += std::abs((target_val - pred_val) / target_val);
                        mape_count += 1.0f;
                    }

                    float denom = std::abs(target_val) + std::abs(pred_val);
                    if (denom > 1e-8f) {
                        smape_sum += std::abs(target_val - pred_val) / denom;
                    }
                }
            }
        }

        metrics.mape = mape_count > 0 ? (mape_sum / mape_count) * 100.0f : 0.0f;
        metrics.smape = (smape_sum / total_weight) * 200.0f;

        // R² score
        float ss_res = squared_diff.sum();
        float target_mean = (targ.array() * effective_mask.array()).sum() / total_weight;
        float ss_tot = ((targ.array() - target_mean).square() * effective_mask.array()).sum();
        metrics.r2_score = ss_tot > 1e-8f ? (1.0f - ss_res / ss_tot) : 0.0f;

        // Per-feature metrics
        if (pred.cols() > 1) {
            metrics.per_feature_mse.resize(pred.cols());
            for (int j = 0; j < pred.cols(); ++j) {
                float feature_weight = effective_mask.col(j).sum();
                if (feature_weight > 0) {
                    metrics.per_feature_mse[j] = (squared_diff.col(j).array() * effective_mask.col(j).array()).sum() / feature_weight;
                }
            }
        }

        // Per-timestep metrics (if sequence structure provided)
        if (batch_size > 0 && seq_len > 0 && pred.rows() == batch_size * seq_len) {
            metrics.per_timestep_mse.resize(seq_len);
            for (int t = 0; t < seq_len; ++t) {
                int start_row = t * batch_size;
                int end_row = start_row + batch_size;

                Eigen::MatrixXf timestep_squared_diff = squared_diff.block(start_row, 0, batch_size, pred.cols());
                Eigen::MatrixXf timestep_mask = effective_mask.block(start_row, 0, batch_size, pred.cols());

                float timestep_weight = timestep_mask.sum();
                if (timestep_weight > 0) {
                    metrics.per_timestep_mse[t] = (timestep_squared_diff.array() * timestep_mask.array()).sum() / timestep_weight;
                }
            }
        }

        return metrics;
    }

    static void print_metrics(const DetailedMetrics& metrics, const std::string& prefix = "") {
        std::cout << prefix << "Evaluation Metrics:" << std::endl;
        std::cout << prefix << "  MSE:    " << std::fixed << std::setprecision(6) << metrics.mse << std::endl;
        std::cout << prefix << "  MAE:    " << metrics.mae << std::endl;
        std::cout << prefix << "  RMSE:   " << metrics.rmse << std::endl;
        std::cout << prefix << "  MAPE:   " << std::setprecision(2) << metrics.mape << "%" << std::endl;
        std::cout << prefix << "  SMAPE:  " << metrics.smape << "%" << std::endl;
        std::cout << prefix << "  R²:     " << std::setprecision(4) << metrics.r2_score << std::endl;
        std::cout << prefix << "  Samples: " << metrics.num_samples << std::endl;

        if (!metrics.per_feature_mse.empty()) {
            std::cout << prefix << "  Per-feature MSE: ";
            for (size_t i = 0; i < metrics.per_feature_mse.size(); ++i) {
                std::cout << std::setprecision(6) << metrics.per_feature_mse[i];
                if (i < metrics.per_feature_mse.size() - 1) std::cout << ", ";
            }
            std::cout << std::endl;
        }
    }
};

} // namespace utils
} // namespace grud

#endif // GRUD_UTILS_DATA_H