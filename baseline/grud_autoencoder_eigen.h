// file: grud_autoencoder_eigen.h
#ifndef TENSOREIGEN_AUTOENCODER_COMPLETE_H
#define TENSOREIGEN_AUTOENCODER_COMPLETE_H

#include "grud_eigen.h" // Your core GRU-D implementation
#include <deque>
#include <memory>
#include <functional>

// ============================================================================
// 1. ENHANCED CONFIGURATION AND ENUMS
// ============================================================================

enum class BottleneckType {
    LAST_HIDDEN,
    MEAN_POOL,
    MAX_POOL,
    ATTENTION_POOL
};

enum class MaskProjectionType {
    MAX_POOL,
    LEARNED,
    ANY_OBSERVED
};

enum class AutoencoderMode {
    RECONSTRUCTION,
    FORECASTING
};

enum class ForecastingMode {
    DIRECT,
    AUTOREGRESSIVE
};

enum class DTPresictionMethod {
    LEARNED,
    LAST_VALUE
};

enum class FeedbackTransformType {
    LINEAR,
    IDENTITY,
    LEARNED
};

struct AutoencoderConfig {
    // Architecture
    int input_size = 12;
    int latent_size = 8;
    int internal_projection_size = 32;

    // Bottleneck
    BottleneckType bottleneck_type = BottleneckType::MEAN_POOL;
    bool use_input_projection = true;
    MaskProjectionType mask_projection_type = MaskProjectionType::MAX_POOL;
    int attention_context_dim = 64;

    // Mode configuration
    AutoencoderMode mode = AutoencoderMode::RECONSTRUCTION;
    int forecast_horizon = 1;
    ForecastingMode forecasting_mode = ForecastingMode::DIRECT;

    // dt prediction
    bool predict_future_dt = false;
    DTPresictionMethod dt_prediction_method = DTPresictionMethod::LAST_VALUE;

    // Autoregressive configuration
    FeedbackTransformType feedback_transform_type = FeedbackTransformType::LINEAR;
    bool pass_mask_to_decoder_rnn = false;

    // Loss configuration
    float loss_ramp_start = 1.0f;
    float loss_ramp_end = 1.0f;
};

// ============================================================================
// 2. ENHANCED MASK PROJECTOR WITH COMPLETE GRADIENT FLOW
// ============================================================================

struct MaskProjectorCache {
    bool used = false;

    // Support both single timestep and sequence projections
    std::optional<MatrixXf> input_mask_single;           // For single (B, F) masks
    std::optional<std::vector<MatrixXf>> input_mask_sequence; // For sequence (B, T, F) masks

    MaskProjectionType projection_type;
    std::optional<LinearCache> learned_cache_single;
    std::optional<std::vector<LinearCache>> learned_cache_sequence;
    std::optional<MatrixXf> weight_matrix;
    std::optional<MatrixXf> pre_sigmoid_output_single;
    std::optional<std::vector<MatrixXf>> pre_sigmoid_output_sequence;

    // For max pooling with weights
    std::optional<MatrixXf> significant_connections;
    std::vector<std::pair<int, int>> group_boundaries; // start_idx, end_idx for each group

    // NEW: Store original values for exact PyTorch backward compatibility
    std::optional<std::vector<MatrixXf>> original_values_sequence; // Store what was actually True in forward
    std::optional<MatrixXf> original_values_single;

    int batch_size = 0;
    int seq_len = 0;
};

class NpMaskProjector {
private:
    int input_size;
    int output_size;
    MaskProjectionType projection_type;
    std::unique_ptr<NpLinear> learned_proj;
    std::mt19937 &gen_ref;

public:
    NpMaskProjector(int in_size, int out_size, MaskProjectionType proj_type, std::mt19937 &gen)
            : input_size(in_size), output_size(out_size), projection_type(proj_type), gen_ref(gen) {
        if (projection_type == MaskProjectionType::LEARNED) {
            learned_proj = std::make_unique<NpLinear>(input_size, output_size, false, &gen_ref);
            learned_proj->weights.setConstant(1.0f / input_size);
        }
    }

    // MAIN INTERFACE: Handle sequences like PyTorch (B, T, F) -> (B, T, F_out)
    std::pair<std::vector<MatrixXf>, MaskProjectorCache> forward_sequence(
            const std::vector<MatrixXf> &mask_seq,
            const std::optional<MatrixXf> &weight_matrix = std::nullopt) {

        MaskProjectorCache cache;
        cache.used = true;
        cache.input_mask_sequence = mask_seq;
        cache.projection_type = projection_type;
        cache.weight_matrix = weight_matrix;
        cache.batch_size = mask_seq.empty() ? 0 : mask_seq[0].rows();
        cache.seq_len = mask_seq.size();

        if (mask_seq.empty()) {
            return {{}, cache};
        }

        // Check if projection is needed
        if (mask_seq[0].cols() == output_size) {
            cache.used = false;
            return {mask_seq, cache};
        }

        std::vector<MatrixXf> projected_seq;
        projected_seq.reserve(mask_seq.size());

        switch (projection_type) {
            case MaskProjectionType::MAX_POOL:
                projected_seq = forward_max_pool_sequence(mask_seq, weight_matrix, cache);
                break;
            case MaskProjectionType::LEARNED:
                projected_seq = forward_learned_sequence(mask_seq, cache);
                break;
            case MaskProjectionType::ANY_OBSERVED:
                projected_seq = forward_any_observed_sequence(mask_seq);
                break;
        }

        return {projected_seq, cache};
    }

    // LEGACY INTERFACE: Single timestep (B, F) -> (B, F_out) for backward compatibility
    std::pair<MatrixXf, MaskProjectorCache> forward(
            const MatrixXf &mask,
            const std::optional<MatrixXf> &weight_matrix = std::nullopt) {

        // Convert to sequence and back for consistency
        std::vector<MatrixXf> mask_seq = {mask};
        auto [projected_seq, cache] = forward_sequence(mask_seq, weight_matrix);

        // Update cache to indicate single timestep
        cache.input_mask_single = mask;
        cache.input_mask_sequence = std::nullopt;

        if (projected_seq.empty()) {
            return {MatrixXf::Zero(mask.rows(), output_size), cache};
        }

        return {projected_seq[0], cache};
    }

    // EXACT PYTORCH BACKWARD BEHAVIOR
    std::vector<MatrixXf> backward_sequence(const std::vector<MatrixXf> &d_projected_seq, const MaskProjectorCache &cache) {
        if (!cache.used) {
            return cache.input_mask_sequence.value_or(std::vector<MatrixXf>{});
        }

        if (!cache.input_mask_sequence.has_value()) {
            throw std::runtime_error("Missing input mask sequence in cache");
        }

        const std::vector<MatrixXf> &input_masks = cache.input_mask_sequence.value();

        if (input_masks[0].cols() == output_size) {
            return d_projected_seq;
        }

        switch (cache.projection_type) {
            case MaskProjectionType::MAX_POOL:
                return backward_max_pool_sequence_pytorch_exact(d_projected_seq, cache);
            case MaskProjectionType::LEARNED:
                return backward_learned_sequence(d_projected_seq, cache);
            case MaskProjectionType::ANY_OBSERVED:
                return backward_any_observed_sequence(d_projected_seq, cache);
        }

        return std::vector<MatrixXf>(input_masks.size(), MatrixXf::Zero(cache.batch_size, input_size));
    }

    // Legacy single timestep backward
    MatrixXf backward(const MatrixXf &d_projected, const MaskProjectorCache &cache) {
        if (cache.input_mask_single.has_value()) {
            // True single timestep call
            std::vector<MatrixXf> d_proj_seq = {d_projected};
            auto d_input_seq = backward_sequence(d_proj_seq, cache);
            return d_input_seq.empty() ? MatrixXf::Zero(d_projected.rows(), input_size) : d_input_seq[0];
        } else if (cache.input_mask_sequence.has_value() && cache.input_mask_sequence.value().size() == 1) {
            // Sequence with single element
            std::vector<MatrixXf> d_proj_seq = {d_projected};
            auto d_input_seq = backward_sequence(d_proj_seq, cache);
            return d_input_seq.empty() ? MatrixXf::Zero(d_projected.rows(), input_size) : d_input_seq[0];
        }

        return MatrixXf::Zero(d_projected.rows(), input_size);
    }

    void zero_grad() {
        if (learned_proj) learned_proj->zero_grad();
    }

    std::vector<std::pair<MatrixXf *, MatrixXf *>> get_params_grads() {
        if (learned_proj && projection_type == MaskProjectionType::LEARNED) {
            return {{&learned_proj->weights, &learned_proj->grad_weights}};
        }
        return {};
    }

    std::vector<std::pair<VectorXf *, VectorXf *>> get_bias_params_grads() {
        if (learned_proj && projection_type == MaskProjectionType::LEARNED && learned_proj->bias.size() > 0) {
            return {{&learned_proj->bias, &learned_proj->grad_bias}};
        }
        return {};
    }

private:
    // EXACT PYTORCH MAX_POOL BEHAVIOR: torch.any() equivalent
    std::vector<MatrixXf> forward_max_pool_sequence(
            const std::vector<MatrixXf> &mask_seq,
            const std::optional<MatrixXf> &weight_matrix,
            MaskProjectorCache &cache) {

        int batch_size = mask_seq[0].rows();
        int seq_len = mask_seq.size();
        std::vector<MatrixXf> result_seq;
        result_seq.reserve(seq_len);

        // Store original values for exact backward pass
        cache.original_values_sequence = mask_seq;

        if (weight_matrix.has_value() && weight_matrix.value().rows() == output_size &&
            weight_matrix.value().cols() == input_size) {
            // Weight-based projection
            MatrixXf abs_weights = weight_matrix.value().array().abs();
            MatrixXf threshold = abs_weights.rowwise().maxCoeff() * 0.1f;
            MatrixXf significant_connections = MatrixXf::Zero(abs_weights.rows(), abs_weights.cols());
            for (int i = 0; i < abs_weights.rows(); ++i) {
                for (int j = 0; j < abs_weights.cols(); ++j) {
                    significant_connections(i, j) = (abs_weights(i, j) > threshold(i, 0)) ? 1.0f : 0.0f;
                }
            }
            cache.significant_connections = significant_connections;

            // Apply to each timestep - EXACT PyTorch behavior
            for (int t = 0; t < seq_len; ++t) {
                MatrixXf result_t(batch_size, output_size);
                const MatrixXf &mask_t = mask_seq[t];

                for (int b = 0; b < batch_size; ++b) {
                    for (int o = 0; o < output_size; ++o) {
                        float any_val = 0.0f;
                        // Equivalent to: torch.any(mask_t[significant_connections[o] > 0.5])
                        for (int i = 0; i < input_size; ++i) {
                            if (significant_connections(o, i) > 0.5f && mask_t(b, i) > 0.5f) {
                                any_val = 1.0f;
                                break;  // any() stops at first True
                            }
                        }
                        result_t(b, o) = any_val;
                    }
                }
                result_seq.push_back(result_t);
            }
        } else {
            // Simple grouping - EXACT PyTorch behavior
            int group_size = input_size / output_size;
            int remainder = input_size % output_size;

            cache.group_boundaries.clear();
            cache.group_boundaries.reserve(output_size);

            int start_idx = 0;
            for (int i = 0; i < output_size; ++i) {
                int current_group_size = group_size + (i < remainder ? 1 : 0);
                int end_idx = start_idx + current_group_size;
                cache.group_boundaries.emplace_back(start_idx, end_idx);
                start_idx = end_idx;
            }

            // Apply to each timestep
            for (int t = 0; t < seq_len; ++t) {
                MatrixXf result_t(batch_size, output_size);
                const MatrixXf &mask_t = mask_seq[t];

                for (int b = 0; b < batch_size; ++b) {
                    for (int i = 0; i < output_size; ++i) {
                        const auto &[start_idx, end_idx] = cache.group_boundaries[i];
                        float any_val = 0.0f;
                        // Equivalent to: torch.any(mask_t[:, start_idx:end_idx], dim=-1)
                        for (int j = start_idx; j < end_idx; ++j) {
                            if (mask_t(b, j) > 0.5f) {
                                any_val = 1.0f;
                                break;  // any() stops at first True
                            }
                        }
                        result_t(b, i) = any_val;
                    }
                }
                result_seq.push_back(result_t);
            }
        }

        return result_seq;
    }

    std::vector<MatrixXf> forward_learned_sequence(const std::vector<MatrixXf> &mask_seq, MaskProjectorCache &cache) {
        std::vector<MatrixXf> result_seq;
        std::vector<LinearCache> linear_caches;
        std::vector<MatrixXf> pre_sigmoid_outputs;

        result_seq.reserve(mask_seq.size());
        linear_caches.reserve(mask_seq.size());
        pre_sigmoid_outputs.reserve(mask_seq.size());

        for (const auto &mask_t : mask_seq) {
            auto [proj_output, linear_cache] = learned_proj->forward(mask_t);
            MatrixXf sigmoid_output = sigmoid(proj_output);

            result_seq.push_back(sigmoid_output);
            linear_caches.push_back(linear_cache);
            pre_sigmoid_outputs.push_back(proj_output);
        }

        cache.learned_cache_sequence = linear_caches;
        cache.pre_sigmoid_output_sequence = pre_sigmoid_outputs;

        return result_seq;
    }

    std::vector<MatrixXf> forward_any_observed_sequence(const std::vector<MatrixXf> &mask_seq) {
        std::vector<MatrixXf> result_seq;
        result_seq.reserve(mask_seq.size());

        for (const auto &mask_t : mask_seq) {
            MatrixXf result_t(mask_t.rows(), output_size);
            for (int b = 0; b < mask_t.rows(); ++b) {
                float any_obs = mask_t.row(b).maxCoeff() > 0.5f ? 1.0f : 0.0f;
                result_t.row(b).setConstant(any_obs);
            }
            result_seq.push_back(result_t);
        }

        return result_seq;
    }

    // EXACT PYTORCH MAX_POOL BACKWARD: Only True elements get gradients
    std::vector<MatrixXf> backward_max_pool_sequence_pytorch_exact(
            const std::vector<MatrixXf> &d_projected_seq, const MaskProjectorCache &cache) {

        if (!cache.original_values_sequence.has_value()) {
            throw std::runtime_error("Missing original values for PyTorch-exact backward");
        }

        const std::vector<MatrixXf> &original_masks = cache.original_values_sequence.value();
        std::vector<MatrixXf> d_input_seq;
        d_input_seq.reserve(d_projected_seq.size());

        if (cache.significant_connections.has_value()) {
            // Weight-based connections
            const MatrixXf &connections = cache.significant_connections.value();

            for (size_t t = 0; t < d_projected_seq.size(); ++t) {
                MatrixXf d_input_t = MatrixXf::Zero(cache.batch_size, input_size);
                const MatrixXf &original_mask_t = original_masks[t];
                const MatrixXf &d_projected_t = d_projected_seq[t];

                for (int b = 0; b < cache.batch_size; ++b) {
                    for (int o = 0; o < output_size; ++o) {
                        float d_o = d_projected_t(b, o);

                        // PYTORCH EXACT: Only elements that were originally True get gradients
                        // This matches torch.any() backward pass exactly
                        for (int i = 0; i < input_size; ++i) {
                            if (connections(o, i) > 0.5f && original_mask_t(b, i) > 0.5f) {
                                d_input_t(b, i) += d_o;  // Full gradient, no division
                            }
                        }
                    }
                }
                d_input_seq.push_back(d_input_t);
            }
        } else {
            // Group-based approach
            for (size_t t = 0; t < d_projected_seq.size(); ++t) {
                MatrixXf d_input_t = MatrixXf::Zero(cache.batch_size, input_size);
                const MatrixXf &original_mask_t = original_masks[t];
                const MatrixXf &d_projected_t = d_projected_seq[t];

                for (int b = 0; b < cache.batch_size; ++b) {
                    for (int i = 0; i < output_size && i < static_cast<int>(cache.group_boundaries.size()); ++i) {
                        const auto &[start_idx, end_idx] = cache.group_boundaries[i];
                        float d_o = d_projected_t(b, i);

                        // PYTORCH EXACT: Only elements that were originally True get gradients
                        for (int j = start_idx; j < end_idx; ++j) {
                            if (original_mask_t(b, j) > 0.5f) {  // Only True elements
                                d_input_t(b, j) += d_o;  // Full gradient, no division
                            }
                        }
                    }
                }
                d_input_seq.push_back(d_input_t);
            }
        }

        return d_input_seq;
    }

    std::vector<MatrixXf> backward_learned_sequence(const std::vector<MatrixXf> &d_projected_seq, const MaskProjectorCache &cache) {
        if (!cache.learned_cache_sequence.has_value() || !cache.pre_sigmoid_output_sequence.has_value()) {
            throw std::runtime_error("Missing learned cache for sequence backward");
        }

        const auto &linear_caches = cache.learned_cache_sequence.value();
        const auto &pre_sigmoid_outputs = cache.pre_sigmoid_output_sequence.value();

        std::vector<MatrixXf> d_input_seq;
        d_input_seq.reserve(d_projected_seq.size());

        for (size_t t = 0; t < d_projected_seq.size(); ++t) {
            MatrixXf sigmoid_out = sigmoid(pre_sigmoid_outputs[t]);
            MatrixXf d_sigmoid = d_projected_seq[t].array() * sigmoid_out.array() * (1.0f - sigmoid_out.array());
            MatrixXf d_input_t = learned_proj->backward(d_sigmoid, linear_caches[t]);
            d_input_seq.push_back(d_input_t);
        }

        return d_input_seq;
    }

    std::vector<MatrixXf> backward_any_observed_sequence(const std::vector<MatrixXf> &d_projected_seq, const MaskProjectorCache &cache) {
        std::vector<MatrixXf> d_input_seq;
        d_input_seq.reserve(d_projected_seq.size());

        for (const auto &d_projected_t : d_projected_seq) {
            MatrixXf d_input_t = MatrixXf::Zero(cache.batch_size, input_size);
            for (int b = 0; b < cache.batch_size; ++b) {
                float total_grad = d_projected_t.row(b).sum() / input_size;
                d_input_t.row(b).setConstant(total_grad);
            }
            d_input_seq.push_back(d_input_t);
        }

        return d_input_seq;
    }
};

// ============================================================================
// 3. COMPLETE ATTENTION POOLING
// ============================================================================

struct AttentionPoolingCache {
    MatrixXf input_x; // (T*B, H) original input
    MatrixXf queries; // (T*B, context_dim)
    MatrixXf scores; // (B, T)
    MatrixXf attn_weights; // (B, T)
    LinearCache query_proj_cache;
    std::optional<MatrixXf> sequence_mask; // (B, T)
    int batch_size;
    int seq_len;
    VectorXf scores_flat; // (T*B,) for backward
};

class NpAttentionPooling {
private:
    int input_dim;
    int context_dim;
    float scale;
    std::unique_ptr<NpLinear> query_proj;
    VectorXf context_vector;
    VectorXf grad_context_vector;

public:
    NpAttentionPooling(int input_d, int context_d, std::mt19937 &gen)
            : input_dim(input_d), context_dim(context_d) {
        scale = std::sqrt(static_cast<float>(context_dim));
        query_proj = std::make_unique<NpLinear>(input_dim, context_dim, true, &gen);

        // Initialize context vector randomly
        context_vector = VectorXf::Zero(context_dim);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        for (int i = 0; i < context_dim; ++i) {
            context_vector(i) = dist(gen);
        }
        grad_context_vector = VectorXf::Zero(context_dim);
    }

    std::pair<MatrixXf, AttentionPoolingCache> forward(const MatrixXf &x_stacked,
                                                       const std::optional<MatrixXf> &sequence_mask,
                                                       int batch_size, int seq_len) {
        AttentionPoolingCache cache;
        cache.input_x = x_stacked;
        cache.batch_size = batch_size;
        cache.seq_len = seq_len;
        cache.sequence_mask = sequence_mask;

        // Project to queries: (T*B, H) -> (T*B, context_dim)
        auto [queries_stacked, query_cache] = query_proj->forward(x_stacked);
        cache.queries = queries_stacked;
        cache.query_proj_cache = query_cache;

        // Compute attention scores: (T*B, context_dim) @ (context_dim,) -> (T*B,)
        VectorXf scores_stacked = (queries_stacked * context_vector) / scale;
        cache.scores_flat = scores_stacked;

        // Reshape scores to (B, T)
        MatrixXf scores_BT(batch_size, seq_len);
        for (int t = 0; t < seq_len; ++t) {
            for (int b = 0; b < batch_size; ++b) {
                int stacked_idx = t * batch_size + b;
                scores_BT(b, t) = scores_stacked(stacked_idx);
            }
        }
        cache.scores = scores_BT;

        // Apply mask if provided
        if (sequence_mask.has_value()) {
            for (int b = 0; b < batch_size; ++b) {
                for (int t = 0; t < seq_len; ++t) {
                    if (sequence_mask.value()(b, t) < 0.5f) {
                        scores_BT(b, t) = -std::numeric_limits<float>::infinity();
                    }
                }
            }
        }

        // Compute softmax attention weights
        MatrixXf attn_weights_BT(batch_size, seq_len);
        for (int b = 0; b < batch_size; ++b) {
            VectorXf scores_b = scores_BT.row(b);
            VectorXf weights_b = softmax_1d(scores_b);
            attn_weights_BT.row(b) = weights_b;
        }
        cache.attn_weights = attn_weights_BT;

        // Compute weighted sum: (B, H)
        MatrixXf pooled_output(batch_size, input_dim);
        pooled_output.setZero();

        for (int b = 0; b < batch_size; ++b) {
            for (int t = 0; t < seq_len; ++t) {
                float weight = attn_weights_BT(b, t);
                int stacked_idx = t * batch_size + b;
                pooled_output.row(b) += weight * x_stacked.row(stacked_idx);
            }
        }

        return {pooled_output, cache};
    }

    MatrixXf backward(const MatrixXf &d_pooled_output, const AttentionPoolingCache &cache) {
        int batch_size = cache.batch_size;
        int seq_len = cache.seq_len;

        // Backward through weighted sum
        MatrixXf d_attn_weights = MatrixXf::Zero(batch_size, seq_len);
        MatrixXf d_x_stacked = MatrixXf::Zero(batch_size * seq_len, input_dim);

        for (int b = 0; b < batch_size; ++b) {
            for (int t = 0; t < seq_len; ++t) {
                int stacked_idx = t * batch_size + b;
                float weight = cache.attn_weights(b, t);

                // Gradient w.r.t. attention weight
                d_attn_weights(b, t) = d_pooled_output.row(b).dot(cache.input_x.row(stacked_idx));

                // Gradient w.r.t. input
                d_x_stacked.row(stacked_idx) = weight * d_pooled_output.row(b);
            }
        }

        // Backward through softmax
        MatrixXf d_scores = MatrixXf::Zero(batch_size, seq_len);
        for (int b = 0; b < batch_size; ++b) {
            VectorXf weights_b = cache.attn_weights.row(b);
            VectorXf d_weights_b = d_attn_weights.row(b);
            VectorXf d_scores_b = softmax_backward_1d(d_weights_b, weights_b);
            d_scores.row(b) = d_scores_b;
        }

        // Apply mask to gradients
        if (cache.sequence_mask.has_value()) {
            for (int b = 0; b < batch_size; ++b) {
                for (int t = 0; t < seq_len; ++t) {
                    if (cache.sequence_mask.value()(b, t) < 0.5f) {
                        d_scores(b, t) = 0.0f;
                    }
                }
            }
        }

        // Backward through attention score computation
        VectorXf d_scores_stacked(batch_size * seq_len);
        for (int t = 0; t < seq_len; ++t) {
            for (int b = 0; b < batch_size; ++b) {
                int stacked_idx = t * batch_size + b;
                d_scores_stacked(stacked_idx) = d_scores(b, t) / scale;
            }
        }

        // Gradient w.r.t. context vector
        grad_context_vector += cache.queries.transpose() * d_scores_stacked;

        // Gradient w.r.t. queries
        MatrixXf d_queries = d_scores_stacked * context_vector.transpose();

        // Backward through query projection
        MatrixXf d_input_from_queries = query_proj->backward(d_queries, cache.query_proj_cache);
        d_x_stacked += d_input_from_queries;

        return d_x_stacked;
    }

    void zero_grad() {
        query_proj->zero_grad();
        grad_context_vector.setZero();
    }

    std::vector<std::pair<MatrixXf *, MatrixXf *> > get_params_grads() {
        return {{&query_proj->weights, &query_proj->grad_weights}};
    }

    std::vector<std::pair<VectorXf *, VectorXf *> > get_vector_params_grads() {
        std::vector<std::pair<VectorXf *, VectorXf *> > result;
        result.push_back({&context_vector, &grad_context_vector});
        if (query_proj->bias.size() > 0) {
            result.push_back({&query_proj->bias, &query_proj->grad_bias});
        }
        return result;
    }

private:
    VectorXf softmax_1d(const VectorXf &x) {
        float max_val = x.maxCoeff();
        VectorXf exp_vals = (x.array() - max_val).exp();
        float sum_exp = exp_vals.sum();
        return exp_vals / sum_exp;
    }

    VectorXf softmax_backward_1d(const VectorXf &d_output, const VectorXf &softmax_output) {
        float sum_term = d_output.dot(softmax_output);
        return softmax_output.array() * (d_output.array() - sum_term);
    }
};

// ============================================================================
// 4. COMPLETE BOTTLENECK AGGREGATION WITH FULL SEQUENCE MASK HANDLING
// ============================================================================

struct BottleneckAggregationCache {
    BottleneckType type;
    MatrixXf hidden_seq_stacked;
    int batch_size;
    int seq_len;
    std::optional<MatrixXf> sequence_mask;
    std::optional<AttentionPoolingCache> attention_cache;
    std::optional<MatrixXf> max_indicators; // For max pooling backward
    std::optional<MatrixXf> valid_step_counts; // For mean pooling with mask
};

class NpBottleneckAggregation {
private:
    BottleneckType aggregation_type;
    std::unique_ptr<NpAttentionPooling> attention_pooling;
    int hidden_dim;

public:
    NpBottleneckAggregation(BottleneckType type, int hidden_d, int context_dim, std::mt19937 &gen)
            : aggregation_type(type), hidden_dim(hidden_d) {
        if (type == BottleneckType::ATTENTION_POOL) {
            attention_pooling = std::make_unique<NpAttentionPooling>(hidden_dim, context_dim, gen);
        }
    }

    std::pair<MatrixXf, BottleneckAggregationCache> forward(const MatrixXf &hidden_seq_stacked,
                                                            const std::optional<MatrixXf> &feature_mask,
                                                            int batch_size, int seq_len) {
        BottleneckAggregationCache cache;
        cache.type = aggregation_type;
        cache.hidden_seq_stacked = hidden_seq_stacked;
        cache.batch_size = batch_size;
        cache.seq_len = seq_len;

        // COMPLETE: Create sequence mask from feature mask
        std::optional<MatrixXf> sequence_mask = create_sequence_mask_from_feature_mask(
                feature_mask, batch_size, seq_len);
        cache.sequence_mask = sequence_mask;

        MatrixXf result;
        switch (aggregation_type) {
            case BottleneckType::LAST_HIDDEN:
                result = forward_last_hidden(hidden_seq_stacked, batch_size, seq_len);
                break;
            case BottleneckType::MEAN_POOL:
                result = forward_mean_pool(hidden_seq_stacked, sequence_mask, batch_size, seq_len, cache);
                break;
            case BottleneckType::MAX_POOL:
                result = forward_max_pool(hidden_seq_stacked, sequence_mask, batch_size, seq_len, cache);
                break;
            case BottleneckType::ATTENTION_POOL:
                result = forward_attention_pool(hidden_seq_stacked, sequence_mask, batch_size, seq_len, cache);
                break;
        }

        return {result, cache};
    }

    MatrixXf backward(const MatrixXf &d_aggregated, const BottleneckAggregationCache &cache) {
        switch (cache.type) {
            case BottleneckType::LAST_HIDDEN:
                return backward_last_hidden(d_aggregated, cache);
            case BottleneckType::MEAN_POOL:
                return backward_mean_pool(d_aggregated, cache);
            case BottleneckType::MAX_POOL:
                return backward_max_pool(d_aggregated, cache);
            case BottleneckType::ATTENTION_POOL:
                return backward_attention_pool(d_aggregated, cache);
        }
        return MatrixXf::Zero(cache.hidden_seq_stacked.rows(), cache.hidden_seq_stacked.cols());
    }

    void zero_grad() {
        if (attention_pooling) attention_pooling->zero_grad();
    }

    std::vector<std::pair<MatrixXf *, MatrixXf *> > get_params_grads() {
        if (attention_pooling) {
            return attention_pooling->get_params_grads();
        }
        return {};
    }

    std::vector<std::pair<VectorXf *, VectorXf *> > get_vector_params_grads() {
        if (attention_pooling) {
            return attention_pooling->get_vector_params_grads();
        }
        return {};
    }

private:
    std::optional<MatrixXf> create_sequence_mask_from_feature_mask(
            const std::optional<MatrixXf> &feature_mask, int batch_size, int seq_len) {
        if (!feature_mask.has_value()) {
            return std::nullopt;
        }

        const MatrixXf &mask = feature_mask.value();
        int mask_cols = mask.cols();

        // Determine the format of the feature mask
        if (mask_cols == seq_len) {
            // Already a sequence mask (B, T)
            return mask;
        } else if (mask_cols % seq_len == 0) {
            // (B, T*F) format - convert to sequence mask
            int features_per_timestep = mask_cols / seq_len;
            MatrixXf sequence_mask(batch_size, seq_len);

            for (int b = 0; b < batch_size; ++b) {
                for (int t = 0; t < seq_len; ++t) {
                    bool any_observed = false;
                    int base_idx = t * features_per_timestep;

                    for (int f = 0; f < features_per_timestep; ++f) {
                        if (mask(b, base_idx + f) > 0.5f) {
                            any_observed = true;
                            break;
                        }
                    }
                    sequence_mask(b, t) = any_observed ? 1.0f : 0.0f;
                }
            }
            return sequence_mask;
        } else {
            // (B, F) format - assume same for all timesteps
            MatrixXf sequence_mask(batch_size, seq_len);
            for (int b = 0; b < batch_size; ++b) {
                bool any_observed = mask.row(b).maxCoeff() > 0.5f;
                sequence_mask.row(b).setConstant(any_observed ? 1.0f : 0.0f);
            }
            return sequence_mask;
        }
    }

    MatrixXf forward_last_hidden(const MatrixXf &hidden_seq, int batch_size, int seq_len) {
        MatrixXf result(batch_size, hidden_dim);
        for (int b = 0; b < batch_size; ++b) {
            int last_idx = (seq_len - 1) * batch_size + b;
            result.row(b) = hidden_seq.row(last_idx);
        }
        return result;
    }

    MatrixXf forward_mean_pool(const MatrixXf &hidden_seq, const std::optional<MatrixXf> &sequence_mask,
                               int batch_size, int seq_len, BottleneckAggregationCache &cache) {
        MatrixXf result(batch_size, hidden_dim);
        MatrixXf valid_counts(batch_size, 1);

        for (int b = 0; b < batch_size; ++b) {
            VectorXf sum_hidden = VectorXf::Zero(hidden_dim);
            float count = 0.0f;

            for (int t = 0; t < seq_len; ++t) {
                int idx = t * batch_size + b;
                float mask_val = 1.0f;

                if (sequence_mask.has_value()) {
                    mask_val = sequence_mask.value()(b, t);
                }

                if (mask_val > 0.5f) {
                    sum_hidden += hidden_seq.row(idx).transpose();
                    count += 1.0f;
                }
            }

            valid_counts(b, 0) = count;
            if (count > 0.0f) {
                result.row(b) = (sum_hidden / count).transpose();
            } else {
                result.row(b).setZero();
            }
        }

        cache.valid_step_counts = valid_counts;
        return result;
    }

    MatrixXf forward_max_pool(const MatrixXf &hidden_seq, const std::optional<MatrixXf> &sequence_mask,
                              int batch_size, int seq_len, BottleneckAggregationCache &cache) {
        MatrixXf result(batch_size, hidden_dim);
        MatrixXf max_indicators = MatrixXf::Zero(batch_size * seq_len, hidden_dim);

        for (int b = 0; b < batch_size; ++b) {
            VectorXf max_vals = VectorXf::Constant(hidden_dim, -std::numeric_limits<float>::infinity());

            // Find max values considering mask
            for (int t = 0; t < seq_len; ++t) {
                int idx = t * batch_size + b;
                float mask_val = 1.0f;

                if (sequence_mask.has_value()) {
                    mask_val = sequence_mask.value()(b, t);
                }

                if (mask_val > 0.5f) {
                    for (int h = 0; h < hidden_dim; ++h) {
                        if (hidden_seq(idx, h) > max_vals(h)) {
                            max_vals(h) = hidden_seq(idx, h);
                        }
                    }
                }
            }

            // Create indicators for gradient flow
            for (int t = 0; t < seq_len; ++t) {
                int idx = t * batch_size + b;
                float mask_val = 1.0f;

                if (sequence_mask.has_value()) {
                    mask_val = sequence_mask.value()(b, t);
                }

                if (mask_val > 0.5f) {
                    for (int h = 0; h < hidden_dim; ++h) {
                        if (std::abs(hidden_seq(idx, h) - max_vals(h)) < 1e-6f) {
                            max_indicators(idx, h) = 1.0f;
                        }
                    }
                }
            }

            result.row(b) = max_vals.transpose();
        }

        cache.max_indicators = max_indicators;
        return result;
    }

    MatrixXf forward_attention_pool(const MatrixXf &hidden_seq, const std::optional<MatrixXf> &sequence_mask,
                                    int batch_size, int seq_len, BottleneckAggregationCache &cache) {
        auto [pooled, attention_cache] = attention_pooling->forward(hidden_seq, sequence_mask, batch_size, seq_len);
        cache.attention_cache = attention_cache;
        return pooled;
    }

    MatrixXf backward_last_hidden(const MatrixXf &d_aggregated, const BottleneckAggregationCache &cache) {
        MatrixXf d_hidden = MatrixXf::Zero(cache.batch_size * cache.seq_len, hidden_dim);
        for (int b = 0; b < cache.batch_size; ++b) {
            int last_idx = (cache.seq_len - 1) * cache.batch_size + b;
            d_hidden.row(last_idx) = d_aggregated.row(b);
        }
        return d_hidden;
    }

    MatrixXf backward_mean_pool(const MatrixXf &d_aggregated, const BottleneckAggregationCache &cache) {
        MatrixXf d_hidden = MatrixXf::Zero(cache.batch_size * cache.seq_len, hidden_dim);

        for (int b = 0; b < cache.batch_size; ++b) {
            float valid_count = 1.0f;
            if (cache.valid_step_counts.has_value()) {
                valid_count = std::max(1.0f, cache.valid_step_counts.value()(b, 0));
            } else {
                valid_count = static_cast<float>(cache.seq_len);
            }

            VectorXf grad_per_timestep = d_aggregated.row(b).transpose() / valid_count;

            for (int t = 0; t < cache.seq_len; ++t) {
                int idx = t * cache.batch_size + b;
                float mask_val = 1.0f;

                if (cache.sequence_mask.has_value()) {
                    mask_val = cache.sequence_mask.value()(b, t);
                }

                if (mask_val > 0.5f) {
                    d_hidden.row(idx) = grad_per_timestep.transpose();
                }
            }
        }
        return d_hidden;
    }

    MatrixXf backward_max_pool(const MatrixXf &d_aggregated, const BottleneckAggregationCache &cache) {
        if (!cache.max_indicators.has_value()) {
            throw std::runtime_error("Max indicators not found in cache");
        }

        MatrixXf d_hidden = MatrixXf::Zero(cache.batch_size * cache.seq_len, hidden_dim);
        const MatrixXf &indicators = cache.max_indicators.value();

        for (int b = 0; b < cache.batch_size; ++b) {
            for (int h = 0; h < hidden_dim; ++h) {
                float d_max_h = d_aggregated(b, h);

                // Count max achievers for this batch and feature
                int count = 0;
                for (int t = 0; t < cache.seq_len; ++t) {
                    int idx = t * cache.batch_size + b;
                    if (indicators(idx, h) > 0.5f) count++;
                }

                // Distribute gradient among max achievers
                if (count > 0) {
                    float grad_per_achiever = d_max_h / count;
                    for (int t = 0; t < cache.seq_len; ++t) {
                        int idx = t * cache.batch_size + b;
                        if (indicators(idx, h) > 0.5f) {
                            d_hidden(idx, h) = grad_per_achiever;
                        }
                    }
                }
            }
        }
        return d_hidden;
    }

    MatrixXf backward_attention_pool(const MatrixXf &d_aggregated, const BottleneckAggregationCache &cache) {
        return attention_pooling->backward(d_aggregated, cache.attention_cache.value());
    }
};

// ============================================================================
// 5. COMPLETE FLEXIBLE TEMPORAL RNN WITH DT GRADIENT COLLECTION
// ============================================================================

struct FlexibleTemporalRNNCache {
    bool has_projection;
    std::vector<LinearCache> input_proj_caches;
    std::vector<MaskProjectorCache> mask_proj_caches;
    NpTemporalRNN::RNNBackwardOutput rnn_backward_output;
    std::vector<MatrixXf> dt_gradients;

    // NEW: Store original mask sequence for proper backward pass
    std::optional<std::vector<std::optional<MatrixXf>>> original_mask_sequence;
};

class NpFlexibleTemporalRNN {
private:
    int actual_input_size;
    int rnn_expected_input_size;
    bool use_projection;
    MaskProjectionType mask_projection_type;
    std::unique_ptr<NpLinear> input_proj;
    std::unique_ptr<NpMaskProjector> mask_projector;
    std::unique_ptr<NpTemporalRNN> rnn;

public:
    NpFlexibleTemporalRNN(const NpTemporalConfig &rnn_config, int actual_in_size,
                          bool use_proj, MaskProjectionType mask_proj_type, std::mt19937 &gen)
            : actual_input_size(actual_in_size), rnn_expected_input_size(rnn_config.in_size),
              use_projection(use_proj), mask_projection_type(mask_proj_type) {
        if (use_projection && actual_input_size != rnn_expected_input_size) {
            input_proj = std::make_unique<NpLinear>(actual_input_size, rnn_expected_input_size, true, &gen);
            mask_projector = std::make_unique<NpMaskProjector>(actual_input_size, rnn_expected_input_size,
                                                               mask_projection_type, gen);
        } else if (!use_projection && actual_input_size != rnn_expected_input_size) {
            throw std::invalid_argument("Input size mismatch without projection");
        }

        rnn = std::make_unique<NpTemporalRNN>(rnn_config, gen);
    }

    std::tuple<MatrixXf, std::vector<MatrixXf>, FlexibleTemporalRNNCache> forward(
            const std::vector<MatrixXf> &X_seq,
            const std::vector<MatrixXf> &dt_seq,
            const std::vector<std::optional<MatrixXf>> &mask_seq,
            const std::vector<MatrixXf> &initial_h) {

        FlexibleTemporalRNNCache cache;
        cache.has_projection = (input_proj != nullptr);
        cache.original_mask_sequence = mask_seq;  // Store for backward pass

        int T_win = X_seq.size();
        std::vector<MatrixXf> X_to_rnn(T_win);
        std::vector<std::optional<MatrixXf>> mask_to_rnn(T_win);

        cache.input_proj_caches.resize(T_win);

        // Project inputs
        for (int t = 0; t < T_win; ++t) {
            if (input_proj) {
                auto [x_proj, proj_cache] = input_proj->forward(X_seq[t]);
                X_to_rnn[t] = x_proj;
                cache.input_proj_caches[t] = proj_cache;
            } else {
                X_to_rnn[t] = X_seq[t];
            }
        }

        // FIXED: Project mask sequence properly if needed
        if (mask_projector && std::any_of(mask_seq.begin(), mask_seq.end(),
                                          [](const auto& m) { return m.has_value(); })) {

            // Collect valid masks for sequence projection
            std::vector<MatrixXf> valid_masks;
            for (const auto& mask_opt : mask_seq) {
                if (mask_opt.has_value()) {
                    valid_masks.push_back(mask_opt.value());
                } else {
                    // Create default mask if missing
                    valid_masks.push_back(MatrixXf::Ones(X_seq[0].rows(), actual_input_size));
                }
            }

            // Project the entire sequence at once (PyTorch-like behavior)
            std::optional<MatrixXf> weights = (mask_projection_type == MaskProjectionType::MAX_POOL)
                                              ? std::make_optional(input_proj->weights)
                                              : std::nullopt;
            auto [proj_masks, mask_cache] = mask_projector->forward_sequence(valid_masks, weights);

            // Convert back to optional sequence
            for (int t = 0; t < T_win; ++t) {
                if (t < static_cast<int>(proj_masks.size())) {
                    mask_to_rnn[t] = proj_masks[t];
                } else {
                    mask_to_rnn[t] = std::nullopt;
                }
            }

            // Store single cache for entire sequence (backward compatibility)
            if (!cache.mask_proj_caches.empty()) {
                cache.mask_proj_caches.resize(1);
            } else {
                cache.mask_proj_caches.resize(1);
            }
            cache.mask_proj_caches[0] = mask_cache;
        } else {
            mask_to_rnn = mask_seq;
            cache.mask_proj_caches.clear();
        }

        auto [H_out, final_h] = rnn->forward(X_to_rnn, dt_seq, mask_to_rnn, initial_h);
        return {H_out, final_h, cache};
    }

    std::pair<std::vector<MatrixXf>, std::vector<MatrixXf>> backward(
            const MatrixXf &d_H_out, FlexibleTemporalRNNCache &cache) {

        // Get complete backward output from RNN
        cache.rnn_backward_output = rnn->backward(d_H_out);
        cache.dt_gradients = cache.rnn_backward_output.d_dt_seq_window;

        int T_win = cache.rnn_backward_output.d_X_seq_window.size();
        std::vector<MatrixXf> d_X_original(T_win);

        if (cache.has_projection && input_proj) {
            // Backward through input projection
            for (int t = 0; t < T_win; ++t) {
                d_X_original[t] = input_proj->backward(cache.rnn_backward_output.d_X_seq_window[t],
                                                       cache.input_proj_caches[t]);
            }

            // FIXED: Backward through mask projection if used
            if (mask_projector && !cache.mask_proj_caches.empty()) {
                const auto& mask_cache = cache.mask_proj_caches[0];

                // Get mask gradients from RNN if available
                if (!cache.rnn_backward_output.d_mask_seq_window.empty()) {
                    // Use sequence backward for exact PyTorch behavior
                    auto d_original_masks = mask_projector->backward_sequence(
                            cache.rnn_backward_output.d_mask_seq_window, mask_cache);

                    // Note: d_original_masks contains gradients w.r.t. original masks
                    // The mask projector's weights are updated through the backward call
                }
            }
        } else {
            d_X_original = cache.rnn_backward_output.d_X_seq_window;
        }

        return {d_X_original, cache.rnn_backward_output.d_initial_h_window};
    }

    void zero_grad() {
        rnn->zero_grad_rnn_params();
        if (input_proj) input_proj->zero_grad();
        if (mask_projector) mask_projector->zero_grad();
    }

    std::vector<std::pair<MatrixXf *, MatrixXf *>> get_params_grads() {
        auto params = rnn->get_rnn_params_grads();
        if (input_proj) {
            params.push_back({&input_proj->weights, &input_proj->grad_weights});
        }
        if (mask_projector && mask_projection_type == MaskProjectionType::LEARNED) {
            auto mask_proj_params = mask_projector->get_params_grads();
            params.insert(params.end(), mask_proj_params.begin(), mask_proj_params.end());
        }
        return params;
    }

    std::vector<std::pair<VectorXf *, VectorXf *>> get_bias_params_grads() {
        auto bias_params = rnn->get_rnn_bias_params_grads();
        auto vector_params = rnn->get_rnn_vector_params_grads();
        bias_params.insert(bias_params.end(), vector_params.begin(), vector_params.end());

        if (input_proj && input_proj->bias.size() > 0) {
            bias_params.push_back({&input_proj->bias, &input_proj->grad_bias});
        }
        if (mask_projector && mask_projection_type == MaskProjectionType::LEARNED) {
            auto mask_proj_bias = mask_projector->get_bias_params_grads();
            bias_params.insert(bias_params.end(), mask_proj_bias.begin(), mask_proj_bias.end());
        }
        return bias_params;
    }
};

// ============================================================================
// 6. FEEDBACK TRANSFORM FOR AUTOREGRESSIVE MODE
// ============================================================================

class NpFeedbackTransform {
private:
    FeedbackTransformType transform_type;
    int input_size;
    int output_size;
    std::unique_ptr<NpLinear> linear_transform;
    std::unique_ptr<NpLinear> learned_hidden;
    std::mt19937 &gen_ref;

public:
    NpFeedbackTransform(FeedbackTransformType type, int in_size, int out_size,
                        int internal_size, std::mt19937 &gen)
            : transform_type(type), input_size(in_size), output_size(out_size), gen_ref(gen) {
        if (type == FeedbackTransformType::LINEAR) {
            linear_transform = std::make_unique<NpLinear>(input_size, output_size, true, &gen_ref);
        } else if (type == FeedbackTransformType::LEARNED) {
            linear_transform = std::make_unique<NpLinear>(input_size, internal_size, true, &gen_ref);
            learned_hidden = std::make_unique<NpLinear>(internal_size, output_size, true, &gen_ref);
        } else if (type == FeedbackTransformType::IDENTITY) {
            if (input_size != output_size) {
                throw std::invalid_argument("Identity transform requires input_size == output_size");
            }
        }
    }

    std::pair<MatrixXf, std::vector<LinearCache> > forward(const MatrixXf &output) {
        std::vector<LinearCache> caches;

        switch (transform_type) {
            case FeedbackTransformType::IDENTITY:
                return {output, caches};

            case FeedbackTransformType::LINEAR: {
                auto [result, cache] = linear_transform->forward(output);
                caches.push_back(cache);
                return {result, caches};
            }

            case FeedbackTransformType::LEARNED: {
                auto [hidden, cache1] = linear_transform->forward(output);
                MatrixXf activated = hidden.array().cwiseMax(0.0f); // ReLU
                auto [result, cache2] = learned_hidden->forward(activated);
                caches.push_back(cache1);
                caches.push_back(cache2);
                // Store the pre-activation values for proper ReLU backward
                LinearCache relu_cache;
                relu_cache.input = hidden; // Store pre-activation for backward
                caches.push_back(relu_cache);
                return {result, caches};
            }
        }
        return {output, caches};
    }

    MatrixXf backward(const MatrixXf &d_output, const std::vector<LinearCache> &caches) {
        switch (transform_type) {
            case FeedbackTransformType::IDENTITY:
                return d_output;

            case FeedbackTransformType::LINEAR:
                if (caches.size() != 1) {
                    throw std::runtime_error("Expected 1 cache for linear transform");
                }
                return linear_transform->backward(d_output, caches[0]);

            case FeedbackTransformType::LEARNED: {
                if (caches.size() != 3) {
                    throw std::runtime_error("Expected 3 caches for learned transform");
                }
                MatrixXf d_activated = learned_hidden->backward(d_output, caches[1]);
                // Fix: Proper ReLU backward using stored pre-activation values
                const MatrixXf &pre_activation = caches[2].input; // Get stored pre-activation
                MatrixXf relu_mask = (pre_activation.array() > 0.0f).cast<float>(); // Create mask
                MatrixXf d_hidden = d_activated.array() * relu_mask.array(); // Apply mask to gradients
                return linear_transform->backward(d_hidden, caches[0]);
            }
        }
        return d_output;
    }

    void zero_grad() {
        if (linear_transform) linear_transform->zero_grad();
        if (learned_hidden) learned_hidden->zero_grad();
    }

    std::vector<std::pair<MatrixXf *, MatrixXf *> > get_params_grads() {
        std::vector<std::pair<MatrixXf *, MatrixXf *> > params;
        if (linear_transform) {
            params.push_back({&linear_transform->weights, &linear_transform->grad_weights});
        }
        if (learned_hidden) {
            params.push_back({&learned_hidden->weights, &learned_hidden->grad_weights});
        }
        return params;
    }

    std::vector<std::pair<VectorXf *, VectorXf *> > get_bias_params_grads() {
        std::vector<std::pair<VectorXf *, VectorXf *> > bias_params;
        if (linear_transform && linear_transform->bias.size() > 0) {
            bias_params.push_back({&linear_transform->bias, &linear_transform->grad_bias});
        }
        if (learned_hidden && learned_hidden->bias.size() > 0) {
            bias_params.push_back({&learned_hidden->bias, &learned_hidden->grad_bias});
        }
        return bias_params;
    }
};


// ============================================================================
// 9. ENHANCED OPTIMIZERS WITH COMPLETE FUNCTIONALITY
// ============================================================================

class NpSimpleSGD {
private:
    float lr;
    float momentum;
    float weight_decay;
    std::vector<MatrixXf> momentum_matrices;
    std::vector<VectorXf> momentum_vectors;
    bool initialized;

public:
    NpSimpleSGD(float learning_rate = 1e-3f, float mom = 0.0f, float wd = 0.0f)
            : lr(learning_rate), momentum(mom), weight_decay(wd), initialized(false) {
    }

    void initialize_buffers(const std::vector<std::pair<MatrixXf *, MatrixXf *> > &matrix_params,
                            const std::vector<std::pair<VectorXf *, VectorXf *> > &vector_params) {
        momentum_matrices.clear();
        momentum_vectors.clear();

        for (const auto &[param, grad]: matrix_params) {
            momentum_matrices.push_back(MatrixXf::Zero(param->rows(), param->cols()));
        }

        for (const auto &[param, grad]: vector_params) {
            momentum_vectors.push_back(VectorXf::Zero(param->size()));
        }

        initialized = true;
    }

    void step(const std::vector<std::pair<MatrixXf *, MatrixXf *> > &matrix_params,
              const std::vector<std::pair<VectorXf *, VectorXf *> > &vector_params) {
        if (!initialized) {
            initialize_buffers(matrix_params, vector_params);
        }

        // Update matrix parameters
        for (size_t i = 0; i < matrix_params.size() && i < momentum_matrices.size(); ++i) {
            auto [param, grad] = matrix_params[i];
            MatrixXf &mom_buf = momentum_matrices[i];

            MatrixXf grad_with_decay = *grad + weight_decay * (*param);
            mom_buf = momentum * mom_buf + grad_with_decay;
            *param -= lr * mom_buf;
        }

        // Update vector parameters
        for (size_t i = 0; i < vector_params.size() && i < momentum_vectors.size(); ++i) {
            auto [param, grad] = vector_params[i];
            VectorXf &mom_buf = momentum_vectors[i];

            VectorXf grad_with_decay = *grad + weight_decay * (*param);
            mom_buf = momentum * mom_buf + grad_with_decay;
            *param -= lr * mom_buf;
        }
    }

    void set_learning_rate(float new_lr) { lr = new_lr; }
    float get_learning_rate() const { return lr; }
};

class NpSimpleAdamW {
private:
    float lr, beta1, beta2, eps, weight_decay;
    int step_count;
    std::vector<MatrixXf> m_matrices, v_matrices;
    std::vector<VectorXf> m_vectors, v_vectors;
    bool initialized;

public:
    NpSimpleAdamW(float learning_rate = 2e-3f, float b1 = 0.9f, float b2 = 0.999f,
                  float epsilon = 1e-8f, float wd = 1e-4f)
            : lr(learning_rate), beta1(b1), beta2(b2), eps(epsilon), weight_decay(wd),
              step_count(0), initialized(false) {
    }

    void initialize_buffers(const std::vector<std::pair<MatrixXf *, MatrixXf *> > &matrix_params,
                            const std::vector<std::pair<VectorXf *, VectorXf *> > &vector_params) {
        m_matrices.clear();
        v_matrices.clear();
        m_vectors.clear();
        v_vectors.clear();

        for (const auto &[param, grad]: matrix_params) {
            m_matrices.push_back(MatrixXf::Zero(param->rows(), param->cols()));
            v_matrices.push_back(MatrixXf::Zero(param->rows(), param->cols()));
        }

        for (const auto &[param, grad]: vector_params) {
            m_vectors.push_back(VectorXf::Zero(param->size()));
            v_vectors.push_back(VectorXf::Zero(param->size()));
        }

        initialized = true;
    }

    void step(const std::vector<std::pair<MatrixXf *, MatrixXf *> > &matrix_params,
              const std::vector<std::pair<VectorXf *, VectorXf *> > &vector_params) {
        if (!initialized) {
            initialize_buffers(matrix_params, vector_params);
        }

        step_count++;

        float bias_correction1 = 1.0f - std::pow(beta1, step_count);
        float bias_correction2 = 1.0f - std::pow(beta2, step_count);
        float corrected_lr = lr * std::sqrt(bias_correction2) / bias_correction1;

        // Update matrix parameters
        for (size_t i = 0; i < matrix_params.size() && i < m_matrices.size(); ++i) {
            auto [param, grad] = matrix_params[i];
            MatrixXf &m = m_matrices[i];
            MatrixXf &v = v_matrices[i];

            MatrixXf grad_with_decay = *grad + weight_decay * (*param);
            m = beta1 * m + (1.0f - beta1) * grad_with_decay;
            v = beta2 * v + (1.0f - beta2) * grad_with_decay.array().square().matrix();
            MatrixXf update = (corrected_lr * m.array() / (v.array().sqrt() + eps)).matrix();
            *param -= update;
        }

        // Update vector parameters
        for (size_t i = 0; i < vector_params.size() && i < m_vectors.size(); ++i) {
            auto [param, grad] = vector_params[i];
            VectorXf &m = m_vectors[i];
            VectorXf &v = v_vectors[i];

            VectorXf grad_with_decay = *grad + weight_decay * (*param);
            m = beta1 * m + (1.0f - beta1) * grad_with_decay;
            v = beta2 * v + (1.0f - beta2) * grad_with_decay.array().square().matrix();
            VectorXf update = (corrected_lr * m.array() / (v.array().sqrt() + eps)).matrix();
            *param -= update;
        }
    }

    void set_learning_rate(float new_lr) { lr = new_lr; }
    float get_learning_rate() const { return lr; }
    int get_step_count() const { return step_count; }

    void reset() {
        step_count = 0;
        initialized = false;
        // FIXED: Clear moment vectors to avoid stale sizes when reusing the optimizer
        m_matrices.clear();
        v_matrices.clear();
        m_vectors.clear();
        v_vectors.clear();
    }
};


// ============================================================================
// 7. COMPLETE TEMPORAL AUTOENCODER WITH ALL FEATURES
// ============================================================================

// ============================================================================
// COMPLETE FIXED NpTemporalAutoencoder - 100% PYTORCH COMPATIBILITY
// ============================================================================

struct AutoencoderForwardResult {
    MatrixXf output_sequence;
    MatrixXf latent;
    float loss;
    MatrixXf encoder_hidden_seq;
    std::vector<MatrixXf> encoder_final_hidden;
    std::vector<MatrixXf> decoder_final_hidden;
};

struct AutoencoderBackwardCache {
    FlexibleTemporalRNNCache encoder_cache;
    BottleneckAggregationCache bottleneck_cache;
    LinearCache bottleneck_linear_cache;
    std::optional<LinearCache> latent_to_hidden_cache;

    // For direct mode
    std::optional<FlexibleTemporalRNNCache> decoder_cache_direct;

    // For autoregressive mode
    std::optional<std::vector<FlexibleTemporalRNNCache>> decoder_caches_autoregressive;
    std::optional<std::vector<std::vector<LinearCache>>> feedback_caches;

    // Store per-timestep output projection caches
    std::vector<LinearCache> output_proj_caches;
    LinearCache output_proj_cache; // Keep for backward compatibility
    std::optional<LinearCache> dt_predictor_cache;

    // Intermediate values needed for backward
    MatrixXf encoder_hidden_stacked;
    MatrixXf aggregated_hidden;
    MatrixXf latent;
    MatrixXf target;
    std::vector<MatrixXf> dt_decode_seq;
    int T_decode;
    int batch_size;
    AutoencoderMode mode_used;
    ForecastingMode forecasting_mode_used;
    bool used_dt_predictor = false;

    // FIXED: Store original mask sequences for exact PyTorch backward compatibility
    std::optional<std::vector<std::optional<MatrixXf>>> original_input_mask_sequence;
    std::optional<std::vector<std::optional<MatrixXf>>> decoder_mask_sequence_used;

    // Store input sequence length for proper mask handling
    int T_input = 0;
};

class NpTemporalAutoencoder {
private:
    NpTemporalConfig rnn_cfg;
    std::unique_ptr<NpFlexibleTemporalRNN> encoder;
    std::unique_ptr<NpBottleneckAggregation> bottleneck_aggregation;
    std::unique_ptr<NpLinear> bottleneck_linear;
    std::unique_ptr<NpFlexibleTemporalRNN> decoder_rnn;
    std::unique_ptr<NpLinear> latent_to_decoder_hidden;
    std::unique_ptr<NpLinear> output_proj;
    std::unique_ptr<NpLinear> dt_predictor;
    std::unique_ptr<NpMaskProjector> decoder_mask_projector;
    std::unique_ptr<NpFeedbackTransform> feedback_transform;
    std::mt19937 &gen_ref;

    // Store cache for backward pass
    AutoencoderBackwardCache forward_cache;

public:
    AutoencoderConfig ae_cfg;

    NpTemporalAutoencoder(const NpTemporalConfig &rnn_config, const AutoencoderConfig &ae_config, std::mt19937 &gen)
            : rnn_cfg(rnn_config), ae_cfg(ae_config), gen_ref(gen) {
        // Create encoder
        NpTemporalConfig encoder_cfg = rnn_config;
        encoder_cfg.in_size = ae_cfg.internal_projection_size;
        encoder = std::make_unique<NpFlexibleTemporalRNN>(encoder_cfg, ae_cfg.input_size,
                                                          ae_cfg.use_input_projection, ae_cfg.mask_projection_type,
                                                          gen);

        // Create bottleneck
        bottleneck_aggregation = std::make_unique<NpBottleneckAggregation>(
                ae_cfg.bottleneck_type, encoder_cfg.hid_size, ae_cfg.attention_context_dim, gen);
        bottleneck_linear = std::make_unique<NpLinear>(encoder_cfg.hid_size, ae_cfg.latent_size, true, &gen);

        // Create decoder
        NpTemporalConfig decoder_cfg = rnn_config;
        decoder_cfg.in_size = ae_cfg.internal_projection_size;
        decoder_rnn = std::make_unique<NpFlexibleTemporalRNN>(decoder_cfg, ae_cfg.latent_size,
                                                              true, ae_cfg.mask_projection_type, gen);

        // Latent to hidden mapping (for non-last_hidden bottlenecks)
        if (ae_cfg.bottleneck_type != BottleneckType::LAST_HIDDEN) {
            latent_to_decoder_hidden = std::make_unique<NpLinear>(
                    ae_cfg.latent_size, decoder_cfg.num_layers * decoder_cfg.hid_size, true, &gen);
        }

        // Output projection
        output_proj = std::make_unique<NpLinear>(decoder_cfg.hid_size, ae_cfg.input_size, true, &gen);

        // Decoder mask projector (optional)
        if (ae_cfg.pass_mask_to_decoder_rnn) {
            decoder_mask_projector = std::make_unique<NpMaskProjector>(
                    ae_cfg.input_size, ae_cfg.latent_size, ae_cfg.mask_projection_type, gen);
        }

        // Future dt prediction (optional)
        if (ae_cfg.predict_future_dt && ae_cfg.dt_prediction_method == DTPresictionMethod::LEARNED) {
            dt_predictor = std::make_unique<NpLinear>(ae_cfg.latent_size, ae_cfg.forecast_horizon, true, &gen);
        }

        // Autoregressive feedback transformation (optional)
        if (ae_cfg.forecasting_mode == ForecastingMode::AUTOREGRESSIVE) {
            feedback_transform = std::make_unique<NpFeedbackTransform>(
                    ae_cfg.feedback_transform_type, ae_cfg.input_size, ae_cfg.latent_size,
                    ae_cfg.internal_projection_size, gen);
        }
    }

    AutoencoderForwardResult forward(const std::vector<MatrixXf> &X_seq,
                                     const std::vector<MatrixXf> &dt_seq,
                                     const std::vector<std::optional<MatrixXf>> &mask_seq,
                                     const std::vector<MatrixXf> &initial_h_encoder,
                                     const std::vector<MatrixXf> &initial_h_decoder,
                                     const std::optional<MatrixXf> &target_sequence = std::nullopt) {
        AutoencoderForwardResult result;

        if (X_seq.empty()) {
            throw std::invalid_argument("Empty input sequence");
        }

        int batch_size = X_seq[0].rows();
        int T_in = X_seq.size();

        // Clear previous cache
        forward_cache = AutoencoderBackwardCache();
        forward_cache.T_input = T_in;

        // FIXED: Store original mask sequence for exact PyTorch behavior
        forward_cache.original_input_mask_sequence = mask_seq;

        // Encode
        auto [encoder_hidden_stacked, encoder_final_h, encoder_cache] =
                encoder->forward(X_seq, dt_seq, mask_seq, initial_h_encoder);

        result.encoder_hidden_seq = encoder_hidden_stacked;
        result.encoder_final_hidden = encoder_final_h;

        // Bottleneck aggregation
        std::optional<MatrixXf> feature_mask_for_bottleneck = std::nullopt;
        if (!mask_seq.empty() && mask_seq[0].has_value()) {
            // Stack masks for bottleneck - this is used for sequence-level aggregation
            MatrixXf stacked_masks(batch_size, T_in * mask_seq[0].value().cols());
            for (int t = 0; t < T_in; ++t) {
                if (mask_seq[t].has_value()) {
                    int start_col = t * mask_seq[t].value().cols();
                    stacked_masks.block(0, start_col, batch_size, mask_seq[t].value().cols()) = mask_seq[t].value();
                } else {
                    int mask_cols = mask_seq[0].value().cols();
                    int start_col = t * mask_cols;
                    stacked_masks.block(0, start_col, batch_size, mask_cols).setOnes();
                }
            }
            feature_mask_for_bottleneck = stacked_masks;
        }

        auto [aggregated_hidden, bottleneck_cache] =
                bottleneck_aggregation->forward(encoder_hidden_stacked, feature_mask_for_bottleneck, batch_size, T_in);

        // Project to latent
        auto [latent, bottleneck_linear_cache] = bottleneck_linear->forward(aggregated_hidden);
        result.latent = latent;

        // Determine decode parameters
        int T_decode;
        MatrixXf target;

        if (ae_cfg.mode == AutoencoderMode::RECONSTRUCTION) {
            T_decode = T_in;
            target = MatrixXf(T_decode * batch_size, ae_cfg.input_size);
            for (int t = 0; t < T_decode; ++t) {
                target.block(t * batch_size, 0, batch_size, ae_cfg.input_size) = X_seq[t];
            }
        } else {
            T_decode = ae_cfg.forecast_horizon;
            if (!target_sequence.has_value()) {
                throw std::invalid_argument("Target sequence required for forecasting");
            }
            target = target_sequence.value();
        }

        // Compute decode dt with full gradient tracking
        auto dt_decode_seq = compute_decode_dt_complete(dt_seq, latent, T_decode, batch_size);

        // Initialize decoder
        std::vector<MatrixXf> decoder_initial_h = initial_h_decoder;
        std::optional<LinearCache> latent_cache;

        if (decoder_initial_h.empty() && ae_cfg.bottleneck_type == BottleneckType::LAST_HIDDEN) {
            decoder_initial_h = encoder_final_h;
        } else if (decoder_initial_h.empty() && latent_to_decoder_hidden) {
            auto [h_flat, l_cache] = latent_to_decoder_hidden->forward(latent);
            latent_cache = l_cache;
            decoder_initial_h.resize(rnn_cfg.num_layers);
            for (int l = 0; l < rnn_cfg.num_layers; ++l) {
                decoder_initial_h[l] = h_flat.block(0, l * rnn_cfg.hid_size, batch_size, rnn_cfg.hid_size);
            }
        }

        // Decode based on mode
        MatrixXf output_sequence;
        std::vector<MatrixXf> decoder_final_h;

        if (ae_cfg.mode == AutoencoderMode::FORECASTING && ae_cfg.forecasting_mode == ForecastingMode::AUTOREGRESSIVE) {
            auto [output_seq, final_h, autoregressive_caches] = decode_autoregressive_complete(
                    latent, dt_decode_seq, T_decode, batch_size, decoder_initial_h, mask_seq);
            output_sequence = output_seq;
            decoder_final_h = final_h;
            forward_cache.decoder_caches_autoregressive = std::get<0>(autoregressive_caches);
            forward_cache.feedback_caches = std::get<1>(autoregressive_caches);
            forward_cache.output_proj_caches = std::get<2>(autoregressive_caches);
        } else {
            auto [decoder_hidden_seq, final_h, decoder_cache] = decode_direct_complete(
                    latent, dt_decode_seq, T_decode, batch_size, decoder_initial_h, mask_seq);

            // Apply output projection to convert hidden states to output
            auto [projected_output, output_cache] = output_proj->forward(decoder_hidden_seq);
            output_sequence = projected_output;
            forward_cache.output_proj_cache = output_cache;

            decoder_final_h = final_h;
            forward_cache.decoder_cache_direct = decoder_cache;
        }

        result.output_sequence = output_sequence;
        result.decoder_final_hidden = decoder_final_h;

        // Compute loss with temporal weighting
        MatrixXf diff = result.output_sequence - target;
        MatrixXf element_loss = diff.array().square();

        // Apply temporal weighting if configured
        if (ae_cfg.loss_ramp_start != 1.0f || ae_cfg.loss_ramp_end != 1.0f) {
            VectorXf time_weights = VectorXf::LinSpaced(T_decode, ae_cfg.loss_ramp_start, ae_cfg.loss_ramp_end);
            for (int t = 0; t < T_decode; ++t) {
                int start_row = t * batch_size;
                element_loss.block(start_row, 0, batch_size, ae_cfg.input_size) *= time_weights(t);
            }
        }

        result.loss = element_loss.mean();

        // Store cache for backward pass
        forward_cache.encoder_cache = encoder_cache;
        forward_cache.bottleneck_cache = bottleneck_cache;
        forward_cache.bottleneck_linear_cache = bottleneck_linear_cache;
        forward_cache.encoder_hidden_stacked = encoder_hidden_stacked;
        forward_cache.aggregated_hidden = aggregated_hidden;
        forward_cache.latent = latent;
        forward_cache.target = target;
        forward_cache.dt_decode_seq = dt_decode_seq;
        forward_cache.T_decode = T_decode;
        forward_cache.batch_size = batch_size;
        forward_cache.mode_used = ae_cfg.mode;
        forward_cache.forecasting_mode_used = ae_cfg.forecasting_mode;

        if (latent_cache.has_value()) {
            forward_cache.latent_to_hidden_cache = latent_cache;
        }

        return result;
    }

    std::tuple<std::vector<MatrixXf>, // d X seq
            std::vector<MatrixXf>, // d encoder h₀
            std::vector<MatrixXf> > // d decoder h₀
    backward(
            const AutoencoderForwardResult &forward_result,
            const std::optional<MatrixXf> &target_mask) {
        const auto &cache = forward_cache;
        const int T_decode = cache.T_decode;
        const int batch_size = cache.batch_size;

        // Compute gradient w.r.t. output
        MatrixXf d_output_sequence = 2.0f * (forward_result.output_sequence - cache.target); // MSE grad

        // Apply temporal ramp weighting
        if (ae_cfg.loss_ramp_start != 1.0f || ae_cfg.loss_ramp_end != 1.0f) {
            const VectorXf ramp = VectorXf::LinSpaced(T_decode, ae_cfg.loss_ramp_start, ae_cfg.loss_ramp_end);
            for (int t = 0; t < T_decode; ++t) {
                d_output_sequence.block(t * batch_size, 0, batch_size, ae_cfg.input_size) *= ramp(t);
            }
        }

        // Apply element mask if provided
        if (target_mask) {
            d_output_sequence.array() *= target_mask->array();
            const float n_valid = std::max(target_mask->sum(), 1.0f);
            d_output_sequence /= n_valid;
        } else {
            // Mean over all elements
            d_output_sequence /= (T_decode * batch_size * ae_cfg.input_size);
        }

        // Backward through decoder based on mode
        const bool autoregressive = cache.mode_used == AutoencoderMode::FORECASTING &&
                                    cache.forecasting_mode_used == ForecastingMode::AUTOREGRESSIVE;

        MatrixXf d_decoder_hidden_from_output;
        std::vector<MatrixXf> d_latent_seq;
        std::vector<MatrixXf> d_decoder_initial_h;
        std::vector<MatrixXf> d_dt_decode_seq;

        if (autoregressive) {
            // For autoregressive: handle per-timestep output projections
            std::tie(d_latent_seq, d_decoder_initial_h, d_dt_decode_seq) =
                    backward_autoregressive_complete(d_output_sequence, cache);
        } else {
            // For direct: single output projection
            d_decoder_hidden_from_output = output_proj->backward(d_output_sequence, cache.output_proj_cache);
            std::tie(d_latent_seq, d_decoder_initial_h) = backward_direct_complete(d_decoder_hidden_from_output, cache);
        }

        // Accumulate gradients w.r.t. latent
        MatrixXf d_latent = MatrixXf::Zero(batch_size, ae_cfg.latent_size);
        for (const auto &d_lat : d_latent_seq) {
            d_latent += d_lat;
        }

        // Backward through latent to hidden mapping if used
        if (cache.latent_to_hidden_cache && latent_to_decoder_hidden) {
            MatrixXf d_h_flat(batch_size, rnn_cfg.num_layers * rnn_cfg.hid_size);
            for (int l = 0; l < static_cast<int>(d_decoder_initial_h.size()); ++l) {
                d_h_flat.block(0, l * rnn_cfg.hid_size, batch_size, rnn_cfg.hid_size) = d_decoder_initial_h[l];
            }
            d_latent += latent_to_decoder_hidden->backward(d_h_flat, *cache.latent_to_hidden_cache);
        }

        // Backward through future dt predictor if used
        if (dt_predictor && cache.dt_predictor_cache && cache.used_dt_predictor && !d_dt_decode_seq.empty()) {
            MatrixXf d_future_dt(batch_size, T_decode);
            d_future_dt.setZero();
            for (int t = 0; t < T_decode && t < static_cast<int>(d_dt_decode_seq.size()); ++t) {
                d_future_dt.col(t) = d_dt_decode_seq[t].col(0);
            }
            d_latent += dt_predictor->backward(d_future_dt, *cache.dt_predictor_cache);
        }

        // Backward through bottleneck and encoder
        MatrixXf d_aggregated_hidden = bottleneck_linear->backward(d_latent, cache.bottleneck_linear_cache);
        MatrixXf d_encoder_hidden_seq = bottleneck_aggregation->backward(d_aggregated_hidden, cache.bottleneck_cache);
        auto [d_X_seq, d_encoder_initial_h] = encoder->backward(d_encoder_hidden_seq, forward_cache.encoder_cache);

        return {d_X_seq, d_encoder_initial_h, d_decoder_initial_h};
    }

    void zero_grad() {
        encoder->zero_grad();
        bottleneck_aggregation->zero_grad();
        bottleneck_linear->zero_grad();
        decoder_rnn->zero_grad();
        if (latent_to_decoder_hidden) latent_to_decoder_hidden->zero_grad();
        output_proj->zero_grad();
        if (dt_predictor) dt_predictor->zero_grad();
        if (decoder_mask_projector) decoder_mask_projector->zero_grad();
        if (feedback_transform) feedback_transform->zero_grad();
    }

    std::vector<std::pair<MatrixXf *, MatrixXf *>> get_all_params_grads() {
        std::vector<std::pair<MatrixXf *, MatrixXf *>> all_params;

        // Encoder parameters
        auto encoder_params = encoder->get_params_grads();
        all_params.insert(all_params.end(), encoder_params.begin(), encoder_params.end());

        // Bottleneck parameters
        auto bottleneck_agg_params = bottleneck_aggregation->get_params_grads();
        all_params.insert(all_params.end(), bottleneck_agg_params.begin(), bottleneck_agg_params.end());
        all_params.push_back({&bottleneck_linear->weights, &bottleneck_linear->grad_weights});

        // Decoder parameters
        auto decoder_params = decoder_rnn->get_params_grads();
        all_params.insert(all_params.end(), decoder_params.begin(), decoder_params.end());

        if (latent_to_decoder_hidden) {
            all_params.push_back({&latent_to_decoder_hidden->weights, &latent_to_decoder_hidden->grad_weights});
        }

        all_params.push_back({&output_proj->weights, &output_proj->grad_weights});

        // Optional components
        if (dt_predictor) {
            all_params.push_back({&dt_predictor->weights, &dt_predictor->grad_weights});
        }

        if (decoder_mask_projector) {
            auto mask_proj_params = decoder_mask_projector->get_params_grads();
            all_params.insert(all_params.end(), mask_proj_params.begin(), mask_proj_params.end());
        }

        if (feedback_transform) {
            auto feedback_params = feedback_transform->get_params_grads();
            all_params.insert(all_params.end(), feedback_params.begin(), feedback_params.end());
        }

        return all_params;
    }

    std::vector<std::pair<VectorXf *, VectorXf *>> get_all_bias_params_grads() {
        std::vector<std::pair<VectorXf *, VectorXf *>> all_bias_params;

        // Encoder bias parameters
        auto encoder_bias = encoder->get_bias_params_grads();
        all_bias_params.insert(all_bias_params.end(), encoder_bias.begin(), encoder_bias.end());

        // Bottleneck bias parameters
        auto bottleneck_agg_bias = bottleneck_aggregation->get_vector_params_grads();
        all_bias_params.insert(all_bias_params.end(), bottleneck_agg_bias.begin(), bottleneck_agg_bias.end());

        if (bottleneck_linear->bias.size() > 0) {
            all_bias_params.push_back({&bottleneck_linear->bias, &bottleneck_linear->grad_bias});
        }

        // Decoder bias parameters
        auto decoder_bias = decoder_rnn->get_bias_params_grads();
        all_bias_params.insert(all_bias_params.end(), decoder_bias.begin(), decoder_bias.end());

        if (latent_to_decoder_hidden && latent_to_decoder_hidden->bias.size() > 0) {
            all_bias_params.push_back({&latent_to_decoder_hidden->bias, &latent_to_decoder_hidden->grad_bias});
        }

        if (output_proj->bias.size() > 0) {
            all_bias_params.push_back({&output_proj->bias, &output_proj->grad_bias});
        }

        // Optional component biases
        if (dt_predictor && dt_predictor->bias.size() > 0) {
            all_bias_params.push_back({&dt_predictor->bias, &dt_predictor->grad_bias});
        }

        if (decoder_mask_projector) {
            auto mask_proj_bias = decoder_mask_projector->get_bias_params_grads();
            all_bias_params.insert(all_bias_params.end(), mask_proj_bias.begin(), mask_proj_bias.end());
        }

        if (feedback_transform) {
            auto feedback_bias = feedback_transform->get_bias_params_grads();
            all_bias_params.insert(all_bias_params.end(), feedback_bias.begin(), feedback_bias.end());
        }

        return all_bias_params;
    }

private:
    std::vector<MatrixXf> compute_decode_dt_complete(const std::vector<MatrixXf> &dt_seq, const MatrixXf &latent,
                                                     int T_decode, int batch_size) {
        std::vector<MatrixXf> dt_decode_seq(T_decode);

        if (!ae_cfg.predict_future_dt || !dt_predictor) {
            // Repeat last dt
            MatrixXf last_dt = dt_seq.back();
            for (int t = 0; t < T_decode; ++t) {
                dt_decode_seq[t] = last_dt;
            }
            forward_cache.used_dt_predictor = false;
        } else {
            // Use learned predictor
            auto [future_dt, dt_cache] = dt_predictor->forward(latent);
            forward_cache.dt_predictor_cache = dt_cache;
            forward_cache.used_dt_predictor = true;

            for (int t = 0; t < T_decode; ++t) {
                MatrixXf dt_t(batch_size, 1);
                for (int b = 0; b < batch_size; ++b) {
                    dt_t(b, 0) = future_dt(b, t);
                }
                dt_decode_seq[t] = dt_t;
            }
        }

        return dt_decode_seq;
    }

    // FIXED: Direct decode with exact PyTorch mask handling
    std::tuple<MatrixXf, std::vector<MatrixXf>, FlexibleTemporalRNNCache> decode_direct_complete(
            const MatrixXf &latent, const std::vector<MatrixXf> &dt_decode_seq,
            int T_decode, int batch_size, const std::vector<MatrixXf> &initial_h_decoder,
            const std::optional<std::vector<std::optional<MatrixXf>>> &original_input_mask_sequence) {

        // Create latent sequence
        std::vector<MatrixXf> latent_seq(T_decode);
        std::vector<std::optional<MatrixXf>> decoder_mask_seq(T_decode, std::nullopt);

        for (int t = 0; t < T_decode; ++t) {
            latent_seq[t] = latent;
        }

        // FIXED: Handle decoder mask with exact PyTorch behavior
        if (ae_cfg.pass_mask_to_decoder_rnn &&
            decoder_mask_projector &&
            original_input_mask_sequence.has_value()) {

            const auto &input_mask_seq = original_input_mask_sequence.value();

            // Collect valid input masks
            std::vector<MatrixXf> valid_input_masks;
            for (const auto& mask_opt : input_mask_seq) {
                if (mask_opt.has_value()) {
                    valid_input_masks.push_back(mask_opt.value());
                } else {
                    valid_input_masks.push_back(MatrixXf::Ones(batch_size, ae_cfg.input_size));
                }
            }

            if (ae_cfg.mode == AutoencoderMode::RECONSTRUCTION) {
                // EXACT PYTORCH: Project each input mask timestep to corresponding decoder timestep
                if (!valid_input_masks.empty()) {
                    auto [projected_masks, unused_cache] = decoder_mask_projector->forward_sequence(valid_input_masks);

                    // Map to decoder timesteps
                    for (int t = 0; t < T_decode; ++t) {
                        if (t < static_cast<int>(projected_masks.size())) {
                            decoder_mask_seq[t] = projected_masks[t];
                        } else if (!projected_masks.empty()) {
                            // Repeat last if decoder sequence is longer
                            decoder_mask_seq[t] = projected_masks.back();
                        }
                    }
                }
            } else {
                // EXACT PYTORCH: For forecasting, use last input mask for all decode steps
                if (!valid_input_masks.empty()) {
                    auto [proj_mask, unused_cache] = decoder_mask_projector->forward(valid_input_masks.back());
                    for (int t = 0; t < T_decode; ++t) {
                        decoder_mask_seq[t] = proj_mask;
                    }
                }
            }

            // Store the decoder mask sequence used for backward pass
            forward_cache.decoder_mask_sequence_used = decoder_mask_seq;
        }

        // Forward through decoder
        auto [decoder_hidden_stacked, decoder_final_h, decoder_cache] =
                decoder_rnn->forward(latent_seq, dt_decode_seq, decoder_mask_seq, initial_h_decoder);

        return {decoder_hidden_stacked, decoder_final_h, decoder_cache};
    }

    // FIXED: Autoregressive decode with exact PyTorch mask handling
    std::tuple<MatrixXf, std::vector<MatrixXf>, std::tuple<std::vector<FlexibleTemporalRNNCache>,
            std::vector<std::vector<LinearCache>>,
            std::vector<LinearCache>>> decode_autoregressive_complete(
            const MatrixXf &latent, const std::vector<MatrixXf> &dt_decode_seq,
            int T_decode, int batch_size, const std::vector<MatrixXf> &initial_h_decoder,
            const std::optional<std::vector<std::optional<MatrixXf>>> &original_input_mask_sequence) {

        if (ae_cfg.forecasting_mode != ForecastingMode::AUTOREGRESSIVE) {
            throw std::invalid_argument("decode_autoregressive_complete called but forecasting_mode is not AUTOREGRESSIVE");
        }

        if (!feedback_transform) {
            throw std::runtime_error("Feedback transform not initialized for autoregressive mode");
        }

        // Initialize decoder hidden states
        std::vector<MatrixXf> decoder_h_states = initial_h_decoder;
        if (decoder_h_states.empty()) {
            decoder_h_states.resize(rnn_cfg.num_layers);
            for (int l = 0; l < rnn_cfg.num_layers; ++l) {
                decoder_h_states[l] = MatrixXf::Zero(batch_size, rnn_cfg.hid_size);
            }
        }

        // Initialize first input (use latent)
        MatrixXf current_input = latent;
        std::vector<FlexibleTemporalRNNCache> step_caches;
        std::vector<std::vector<LinearCache>> feedback_caches_all;
        std::vector<MatrixXf> outputs_all;
        std::vector<LinearCache> output_caches_all;

        // FIXED: Handle decoder mask with exact PyTorch behavior for autoregressive
        std::optional<MatrixXf> projected_mask_for_all_steps = std::nullopt;
        if (ae_cfg.pass_mask_to_decoder_rnn &&
            decoder_mask_projector &&
            original_input_mask_sequence.has_value()) {

            const auto &input_mask_seq = original_input_mask_sequence.value();

            // EXACT PYTORCH: Use last known mask (matches Python's [:, -1:, :] behavior)
            MatrixXf last_known_mask;
            if (!input_mask_seq.empty() && input_mask_seq.back().has_value()) {
                last_known_mask = input_mask_seq.back().value();
            } else {
                last_known_mask = MatrixXf::Ones(batch_size, ae_cfg.input_size);
            }

            auto [proj_last_mask, unused_cache] = decoder_mask_projector->forward(last_known_mask);
            projected_mask_for_all_steps = proj_last_mask;
        }

        step_caches.reserve(T_decode);
        feedback_caches_all.reserve(T_decode);
        outputs_all.reserve(T_decode);
        output_caches_all.reserve(T_decode);

        for (int t = 0; t < T_decode; ++t) {
            // Single timestep forward
            std::vector<MatrixXf> input_seq = {current_input};
            std::vector<MatrixXf> dt_seq = {dt_decode_seq[t]};
            std::vector<std::optional<MatrixXf>> mask_seq = {std::nullopt};

            // Apply mask for this timestep if configured
            if (projected_mask_for_all_steps.has_value()) {
                mask_seq[0] = projected_mask_for_all_steps.value();
            }

            auto [step_hidden, step_final_h, step_cache] = decoder_rnn->forward(
                    input_seq, dt_seq, mask_seq, decoder_h_states);
            step_caches.push_back(step_cache);
            decoder_h_states = step_final_h;

            // Project to output
            auto [step_output, output_cache] = output_proj->forward(step_hidden);
            outputs_all.push_back(step_output);
            output_caches_all.push_back(output_cache);

            // Prepare next input (transform output back to latent space)
            if (t < T_decode - 1) {
                auto [next_input, feedback_cache] = feedback_transform->forward(step_output);
                current_input = next_input;
                feedback_caches_all.push_back(feedback_cache);
            } else {
                feedback_caches_all.push_back({}); // Empty cache for last timestep
            }
        }

        // Stack outputs
        MatrixXf output_sequence(T_decode * batch_size, ae_cfg.input_size);
        for (int t = 0; t < T_decode; ++t) {
            output_sequence.block(t * batch_size, 0, batch_size, ae_cfg.input_size) = outputs_all[t];
        }

        return {output_sequence, decoder_h_states, {step_caches, feedback_caches_all, output_caches_all}};
    }

    std::pair<std::vector<MatrixXf>, std::vector<MatrixXf>> backward_direct_complete(
            const MatrixXf &d_decoder_hidden, const AutoencoderBackwardCache &cache) {
        if (!cache.decoder_cache_direct.has_value()) {
            throw std::runtime_error("Direct decoder cache not available");
        }

        // Backward through decoder RNN
        auto [d_latent_seq, d_decoder_initial_h] = decoder_rnn->backward(d_decoder_hidden,
                                                                         const_cast<FlexibleTemporalRNNCache &>(cache.decoder_cache_direct.value()));

        return {d_latent_seq, d_decoder_initial_h};
    }

    std::tuple<std::vector<MatrixXf>, std::vector<MatrixXf>, std::vector<MatrixXf>> backward_autoregressive_complete(
            const MatrixXf &d_decoder_hidden, const AutoencoderBackwardCache &cache) {
        if (!cache.decoder_caches_autoregressive.has_value() || !cache.feedback_caches.has_value()) {
            throw std::runtime_error("Autoregressive decoder caches not available");
        }

        const auto &step_caches = cache.decoder_caches_autoregressive.value();
        const auto &feedback_caches = cache.feedback_caches.value();
        int T_decode = cache.T_decode;
        int batch_size = cache.batch_size;

        std::vector<MatrixXf> d_latent_seq;
        std::vector<MatrixXf> d_decoder_initial_h;
        std::vector<MatrixXf> d_dt_decode_seq(T_decode);

        // Initialize gradients
        MatrixXf d_next_input = MatrixXf::Zero(batch_size, ae_cfg.latent_size);

        // Backward through timesteps in reverse order
        for (int t = T_decode - 1; t >= 0; --t) {
            // Get gradient for this timestep's output
            MatrixXf d_step_output = d_decoder_hidden.block(t * batch_size, 0, batch_size, ae_cfg.input_size);

            // Backward through feedback transform (except for last timestep)
            if (t < T_decode - 1 && feedback_transform) {
                MatrixXf d_feedback_input = feedback_transform->backward(d_next_input, feedback_caches[t]);
                d_step_output += d_feedback_input;
            }

            // Backward through output projection
            MatrixXf d_step_hidden;
            if (t < static_cast<int>(cache.output_proj_caches.size())) {
                d_step_hidden = output_proj->backward(d_step_output, cache.output_proj_caches[t]);
            } else {
                // Fallback to general cache if per-timestep cache not available
                d_step_hidden = output_proj->backward(d_step_output, cache.output_proj_cache);
            }

            // Backward through decoder RNN for this timestep
            auto [d_step_latent_seq, d_step_decoder_h] = decoder_rnn->backward(d_step_hidden,
                                                                               const_cast<FlexibleTemporalRNNCache &>(step_caches[t]));

            // Accumulate gradients
            if (!d_step_latent_seq.empty()) {
                d_latent_seq.insert(d_latent_seq.begin(), d_step_latent_seq.begin(), d_step_latent_seq.end());
                d_next_input = d_step_latent_seq[0]; // For next (previous) timestep
            }

            if (t == 0) {
                d_decoder_initial_h = d_step_decoder_h;
            }

            // Get dt gradients from decoder step cache
            if (t < static_cast<int>(step_caches.size())) {
                d_dt_decode_seq[t] = step_caches[t].dt_gradients.empty()
                                     ? MatrixXf::Zero(batch_size, 1)
                                     : step_caches[t].dt_gradients[0];
            } else {
                d_dt_decode_seq[t] = MatrixXf::Zero(batch_size, 1);
            }
        }

        return {d_latent_seq, d_decoder_initial_h, d_dt_decode_seq};
    }
};

// ============================================================================
// 8. COMPLETE ONLINE LEARNER WITH FULL FUNCTIONALITY
// ============================================================================

class NpOnlineLearner {
private:
    std::unique_ptr<NpTemporalAutoencoder> autoencoder;
    std::unique_ptr<NpSimpleAdamW> optimizer;
    NpTemporalConfig opt_cfg;

    // Streaming state
    std::vector<MatrixXf> h_states_encoder;
    std::vector<MatrixXf> h_states_decoder;

    // TBPTT window
    std::deque<MatrixXf> win_X;
    std::deque<MatrixXf> win_dt;
    std::deque<std::optional<MatrixXf> > win_mask;
    std::deque<MatrixXf> win_targets;

public:
    NpOnlineLearner(std::unique_ptr<NpTemporalAutoencoder> ae, const NpTemporalConfig &cfg)
            : autoencoder(std::move(ae)), opt_cfg(cfg) {
        optimizer = std::make_unique<NpSimpleAdamW>(cfg.lr, cfg.beta1, cfg.beta2, cfg.eps_adam, cfg.weight_decay);
        reset_streaming_state();
    }

    void reset_streaming_state(int batch_size = 1) {
        h_states_encoder.clear();
        h_states_decoder.clear();

        for (int l = 0; l < opt_cfg.num_layers; ++l) {
            h_states_encoder.push_back(MatrixXf::Zero(batch_size, opt_cfg.hid_size));
            h_states_decoder.push_back(MatrixXf::Zero(batch_size, opt_cfg.hid_size));
        }

        win_X.clear();
        win_dt.clear();
        win_mask.clear();
        win_targets.clear();
    }

    std::pair<float, MatrixXf> step_stream(const MatrixXf &x_t, const MatrixXf &dt_t,
                                           const std::optional<MatrixXf> &mask_t = std::nullopt,
                                           const std::optional<MatrixXf> &target_t = std::nullopt) {
        int current_bs = x_t.rows();
        if (h_states_encoder.empty() || h_states_encoder[0].rows() != current_bs) {
            reset_streaming_state(current_bs);
        }

        // Update window
        win_X.push_back(x_t);
        win_dt.push_back(dt_t);
        win_mask.push_back(mask_t);
        if (target_t.has_value()) {
            win_targets.push_back(target_t.value());
        }

        // Trim window
        while (static_cast<int>(win_X.size()) > opt_cfg.tbptt_steps) {
            win_X.pop_front();
            win_dt.pop_front();
            win_mask.pop_front();
            if (!win_targets.empty()) win_targets.pop_front();
        }

        if (static_cast<int>(win_X.size()) < 2) {
            return {0.0f, x_t};
        }

        // Prepare sequences
        std::vector<MatrixXf> X_seq(win_X.begin(), win_X.end());
        std::vector<MatrixXf> dt_seq(win_dt.begin(), win_dt.end());
        std::vector<std::optional<MatrixXf> > mask_seq(win_mask.begin(), win_mask.end());

        std::optional<MatrixXf> target_seq;
        if (!win_targets.empty()) {
            target_seq = win_targets.back();
        }

        // COMPLETE: Get all parameters for optimization
        auto matrix_params = autoencoder->get_all_params_grads();
        auto vector_params = autoencoder->get_all_bias_params_grads();

        // Initialize optimizer if needed
        static bool optimizer_initialized = false;
        if (!optimizer_initialized) {
            optimizer->initialize_buffers(matrix_params, vector_params);
            optimizer_initialized = true;
        }

        // Zero gradients
        for (auto [param, grad]: matrix_params) {
            grad->setZero();
        }
        for (auto [param, grad]: vector_params) {
            grad->setZero();
        }
        autoencoder->zero_grad();

        // Forward pass
        auto result = autoencoder->forward(X_seq, dt_seq, mask_seq, h_states_encoder, h_states_decoder, target_seq);

        // COMPLETE: Full backward pass and optimization
        if (std::isfinite(result.loss)) {
            // Create target mask for reconstruction mode
            std::optional<MatrixXf> target_mask;
            if (autoencoder->ae_cfg.mode == AutoencoderMode::RECONSTRUCTION && !mask_seq.empty()) {
                // Create stacked target mask from input masks
                int T_decode = static_cast<int>(mask_seq.size());
                target_mask = MatrixXf(T_decode * current_bs, autoencoder->ae_cfg.input_size);
                for (int t = 0; t < T_decode; ++t) {
                    if (mask_seq[t].has_value()) {
                        target_mask.value().block(t * current_bs, 0, current_bs, autoencoder->ae_cfg.input_size) =
                                mask_seq[t].value();
                    } else {
                        target_mask.value().block(t * current_bs, 0, current_bs, autoencoder->ae_cfg.input_size).
                                setOnes();
                    }
                }
            }

            // COMPLETE: Full backward pass
            auto [d_X_seq, d_enc_h, d_dec_h] = autoencoder->backward(result, target_mask);

            // Gradient clipping if configured
            if (opt_cfg.clip_grad_norm.has_value()) {
                float total_norm = 0.0f;
                for (auto [param, grad]: matrix_params) {
                    total_norm += grad->array().square().sum();
                }
                for (auto [param, grad]: vector_params) {
                    total_norm += grad->array().square().sum();
                }
                total_norm = std::sqrt(total_norm);

                if (total_norm > opt_cfg.clip_grad_norm.value()) {
                    float scale = opt_cfg.clip_grad_norm.value() / total_norm;
                    for (auto [param, grad]: matrix_params) {
                        *grad *= scale;
                    }
                    for (auto [param, grad]: vector_params) {
                        *grad *= scale;
                    }
                }
            }

            // COMPLETE: Optimizer step
            optimizer->step(matrix_params, vector_params);
        }

        // Update streaming states
        h_states_encoder = result.encoder_final_hidden;
        h_states_decoder = result.decoder_final_hidden;

        // Extract prediction
        MatrixXf prediction;
        if (autoencoder->ae_cfg.mode == AutoencoderMode::RECONSTRUCTION) {
            // Get last timestep for reconstruction
            int output_timesteps = result.output_sequence.rows() / current_bs;
            if (output_timesteps > 0) {
                prediction = result.output_sequence.block((output_timesteps - 1) * current_bs, 0,
                                                          current_bs, result.output_sequence.cols());
            } else {
                prediction = x_t; // Fallback
            }
        } else {
            // For forecasting, return the full forecast
            prediction = result.output_sequence;
        }

        return {result.loss, prediction};
    }

    // COMPLETE: Prediction-only method without learning
    MatrixXf predict_single(const MatrixXf &x_t, const MatrixXf &dt_t,
                            const std::optional<MatrixXf> &mask_t = std::nullopt) {
        int current_bs = x_t.rows();
        if (h_states_encoder.empty() || h_states_encoder[0].rows() != current_bs) {
            reset_streaming_state(current_bs);
        }

        // Prepare single-step input
        std::vector<MatrixXf> X_seq = {x_t};
        std::vector<MatrixXf> dt_seq = {dt_t};
        std::vector<std::optional<MatrixXf> > mask_seq = {mask_t};

        // Forward pass with streaming states
        auto result = autoencoder->forward(X_seq, dt_seq, mask_seq, h_states_encoder, h_states_decoder);

        // Update streaming states (detached from gradients)
        h_states_encoder = result.encoder_final_hidden;
        h_states_decoder = result.decoder_final_hidden;

        // Return prediction
        if (autoencoder->ae_cfg.mode == AutoencoderMode::RECONSTRUCTION) {
            return result.output_sequence;
        } else {
            return result.output_sequence; // Full forecast
        }
    }

    // COMPLETE: Batch processing method
    std::pair<float, std::vector<MatrixXf> > process_batch(
            const std::vector<MatrixXf> &X_batch,
            const std::vector<MatrixXf> &dt_batch,
            const std::vector<std::optional<MatrixXf> > &mask_batch,
            const std::vector<std::optional<MatrixXf> > &target_batch = {}) {
        if (X_batch.empty()) {
            return {0.0f, {}};
        }

        int batch_size = X_batch[0].rows();
        reset_streaming_state(batch_size);

        std::vector<MatrixXf> predictions;
        float total_loss = 0.0f;
        int valid_steps = 0;

        for (size_t i = 0; i < X_batch.size(); ++i) {
            std::optional<MatrixXf> target_i = std::nullopt;
            if (i < target_batch.size() && target_batch[i].has_value()) {
                target_i = target_batch[i];
            }

            std::optional<MatrixXf> mask_i = std::nullopt;
            if (i < mask_batch.size()) {
                mask_i = mask_batch[i];
            }

            auto [loss, pred] = step_stream(X_batch[i], dt_batch[i], mask_i, target_i);
            predictions.push_back(pred);

            if (std::isfinite(loss)) {
                total_loss += loss;
                valid_steps++;
            }
        }

        float avg_loss = valid_steps > 0 ? total_loss / valid_steps : 0.0f;
        return {avg_loss, predictions};
    }

    // Getters for inspection
    const std::vector<MatrixXf> &get_encoder_states() const { return h_states_encoder; }
    const std::vector<MatrixXf> &get_decoder_states() const { return h_states_decoder; }
    size_t get_window_size() const { return win_X.size(); }

    // Configuration access
    const NpTemporalConfig &get_config() const { return opt_cfg; }
    const NpTemporalAutoencoder &get_autoencoder() const { return *autoencoder; }
    NpTemporalAutoencoder &get_autoencoder_mutable() { return *autoencoder; }
};


// ============================================================================
// 10. COMPLETE USAGE EXAMPLE AND FACTORY FUNCTIONS
// ============================================================================

// Factory function for creating complete autoencoder configurations
std::unique_ptr<NpTemporalAutoencoder> create_autoencoder(
        const std::string &mode,
        int input_size,
        int latent_size,
        int hidden_size,
        int num_layers,
        const std::string &bottleneck_type,
        bool use_attention,
        bool autoregressive,
        int forecast_horizon,
        std::mt19937 &gen) {
    // Base RNN configuration
    NpTemporalConfig rnn_cfg;
    rnn_cfg.in_size = 32; // Internal projection size
    rnn_cfg.hid_size = hidden_size;
    rnn_cfg.num_layers = num_layers;
    rnn_cfg.use_exponential_decay = true;
    rnn_cfg.layer_norm = true;
    rnn_cfg.dropout = 0.1f;
    rnn_cfg.final_dropout = 0.1f;

    // Autoencoder configuration
    AutoencoderConfig ae_cfg;
    ae_cfg.input_size = input_size;
    ae_cfg.latent_size = latent_size;
    ae_cfg.internal_projection_size = 32;
    ae_cfg.use_input_projection = true;
    ae_cfg.forecast_horizon = forecast_horizon;

    // Set mode
    if (mode == "reconstruction") {
        ae_cfg.mode = AutoencoderMode::RECONSTRUCTION;
    } else if (mode == "forecasting") {
        ae_cfg.mode = AutoencoderMode::FORECASTING;
        if (autoregressive) {
            ae_cfg.forecasting_mode = ForecastingMode::AUTOREGRESSIVE;
            ae_cfg.feedback_transform_type = FeedbackTransformType::LINEAR;
        }
    }

    // Set bottleneck type
    if (bottleneck_type == "last_hidden") {
        ae_cfg.bottleneck_type = BottleneckType::LAST_HIDDEN;
    } else if (bottleneck_type == "mean_pool") {
        ae_cfg.bottleneck_type = BottleneckType::MEAN_POOL;
    } else if (bottleneck_type == "max_pool") {
        ae_cfg.bottleneck_type = BottleneckType::MAX_POOL;
    } else if (bottleneck_type == "attention_pool" || use_attention) {
        ae_cfg.bottleneck_type = BottleneckType::ATTENTION_POOL;
        ae_cfg.attention_context_dim = hidden_size;
    }

    return std::make_unique<NpTemporalAutoencoder>(rnn_cfg, ae_cfg, gen);
}

// FIXED: Overload with default parameters that uses a static RNG to avoid memory leak
std::unique_ptr<NpTemporalAutoencoder> create_autoencoder(
        const std::string &mode = "reconstruction",
        int input_size = 12,
        int latent_size = 8,
        int hidden_size = 64,
        int num_layers = 2,
        const std::string &bottleneck_type = "mean_pool",
        bool use_attention = false,
        bool autoregressive = false,
        int forecast_horizon = 1) {
    // Use a static generator to avoid memory leaks
    static std::mt19937 default_gen(42);
    return create_autoencoder(mode, input_size, latent_size, hidden_size, num_layers,
                              bottleneck_type, use_attention, autoregressive, forecast_horizon, default_gen);
}

// Factory function for creating online learners
std::unique_ptr<NpOnlineLearner> create_online_learner(
        std::unique_ptr<NpTemporalAutoencoder> autoencoder,
        float learning_rate = 2e-3f,
        int tbptt_steps = 20,
        float clip_grad_norm = 5.0f) {
    NpTemporalConfig opt_cfg;
    opt_cfg.lr = learning_rate;
    opt_cfg.tbptt_steps = tbptt_steps;
    opt_cfg.clip_grad_norm = clip_grad_norm;
    opt_cfg.weight_decay = 1e-4f;
    opt_cfg.beta1 = 0.9f;
    opt_cfg.beta2 = 0.999f;
    opt_cfg.eps_adam = 1e-8f;

    return std::make_unique<NpOnlineLearner>(std::move(autoencoder), opt_cfg);
}

#endif // TENSOREIGEN_AUTOENCODER_COMPLETE_H