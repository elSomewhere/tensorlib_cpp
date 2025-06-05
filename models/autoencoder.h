//
// grud/models/autoencoder.h - Complete GRU-D autoencoder models
//

#ifndef GRUD_MODELS_AUTOENCODER_H
#define GRUD_MODELS_AUTOENCODER_H

#include "../core/module.h"
#include "../core/tape.h"
#include "../layers/basic.h"
#include "../layers/temporal.h"
#include <memory>
#include <vector>
#include <optional>

namespace grud {
namespace models {

// ============================================================================
// AUTOENCODER CONFIGURATION
// ============================================================================

enum class BottleneckType {
    LAST_HIDDEN,
    MEAN_POOL,
    MAX_POOL,
    ATTENTION_POOL
};

enum class AutoencoderMode {
    RECONSTRUCTION,
    FORECASTING
};

struct AutoencoderConfig {
    // Architecture
    int input_size = 12;
    int latent_size = 8;
    int hidden_size = 64;
    int num_layers = 2;

    // Bottleneck
    BottleneckType bottleneck_type = BottleneckType::MEAN_POOL;
    bool use_input_projection = true;
    int internal_projection_size = 32;

    // Mode configuration
    AutoencoderMode mode = AutoencoderMode::RECONSTRUCTION;
    int forecast_horizon = 1;

    // Regularization
    float dropout = 0.1f;
    float final_dropout = 0.1f;
    bool layer_norm = true;

    // Temporal settings
    bool use_exponential_decay = true;
    float softclip_threshold = 3.0f;
    float min_log_gamma = -10.0f;
};

// ============================================================================
// TEMPORAL ENCODER
// ============================================================================

/**
 * Multi-layer temporal encoder using GRU-D
 */
class TemporalEncoder : public Module {
public:
    std::vector<std::unique_ptr<layers::TemporalRNNLayer>> rnn_layers;
    std::unique_ptr<layers::Linear> input_projection;
    std::unique_ptr<layers::Dropout> final_dropout;

private:
    AutoencoderConfig config_;
    std::mt19937& gen_;

public:
    TemporalEncoder(const AutoencoderConfig& config, std::mt19937& gen)
        : config_(config), gen_(gen) {

        // Input projection if needed
        if (config_.use_input_projection && config_.input_size != config_.internal_projection_size) {
            input_projection = std::make_unique<layers::Linear>(
                config_.input_size, config_.internal_projection_size, true, &gen_);
        }

        // Create RNN layers
        int current_input_size = config_.use_input_projection ?
            config_.internal_projection_size : config_.input_size;

        for (int i = 0; i < config_.num_layers; ++i) {
            auto layer = std::make_unique<layers::TemporalRNNLayer>(
                current_input_size, config_.hidden_size, i,
                config_.layer_norm, config_.dropout, config_.use_exponential_decay, &gen_);
            rnn_layers.push_back(std::move(layer));
            current_input_size = config_.hidden_size;  // Subsequent layers use hidden size
        }

        // Final dropout
        if (config_.final_dropout > 0.0f) {
            final_dropout = std::make_unique<layers::Dropout>(config_.final_dropout, gen_);
        }
    }

    /**
     * Forward pass through encoder
     * @param X_seq Sequence of input matrices
     * @param dt_seq Sequence of time differences
     * @param mask_seq Optional sequence of masks
     * @param initial_hidden Initial hidden states for each layer
     * @return Tuple of (output_sequence, final_hidden_states, tape_indices)
     */
    std::tuple<Eigen::MatrixXf, std::vector<Eigen::MatrixXf>, std::vector<size_t>>
    forward_sequence(
        const std::vector<Eigen::MatrixXf>& X_seq,
        const std::vector<Eigen::MatrixXf>& dt_seq,
        const std::vector<std::optional<Eigen::MatrixXf>>& mask_seq,
        const std::vector<Eigen::MatrixXf>& initial_hidden,
        Tape& tape) {

        if (X_seq.empty()) {
            throw std::invalid_argument("Empty input sequence");
        }

        int seq_len = X_seq.size();
        int batch_size = X_seq[0].rows();

        // Initialize hidden states
        std::vector<Eigen::MatrixXf> current_hidden = initial_hidden;
        if (current_hidden.empty()) {
            current_hidden.resize(config_.num_layers);
            for (int i = 0; i < config_.num_layers; ++i) {
                current_hidden[i] = Eigen::MatrixXf::Zero(batch_size, config_.hidden_size);
            }
        }

        // Storage for outputs and tape indices
        std::vector<Eigen::MatrixXf> sequence_outputs;
        std::vector<size_t> tape_indices;
        sequence_outputs.reserve(seq_len);
        tape_indices.reserve(seq_len * config_.num_layers);

        // Process each timestep
        for (int t = 0; t < seq_len; ++t) {
            Eigen::MatrixXf layer_input = X_seq[t];

            // Apply input projection if configured
            if (input_projection) {
                Context proj_ctx(input_projection.get());
                layer_input = input_projection->forward(layer_input, proj_ctx);
                tape_indices.push_back(tape.push(std::move(proj_ctx)));
            }

            // Forward through RNN layers
            for (int layer_idx = 0; layer_idx < config_.num_layers; ++layer_idx) {
                auto& layer = rnn_layers[layer_idx];

                // Only apply mask to first layer
                std::optional<Eigen::MatrixXf> layer_mask = std::nullopt;
                if (layer_idx == 0 && t < static_cast<int>(mask_seq.size())) {
                    layer_mask = mask_seq[t];
                }

                auto [layer_output, layer_ctx] = layer->forward_temporal(
                    layer_input, current_hidden[layer_idx], dt_seq[t], layer_mask);

                current_hidden[layer_idx] = layer_output;
                layer_input = layer_output;

                tape_indices.push_back(tape.push(std::move(layer_ctx)));
            }

            sequence_outputs.push_back(layer_input);
        }

        // Stack sequence outputs: (seq_len * batch_size, hidden_size)
        Eigen::MatrixXf stacked_output(seq_len * batch_size, config_.hidden_size);
        for (int t = 0; t < seq_len; ++t) {
            stacked_output.block(t * batch_size, 0, batch_size, config_.hidden_size) = sequence_outputs[t];
        }

        // Apply final dropout if configured
        if (final_dropout && training_mode) {
            Context dropout_ctx(final_dropout.get());
            stacked_output = final_dropout->forward(stacked_output, dropout_ctx);
            tape_indices.push_back(tape.push(std::move(dropout_ctx)));
        }

        return {stacked_output, current_hidden, tape_indices};
    }

    // Standard Module interface (not used for sequence processing)
    Eigen::MatrixXf forward(const Eigen::MatrixXf& input, Context& ctx) override {
        throw std::runtime_error("TemporalEncoder requires forward_sequence method");
    }

    Eigen::MatrixXf backward(const Eigen::MatrixXf& grad_output, const Context& ctx) override {
        throw std::runtime_error("TemporalEncoder backward pass handled by autograd system");
    }

    std::vector<Module*> children() override {
        std::vector<Module*> child_modules;
        if (input_projection) child_modules.push_back(input_projection.get());
        for (auto& layer : rnn_layers) {
            child_modules.push_back(layer.get());
        }
        if (final_dropout) child_modules.push_back(final_dropout.get());
        return child_modules;
    }

    std::string name() const override {
        return "TemporalEncoder(" + std::to_string(config_.num_layers) + " layers)";
    }
};

// ============================================================================
// BOTTLENECK AGGREGATION
// ============================================================================

/**
 * Aggregates temporal sequence into a fixed-size representation
 */
class BottleneckAggregation : public Module {
private:
    BottleneckType aggregation_type_;
    int hidden_size_;
    int latent_size_;
    std::unique_ptr<layers::Linear> projection;

public:
    BottleneckAggregation(BottleneckType type, int hidden_size, int latent_size, std::mt19937& gen)
        : aggregation_type_(type), hidden_size_(hidden_size), latent_size_(latent_size) {

        projection = std::make_unique<layers::Linear>(hidden_size, latent_size, true, &gen);
    }

    /**
     * Aggregate sequence into fixed representation
     * @param sequence_stacked Stacked sequence (seq_len * batch_size, hidden_size)
     * @param batch_size Batch size
     * @param seq_len Sequence length
     */
    std::pair<Eigen::MatrixXf, Context> aggregate_sequence(
        const Eigen::MatrixXf& sequence_stacked,
        int batch_size, int seq_len) {

        Context ctx(this);
        Eigen::MatrixXf aggregated;

        switch (aggregation_type_) {
            case BottleneckType::LAST_HIDDEN: {
                // Take the last timestep
                aggregated = Eigen::MatrixXf(batch_size, hidden_size_);
                for (int b = 0; b < batch_size; ++b) {
                    int last_idx = (seq_len - 1) * batch_size + b;
                    aggregated.row(b) = sequence_stacked.row(last_idx);
                }
                break;
            }

            case BottleneckType::MEAN_POOL: {
                // Average over timesteps
                aggregated = Eigen::MatrixXf::Zero(batch_size, hidden_size_);
                for (int b = 0; b < batch_size; ++b) {
                    for (int t = 0; t < seq_len; ++t) {
                        int idx = t * batch_size + b;
                        aggregated.row(b) += sequence_stacked.row(idx);
                    }
                    aggregated.row(b) /= static_cast<float>(seq_len);
                }
                break;
            }

            case BottleneckType::MAX_POOL: {
                // Max over timesteps
                aggregated = Eigen::MatrixXf(batch_size, hidden_size_);
                for (int b = 0; b < batch_size; ++b) {
                    aggregated.row(b) = sequence_stacked.row(b);  // Initialize with first timestep
                    for (int t = 1; t < seq_len; ++t) {
                        int idx = t * batch_size + b;
                        aggregated.row(b) = aggregated.row(b).cwiseMax(sequence_stacked.row(idx));
                    }
                }
                break;
            }

            case BottleneckType::ATTENTION_POOL: {
                // For now, fallback to mean pooling
                // Full attention would require additional parameters
                aggregated = Eigen::MatrixXf::Zero(batch_size, hidden_size_);
                for (int b = 0; b < batch_size; ++b) {
                    for (int t = 0; t < seq_len; ++t) {
                        int idx = t * batch_size + b;
                        aggregated.row(b) += sequence_stacked.row(idx);
                    }
                    aggregated.row(b) /= static_cast<float>(seq_len);
                }
                break;
            }
        }

        // Save for backward pass
        ctx.save_for_backward(sequence_stacked);
        ctx.save_for_backward(Eigen::MatrixXf::Constant(1, 2, batch_size));  // Save batch_size
        ctx.save_for_backward(Eigen::MatrixXf::Constant(1, 2, seq_len));     // Save seq_len

        // Project to latent space
        Context proj_ctx(projection.get());
        Eigen::MatrixXf latent = projection->forward(aggregated, proj_ctx);

        return {latent, ctx};
    }

    Eigen::MatrixXf forward(const Eigen::MatrixXf& input, Context& ctx) override {
        throw std::runtime_error("BottleneckAggregation requires aggregate_sequence method");
    }

    Eigen::MatrixXf backward(const Eigen::MatrixXf& grad_output, const Context& ctx) override {
        throw std::runtime_error("BottleneckAggregation backward pass handled by autograd system");
    }

    std::vector<Module*> children() override {
        return {projection.get()};
    }

    std::string name() const override {
        return "BottleneckAggregation(" + std::to_string(static_cast<int>(aggregation_type_)) + ")";
    }
};

// ============================================================================
// TEMPORAL DECODER
// ============================================================================

/**
 * Multi-layer temporal decoder using GRU-D
 */
class TemporalDecoder : public Module {
public:
    std::vector<std::unique_ptr<layers::TemporalRNNLayer>> rnn_layers;
    std::unique_ptr<layers::Linear> latent_projection;
    std::unique_ptr<layers::Linear> output_projection;
    std::unique_ptr<layers::Linear> hidden_state_init;

private:
    AutoencoderConfig config_;
    std::mt19937& gen_;

public:
    TemporalDecoder(const AutoencoderConfig& config, std::mt19937& gen)
        : config_(config), gen_(gen) {

        // Project latent to decoder input size
        latent_projection = std::make_unique<layers::Linear>(
            config_.latent_size, config_.internal_projection_size, true, &gen_);

        // Create RNN layers
        for (int i = 0; i < config_.num_layers; ++i) {
            auto layer = std::make_unique<layers::TemporalRNNLayer>(
                config_.internal_projection_size, config_.hidden_size, i,
                config_.layer_norm, config_.dropout, config_.use_exponential_decay, &gen_);
            rnn_layers.push_back(std::move(layer));
        }

        // Project hidden states to output
        output_projection = std::make_unique<layers::Linear>(
            config_.hidden_size, config_.input_size, true, &gen_);

        // Initialize hidden states from latent (for non-last-hidden bottlenecks)
        if (config_.bottleneck_type != BottleneckType::LAST_HIDDEN) {
            hidden_state_init = std::make_unique<layers::Linear>(
                config_.latent_size, config_.num_layers * config_.hidden_size, true, &gen_);
        }
    }

    /**
     * Decode sequence from latent representation
     */
    std::tuple<Eigen::MatrixXf, std::vector<Eigen::MatrixXf>, std::vector<size_t>>
    decode_sequence(
        const Eigen::MatrixXf& latent,
        const std::vector<Eigen::MatrixXf>& dt_seq,
        const std::vector<Eigen::MatrixXf>& initial_hidden,
        int decode_length,
        Tape& tape) {

        int batch_size = latent.rows();

        // Initialize decoder hidden states
        std::vector<Eigen::MatrixXf> current_hidden = initial_hidden;
        if (current_hidden.empty()) {
            if (hidden_state_init) {
                // Initialize from latent
                Context init_ctx(hidden_state_init.get());
                Eigen::MatrixXf h_flat = hidden_state_init->forward(latent, init_ctx);
                tape.push(std::move(init_ctx));

                current_hidden.resize(config_.num_layers);
                for (int i = 0; i < config_.num_layers; ++i) {
                    current_hidden[i] = h_flat.block(0, i * config_.hidden_size,
                                                    batch_size, config_.hidden_size);
                }
            } else {
                // Zero initialization
                current_hidden.resize(config_.num_layers);
                for (int i = 0; i < config_.num_layers; ++i) {
                    current_hidden[i] = Eigen::MatrixXf::Zero(batch_size, config_.hidden_size);
                }
            }
        }

        std::vector<Eigen::MatrixXf> sequence_outputs;
        std::vector<size_t> tape_indices;
        sequence_outputs.reserve(decode_length);

        for (int t = 0; t < decode_length; ++t) {
            // Project latent to decoder input
            Context latent_proj_ctx(latent_projection.get());
            Eigen::MatrixXf decoder_input = latent_projection->forward(latent, latent_proj_ctx);
            tape_indices.push_back(tape.push(std::move(latent_proj_ctx)));

            // Forward through RNN layers
            for (int layer_idx = 0; layer_idx < config_.num_layers; ++layer_idx) {
                auto& layer = rnn_layers[layer_idx];

                auto [layer_output, layer_ctx] = layer->forward_temporal(
                    decoder_input, current_hidden[layer_idx], dt_seq[t]);

                current_hidden[layer_idx] = layer_output;
                decoder_input = layer_output;

                tape_indices.push_back(tape.push(std::move(layer_ctx)));
            }

            // Project to output space
            Context output_proj_ctx(output_projection.get());
            Eigen::MatrixXf output = output_projection->forward(decoder_input, output_proj_ctx);
            tape_indices.push_back(tape.push(std::move(output_proj_ctx)));

            sequence_outputs.push_back(output);
        }

        // Stack outputs
        Eigen::MatrixXf stacked_output(decode_length * batch_size, config_.input_size);
        for (int t = 0; t < decode_length; ++t) {
            stacked_output.block(t * batch_size, 0, batch_size, config_.input_size) = sequence_outputs[t];
        }

        return {stacked_output, current_hidden, tape_indices};
    }

    Eigen::MatrixXf forward(const Eigen::MatrixXf& input, Context& ctx) override {
        throw std::runtime_error("TemporalDecoder requires decode_sequence method");
    }

    Eigen::MatrixXf backward(const Eigen::MatrixXf& grad_output, const Context& ctx) override {
        throw std::runtime_error("TemporalDecoder backward pass handled by autograd system");
    }

    std::vector<Module*> children() override {
        std::vector<Module*> child_modules = {latent_projection.get(), output_projection.get()};
        if (hidden_state_init) child_modules.push_back(hidden_state_init.get());
        for (auto& layer : rnn_layers) {
            child_modules.push_back(layer.get());
        }
        return child_modules;
    }

    std::string name() const override {
        return "TemporalDecoder(" + std::to_string(config_.num_layers) + " layers)";
    }
};

// ============================================================================
// COMPLETE TEMPORAL AUTOENCODER
// ============================================================================

/**
 * Complete GRU-D autoencoder with encoder, bottleneck, and decoder
 */
class TemporalAutoencoder : public Module {
public:
    std::unique_ptr<TemporalEncoder> encoder;
    std::unique_ptr<BottleneckAggregation> bottleneck;
    std::unique_ptr<TemporalDecoder> decoder;

    AutoencoderConfig config;

private:
    std::mt19937& gen_;

public:
    TemporalAutoencoder(const AutoencoderConfig& ae_config, std::mt19937& gen)
        : config(ae_config), gen_(gen) {

        encoder = std::make_unique<TemporalEncoder>(config, gen_);
        bottleneck = std::make_unique<BottleneckAggregation>(
            config.bottleneck_type, config.hidden_size, config.latent_size, gen_);
        decoder = std::make_unique<TemporalDecoder>(config, gen_);
    }

    /**
     * Complete forward pass through autoencoder
     */
    struct ForwardResult {
        Eigen::MatrixXf output_sequence;
        Eigen::MatrixXf latent;
        std::vector<Eigen::MatrixXf> encoder_final_hidden;
        std::vector<Eigen::MatrixXf> decoder_final_hidden;
        std::vector<size_t> tape_indices;  // For backward pass
    };

    ForwardResult forward_autoencoder(
        const std::vector<Eigen::MatrixXf>& X_seq,
        const std::vector<Eigen::MatrixXf>& dt_seq,
        const std::vector<std::optional<Eigen::MatrixXf>>& mask_seq,
        const std::vector<Eigen::MatrixXf>& initial_h_encoder,
        const std::vector<Eigen::MatrixXf>& initial_h_decoder,
        Tape& tape) {

        ForwardResult result;

        // Encode
        auto [encoder_output, encoder_final_h, encoder_tape_indices] =
            encoder->forward_sequence(X_seq, dt_seq, mask_seq, initial_h_encoder, tape);

        result.encoder_final_hidden = encoder_final_h;
        result.tape_indices.insert(result.tape_indices.end(),
                                  encoder_tape_indices.begin(), encoder_tape_indices.end());

        // Bottleneck aggregation
        auto [latent, bottleneck_ctx] = bottleneck->aggregate_sequence(
            encoder_output, X_seq[0].rows(), X_seq.size());
        result.latent = latent;
        result.tape_indices.push_back(tape.push(std::move(bottleneck_ctx)));

        // Decode
        int decode_length = (config.mode == AutoencoderMode::RECONSTRUCTION) ?
                           X_seq.size() : config.forecast_horizon;

        // Create dt sequence for decoding
        std::vector<Eigen::MatrixXf> decode_dt_seq;
        if (config.mode == AutoencoderMode::RECONSTRUCTION) {
            decode_dt_seq = dt_seq;
        } else {
            // For forecasting, repeat the last dt
            Eigen::MatrixXf last_dt = dt_seq.back();
            decode_dt_seq.resize(decode_length, last_dt);
        }

        // Initialize decoder hidden states
        std::vector<Eigen::MatrixXf> decoder_init_hidden = initial_h_decoder;
        if (decoder_init_hidden.empty() && config.bottleneck_type == BottleneckType::LAST_HIDDEN) {
            decoder_init_hidden = encoder_final_h;
        }

        auto [decoder_output, decoder_final_h, decoder_tape_indices] =
            decoder->decode_sequence(latent, decode_dt_seq, decoder_init_hidden, decode_length, tape);

        result.output_sequence = decoder_output;
        result.decoder_final_hidden = decoder_final_h;
        result.tape_indices.insert(result.tape_indices.end(),
                                  decoder_tape_indices.begin(), decoder_tape_indices.end());

        return result;
    }

    Eigen::MatrixXf forward(const Eigen::MatrixXf& input, Context& ctx) override {
        throw std::runtime_error("TemporalAutoencoder requires forward_autoencoder method");
    }

    Eigen::MatrixXf backward(const Eigen::MatrixXf& grad_output, const Context& ctx) override {
        throw std::runtime_error("TemporalAutoencoder backward pass handled by autograd system");
    }

    std::vector<Module*> children() override {
        return {encoder.get(), bottleneck.get(), decoder.get()};
    }

    std::string name() const override {
        return "TemporalAutoencoder(" + std::to_string(config.latent_size) + "D latent)";
    }
};

} // namespace models
} // namespace grud

#endif // GRUD_MODELS_AUTOENCODER_H