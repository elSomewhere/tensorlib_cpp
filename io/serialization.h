//
// grud/io/serialization.h - Model serialization and persistence
//

#ifndef GRUD_IO_SERIALIZATION_H
#define GRUD_IO_SERIALIZATION_H

#include "../grud.h"
#include <fstream>
#include <sstream>
#include <iomanip>
#include <map>
#include <vector>
#include <string>

namespace grud {
namespace io {

// ============================================================================
// BINARY SERIALIZATION FORMAT
// ============================================================================

/**
 * Simple binary format for storing matrices:
 * [4 bytes: rows] [4 bytes: cols] [rows*cols*4 bytes: data in row-major order]
 */
class BinaryWriter {
private:
    std::ofstream file_;

public:
    BinaryWriter(const std::string& filename) : file_(filename, std::ios::binary) {
        if (!file_) {
            throw std::runtime_error("Failed to open file for writing: " + filename);
        }
    }

    void write_int32(int32_t value) {
        file_.write(reinterpret_cast<const char*>(&value), sizeof(value));
    }

    void write_float(float value) {
        file_.write(reinterpret_cast<const char*>(&value), sizeof(value));
    }

    void write_string(const std::string& str) {
        int32_t length = static_cast<int32_t>(str.length());
        write_int32(length);
        file_.write(str.c_str(), length);
    }

    void write_matrix(const Eigen::MatrixXf& matrix) {
        write_int32(static_cast<int32_t>(matrix.rows()));
        write_int32(static_cast<int32_t>(matrix.cols()));

        // Write data in row-major order for consistency
        for (int i = 0; i < matrix.rows(); ++i) {
            for (int j = 0; j < matrix.cols(); ++j) {
                write_float(matrix(i, j));
            }
        }
    }

    bool good() const { return file_.good(); }
};

class BinaryReader {
private:
    std::ifstream file_;

public:
    BinaryReader(const std::string& filename) : file_(filename, std::ios::binary) {
        if (!file_) {
            throw std::runtime_error("Failed to open file for reading: " + filename);
        }
    }

    int32_t read_int32() {
        int32_t value;
        file_.read(reinterpret_cast<char*>(&value), sizeof(value));
        if (!file_) {
            throw std::runtime_error("Failed to read int32 from file");
        }
        return value;
    }

    float read_float() {
        float value;
        file_.read(reinterpret_cast<char*>(&value), sizeof(value));
        if (!file_) {
            throw std::runtime_error("Failed to read float from file");
        }
        return value;
    }

    std::string read_string() {
        int32_t length = read_int32();
        if (length < 0 || length > 1000000) { // Sanity check
            throw std::runtime_error("Invalid string length in file");
        }

        std::string str(length, '\0');
        file_.read(&str[0], length);
        if (!file_) {
            throw std::runtime_error("Failed to read string from file");
        }
        return str;
    }

    Eigen::MatrixXf read_matrix() {
        int32_t rows = read_int32();
        int32_t cols = read_int32();

        if (rows < 0 || cols < 0 || rows > 100000 || cols > 100000) {
            throw std::runtime_error("Invalid matrix dimensions in file");
        }

        Eigen::MatrixXf matrix(rows, cols);

        // Read data in row-major order
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                matrix(i, j) = read_float();
            }
        }

        return matrix;
    }

    bool good() const { return file_.good(); }
};

// ============================================================================
// MODEL CHECKPOINT FORMAT
// ============================================================================

struct CheckpointHeader {
    std::string framework_version;
    std::string model_type;
    int64_t timestamp;
    std::map<std::string, std::string> metadata;

    void write(BinaryWriter& writer) const {
        writer.write_string(framework_version);
        writer.write_string(model_type);
        writer.write_int32(static_cast<int32_t>(timestamp));

        // Write metadata
        writer.write_int32(static_cast<int32_t>(metadata.size()));
        for (const auto& [key, value] : metadata) {
            writer.write_string(key);
            writer.write_string(value);
        }
    }

    void read(BinaryReader& reader) {
        framework_version = reader.read_string();
        model_type = reader.read_string();
        timestamp = reader.read_int32();

        // Read metadata
        int32_t num_metadata = reader.read_int32();
        metadata.clear();
        for (int i = 0; i < num_metadata; ++i) {
            std::string key = reader.read_string();
            std::string value = reader.read_string();
            metadata[key] = value;
        }
    }
};

// ============================================================================
// AUTOENCODER SERIALIZATION
// ============================================================================

class AutoencoderSerializer {
public:
    /**
     * Save autoencoder model to file
     */
    static void save(const models::TemporalAutoencoder& model,
                    const std::string& filename,
                    const std::map<std::string, std::string>& metadata = {}) {

        BinaryWriter writer(filename);

        // Write header
        CheckpointHeader header;
        header.framework_version = get_version().to_string();
        header.model_type = "TemporalAutoencoder";
        header.timestamp = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        header.metadata = metadata;
        header.write(writer);

        // Write model configuration
        write_config(writer, model.config);

        // Write model parameters
        write_parameters(writer, model);

        if (!writer.good()) {
            throw std::runtime_error("Error writing to checkpoint file");
        }
    }

    /**
     * Load autoencoder model from file
     */
    static std::unique_ptr<models::TemporalAutoencoder> load(
        const std::string& filename,
        std::mt19937& gen,
        CheckpointHeader* header_out = nullptr) {

        BinaryReader reader(filename);

        // Read header
        CheckpointHeader header;
        header.read(reader);

        if (header.model_type != "TemporalAutoencoder") {
            throw std::runtime_error("Invalid model type in checkpoint: " + header.model_type);
        }

        if (header_out) {
            *header_out = header;
        }

        // Read model configuration
        models::AutoencoderConfig config = read_config(reader);

        // Create model with loaded configuration
        auto model = std::make_unique<models::TemporalAutoencoder>(config, gen);

        // Load parameters
        load_parameters(reader, *model);

        return model;
    }

private:
    static void write_config(BinaryWriter& writer, const models::AutoencoderConfig& config) {
        writer.write_int32(config.input_size);
        writer.write_int32(config.latent_size);
        writer.write_int32(config.hidden_size);
        writer.write_int32(config.num_layers);
        writer.write_int32(static_cast<int32_t>(config.bottleneck_type));
        writer.write_int32(config.internal_projection_size);
        writer.write_int32(static_cast<int32_t>(config.mode));
        writer.write_int32(config.forecast_horizon);
        writer.write_float(config.dropout);
        writer.write_float(config.final_dropout);
        writer.write_int32(config.layer_norm ? 1 : 0);
        writer.write_int32(config.use_exponential_decay ? 1 : 0);
        writer.write_float(config.softclip_threshold);
        writer.write_float(config.min_log_gamma);
        writer.write_int32(config.use_input_projection ? 1 : 0);
    }

    static models::AutoencoderConfig read_config(BinaryReader& reader) {
        models::AutoencoderConfig config;
        config.input_size = reader.read_int32();
        config.latent_size = reader.read_int32();
        config.hidden_size = reader.read_int32();
        config.num_layers = reader.read_int32();
        config.bottleneck_type = static_cast<models::BottleneckType>(reader.read_int32());
        config.internal_projection_size = reader.read_int32();
        config.mode = static_cast<models::AutoencoderMode>(reader.read_int32());
        config.forecast_horizon = reader.read_int32();
        config.dropout = reader.read_float();
        config.final_dropout = reader.read_float();
        config.layer_norm = (reader.read_int32() != 0);
        config.use_exponential_decay = (reader.read_int32() != 0);
        config.softclip_threshold = reader.read_float();
        config.min_log_gamma = reader.read_float();
        config.use_input_projection = (reader.read_int32() != 0);
        return config;
    }

    static void write_parameters(BinaryWriter& writer, const models::TemporalAutoencoder& model) {
        auto params = const_cast<models::TemporalAutoencoder&>(model).all_parameters();

        writer.write_int32(static_cast<int32_t>(params.size()));

        for (const auto* param : params) {
            writer.write_string(param->name);
            writer.write_matrix(param->value);
        }
    }

    static void load_parameters(BinaryReader& reader, models::TemporalAutoencoder& model) {
        auto params = model.all_parameters();

        int32_t num_params = reader.read_int32();

        // Create name to parameter mapping
        std::map<std::string, Param*> param_map;
        for (auto* param : params) {
            param_map[param->name] = param;
        }

        // Load parameters by name
        for (int i = 0; i < num_params; ++i) {
            std::string param_name = reader.read_string();
            Eigen::MatrixXf param_value = reader.read_matrix();

            auto it = param_map.find(param_name);
            if (it != param_map.end()) {
                if (it->second->value.rows() == param_value.rows() &&
                    it->second->value.cols() == param_value.cols()) {
                    it->second->value = param_value;
                } else {
                    std::cerr << "Warning: Parameter " << param_name
                              << " size mismatch. Expected ("
                              << it->second->value.rows() << ", " << it->second->value.cols()
                              << "), got (" << param_value.rows() << ", " << param_value.cols()
                              << ")" << std::endl;
                }
            } else {
                std::cerr << "Warning: Unknown parameter " << param_name
                          << " in checkpoint" << std::endl;
            }
        }
    }
};

// ============================================================================
// TRAINING STATE SERIALIZATION
// ============================================================================

struct TrainingState {
    int epoch = 0;
    int step = 0;
    float best_loss = std::numeric_limits<float>::infinity();
    std::vector<float> train_losses;
    std::vector<float> val_losses;
    std::map<std::string, float> optimizer_state;

    void save(const std::string& filename) const {
        BinaryWriter writer(filename);

        writer.write_int32(epoch);
        writer.write_int32(step);
        writer.write_float(best_loss);

        // Write loss history
        writer.write_int32(static_cast<int32_t>(train_losses.size()));
        for (float loss : train_losses) {
            writer.write_float(loss);
        }

        writer.write_int32(static_cast<int32_t>(val_losses.size()));
        for (float loss : val_losses) {
            writer.write_float(loss);
        }

        // Write optimizer state
        writer.write_int32(static_cast<int32_t>(optimizer_state.size()));
        for (const auto& [key, value] : optimizer_state) {
            writer.write_string(key);
            writer.write_float(value);
        }
    }

    void load(const std::string& filename) {
        BinaryReader reader(filename);

        epoch = reader.read_int32();
        step = reader.read_int32();
        best_loss = reader.read_float();

        // Read loss history
        int32_t num_train_losses = reader.read_int32();
        train_losses.clear();
        train_losses.reserve(num_train_losses);
        for (int i = 0; i < num_train_losses; ++i) {
            train_losses.push_back(reader.read_float());
        }

        int32_t num_val_losses = reader.read_int32();
        val_losses.clear();
        val_losses.reserve(num_val_losses);
        for (int i = 0; i < num_val_losses; ++i) {
            val_losses.push_back(reader.read_float());
        }

        // Read optimizer state
        int32_t num_optimizer_state = reader.read_int32();
        optimizer_state.clear();
        for (int i = 0; i < num_optimizer_state; ++i) {
            std::string key = reader.read_string();
            float value = reader.read_float();
            optimizer_state[key] = value;
        }
    }
};

// ============================================================================
// CHECKPOINT MANAGER
// ============================================================================

class CheckpointManager {
private:
    std::string checkpoint_dir_;
    int max_checkpoints_;

public:
    CheckpointManager(const std::string& checkpoint_dir, int max_checkpoints = 5)
        : checkpoint_dir_(checkpoint_dir), max_checkpoints_(max_checkpoints) {

        // Create directory if it doesn't exist (platform-specific code omitted)
        // In practice, you'd use std::filesystem or platform-specific APIs
    }

    /**
     * Save complete training checkpoint
     */
    void save_checkpoint(const models::TemporalAutoencoder& model,
                        const TrainingState& training_state,
                        const std::string& checkpoint_name = "") {

        std::string name = checkpoint_name;
        if (name.empty()) {
            name = "checkpoint_epoch_" + std::to_string(training_state.epoch);
        }

        std::string model_file = checkpoint_dir_ + "/" + name + "_model.grud";
        std::string state_file = checkpoint_dir_ + "/" + name + "_state.grud";

        // Save model
        std::map<std::string, std::string> metadata;
        metadata["epoch"] = std::to_string(training_state.epoch);
        metadata["step"] = std::to_string(training_state.step);
        metadata["best_loss"] = std::to_string(training_state.best_loss);

        AutoencoderSerializer::save(model, model_file, metadata);

        // Save training state
        training_state.save(state_file);

        // Clean up old checkpoints
        cleanup_old_checkpoints();

        std::cout << "Saved checkpoint: " << name << std::endl;
    }

    /**
     * Load complete training checkpoint
     */
    std::pair<std::unique_ptr<models::TemporalAutoencoder>, TrainingState>
    load_checkpoint(const std::string& checkpoint_name, std::mt19937& gen) {

        std::string model_file = checkpoint_dir_ + "/" + checkpoint_name + "_model.grud";
        std::string state_file = checkpoint_dir_ + "/" + checkpoint_name + "_state.grud";

        // Load model
        CheckpointHeader header;
        auto model = AutoencoderSerializer::load(model_file, gen, &header);

        // Load training state
        TrainingState training_state;
        training_state.load(state_file);

        std::cout << "Loaded checkpoint: " << checkpoint_name << std::endl;
        std::cout << "  Epoch: " << training_state.epoch << std::endl;
        std::cout << "  Step: " << training_state.step << std::endl;
        std::cout << "  Best loss: " << training_state.best_loss << std::endl;

        return {std::move(model), training_state};
    }

    /**
     * Get list of available checkpoints
     */
    std::vector<std::string> list_checkpoints() const {
        std::vector<std::string> checkpoints;

        // In practice, you'd scan the directory for .grud files
        // This is a simplified version

        return checkpoints;
    }

    /**
     * Find the latest checkpoint
     */
    std::string find_latest_checkpoint() const {
        auto checkpoints = list_checkpoints();
        if (checkpoints.empty()) {
            return "";
        }

        // Sort by modification time or parse epoch numbers
        // For simplicity, just return the last one alphabetically
        std::sort(checkpoints.begin(), checkpoints.end());
        return checkpoints.back();
    }

private:
    void cleanup_old_checkpoints() {
        // Remove old checkpoints beyond max_checkpoints_
        // Implementation would scan directory and remove oldest files
    }
};

// ============================================================================
// HUMAN-READABLE EXPORT
// ============================================================================

class ModelExporter {
public:
    /**
     * Export model parameters to human-readable text format
     */
    static void export_to_text(const models::TemporalAutoencoder& model,
                              const std::string& filename) {
        std::ofstream file(filename);
        if (!file) {
            throw std::runtime_error("Failed to open file for writing: " + filename);
        }

        file << "# GRU-D Temporal Autoencoder Parameters\n";
        file << "# Framework Version: " << get_version().to_string() << "\n";
        file << "# Export Time: " << std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count() << "\n\n";

        // Model configuration
        file << "[Configuration]\n";
        file << "input_size = " << model.config.input_size << "\n";
        file << "latent_size = " << model.config.latent_size << "\n";
        file << "hidden_size = " << model.config.hidden_size << "\n";
        file << "num_layers = " << model.config.num_layers << "\n";
        file << "bottleneck_type = " << static_cast<int>(model.config.bottleneck_type) << "\n";
        file << "mode = " << static_cast<int>(model.config.mode) << "\n";
        file << "forecast_horizon = " << model.config.forecast_horizon << "\n";
        file << "\n";

        // Parameters
        auto params = const_cast<models::TemporalAutoencoder&>(model).all_parameters();

        for (size_t i = 0; i < params.size(); ++i) {
            const auto* param = params[i];

            file << "[Parameter_" << i << "]\n";
            file << "name = " << param->name << "\n";
            file << "shape = (" << param->value.rows() << ", " << param->value.cols() << ")\n";
            file << "data = \n";

            file << std::fixed << std::setprecision(6);
            for (int row = 0; row < param->value.rows(); ++row) {
                for (int col = 0; col < param->value.cols(); ++col) {
                    file << param->value(row, col);
                    if (col < param->value.cols() - 1) file << ", ";
                }
                file << "\n";
            }
            file << "\n";
        }
    }

    /**
     * Export model summary statistics
     */
    static void export_summary(const models::TemporalAutoencoder& model,
                              const std::string& filename) {
        std::ofstream file(filename);
        if (!file) {
            throw std::runtime_error("Failed to open file for writing: " + filename);
        }

        file << "GRU-D Temporal Autoencoder Summary\n";
        file << "==================================\n\n";

        // Model info
        file << "Architecture:\n";
        file << "  Input size: " << model.config.input_size << "\n";
        file << "  Latent size: " << model.config.latent_size << "\n";
        file << "  Hidden size: " << model.config.hidden_size << "\n";
        file << "  Number of layers: " << model.config.num_layers << "\n";
        file << "  Total parameters: " << model.num_parameters() << "\n\n";

        // Parameter statistics
        auto params = const_cast<models::TemporalAutoencoder&>(model).all_parameters();

        file << "Parameter Statistics:\n";
        for (const auto* param : params) {
            float mean = param->value.mean();
            float std_dev = std::sqrt((param->value.array() - mean).square().mean());
            float min_val = param->value.minCoeff();
            float max_val = param->value.maxCoeff();

            file << "  " << param->name << ":\n";
            file << "    Shape: (" << param->value.rows() << ", " << param->value.cols() << ")\n";
            file << "    Mean: " << std::fixed << std::setprecision(6) << mean << "\n";
            file << "    Std: " << std_dev << "\n";
            file << "    Min: " << min_val << "\n";
            file << "    Max: " << max_val << "\n";
        }

        file << "\nConfiguration:\n";
        file << "  Bottleneck: " << static_cast<int>(model.config.bottleneck_type) << "\n";
        file << "  Mode: " << static_cast<int>(model.config.mode) << "\n";
        file << "  Dropout: " << model.config.dropout << "\n";
        file << "  Layer norm: " << (model.config.layer_norm ? "Yes" : "No") << "\n";
        file << "  Exponential decay: " << (model.config.use_exponential_decay ? "Yes" : "No") << "\n";
    }
};

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/**
 * Quick save function
 */
inline void save_model(const models::TemporalAutoencoder& model, const std::string& filename) {
    AutoencoderSerializer::save(model, filename);
}

/**
 * Quick load function
 */
inline std::unique_ptr<models::TemporalAutoencoder> load_model(const std::string& filename, std::mt19937& gen) {
    return AutoencoderSerializer::load(filename, gen);
}

/**
 * Get file size in bytes
 */
inline size_t get_file_size(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file) {
        return 0;
    }
    return static_cast<size_t>(file.tellg());
}

/**
 * Check if file exists
 */
inline bool file_exists(const std::string& filename) {
    std::ifstream file(filename);
    return file.good();
}

} // namespace io
} // namespace grud

#endif // GRUD_IO_SERIALIZATION_H