# GRU-D Autoencoder Framework v2.0

A complete C++ implementation of GRU-D (Gated Recurrent Unit with Decay) autoencoders featuring automatic differentiation, streaming training, and comprehensive temporal modeling capabilities.

## 🌟 Key Features

- **Automatic Differentiation**: Modern tape-based autograd system with numerical gradient checking
- **Temporal Modeling**: Full GRU-D implementation with exponential decay and missing value imputation
- **Flexible Architectures**: Support for reconstruction and forecasting autoencoders
- **Streaming Training**: Online learning with Truncated Backpropagation Through Time (TBPTT)
- **Production Ready**: Model serialization, checkpointing, and comprehensive testing
- **Header-Only**: Single-file include with no external dependencies except Eigen

## 📚 Table of Contents

1. [Quick Start](#quick-start)
2. [Installation](#installation)
3. [Architecture Overview](#architecture-overview)
4. [Basic Usage](#basic-usage)
5. [Advanced Features](#advanced-features)
6. [Training](#training)
7. [Model Persistence](#model-persistence)
8. [Examples](#examples)
9. [API Reference](#api-reference)
10. [Contributing](#contributing)

## 🚀 Quick Start

```cpp
#include "grud.h"

int main() {
    // Verify framework installation
    if (!grud::diagnostics::self_test()) {
        return 1;
    }
    
    // Create a reconstruction autoencoder
    std::mt19937 gen(42);
    auto model = grud::quick::reconstruction_model(12, 6, 32, &gen);
    
    // Train on synthetic data
    auto metrics = grud::quick::quick_train(std::move(model), 20, 4, 8, &gen);
    
    std::cout << "Training completed with loss: " << metrics.loss << std::endl;
    return 0;
}
```

## 💻 Installation

### Prerequisites
- C++17 compatible compiler (GCC 7+, Clang 5+, MSVC 2017+)
- Eigen 3.3+ (header-only linear algebra library)

### Basic Setup
1. Download Eigen: `git clone https://gitlab.com/libeigen/eigen.git`
2. Download GRU-D framework (all header files)
3. Include path setup:
   ```cpp
   #include <Eigen/Dense>  // Ensure Eigen is in include path
   #include "grud.h"       // Single header includes everything
   ```

### CMake Integration
```cmake
find_package(Eigen3 REQUIRED)
add_executable(my_app main.cpp)
target_link_libraries(my_app Eigen3::Eigen)
target_include_directories(my_app PRIVATE path/to/grud/headers)
```

## 🏗 Architecture Overview

### Core Components

```
grud/
├── core/
│   ├── module.h          # Base classes (Module, Param, Context)
│   ├── tape.h            # Autograd tape system
│   └── checkgrad.h       # Numerical gradient checking
├── layers/
│   ├── basic.h           # Linear, LayerNorm, Dropout, Activations
│   └── temporal.h        # GRU-D cells and temporal layers
├── models/
│   └── autoencoder.h     # Complete autoencoder models
├── optim/
│   └── optimizers.h      # SGD, AdamW, loss functions
├── training/
│   └── trainer.h         # Online and batch training
├── io/
│   └── serialization.h   # Model persistence
├── utils/
│   └── data.h            # Preprocessing and utilities
├── testing/
│   └── tests.h           # Comprehensive test suite
└── grud.h                # Single include header
```

### Design Principles

1. **Module-Based Architecture**: Every component inherits from `Module` base class
2. **Tape-Based Autograd**: Computational graph stored on tape for automatic differentiation
3. **Memory Efficient**: TBPTT for streaming data, minimal memory footprint
4. **Type Safety**: Strong typing with Eigen matrices, no raw pointers
5. **Testable**: Gradient checking for every layer, comprehensive test suite

## 📖 Basic Usage

### Creating Models

```cpp
std::mt19937 gen(42);

// Reconstruction autoencoder
auto reconstruction_ae = grud::factory::create_autoencoder(
    12,  // input_size
    6,   // latent_size  
    32,  // hidden_size
    2,   // num_layers
    grud::models::BottleneckType::MEAN_POOL,
    grud::models::AutoencoderMode::RECONSTRUCTION
);

// Forecasting autoencoder
auto forecasting_ae = grud::factory::create_autoencoder(
    8,   // input_size
    4,   // latent_size
    24,  // hidden_size
    1,   // num_layers
    grud::models::BottleneckType::LAST_HIDDEN,
    grud::models::AutoencoderMode::FORECASTING,
    5    // forecast_horizon
);
```

### Forward Pass

```cpp
// Prepare data
std::vector<Eigen::MatrixXf> X_seq = {/* your time series data */};
std::vector<Eigen::MatrixXf> dt_seq = {/* time differences */};
std::vector<std::optional<Eigen::MatrixXf>> mask_seq = {/* missing value masks */};

// Forward pass
grud::Tape tape;
auto result = model->forward_autoencoder(X_seq, dt_seq, mask_seq, {}, {}, tape);

std::cout << "Output shape: " << result.output_sequence.rows() 
          << " x " << result.output_sequence.cols() << std::endl;
std::cout << "Latent shape: " << result.latent.rows() 
          << " x " << result.latent.cols() << std::endl;
```

### Training

```cpp
// Create trainer
auto trainer = grud::factory::create_trainer(
    std::move(model), 
    1e-3f,      // learning_rate
    "adamw",    // optimizer
    "mse"       // loss_function
);

// Online training
for (int step = 0; step < 1000; ++step) {
    auto [loss, prediction] = trainer->step_online(x_t, dt_t, mask_t, target_t);
    
    if (step % 100 == 0) {
        std::cout << "Step " << step << ", Loss: " << loss << std::endl;
    }
}
```

## 🎯 Advanced Features

### Missing Value Imputation

GRU-D automatically handles missing values through learned imputation:

```cpp
// Create mask (1 = observed, 0 = missing)
Eigen::MatrixXf mask(batch_size, input_size);
mask.setOnes();
mask(0, 2) = 0.0f;  // Mark feature 2 as missing for batch 0

// Model automatically imputes missing values during forward pass
auto result = model->forward_autoencoder(X_seq, dt_seq, {mask}, {}, {}, tape);
```

### Exponential Decay

Temporal decay of hidden states based on time intervals:

```cpp
grud::models::AutoencoderConfig config;
config.use_exponential_decay = true;
config.softclip_threshold = 3.0f;  // Prevent numerical instability
config.min_log_gamma = -10.0f;     // Minimum decay rate

auto model = std::make_unique<grud::models::TemporalAutoencoder>(config, gen);
```

### Custom Loss Functions

```cpp
// Huber loss for robustness to outliers
auto trainer = grud::factory::create_trainer(
    std::move(model), 1e-3f, "adamw", "huber");

// Custom loss with temporal weighting
grud::loss::MSELoss base_loss;
auto temporal_loss = std::make_unique<grud::loss::TemporalLoss>(
    std::make_unique<grud::loss::MSELoss>(), 
    0.5f,  // ramp_start
    1.0f   // ramp_end
);
```

### Bottleneck Aggregation

Multiple ways to aggregate temporal sequences:

```cpp
// Last hidden state (good for forecasting)
config.bottleneck_type = grud::models::BottleneckType::LAST_HIDDEN;

// Mean pooling (good for reconstruction)
config.bottleneck_type = grud::models::BottleneckType::MEAN_POOL;

// Max pooling (captures peak activations)
config.bottleneck_type = grud::models::BottleneckType::MAX_POOL;

// Attention pooling (learned importance weights)
config.bottleneck_type = grud::models::BottleneckType::ATTENTION_POOL;
```

## 🎓 Training

### Batch Training

```cpp
// Generate dataset
auto train_data = grud::training::generate_synthetic_dataset(100, 4, 10, 12, gen);
auto val_data = grud::training::generate_synthetic_dataset(20, 4, 10, 12, gen);

// Create batch trainer
auto batch_trainer = grud::factory::create_batch_trainer(
    std::move(model), 50, 1e-3f, "adamw", &gen);

// Train
batch_trainer->train(train_data, val_data);

// Get training history
const auto& train_losses = batch_trainer->get_train_losses();
const auto& val_losses = batch_trainer->get_val_losses();
```

### Online/Streaming Training

```cpp
auto online_trainer = grud::factory::create_trainer(std::move(model));

// Stream data timestep by timestep
for (const auto& timestep : data_stream) {
    auto [loss, prediction] = online_trainer->step_online(
        timestep.x, timestep.dt, timestep.mask, timestep.target);
    
    // Use prediction for downstream tasks
    process_prediction(prediction);
}
```

### Learning Rate Scheduling

```cpp
grud::training::TrainingConfig config;
config.use_lr_scheduler = true;
config.scheduler_type = "cosine";
config.T_max = 100;  // For cosine annealing

// Step decay
config.scheduler_type = "step";
config.step_size = 30;
config.gamma = 0.1f;
```

## 💾 Model Persistence

### Save and Load Models

```cpp
// Save model
grud::io::save_model(*model, "my_model.grud");

// Load model
std::mt19937 gen(42);
auto loaded_model = grud::io::load_model("my_model.grud", gen);
```

### Checkpointing

```cpp
grud::io::CheckpointManager checkpoint_manager("./checkpoints", 5);

// Save checkpoint during training
grud::io::TrainingState state;
state.epoch = current_epoch;
state.step = current_step;
state.best_loss = best_loss;

checkpoint_manager.save_checkpoint(*model, state, "epoch_10");

// Resume training
auto [resumed_model, resumed_state] = checkpoint_manager.load_checkpoint("epoch_10", gen);
```

### Export for Analysis

```cpp
// Human-readable text export
grud::io::ModelExporter::export_to_text(*model, "model_params.txt");

// Summary statistics
grud::io::ModelExporter::export_summary(*model, "model_summary.txt");
```

## 📊 Examples

### Time Series Reconstruction

```cpp
#include "grud.h"

int main() {
    std::mt19937 gen(42);
    
    // Create model for 12-dimensional time series
    auto model = grud::quick::reconstruction_model(12, 6, 64, &gen);
    
    // Generate synthetic data with missing values
    auto data = grud::utils::SequenceGenerator::generate_sinusoidal(4, 20, 12, 0.1f, gen);
    
    // Add missing values
    for (auto& mask : data.mask_seq) {
        mask = grud::utils::MaskGenerator::random_mask(4, 12, 0.2f, gen);
    }
    
    // Train model
    auto trainer = grud::factory::create_trainer(std::move(model), 1e-3f, "adamw", "mse", &gen);
    auto [loss, predictions] = trainer->process_batch(data);
    
    std::cout << "Reconstruction loss: " << loss << std::endl;
    
    return 0;
}
```

### Multi-step Forecasting

```cpp
#include "grud.h"

int main() {
    std::mt19937 gen(42);
    
    // Create forecasting model
    auto model = grud::quick::forecasting_model(8, 5, 4, 32, &gen);  // Predict 5 steps ahead
    
    // Generate autoregressive data
    std::vector<float> ar_coeffs = {0.7f, -0.2f, 0.1f};
    auto data = grud::utils::SequenceGenerator::generate_autoregressive(2, 15, 8, ar_coeffs, 0.05f, gen);
    
    // Use first 10 timesteps to predict next 5
    std::vector<Eigen::MatrixXf> history(data.X_seq.begin(), data.X_seq.begin() + 10);
    std::vector<Eigen::MatrixXf> history_dt(data.dt_seq.begin(), data.dt_seq.begin() + 10);
    std::vector<std::optional<Eigen::MatrixXf>> history_mask(data.mask_seq.begin(), data.mask_seq.begin() + 10);
    
    // Create target (next 5 timesteps)
    Eigen::MatrixXf target(5 * 2, 8);  // 5 timesteps * 2 batch size
    for (int t = 0; t < 5; ++t) {
        target.block(t * 2, 0, 2, 8) = data.X_seq[10 + t];
    }
    
    // Forward pass
    grud::Tape tape;
    auto result = model->forward_autoencoder(history, history_dt, history_mask, {}, {}, tape);
    
    // Evaluate forecast
    auto metrics = grud::utils::MetricsCalculator::calculate_metrics(result.output_sequence, target);
    grud::utils::MetricsCalculator::print_metrics(metrics, "Forecast ");
    
    return 0;
}
```

### Data Preprocessing Pipeline

```cpp
#include "grud.h"

int main() {
    std::mt19937 gen(42);
    
    // Generate raw data
    auto raw_data = grud::utils::SequenceGenerator::generate_random_walk(10, 50, 6, 1.0f, gen);
    
    // Extract data matrices
    std::vector<Eigen::MatrixXf> data_matrices;
    for (const auto& x : raw_data.X_seq) {
        data_matrices.push_back(x);
    }
    
    // Normalize data
    grud::utils::StandardScaler scaler;
    auto normalized_data = scaler.fit_transform(data_matrices);
    
    // Split dataset
    std::vector<grud::training::Batch> dataset;
    for (size_t i = 0; i < normalized_data.size() - 10; i += 10) {
        grud::training::Batch batch;
        for (int j = 0; j < 10; ++j) {
            batch.X_seq.push_back(normalized_data[i + j]);
            batch.dt_seq.push_back(Eigen::MatrixXf::Ones(10, 1));
            batch.mask_seq.push_back(Eigen::MatrixXf::Ones(10, 6));
        }
        dataset.push_back(batch);
    }
    
    auto splits = grud::utils::DatasetSplitter::train_val_test_split(dataset, 0.7f, 0.2f, gen);
    
    std::cout << "Train batches: " << splits.train.size() << std::endl;
    std::cout << "Val batches: " << splits.val.size() << std::endl;
    std::cout << "Test batches: " << splits.test.size() << std::endl;
    
    return 0;
}
```

## 🧪 Testing and Validation

### Run Test Suite

```cpp
#include "grud.h"

int main() {
    // Run comprehensive test suite
    grud::testing::run_all_tests();
    
    // Run quick self-test
    if (grud::diagnostics::self_test()) {
        std::cout << "✅ Framework working correctly" << std::endl;
    }
    
    return 0;
}
```

### Gradient Checking

```cpp
// Check gradients for any module
std::mt19937 gen(42);
auto linear = std::make_unique<grud::Linear>(10, 5, true, &gen);
Eigen::MatrixXf input = Eigen::MatrixXf::Random(3, 10);

auto result = grud::checkgrad::check_gradients(*linear, input, 1e-5f, 1e-3f, 1e-8f, true);

if (result.passed) {
    std::cout << "✅ Gradients correct" << std::endl;
} else {
    std::cout << "❌ Gradient check failed" << std::endl;
    std::cout << "Max relative error: " << result.max_relative_error << std::endl;
}
```

### Benchmarking

```cpp
// Performance benchmark
std::mt19937 gen(42);
auto model = grud::quick::reconstruction_model(32, 16, 64, &gen);

grud::quick::benchmark(*model, 16, 32, 100);
```

## 📖 API Reference

### Core Classes

| Class | Description |
|-------|-------------|
| `grud::Module` | Base class for all neural network components |
| `grud::Param` | Trainable parameter with value and gradient |
| `grud::Context` | Stores intermediate values for backward pass |
| `grud::Tape` | Computational graph for automatic differentiation |

### Layers

| Layer | Description |
|-------|-------------|
| `grud::Linear` | Fully connected layer: y = xW^T + b |
| `grud::LayerNorm` | Layer normalization |
| `grud::Dropout` | Dropout regularization |
| `grud::ReLU`, `grud::Tanh`, `grud::Sigmoid` | Activation functions |
| `grud::GRUDCell` | GRU with decay and imputation |
| `grud::TemporalRNNLayer` | Complete temporal layer with normalization |

### Models

| Model | Description |
|-------|-------------|
| `grud::TemporalAutoencoder` | Complete autoencoder with encoder/decoder |
| `grud::TemporalEncoder` | Multi-layer temporal encoder |
| `grud::TemporalDecoder` | Multi-layer temporal decoder |
| `grud::BottleneckAggregation` | Sequence aggregation module |

### Training

| Class | Description |
|-------|-------------|
| `grud::OnlineTrainer` | Streaming training with TBPTT |
| `grud::BatchTrainer` | Offline batch training |
| `grud::SGD`, `grud::AdamW` | Optimizers |
| `grud::MSELoss`, `grud::MAELoss`, `grud::HuberLoss` | Loss functions |

### Utilities

| Class | Description |
|-------|-------------|
| `grud::utils::StandardScaler` | Data normalization |
| `grud::utils::MaskGenerator` | Missing value patterns |
| `grud::utils::SequenceGenerator` | Synthetic data generation |
| `grud::utils::MetricsCalculator` | Evaluation metrics |

## 🤝 Contributing

### Development Setup

1. Clone repository with submodules
2. Install Eigen 3.3+
3. Set up build system (CMake recommended)
4. Run test suite to verify setup

### Code Style

- C++17 standard
- Header-only implementation
- Comprehensive documentation
- Gradient checking for all layers
- Exception safety guarantees

### Testing

All contributions must include:
- Unit tests for new functionality
- Gradient checking for new layers
- Performance benchmarks for significant changes
- Documentation updates

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Eigen** for efficient linear algebra
- **Original GRU-D paper** by Che et al. (2018)
- **PyTorch** for autograd inspiration
- **Research community** for temporal modeling advances

## 📞 Support

- **Documentation**: See examples/ directory
- **Issues**: GitHub issue tracker
- **Discussions**: GitHub discussions
- **Performance**: Use built-in profiling tools

---

**GRU-D Autoencoder Framework v2.0** - Built for production temporal modeling with C++ performance and Python-like ease of use.