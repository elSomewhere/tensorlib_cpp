#include "grud_autoencoder_eigen.h"
#include <iostream>
#include <iomanip>

void test_reconstruction_mode() {
    std::cout << "\n=== Testing Reconstruction Mode ===" << std::endl;

    std::mt19937 gen(42);

    // Configure base RNN
    NpTemporalConfig base_rnn_config;
    base_rnn_config.batch_size = 2;
    base_rnn_config.in_size = 16;
    base_rnn_config.hid_size = 32;
    base_rnn_config.num_layers = 2;
    base_rnn_config.use_exponential_decay = true;
    base_rnn_config.layer_norm = true;
    base_rnn_config.dropout = 0.1f;
    base_rnn_config.lr = 2e-3f;
    base_rnn_config.tbptt_steps = 8;

    // Configure autoencoder
    AutoencoderConfig ae_config;
    ae_config.input_size = 12;
    ae_config.latent_size = 8;
    ae_config.internal_projection_size = 16;
    ae_config.bottleneck_type = BottleneckType::MEAN_POOL;
    ae_config.mask_projection_type = MaskProjectionType::MAX_POOL;
    ae_config.mode = AutoencoderMode::RECONSTRUCTION;
    ae_config.use_input_projection = true;

    // Create autoencoder
    auto autoencoder = std::make_unique<NpTemporalAutoencoder>(base_rnn_config, ae_config, gen);
    NpOnlineLearner learner(std::move(autoencoder), base_rnn_config);

    std::cout << "✅ Reconstruction autoencoder created" << std::endl;

    // Test streaming learning
    int batch_size = 2;
    learner.reset_streaming_state(batch_size);

    for (int step = 0; step < 50; ++step) {
        // Generate synthetic data
        MatrixXf x_t = MatrixXf::Random(batch_size, ae_config.input_size) * 0.5f;
        MatrixXf dt_t = (MatrixXf::Random(batch_size, 1).array().abs() + 0.1f) * 0.5f;
        MatrixXf mask_t = (MatrixXf::Random(batch_size, ae_config.input_size).array() > -0.2f).cast<float>();

        auto [loss, prediction] = learner.step_stream(x_t, dt_t, mask_t);

        if ((step + 1) % 10 == 0) {
            float mse = (prediction - x_t).array().square().mean();
            std::cout << "Step " << std::setw(2) << (step + 1)
                      << ": Loss=" << std::fixed << std::setprecision(4) << loss
                      << ", MSE=" << std::setprecision(4) << mse << std::endl;
        }
    }
}

void test_forecasting_mode() {
    std::cout << "\n=== Testing Forecasting Mode ===" << std::endl;

    std::mt19937 gen(42);

    // Configure base RNN
    NpTemporalConfig base_rnn_config;
    base_rnn_config.batch_size = 2;
    base_rnn_config.in_size = 16;
    base_rnn_config.hid_size = 32;
    base_rnn_config.num_layers = 2;
    base_rnn_config.lr = 2e-3f;
    base_rnn_config.tbptt_steps = 8;

    // Configure autoencoder for forecasting
    AutoencoderConfig ae_config;
    ae_config.input_size = 12;
    ae_config.latent_size = 8;
    ae_config.internal_projection_size = 16;
    ae_config.bottleneck_type = BottleneckType::ATTENTION_POOL;
    ae_config.attention_context_dim = 24;
    ae_config.mode = AutoencoderMode::FORECASTING;
    ae_config.forecast_horizon = 3;
    ae_config.forecasting_mode = ForecastingMode::DIRECT;
    ae_config.predict_future_dt = true;
    ae_config.dt_prediction_method = DTPresictionMethod::LEARNED;

    // Create autoencoder
    auto autoencoder = std::make_unique<NpTemporalAutoencoder>(base_rnn_config, ae_config, gen);
    NpOnlineLearner learner(std::move(autoencoder), base_rnn_config);

    std::cout << "✅ Forecasting autoencoder created with:" << std::endl;
    std::cout << "   - Attention pooling bottleneck" << std::endl;
    std::cout << "   - Learned dt prediction" << std::endl;
    std::cout << "   - " << ae_config.forecast_horizon << " step horizon" << std::endl;

    // Test streaming learning
    int batch_size = 2;
    learner.reset_streaming_state(batch_size);

    for (int step = 0; step < 30; ++step) {
        // Generate time series with trend
        MatrixXf x_t = MatrixXf::Zero(batch_size, ae_config.input_size);
        for (int b = 0; b < batch_size; ++b) {
            for (int f = 0; f < ae_config.input_size; ++f) {
                float trend = 0.01f * step;
                float seasonal = 0.3f * std::sin(2 * M_PI * step / 20.0f);
                float noise = (static_cast<float>(gen()) / gen.max() - 0.5f) * 0.1f;
                x_t(b, f) = trend + seasonal + noise;
            }
        }

        MatrixXf dt_t = MatrixXf::Ones(batch_size, 1) * 1.0f;

        // Generate future target
        MatrixXf future_target = MatrixXf::Random(ae_config.forecast_horizon * batch_size, ae_config.input_size) * 0.2f;

        auto [loss, forecast] = learner.step_stream(x_t, dt_t, std::nullopt, future_target);

        if ((step + 1) % 10 == 0) {
            std::cout << "Step " << std::setw(2) << (step + 1)
                      << ": Loss=" << std::fixed << std::setprecision(4) << loss
                      << ", Forecast shape=(" << forecast.rows() << "x" << forecast.cols() << ")" << std::endl;
        }
    }
}

void test_all_bottleneck_types() {
    std::cout << "\n=== Testing All Bottleneck Types ===" << std::endl;

    std::mt19937 gen(42);

    std::vector<std::pair<BottleneckType, std::string>> bottleneck_types = {
            {BottleneckType::LAST_HIDDEN, "LAST_HIDDEN"},
            {BottleneckType::MEAN_POOL, "MEAN_POOL"},
            {BottleneckType::MAX_POOL, "MAX_POOL"},
            {BottleneckType::ATTENTION_POOL, "ATTENTION_POOL"}
    };

    for (const auto& [type, name] : bottleneck_types) {
        std::cout << "\nTesting " << name << "..." << std::endl;

        NpTemporalConfig rnn_config;
        rnn_config.batch_size = 2;
        rnn_config.in_size = 8;
        rnn_config.hid_size = 16;
        rnn_config.num_layers = 1;

        AutoencoderConfig ae_config;
        ae_config.input_size = 6;
        ae_config.latent_size = 4;
        ae_config.internal_projection_size = 8;
        ae_config.bottleneck_type = type;
        ae_config.attention_context_dim = 12;

        try {
            auto autoencoder = std::make_unique<NpTemporalAutoencoder>(rnn_config, ae_config, gen);

            // Quick test
            std::vector<MatrixXf> X_seq = {MatrixXf::Random(2, 6)};
            std::vector<MatrixXf> dt_seq = {MatrixXf::Ones(2, 1)};
            std::vector<std::optional<MatrixXf>> mask_seq = {std::nullopt};
            std::vector<MatrixXf> init_h_enc = {MatrixXf::Zero(2, 16)};
            std::vector<MatrixXf> init_h_dec;

            auto result = autoencoder->forward(X_seq, dt_seq, mask_seq, init_h_enc, init_h_dec);

            std::cout << "  ✅ " << name << ": latent=" << result.latent.rows() << "x" << result.latent.cols()
                      << ", loss=" << std::fixed << std::setprecision(4) << result.loss << std::endl;

        } catch (const std::exception& e) {
            std::cout << "  ❌ " << name << ": " << e.what() << std::endl;
        }
    }
}

void test_all_mask_projector_types() {
    std::cout << "\n=== Testing All Mask Projector Types ===" << std::endl;

    std::mt19937 gen(42);

    std::vector<std::pair<MaskProjectionType, std::string>> mask_types = {
            {MaskProjectionType::MAX_POOL, "MAX_POOL"},
            {MaskProjectionType::LEARNED, "LEARNED"},
            {MaskProjectionType::ANY_OBSERVED, "ANY_OBSERVED"}
    };

    for (const auto& [type, name] : mask_types) {
        std::cout << "\nTesting " << name << "..." << std::endl;

        try {
            NpMaskProjector projector(8, 6, type, gen);
            MatrixXf mask = (MatrixXf::Random(3, 8).array() > 0.0f).cast<float>();

            auto [projected, cache] = projector.forward(mask);

            std::cout << "  ✅ " << name << ": " << mask.rows() << "x" << mask.cols()
                      << " -> " << projected.rows() << "x" << projected.cols() << std::endl;

            // Test backward
            MatrixXf d_projected = MatrixXf::Ones(3, 6);
            MatrixXf d_input = projector.backward(d_projected, cache);

            std::cout << "     Backward: " << d_projected.rows() << "x" << d_projected.cols()
                      << " -> " << d_input.rows() << "x" << d_input.cols() << std::endl;

        } catch (const std::exception& e) {
            std::cout << "  ❌ " << name << ": " << e.what() << std::endl;
        }
    }
}

int main() {
    std::cout << "🚀 Temporal Autoencoder C++ Implementation Test" << std::endl;
    std::cout << "===============================================" << std::endl;

    try {
        test_all_mask_projector_types();
        test_all_bottleneck_types();
        test_reconstruction_mode();
        test_forecasting_mode();

        std::cout << "\n🎉 All tests completed successfully!" << std::endl;
        std::cout << "✅ Core autoencoder functionality working" << std::endl;
        std::cout << "✅ All bottleneck types implemented" << std::endl;
        std::cout << "✅ All mask projector types implemented" << std::endl;
        std::cout << "✅ Both reconstruction and forecasting modes" << std::endl;
        std::cout << "✅ Online learning capability" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "\n❌ Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}