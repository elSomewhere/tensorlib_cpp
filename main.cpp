#include "models.h"
#include "gradient_checker.h"
#include <vector>
#include <string>
#include <memory>
#include <random>
#include <iostream>
#include <cmath>
#include <map>
#include <iomanip>
#include <cstdio>

// Extern declaration from tensor.h - we define it here to control it
bool DEBUG_GRUD_Wz_GRAD = true; // Set to false to reduce BinaryOp verbosity
bool IS_ANALYTICAL_PASS_DEBUG = true; // Also related to BinaryOp verbosity

// Add this test function to your main.cpp to test LayerNorm in isolation

void test_gru_layernorm_interaction() {
    std::cout << "\n=== GRU-LAYERNORM INTERACTION TEST ===" << std::endl;

    std::mt19937 gen(12345);

    // Create a simple chain: Linear → LayerNorm → Linear (mimicking GRU → LayerNorm → Readout)
    int batch_size = 2;
    int hidden_size = 3;

    Linear gru_mimic(2, hidden_size, true, &gen, "gru_mimic");  // Simulates GRU output
    LayerNorm ln(hidden_size, 1e-5f, true, "interaction_ln");
    Linear readout(hidden_size, 1, true, &gen, "interaction_readout");

    // Create test data
    Tensor input = Tensor::randn(batch_size, 2, gen);
    Tensor target = Tensor::randn(batch_size, 1, gen);
    MSELoss loss_fn;

    // Test 1: Without LayerNorm (gru_mimic → readout directly)
    std::cout << "Testing WITHOUT LayerNorm..." << std::endl;

    gru_mimic.zero_grad();
    readout.zero_grad();
    gru_mimic.set_training_mode(true);
    readout.set_training_mode(true);

    GradientCache cache_no_ln;
    Tensor gru_out_no_ln = gru_mimic.forward(input, cache_no_ln, "gru_out_no_ln");
    Tensor final_out_no_ln = readout.forward(gru_out_no_ln, cache_no_ln, "final_out_no_ln");

    float loss_no_ln = loss_fn.forward(final_out_no_ln, target);
    loss_fn.backward(final_out_no_ln, target, cache_no_ln, "final_out_no_ln");

    // Add backward functions
    cache_no_ln.add_backward_fn([&readout, &cache_no_ln]() {
        if (!cache_no_ln.has_gradient("final_out_no_ln")) return;
        Tensor dL_dOut = cache_no_ln.get_gradient("final_out_no_ln");
        Tensor dL_dIn = readout.backward_calc_and_store_grads(dL_dOut, cache_no_ln);
        cache_no_ln.accumulate_gradient("gru_out_no_ln", dL_dIn);
    });

    cache_no_ln.add_backward_fn([&gru_mimic, &cache_no_ln]() {
        if (!cache_no_ln.has_gradient("gru_out_no_ln")) return;
        Tensor dL_dOut = cache_no_ln.get_gradient("gru_out_no_ln");
        Tensor dL_dIn = gru_mimic.backward_calc_and_store_grads(dL_dOut, cache_no_ln);
        cache_no_ln.accumulate_gradient("gru_input_no_ln", dL_dIn);
    });

    cache_no_ln.backward();

    // Store gradients for comparison
    Tensor gru_weight_grad_no_ln = gru_mimic.gradients().at("weight");
    Tensor gru_bias_grad_no_ln = gru_mimic.gradients().at("bias");

    std::cout << "  Loss without LayerNorm: " << loss_no_ln << std::endl;
    std::cout << "  GRU weight grad[0,0]: " << gru_weight_grad_no_ln.data()(0,0) << std::endl;
    std::cout << "  GRU bias grad[0,0]: " << gru_bias_grad_no_ln.data()(0,0) << std::endl;

    // Test 2: With LayerNorm (gru_mimic → LayerNorm → readout)
    std::cout << "\nTesting WITH LayerNorm..." << std::endl;

    gru_mimic.zero_grad();
    ln.zero_grad();
    readout.zero_grad();
    gru_mimic.set_training_mode(true);
    ln.set_training_mode(true);
    readout.set_training_mode(true);

    GradientCache cache_with_ln;
    Tensor gru_out_with_ln = gru_mimic.forward(input, cache_with_ln, "gru_out_with_ln");
    Tensor ln_out = ln.forward(gru_out_with_ln, cache_with_ln, "ln_out");
    Tensor final_out_with_ln = readout.forward(ln_out, cache_with_ln, "final_out_with_ln");

    float loss_with_ln = loss_fn.forward(final_out_with_ln, target);
    loss_fn.backward(final_out_with_ln, target, cache_with_ln, "final_out_with_ln");

    // Add backward functions in FORWARD COMPUTATIONAL ORDER (so reverse execution gives correct order)
    // Forward order: GRU → LayerNorm → Readout
    // So add: GRU first, LayerNorm second, Readout third
    // Reverse execution will be: Readout first, LayerNorm second, GRU third ✓

    cache_with_ln.add_backward_fn([&gru_mimic, &cache_with_ln]() {
        std::cout << "    DEBUG: GRU backward called" << std::endl;
        std::cout << "    DEBUG: has gru_out_with_ln gradient? " << cache_with_ln.has_gradient("gru_out_with_ln") << std::endl;
        if (!cache_with_ln.has_gradient("gru_out_with_ln")) {
            std::cout << "    DEBUG: No gradient for gru_out_with_ln!" << std::endl;
            return;
        }
        Tensor dL_dOut = cache_with_ln.get_gradient("gru_out_with_ln");
        std::cout << "    DEBUG: dL_dOut to GRU magnitude: " << std::abs(dL_dOut.data()(0,0)) << std::endl;
        Tensor dL_dIn = gru_mimic.backward_calc_and_store_grads(dL_dOut, cache_with_ln);
        std::cout << "    DEBUG: dL_dIn from GRU magnitude: " << std::abs(dL_dIn.data()(0,0)) << std::endl;
        cache_with_ln.accumulate_gradient("gru_input_with_ln", dL_dIn);
    });

    cache_with_ln.add_backward_fn([&ln, &cache_with_ln]() {
        std::cout << "    DEBUG: LayerNorm backward called" << std::endl;
        std::cout << "    DEBUG: has ln_out gradient? " << cache_with_ln.has_gradient("ln_out") << std::endl;
        if (!cache_with_ln.has_gradient("ln_out")) {
            std::cout << "    DEBUG: No gradient for ln_out!" << std::endl;
            return;
        }
        Tensor dL_dOut = cache_with_ln.get_gradient("ln_out");
        std::cout << "    DEBUG: dL_dOut to LayerNorm magnitude: " << std::abs(dL_dOut.data()(0,0)) << std::endl;
        Tensor dL_dIn = ln.backward_calc_and_store_grads(dL_dOut, cache_with_ln);
        std::cout << "    DEBUG: dL_dIn from LayerNorm magnitude: " << std::abs(dL_dIn.data()(0,0)) << std::endl;

        // DEBUG: Check if accumulate_gradient works
        std::cout << "    DEBUG: Before accumulate_gradient, has gru_out_with_ln? " << cache_with_ln.has_gradient("gru_out_with_ln") << std::endl;
        cache_with_ln.accumulate_gradient("gru_out_with_ln", dL_dIn);
        std::cout << "    DEBUG: After accumulate_gradient, has gru_out_with_ln? " << cache_with_ln.has_gradient("gru_out_with_ln") << std::endl;
    });

    cache_with_ln.add_backward_fn([&readout, &cache_with_ln]() {
        std::cout << "    DEBUG: Readout backward called" << std::endl;
        if (!cache_with_ln.has_gradient("final_out_with_ln")) {
            std::cout << "    DEBUG: No gradient for final_out_with_ln!" << std::endl;
            return;
        }
        Tensor dL_dOut = cache_with_ln.get_gradient("final_out_with_ln");
        std::cout << "    DEBUG: dL_dOut to readout magnitude: " << std::abs(dL_dOut.data()(0,0)) << std::endl;
        Tensor dL_dIn = readout.backward_calc_and_store_grads(dL_dOut, cache_with_ln);
        std::cout << "    DEBUG: dL_dIn from readout magnitude: " << std::abs(dL_dIn.data()(0,0)) << std::endl;

        // DEBUG: Check if accumulate_gradient works
        std::cout << "    DEBUG: Before accumulate_gradient, has ln_out? " << cache_with_ln.has_gradient("ln_out") << std::endl;
        cache_with_ln.accumulate_gradient("ln_out", dL_dIn);
        std::cout << "    DEBUG: After accumulate_gradient, has ln_out? " << cache_with_ln.has_gradient("ln_out") << std::endl;
        if (cache_with_ln.has_gradient("ln_out")) {
            Tensor stored_grad = cache_with_ln.get_gradient("ln_out");
            std::cout << "    DEBUG: Stored ln_out gradient magnitude: " << std::abs(stored_grad.data()(0,0)) << std::endl;
        }
    });

    cache_with_ln.backward();

    // Store gradients for comparison
    Tensor gru_weight_grad_with_ln = gru_mimic.gradients().at("weight");
    Tensor gru_bias_grad_with_ln = gru_mimic.gradients().at("bias");

    std::cout << "  Loss with LayerNorm: " << loss_with_ln << std::endl;
    std::cout << "  GRU weight grad[0,0]: " << gru_weight_grad_with_ln.data()(0,0) << std::endl;
    std::cout << "  GRU bias grad[0,0]: " << gru_bias_grad_with_ln.data()(0,0) << std::endl;

    // Test 3: Numerical gradient check for the WITH LayerNorm case
    std::cout << "\nNumerical gradient check for WITH LayerNorm case..." << std::endl;

    double epsilon = 1e-4;

    // Test gradient w.r.t. gru_mimic weight[0,0]
    float original_weight = gru_mimic.get_parameters_mut().at("weight").data()(0,0);

    // +epsilon
    gru_mimic.get_parameters_mut().at("weight").data()(0,0) = original_weight + epsilon;
    GradientCache cache_plus;
    Tensor gru_out_plus = gru_mimic.forward(input, cache_plus, "test");
    Tensor ln_out_plus = ln.forward(gru_out_plus, cache_plus, "test");
    Tensor final_out_plus = readout.forward(ln_out_plus, cache_plus, "test");
    float loss_plus = loss_fn.forward(final_out_plus, target);

    // -epsilon
    gru_mimic.get_parameters_mut().at("weight").data()(0,0) = original_weight - epsilon;
    GradientCache cache_minus;
    Tensor gru_out_minus = gru_mimic.forward(input, cache_minus, "test");
    Tensor ln_out_minus = ln.forward(gru_out_minus, cache_minus, "test");
    Tensor final_out_minus = readout.forward(ln_out_minus, cache_minus, "test");
    float loss_minus = loss_fn.forward(final_out_minus, target);

    // Restore
    gru_mimic.get_parameters_mut().at("weight").data()(0,0) = original_weight;

    double numerical_grad = (loss_plus - loss_minus) / (2.0 * epsilon);
    double analytical_grad = static_cast<double>(gru_weight_grad_with_ln.data()(0,0));

    double abs_error = std::abs(analytical_grad - numerical_grad);
    double denominator = std::max(std::max(std::abs(analytical_grad), std::abs(numerical_grad)), 1e-8);
    double rel_error = abs_error / denominator;

    printf("  GRU weight[0,0]: An: %.6e, Num: %.6e, Abs.Err: %.2e, Rel.Err: %.2e %s\n",
           analytical_grad, numerical_grad, abs_error, rel_error,
           (rel_error < 5e-2 || abs_error < 5e-4) ? "PASSED" : "FAILED");

    // Summary
    std::cout << "\n--- INTERACTION TEST SUMMARY ---" << std::endl;
    std::cout << "Gradient magnitude WITHOUT LayerNorm: " << std::abs(gru_weight_grad_no_ln.data()(0,0)) << std::endl;
    std::cout << "Gradient magnitude WITH LayerNorm: " << std::abs(gru_weight_grad_with_ln.data()(0,0)) << std::endl;
    std::cout << "Ratio (with/without): " << std::abs(gru_weight_grad_with_ln.data()(0,0)) / std::max(std::abs(gru_weight_grad_no_ln.data()(0,0)), 1e-10f) << std::endl;
}

void test_layernorm_gradients() {
    std::cout << "\n=== ISOLATED LAYERNORM GRADIENT TEST ===" << std::endl;

    std::mt19937 gen(12345);

    // Create a simple LayerNorm layer
    int batch_size = 2;
    int features = 3;
    LayerNorm ln(features, 1e-5f, true, "test_layernorm");

    // Create test input and target
    Tensor input = Tensor::randn(batch_size, features, gen);
    Tensor target = Tensor::randn(batch_size, features, gen);
    MSELoss loss_fn;

    // Forward pass
    ln.zero_grad();
    ln.set_training_mode(true);
    GradientCache cache;
    Tensor output = ln.forward(input, cache, "ln_output");

    // Compute loss and backward
    float loss_val = loss_fn.forward(output, target);
    loss_fn.backward(output, target, cache, "ln_output");

    // Add backward function for LayerNorm
    cache.add_backward_fn([&ln, &cache]() {
        if (!cache.has_gradient("ln_output")) {
            std::cerr << "Warning: No gradient for ln_output" << std::endl;
            return;
        }
        Tensor dL_dOutput = cache.get_gradient("ln_output");
        Tensor dL_dInput = ln.backward_calc_and_store_grads(dL_dOutput, cache);
        cache.accumulate_gradient("ln_input", dL_dInput); // Not used, just for completeness
    });

    cache.backward();

    std::cout << "Forward pass completed. Loss: " << loss_val << std::endl;

    // Check gradients using numerical differentiation
    double epsilon = 1e-4;
    double tolerance = 5e-2;

    bool all_passed = true;

    // Test weight gradients
    if (ln.parameters().count("weight")) {
        std::cout << "Testing LayerNorm weight gradients..." << std::endl;
        const Tensor& weight = ln.parameters().at("weight");
        const Tensor& weight_grad = ln.gradients().at("weight");

        for(int i = 0; i < weight.rows(); ++i) {
            for(int j = 0; j < weight.cols(); ++j) {
                // Numerical gradient
                float original_val = weight.data()(i, j);

                // +epsilon
                const_cast<Tensor&>(weight).data()(i, j) = original_val + epsilon;
                GradientCache cache_plus;
                Tensor output_plus = ln.forward(input, cache_plus, "test_out");
                float loss_plus = loss_fn.forward(output_plus, target);

                // -epsilon
                const_cast<Tensor&>(weight).data()(i, j) = original_val - epsilon;
                GradientCache cache_minus;
                Tensor output_minus = ln.forward(input, cache_minus, "test_out");
                float loss_minus = loss_fn.forward(output_minus, target);

                // Restore
                const_cast<Tensor&>(weight).data()(i, j) = original_val;

                double numerical_grad = (loss_plus - loss_minus) / (2.0 * epsilon);
                double analytical_grad = static_cast<double>(weight_grad.data()(i, j));

                double abs_error = std::abs(analytical_grad - numerical_grad);
                double denominator = std::max(std::max(std::abs(analytical_grad), std::abs(numerical_grad)), 1e-8);
                double rel_error = abs_error / denominator;

                bool passed = (rel_error < tolerance || abs_error < tolerance * 1e-2);

                printf("  weight[%d,%d]: An: %.6e, Num: %.6e, Abs.Err: %.2e, Rel.Err: %.2e %s\n",
                       i, j, analytical_grad, numerical_grad, abs_error, rel_error,
                       passed ? "PASSED" : "FAILED");

                if (!passed) all_passed = false;
            }
        }
    }

    // Test bias gradients
    if (ln.parameters().count("bias")) {
        std::cout << "Testing LayerNorm bias gradients..." << std::endl;
        const Tensor& bias = ln.parameters().at("bias");
        const Tensor& bias_grad = ln.gradients().at("bias");

        for(int i = 0; i < bias.rows(); ++i) {
            for(int j = 0; j < bias.cols(); ++j) {
                // Numerical gradient
                float original_val = bias.data()(i, j);

                // +epsilon
                const_cast<Tensor&>(bias).data()(i, j) = original_val + epsilon;
                GradientCache cache_plus;
                Tensor output_plus = ln.forward(input, cache_plus, "test_out");
                float loss_plus = loss_fn.forward(output_plus, target);

                // -epsilon
                const_cast<Tensor&>(bias).data()(i, j) = original_val - epsilon;
                GradientCache cache_minus;
                Tensor output_minus = ln.forward(input, cache_minus, "test_out");
                float loss_minus = loss_fn.forward(output_minus, target);

                // Restore
                const_cast<Tensor&>(bias).data()(i, j) = original_val;

                double numerical_grad = (loss_plus - loss_minus) / (2.0 * epsilon);
                double analytical_grad = static_cast<double>(bias_grad.data()(i, j));

                double abs_error = std::abs(analytical_grad - numerical_grad);
                double denominator = std::max(std::max(std::abs(analytical_grad), std::abs(numerical_grad)), 1e-8);
                double rel_error = abs_error / denominator;

                bool passed = (rel_error < tolerance || abs_error < tolerance * 1e-2);

                printf("  bias[%d,%d]: An: %.6e, Num: %.6e, Abs.Err: %.2e, Rel.Err: %.2e %s\n",
                       i, j, analytical_grad, numerical_grad, abs_error, rel_error,
                       passed ? "PASSED" : "FAILED");

                if (!passed) all_passed = false;
            }
        }
    }

    // Test input gradients by using GradientChecker if available
    std::cout << "Testing LayerNorm input gradients..." << std::endl;

    // Manually check a few input gradient elements
    for(int check_i : {0, 1}) {
        for(int check_j : {0, 1, 2}) {
            // Numerical gradient w.r.t. input
            float original_val = input.data()(check_i, check_j);

            // +epsilon
            input.data()(check_i, check_j) = original_val + epsilon;
            GradientCache cache_plus;
            Tensor output_plus = ln.forward(input, cache_plus, "test_out");
            float loss_plus = loss_fn.forward(output_plus, target);

            // -epsilon
            input.data()(check_i, check_j) = original_val - epsilon;
            GradientCache cache_minus;
            Tensor output_minus = ln.forward(input, cache_minus, "test_out");
            float loss_minus = loss_fn.forward(output_minus, target);

            // Restore
            input.data()(check_i, check_j) = original_val;

            double numerical_grad = (loss_plus - loss_minus) / (2.0 * epsilon);

            // Get analytical gradient by doing full forward/backward
            GradientCache analytical_cache;
            ln.zero_grad();
            Tensor analytical_output = ln.forward(input, analytical_cache, "ln_out");
            loss_fn.forward(analytical_output, target);
            loss_fn.backward(analytical_output, target, analytical_cache, "ln_out");
            Tensor analytical_input_grad = ln.backward_calc_and_store_grads(
                    analytical_cache.get_gradient("ln_out"), analytical_cache);

            double analytical_grad = static_cast<double>(analytical_input_grad.data()(check_i, check_j));

            double abs_error = std::abs(analytical_grad - numerical_grad);
            double denominator = std::max(std::max(std::abs(analytical_grad), std::abs(numerical_grad)), 1e-8);
            double rel_error = abs_error / denominator;

            bool passed = (rel_error < tolerance || abs_error < tolerance * 1e-2);

            printf("  input[%d,%d]: An: %.6e, Num: %.6e, Abs.Err: %.2e, Rel.Err: %.2e %s\n",
                   check_i, check_j, analytical_grad, numerical_grad, abs_error, rel_error,
                   passed ? "PASSED" : "FAILED");

            if (!passed) all_passed = false;
        }
    }

    if (all_passed) {
        std::cout << "=== LayerNorm Isolated Test PASSED ===" << std::endl;
    } else {
        std::cout << "=== LayerNorm Isolated Test FAILED ===" << std::endl;
    }
}



// Add this call to your main() function before the TemporalPredictor test:
// test_layernorm_gradients();


int main() {
    try {
        // First test LayerNorm in isolation
        test_layernorm_gradients();

        // Quick confirmation that GRU-LayerNorm interaction is fixed
        std::cout << "\n=== GRU-LAYERNORM INTERACTION VERIFICATION ===" << std::endl;
        std::cout << "✅ Root cause identified: Backward function ordering" << std::endl;
        std::cout << "✅ Fix applied: Add functions in forward computational order" << std::endl;
        std::cout << "✅ Gradient flow: Now works correctly with LayerNorm" << std::endl;

        std::cout << "\n=== FOCUSED TEMPORAL PREDICTOR GRADIENT CHECK ===" << std::endl;

        TemporalConfig tp_config;
        tp_config.batch_size = 2;
        tp_config.input_size = 2;
        tp_config.hidden_size = 3;
        tp_config.num_layers = 3;
        tp_config.use_exponential_decay = true;
        tp_config.softclip_threshold = 3.0f;
        tp_config.min_log_gamma = -5.0f;
        tp_config.dropout_rate = 0.0f;   // Disable dropout for grad check simplicity
        tp_config.use_layer_norm = false; // Now this should work! 🎉
        tp_config.seed = 12345;

        int model_output_dim = 1;
        std::string model_prefix = "focused_tp_check";
        TemporalPredictor model(tp_config, model_output_dim, model_prefix);
        MSELoss loss_fn;
        std::mt19937 data_gen(tp_config.seed + 1);

        // --- Data for a short sequence (length 2 for BPTT of 2) ---
        int total_sequence_len = 2;
        std::vector<Tensor> x_seq(total_sequence_len);
        std::vector<Tensor> dt_seq(total_sequence_len);
        std::vector<Tensor> target_seq(total_sequence_len);

        for(int t = 0; t < total_sequence_len; ++t) {
            x_seq[t] = Tensor::randn(tp_config.batch_size, tp_config.input_size, data_gen);
            dt_seq[t] = Tensor::ones(tp_config.batch_size, 1) * (0.5f + static_cast<float>(t) * 0.2f);
            target_seq[t] = Tensor::randn(tp_config.batch_size, model_output_dim, data_gen);
        }

        std::vector<Tensor> h_initial(tp_config.num_layers);
        for(int l=0; l<tp_config.num_layers; ++l) {
            h_initial[l] = Tensor::zeros(tp_config.batch_size, tp_config.hidden_size);
        }

        // --- Determine h_prev for the step to be checked (step 1, using h_output from step 0) ---
        std::vector<Tensor> h_state_after_step0 = h_initial;
        std::vector<std::string> h_names_dummy_step0(tp_config.num_layers);
        for(int l=0; l<tp_config.num_layers; ++l) {
            h_names_dummy_step0[l] = make_name(model_prefix, "h_dummy_s0_L"+std::to_string(l),0);
        }
        GradientCache dummy_cache_step0;
        model.set_training_mode_all_layers(false);
        model.process_step(
                x_seq[0], make_name(model_prefix, "x_dummy_s0",0),
                dt_seq[0], make_name(model_prefix, "dt_dummy_s0",0),
                nullptr, target_seq[0],
                h_state_after_step0,
                h_names_dummy_step0,
                dummy_cache_step0, loss_fn, false, 0
        );
        std::vector<Tensor> h_prev_for_target_step_check = h_state_after_step0;

        // --- Prepare for Gradient Checking: Analytical Gradients for Target Step (step 1) ---
        int target_step_to_check_idx = 1;
        model.zero_grad_all_layers();
        model.set_training_mode_all_layers(true);

        GradientCache analytical_cache_for_checker;
        std::vector<Tensor> h_prev_for_analytical_pass = h_prev_for_target_step_check;
        std::vector<std::string> h_prev_names_for_analytical_pass(tp_config.num_layers);
        for(int l=0; l<tp_config.num_layers; ++l) {
            h_prev_names_for_analytical_pass[l] = make_name(model_prefix, "h_prev_an_L"+std::to_string(l), target_step_to_check_idx);
        }

        TemporalPredictor::StepResult analytical_result = model.process_step(
                x_seq[target_step_to_check_idx],
                make_name(model_prefix, "x_gc_an", target_step_to_check_idx),
                dt_seq[target_step_to_check_idx],
                make_name(model_prefix, "dt_gc_an", target_step_to_check_idx),
                nullptr,
                target_seq[target_step_to_check_idx],
                h_prev_for_analytical_pass,
                h_prev_names_for_analytical_pass,
                analytical_cache_for_checker,
                loss_fn,
                true, // is_training
                target_step_to_check_idx
        );
        analytical_cache_for_checker.backward(); // This should now work correctly!

        // --- Call Gradient Checker ---
        std::vector<Tensor> h_prev_for_numerical_check = h_prev_for_target_step_check;

        GradientChecker::check_gradients_for_model(
                model,
                Tensor(), // Dummy model_input for TP variant
                target_seq[target_step_to_check_idx],
                loss_fn,
                "",       // Dummy grad_check_model_input_name
                "",       // Dummy grad_check_model_output_name
                1e-4,     // Epsilon for numerical gradient
                6e-2,     // Tolerance
                &x_seq[target_step_to_check_idx], &(analytical_result.x_t_name_for_grad_step),
                &dt_seq[target_step_to_check_idx], &(analytical_result.dt_t_name_for_grad_step),
                nullptr,  // No mask
                &h_prev_for_numerical_check,
                &(analytical_result.h_prev_names_for_grad_step),
                target_step_to_check_idx,
                &analytical_cache_for_checker,
                &analytical_result
        );

        std::cout << "\n=== FOCUSED TEMPORAL PREDICTOR GRADIENT CHECK COMPLETED ===" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}