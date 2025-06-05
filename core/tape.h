//
// grud/core/tape.h - Autograd tape system
//

#ifndef GRUD_CORE_TAPE_H
#define GRUD_CORE_TAPE_H

#include "module.h"
#include <vector>
#include <memory>
#include <iostream>

namespace grud {

// ============================================================================
// TAPE SYSTEM
// ============================================================================

/**
 * Computational graph tape for automatic differentiation
 * Stores sequence of operations for backward pass
 */
class Tape {
public:
    /**
     * Clear all recorded operations
     */
    void clear() {
        nodes.clear();
    }

    /**
     * Push a new context onto the tape
     * @param ctx Context to add
     * @return Index of the added context
     */
    size_t push(Context&& ctx) {
        size_t idx = nodes.size();
        nodes.emplace_back(std::move(ctx));
        return idx;
    }

    /**
     * Get context at given index
     * @param idx Index of context
     * @return Reference to context
     */
    const Context& get(size_t idx) const {
        if (idx >= nodes.size()) {
            throw std::out_of_range("Tape index out of range");
        }
        return nodes[idx];
    }

    /**
     * Get mutable context at given index
     */
    Context& get_mutable(size_t idx) {
        if (idx >= nodes.size()) {
            throw std::out_of_range("Tape index out of range");
        }
        return nodes[idx];
    }

    /**
     * Get number of operations on tape
     */
    size_t size() const {
        return nodes.size();
    }

    /**
     * Check if tape is empty
     */
    bool empty() const {
        return nodes.empty();
    }

    /**
     * Reserve space for operations (optimization for TBPTT)
     */
    void reserve(size_t capacity) {
        nodes.reserve(capacity);
    }

    /**
     * Get the last context index
     */
    size_t last_index() const {
        if (empty()) {
            throw std::runtime_error("Cannot get last index of empty tape");
        }
        return size() - 1;
    }

private:
    std::vector<Context> nodes;
};

// ============================================================================
// AUTOGRAD ENGINE
// ============================================================================

namespace autograd {

    /**
     * Run backward pass through the computational graph
     * @param tape The computational tape
     * @param last_idx Index of the last operation (typically output)
     * @param grad_output Initial gradient (typically gradient of loss w.r.t. output)
     */
    inline void backward(Tape& tape, size_t last_idx, const Eigen::MatrixXf& grad_output) {
        if (tape.empty()) {
            return;
        }

        if (last_idx >= tape.size()) {
            throw std::out_of_range("Last index out of range for tape");
        }

        // Storage for gradients flowing backward
        std::vector<Eigen::MatrixXf> gradients(tape.size());

        // Initialize the gradient at the output
        gradients[last_idx] = grad_output;

        // Backward pass: traverse tape in reverse order
        for (size_t i = last_idx + 1; i > 0; --i) {
            size_t idx = i - 1;
            const Context& ctx = tape.get(idx);

            if (!ctx.op) {
                continue; // Skip contexts without operations
            }

            // Get gradient flowing into this operation
            const Eigen::MatrixXf& grad_out = gradients[idx];

            if (grad_out.size() == 0) {
                continue; // No gradient to propagate
            }

            // Compute gradient w.r.t. input using the module's backward method
            try {
                Eigen::MatrixXf grad_input = ctx.op->backward(grad_out, ctx);

                // Propagate gradient to parent operations
                // For now, we assume linear chain (each operation has one parent)
                // This can be extended for more complex graphs using ctx.child_indices
                if (idx > 0) {
                    if (gradients[idx - 1].size() == 0) {
                        gradients[idx - 1] = grad_input;
                    } else {
                        // Accumulate gradients if multiple paths lead to the same node
                        gradients[idx - 1] += grad_input;
                    }
                }

            } catch (const std::exception& e) {
                std::cerr << "Error in backward pass at index " << idx << ": " << e.what() << std::endl;
                throw;
            }
        }
    }

    /**
     * Convenience function for single-step backward
     */
    inline void backward(Tape& tape, const Eigen::MatrixXf& grad_output) {
        if (!tape.empty()) {
            backward(tape, tape.last_index(), grad_output);
        }
    }

    /**
     * Run forward and backward pass with automatic gradient computation
     * This is a convenience function for simple cases
     */
    template<typename ModuleType>
    std::pair<Eigen::MatrixXf, Eigen::MatrixXf> forward_backward(
        ModuleType& module,
        const Eigen::MatrixXf& input,
        const Eigen::MatrixXf& grad_output) {

        Tape tape;
        Context ctx(&module);

        // Forward pass
        Eigen::MatrixXf output = module.forward(input, ctx);
        tape.push(std::move(ctx));

        // Backward pass
        backward(tape, grad_output);

        // The input gradient is stored in the module's backward pass
        // For this simple case, we need to call backward again to get the input gradient
        Context dummy_ctx(&module);
        Eigen::MatrixXf input_grad = module.backward(grad_output, tape.get(0));

        return {output, input_grad};
    }

} // namespace autograd

// ============================================================================
// SEQUENTIAL MODULE
// ============================================================================

/**
 * Sequential container that applies modules in order
 * Useful for building networks layer by layer
 */
class Sequential : public Module {
public:
    Sequential() = default;

    /**
     * Add a module to the sequence
     */
    void add(std::unique_ptr<Module> module) {
        modules_.push_back(std::move(module));
    }

    /**
     * Forward pass through all modules in sequence
     */
    Eigen::MatrixXf forward(const Eigen::MatrixXf& input, Context& ctx) override {
        if (modules_.empty()) {
            return input;
        }

        Eigen::MatrixXf current_input = input;
        ctx.child_indices.reserve(modules_.size());

        // Get tape from context (assuming it's stored somewhere accessible)
        // For now, we'll create contexts locally and store their indices
        std::vector<Context> local_contexts;
        local_contexts.reserve(modules_.size());

        for (size_t i = 0; i < modules_.size(); ++i) {
            Context module_ctx(modules_[i].get());
            current_input = modules_[i]->forward(current_input, module_ctx);
            local_contexts.push_back(std::move(module_ctx));
        }

        // Save the local contexts for backward pass
        ctx.saved.reserve(local_contexts.size());
        for (auto& local_ctx : local_contexts) {
            // We need a way to store contexts - for now, we'll save the intermediate values
            // In a full implementation, this would be handled by the tape system
            ctx.save_for_backward(current_input); // Save output for each layer
        }

        return current_input;
    }

    /**
     * Backward pass through all modules in reverse order
     */
    Eigen::MatrixXf backward(const Eigen::MatrixXf& grad_output, const Context& ctx) override {
        if (modules_.empty()) {
            return grad_output;
        }

        Eigen::MatrixXf current_grad = grad_output;

        // In a full implementation, this would use the tape system
        // For now, we'll do a simple reverse pass
        for (int i = static_cast<int>(modules_.size()) - 1; i >= 0; --i) {
            // Note: This is simplified - in the full tape system,
            // we would retrieve the stored context for each module
            Context dummy_ctx(modules_[i].get());
            current_grad = modules_[i]->backward(current_grad, dummy_ctx);
        }

        return current_grad;
    }

    /**
     * Get all parameters from all modules
     */
    std::vector<Param*> params() override {
        std::vector<Param*> all_params;
        for (auto& module : modules_) {
            auto module_params = module->params();
            all_params.insert(all_params.end(), module_params.begin(), module_params.end());
        }
        return all_params;
    }

    /**
     * Get all child modules
     */
    std::vector<Module*> children() override {
        std::vector<Module*> child_ptrs;
        child_ptrs.reserve(modules_.size());
        for (auto& module : modules_) {
            child_ptrs.push_back(module.get());
        }
        return child_ptrs;
    }

    std::string name() const override {
        return "Sequential(" + std::to_string(modules_.size()) + ")";
    }

    /**
     * Get number of modules
     */
    size_t size() const {
        return modules_.size();
    }

    /**
     * Get module at index
     */
    Module* operator[](size_t idx) {
        if (idx >= modules_.size()) {
            throw std::out_of_range("Sequential index out of range");
        }
        return modules_[idx].get();
    }

    const Module* operator[](size_t idx) const {
        if (idx >= modules_.size()) {
            throw std::out_of_range("Sequential index out of range");
        }
        return modules_[idx].get();
    }

private:
    std::vector<std::unique_ptr<Module>> modules_;
};

// ============================================================================
// TAPE-AWARE FORWARD/BACKWARD UTILITIES
// ============================================================================

/**
 * Helper class for managing tape-aware computations
 */
class TapeManager {
public:
    TapeManager() = default;

    /**
     * Run a module with automatic tape management
     */
    template<typename ModuleType>
    Eigen::MatrixXf forward_with_tape(
        ModuleType& module,
        const Eigen::MatrixXf& input,
        Tape& tape) {

        Context ctx(&module);
        Eigen::MatrixXf output = module.forward(input, ctx);
        tape.push(std::move(ctx));
        return output;
    }

    /**
     * Run sequential modules with tape management
     */
    Eigen::MatrixXf forward_sequence(
        const std::vector<Module*>& modules,
        const Eigen::MatrixXf& input,
        Tape& tape) {

        Eigen::MatrixXf current_input = input;

        for (Module* module : modules) {
            current_input = forward_with_tape(*module, current_input, tape);
        }

        return current_input;
    }

    /**
     * Clear tape and prepare for new computation
     */
    void reset_tape(Tape& tape) {
        tape.clear();
    }
};

} // namespace grud

#endif // GRUD_CORE_TAPE_H