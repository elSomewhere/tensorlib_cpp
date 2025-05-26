// file: tensor.h
#pragma once

#include <eigen/Dense>
#include <vector>
#include <deque>
#include <memory>
#include <cmath>
#include <random>
#include <iostream>
#include <unordered_map>
#include <functional>
#include <type_traits>
#include <string>
#include <stdexcept>
#include <algorithm> // For std::max, std::transform
#include <numeric>   // For std::iota
#include <map>       // For AdamOptimizer states, GRUDCell key storage (though less used now)

// Using Eigen namespace locally
using Eigen::MatrixXf;
using Eigen::Vector2i;
using Eigen::ArrayXf;
using Eigen::ArrayXXf;

// --- Forward Declarations ---
class Tensor;
template<typename Derived> class TensorBase;
template<typename LHS, typename RHS, typename Op> class BinaryOp;
struct GradientCache;
class Layer;
class GRUDCell;


// --- TensorBase Definition ---
template<typename Derived>
class TensorBase {
protected:
    MatrixXf data_;

public:
    TensorBase() : data_(0,0) {}
    TensorBase(MatrixXf data) : data_(std::move(data)) {}
    TensorBase(int rows, int cols) : data_(rows, cols) {}

    const MatrixXf& data() const { return data_; }
    MatrixXf& data() { return data_; }
    int rows() const { return data_.rows(); }
    int cols() const { return data_.cols(); }
    Vector2i shape() const { return Vector2i(rows(), cols()); }
    bool empty() const { return data_.size() == 0; }


    auto operator+(const TensorBase& other) const;
    auto operator-(const TensorBase& other) const;
    auto operator*(const TensorBase& other) const;

    Derived operator-() const { return Derived(-data_); }

    Derived& operator+=(const TensorBase& other);
    Derived& operator*=(const TensorBase& other);
    Derived& operator-=(const TensorBase& other);

    Tensor matmul(const TensorBase& other) const;
};


// --- Tensor Class Definition (Shell) ---
class Tensor : public TensorBase<Tensor> {
public:
    using TensorBase<Tensor>::TensorBase;
    Tensor(float scalar);
    template<typename LHS, typename RHS, typename Op>
    Tensor(const BinaryOp<LHS, RHS, Op>& op_expr);
    template<typename LHS, typename RHS, typename Op>
    Tensor& operator=(const BinaryOp<LHS, RHS, Op>& op_expr);

    static Tensor zeros(int rows, int cols);
    static Tensor ones(int rows, int cols);
    static Tensor randn(int rows, int cols, std::mt19937& gen);
    static Tensor glorot_uniform(int in_dim, int out_dim, std::mt19937& gen);

    Tensor broadcast_to(int target_rows, int target_cols) const;
    Tensor transpose() const;
    Tensor sum(int axis = -1) const;
    Tensor mean(int axis = -1) const;
    Tensor sqrt() const;
    Tensor pow(float exponent) const;
    Tensor exp() const;
    Tensor log() const;
    Tensor abs() const;
    Tensor cwiseMax(float val) const;
    Tensor cwiseMin(float val) const;
};


// --- BinaryOp Class Definition (Shell) ---
template<typename LHS, typename RHS, typename Op>
class BinaryOp {
    typename std::decay<LHS>::type lhs_;
    typename std::decay<RHS>::type rhs_;
    Op op_;

public:
    BinaryOp(const LHS& lhs, const RHS& rhs, Op op)
            : lhs_(lhs), rhs_(rhs), op_(op) {}
    Tensor eval() const;
    Vector2i shape() const {
        auto lhs_shape = lhs_.shape();
        auto rhs_shape = rhs_.shape();
        return Vector2i(std::max(lhs_shape[0], rhs_shape[0]),
                        std::max(lhs_shape[1], rhs_shape[1]));
    }
};


// --- TensorBase operator definitions (returning BinaryOp) ---
template<typename Derived>
auto TensorBase<Derived>::operator+(const TensorBase& other) const {
    return BinaryOp<Derived, Derived, std::plus<float>>(
            static_cast<const Derived&>(*this), static_cast<const Derived&>(other), std::plus<float>{}
    );
}

template<typename Derived>
auto TensorBase<Derived>::operator-(const TensorBase& other) const {
    return BinaryOp<Derived, Derived, std::minus<float>>(
            static_cast<const Derived&>(*this), static_cast<const Derived&>(other), std::minus<float>{}
    );
}

template<typename Derived>
auto TensorBase<Derived>::operator*(const TensorBase& other) const {
    return BinaryOp<Derived, Derived, std::multiplies<float>>(
            static_cast<const Derived&>(*this), static_cast<const Derived&>(other), std::multiplies<float>{}
    );
}

template<typename Derived>
Tensor TensorBase<Derived>::matmul(const TensorBase& other_base) const {
    const Derived& other_derived = static_cast<const Derived&>(other_base);

    if (this->cols() != other_derived.rows()) {
        std::cerr << "ERROR matmul: LHS shape (" << this->rows() << "," << this->cols()
                  << "), RHS shape (" << other_derived.rows() << "," << other_derived.cols() << ")" << std::endl;
        throw std::invalid_argument("Matrix dimensions mismatch for matmul");
    }
    return Tensor(this->data_ * other_derived.data());
}


// --- BinaryOp::eval() definition ---
template<typename LHS_Type, typename RHS_Type, typename Op_Type>
Tensor BinaryOp<LHS_Type, RHS_Type, Op_Type>::eval() const {
    const MatrixXf& lhs_data = lhs_.data();
    const MatrixXf& rhs_data = rhs_.data();
    Vector2i lhs_s = lhs_.shape();
    Vector2i rhs_s = rhs_.shape();

    int out_rows = std::max(lhs_s[0], rhs_s[0]);
    int out_cols = std::max(lhs_s[1], rhs_s[1]);

    auto lhs_broadcast_data = (lhs_s[0] == out_rows && lhs_s[1] == out_cols)
                              ? lhs_data : Tensor(lhs_data).broadcast_to(out_rows, out_cols).data();
    auto rhs_broadcast_data = (rhs_s[0] == out_rows && rhs_s[1] == out_cols)
                              ? rhs_data : Tensor(rhs_data).broadcast_to(out_rows, out_cols).data();

    MatrixXf result_matrix_data(out_rows, out_cols);
    if constexpr (std::is_same_v<Op_Type, std::plus<float>>) {
        result_matrix_data = lhs_broadcast_data + rhs_broadcast_data;
    } else if constexpr (std::is_same_v<Op_Type, std::minus<float>>) {
        result_matrix_data = lhs_broadcast_data - rhs_broadcast_data;
    } else if constexpr (std::is_same_v<Op_Type, std::multiplies<float>>) {
        result_matrix_data = lhs_broadcast_data.array() * rhs_broadcast_data.array();
    } else if constexpr (std::is_same_v<Op_Type, std::divides<float>>) { // Added for Tensor / Tensor
        result_matrix_data = lhs_broadcast_data.array() / rhs_broadcast_data.array();
    }
    else {
        // This generic path might not be needed if we explicitly handle common ops
        result_matrix_data = lhs_broadcast_data.binaryExpr(rhs_broadcast_data,
                                                           [this](float a, float b) { return op_(a, b); });
    }
    return Tensor(std::move(result_matrix_data));
}

// --- Tensor specific constructor definitions ---
inline Tensor::Tensor(float scalar) : TensorBase(MatrixXf::Constant(1, 1, scalar)) {}
template<typename LHS, typename RHS, typename Op>
Tensor::Tensor(const BinaryOp<LHS, RHS, Op>& op_expr) : TensorBase(op_expr.eval().data_){}

template<typename LHS, typename RHS, typename Op>
Tensor& Tensor::operator=(const BinaryOp<LHS, RHS, Op>& op_expr) {
    data_ = op_expr.eval().data();
    return *this;
}

// --- TensorBase in-place operator definitions ---
template<typename Derived>
Derived& TensorBase<Derived>::operator+=(const TensorBase& other) {
    Tensor result = Tensor(static_cast<const Derived&>(*this) + static_cast<const Derived&>(other));
    data_ = std::move(result.data());
    return static_cast<Derived&>(*this);
}
template<typename Derived>
Derived& TensorBase<Derived>::operator*=(const TensorBase& other) {
    Tensor result = Tensor(static_cast<const Derived&>(*this) * static_cast<const Derived&>(other));
    data_ = std::move(result.data());
    return static_cast<Derived&>(*this);
}
template<typename Derived>
Derived& TensorBase<Derived>::operator-=(const TensorBase& other) {
    Tensor result = Tensor(static_cast<const Derived&>(*this) - static_cast<const Derived&>(other));
    data_ = std::move(result.data());
    return static_cast<Derived&>(*this);
}

// --- Tensor static factory method definitions ---
inline Tensor Tensor::zeros(int rows, int cols) { return Tensor(MatrixXf::Zero(rows, cols)); }
inline Tensor Tensor::ones(int rows, int cols) { return Tensor(MatrixXf::Ones(rows, cols)); }
inline Tensor Tensor::randn(int rows, int cols, std::mt19937& gen) {
    MatrixXf d(rows, cols); std::normal_distribution<float> dist(0.f,1.f);
    for (int i=0; i<d.size(); ++i) d.data()[i] = dist(gen); return Tensor(std::move(d));
}
inline Tensor Tensor::glorot_uniform(int in, int out, std::mt19937& gen) {
    float l=std::sqrt(6.f/(in+out)); MatrixXf d(out,in); std::uniform_real_distribution<float> dist(-l,l);
    for (int i=0; i<d.size(); ++i) d.data()[i] = dist(gen); return Tensor(std::move(d));
}

// --- Tensor method definitions ---
inline Tensor Tensor::broadcast_to(int tr, int tc) const {
    if (data_.rows()==tr && data_.cols()==tc) return *this; MatrixXf r(tr,tc);
    if (data_.rows()==1 && data_.cols()==tc) r=data_.replicate(tr,1);
    else if (data_.cols()==1 && data_.rows()==tr) r=data_.replicate(1,tc);
    else if (data_.rows()==1 && data_.cols()==1) r.setConstant(data_(0,0));
    else if (data_.rows()==tr && data_.cols()==1) r=data_.replicate(1,tc); // Added this case
    else throw std::invalid_argument("Invalid broadcast_to dimensions for tensor (" + std::to_string(data_.rows()) + "," + std::to_string(data_.cols()) + ") to (" + std::to_string(tr) + "," + std::to_string(tc) + ")");
    return Tensor(std::move(r));
}
inline Tensor Tensor::transpose() const { return Tensor(data_.transpose()); }
inline Tensor Tensor::sum(int ax) const {
    if (ax==-1) return Tensor(MatrixXf::Constant(1,1,data_.sum())); if (ax==0) return Tensor(data_.colwise().sum());
    if (ax==1) return Tensor(data_.rowwise().sum()); throw std::invalid_argument("Sum: invalid axis");
}
inline Tensor Tensor::mean(int ax) const {
    if (ax==-1) return Tensor(MatrixXf::Constant(1,1,data_.mean())); if (ax==0) return Tensor(data_.colwise().mean());
    if (ax==1) return Tensor(data_.rowwise().mean()); throw std::invalid_argument("Mean: invalid axis");
}
inline Tensor Tensor::sqrt() const { return Tensor(data_.array().sqrt().matrix());}
inline Tensor Tensor::pow(float e) const { return Tensor(data_.array().pow(e).matrix());}
inline Tensor Tensor::exp() const { return Tensor(data_.array().exp().matrix());}
inline Tensor Tensor::log() const { return Tensor(data_.array().log().matrix());}
inline Tensor Tensor::abs() const { return Tensor(data_.array().abs().matrix());}
inline Tensor Tensor::cwiseMax(float v) const { return Tensor(data_.array().cwiseMax(v).matrix());}
inline Tensor Tensor::cwiseMin(float v) const { return Tensor(data_.array().cwiseMin(v).matrix());}


// --- Free function operators ---
inline Tensor operator*(const Tensor& t, float s) { return Tensor(t.data() * s); }
inline Tensor operator*(float s, const Tensor& t) { return Tensor(s * t.data()); }
inline Tensor operator+(const Tensor& t, float s) { return Tensor(MatrixXf(t.data().array() + s)); }
inline Tensor operator+(float s, const Tensor& t) { return Tensor(MatrixXf(s + t.data().array())); }
inline Tensor operator-(const Tensor& t, float s) { return Tensor(MatrixXf(t.data().array() - s)); }
inline Tensor operator-(float s, const Tensor& t) { return Tensor(MatrixXf(s - t.data().array())); }
inline Tensor operator/(const Tensor& t, float s) { return Tensor(t.data() / s); }

// Tensor op BinaryOp (evaluates BinaryOp then applies op)
template<typename R_LHS, typename R_RHS, typename R_Op>
Tensor operator+(const Tensor& lhs, const BinaryOp<R_LHS,R_RHS,R_Op>& rhs_op) { return lhs + rhs_op.eval(); }
template<typename L_LHS, typename L_RHS, typename L_Op>
Tensor operator+(const BinaryOp<L_LHS,L_RHS,L_Op>& lhs_op, const Tensor& rhs) { return lhs_op.eval() + rhs; }
template<typename L_LHS, typename L_RHS, typename L_Op, typename R_LHS, typename R_RHS, typename R_Op>
Tensor operator+(const BinaryOp<L_LHS,L_RHS,L_Op>& lhs_op, const BinaryOp<R_LHS,R_RHS,R_Op>& rhs_op) { return lhs_op.eval() + rhs_op.eval(); }

template<typename R_LHS, typename R_RHS, typename R_Op>
Tensor operator-(const Tensor& lhs, const BinaryOp<R_LHS,R_RHS,R_Op>& rhs_op) { return lhs - rhs_op.eval(); }
template<typename L_LHS, typename L_RHS, typename L_Op>
Tensor operator-(const BinaryOp<L_LHS,L_RHS,L_Op>& lhs_op, const Tensor& rhs) { return lhs_op.eval() - rhs; }
template<typename L_LHS, typename L_RHS, typename L_Op, typename R_LHS, typename R_RHS, typename R_Op>
Tensor operator-(const BinaryOp<L_LHS,L_RHS,L_Op>& lhs_op, const BinaryOp<R_LHS,R_RHS,R_Op>& rhs_op) { return lhs_op.eval() - rhs_op.eval(); }

template<typename R_LHS, typename R_RHS, typename R_Op>
Tensor operator*(const Tensor& lhs, const BinaryOp<R_LHS,R_RHS,R_Op>& rhs_op) { return lhs * rhs_op.eval(); }
template<typename L_LHS, typename L_RHS, typename L_Op>
Tensor operator*(const BinaryOp<L_LHS,L_RHS,L_Op>& lhs_op, const Tensor& rhs) { return lhs_op.eval() * rhs; }
template<typename L_LHS, typename L_RHS, typename L_Op, typename R_LHS, typename R_RHS, typename R_Op>
Tensor operator*(const BinaryOp<L_LHS,L_RHS,L_Op>& lhs_op, const BinaryOp<R_LHS,R_RHS,R_Op>& rhs_op) { return lhs_op.eval() * rhs_op.eval(); }


// Element-wise division of Tensors (returns a BinaryOp for lazy evaluation)
inline auto operator/(const Tensor& lhs, const Tensor& rhs) {
    return BinaryOp<Tensor, Tensor, std::divides<float>>(lhs, rhs, std::divides<float>{});
}


// ============================================================================
// GRADIENT CACHE
// ============================================================================
struct GradientCache {
    std::unordered_map<std::string, Tensor> cached_tensors;
    std::unordered_map<std::string, Tensor> propagated_gradients;
    std::vector<std::function<void()>> backward_fns;

    void cache_tensor(const std::string& key, Tensor tensor) {
        cached_tensors[key] = std::move(tensor);
    }
    const Tensor& get_cached_tensor(const std::string& key) const {
        auto it = cached_tensors.find(key);
        if (it == cached_tensors.end()) {
            throw std::runtime_error("Cached tensor not found: " + key);
        }
        return it->second;
    }
    bool has_cached_tensor(const std::string& key) const {
        return cached_tensors.count(key);
    }

    void set_gradient(const std::string& tensor_name, Tensor grad) {
        if (grad.empty()) {
            std::cerr << "Warning: Setting empty gradient for " << tensor_name << std::endl;
        }
        propagated_gradients[tensor_name] = std::move(grad);
    }

    void accumulate_gradient(const std::string& tensor_name, const Tensor& grad_contribution) {
        if (grad_contribution.empty()) {
            // std::cerr << "Warning: Accumulating empty gradient for " << tensor_name << std::endl;
            if(!has_gradient(tensor_name)){ // If it's the first time and it's empty, we might need to initialize it.
                // This case is tricky. For now, let's assume non-empty contributions if key DNE.
                //propagated_gradients[tensor_name] = Tensor::zeros(grad_contribution.rows(), grad_contribution.cols()); // Needs shape info
            }
            return; // Don't add empty tensor
        }
        auto it = propagated_gradients.find(tensor_name);
        if (it == propagated_gradients.end()) {
            propagated_gradients[tensor_name] = grad_contribution;
        } else {
            if (it->second.shape() != grad_contribution.shape()) {
                throw std::runtime_error("Gradient shape mismatch for accumulation on " + tensor_name +
                                         ": existing (" + std::to_string(it->second.rows()) + "," + std::to_string(it->second.cols()) +
                                         "), new (" + std::to_string(grad_contribution.rows()) + "," + std::to_string(grad_contribution.cols()) + ")");
            }
            it->second += grad_contribution;
        }
    }

    Tensor get_gradient(const std::string& tensor_name) const {
        auto it = propagated_gradients.find(tensor_name);
        if (it == propagated_gradients.end()) {
            // Consider returning an empty tensor or a special "NoGradient" object
            // For now, strict: throw. The user of get_gradient should check with has_gradient or know it exists.
            throw std::runtime_error("Gradient not found for tensor: " + tensor_name);
        }
        return it->second;
    }
    // Overload to get gradient or a zero tensor if not found (requires shape)
    Tensor get_gradient_or_zeros(const std::string& tensor_name, int rows, int cols) const {
        auto it = propagated_gradients.find(tensor_name);
        if (it == propagated_gradients.end()) {
            return Tensor::zeros(rows, cols);
        }
        return it->second;
    }


    bool has_gradient(const std::string& tensor_name) const {
        return propagated_gradients.count(tensor_name);
    }

    void add_backward_fn(std::function<void()> fn) {
        backward_fns.push_back(std::move(fn));
    }
    void clear_gradients() {
        propagated_gradients.clear();
    }
    void clear_cached_tensors() {
        cached_tensors.clear();
    }
    void clear_fns() {
        backward_fns.clear();
    }
    void clear_all_except_parameters() { // More descriptive name
        clear_cached_tensors();
        clear_gradients();
        clear_fns();
    }
    void backward() {
        for (auto it = backward_fns.rbegin(); it != backward_fns.rend(); ++it) {
            (*it)();
        }
    }
};

// ============================================================================
// LAYERS
// ============================================================================
class Layer {
protected:
    std::unordered_map<std::string, Tensor> parameters_;
    std::unordered_map<std::string, Tensor> gradients_;
    bool training_mode_;
    std::string name_; // Unique name for this layer instance
public:
    Layer(std::string name) : training_mode_(true), name_(std::move(name)) {}
    virtual ~Layer() = default;

    // output_tensor_name: A unique name the caller (MLP, GRUDCell) assigns to this layer's output.
    //                     This layer will cache its own intermediate values using keys derived from its 'name_'.
    virtual Tensor forward(const Tensor& input, GradientCache& cache, const std::string& output_tensor_name) = 0;

    // dL_dOutput: Gradient w.r.t. this layer's output.
    // cache: GradientCache to retrieve saved tensors.
    // This method calculates parameter gradients (updates this->gradients_) and returns dL_dInput.
    virtual Tensor backward_calc_and_store_grads(const Tensor& dL_dOutput, GradientCache& cache) = 0;

    void set_training_mode(bool training) { training_mode_ = training; }
    bool is_training() const { return training_mode_; }
    const std::unordered_map<std::string, Tensor>& parameters() const { return parameters_; }
    std::unordered_map<std::string, Tensor>& get_parameters_mut() { return parameters_; } // For optimizer
    const std::unordered_map<std::string, Tensor>& gradients() const { return gradients_; }
    void zero_grad() { for (auto& [k,v] : gradients_) v.data().setZero(); }
    const std::string& name() const { return name_; }
protected:
    void register_parameter(const std::string& pname, Tensor p) {
        parameters_[pname] = std::move(p);
        gradients_[pname] = Tensor::zeros(parameters_[pname].rows(), parameters_[pname].cols());
    }
    // Helper for namespacing cache entries for this layer instance
    std::string get_cache_key(const std::string& suffix) const { return name_ + "_" + suffix; }
};

class TanhLayer : public Layer {
public:
    TanhLayer(const std::string& name) : Layer(name) {}

    Tensor forward(const Tensor& input, GradientCache& cache, const std::string& output_tensor_name) override {
        Tensor output = Tensor(input.data().array().tanh().matrix());
        // Cache the output of this layer (which is the input to Tanh's derivative calculation)
        cache.cache_tensor(get_cache_key("output_for_grad"), output);
        return output;
    }

    Tensor backward_calc_and_store_grads(const Tensor& dL_dOutput, GradientCache& cache) override {
        Tensor original_output = cache.get_cached_tensor(get_cache_key("output_for_grad"));
        // derivative of tanh(x) is 1 - tanh(x)^2 = 1 - output^2
        Tensor output_squared = Tensor(original_output * original_output); // Element-wise
        Tensor tanhx_grad_val = Tensor(1.0f) - output_squared;
        Tensor dL_dInput = dL_dOutput * tanhx_grad_val; // Element-wise
        return dL_dInput;
    }
};

class SigmoidLayer : public Layer {
public:
    SigmoidLayer(const std::string& name) : Layer(name) {}
    Tensor forward(const Tensor& input, GradientCache& cache, const std::string& output_tensor_name) override {
        // Clip input to avoid extreme values in exp causing NaN/inf
        MatrixXf clipped_input_data = input.data().array().cwiseMax(-80.0f).cwiseMin(80.0f);
        Tensor output = Tensor((1.0f / (1.0f + (-clipped_input_data.array()).exp())).matrix());
        cache.cache_tensor(get_cache_key("output_for_grad"), output);
        return output;
    }
    Tensor backward_calc_and_store_grads(const Tensor& dL_dOutput, GradientCache& cache) override {
        Tensor original_output = cache.get_cached_tensor(get_cache_key("output_for_grad"));
        // derivative of sigmoid(x) is sigmoid(x) * (1 - sigmoid(x)) = output * (1 - output)
        Tensor term_in_paren = Tensor(1.0f) - original_output;
        Tensor sigx_grad_val = original_output * term_in_paren; // Element-wise
        Tensor dL_dInput = dL_dOutput * sigx_grad_val; // Element-wise
        return dL_dInput;
    }
};

class SoftplusLayer : public Layer {
public:
    SoftplusLayer(const std::string& name) : Layer(name) {}
    Tensor forward(const Tensor& input, GradientCache& cache, const std::string& output_tensor_name) override {
        // Clip input to avoid issues with exp
        MatrixXf x_clipped = input.data().array().cwiseMax(-20.0f).cwiseMin(20.0f).matrix();
        Tensor output = Tensor(MatrixXf((1.0f + x_clipped.array().exp()).log()));
        // Softplus derivative is sigmoid(input). So we need original input for backward.
        cache.cache_tensor(get_cache_key("input_for_grad"), input);
        return output;
    }
    Tensor backward_calc_and_store_grads(const Tensor& dL_dOutput, GradientCache& cache) override {
        Tensor original_input = cache.get_cached_tensor(get_cache_key("input_for_grad"));
        // derivative of softplus(x) is sigmoid(x) = 1 / (1 + exp(-x))
        MatrixXf clipped_orig_input_data = original_input.data().array().cwiseMax(-80.0f).cwiseMin(80.0f);
        Tensor softplus_grad_val = Tensor((1.0f / (1.0f + (-clipped_orig_input_data.array()).exp())).matrix());
        Tensor dL_dInput = dL_dOutput * softplus_grad_val; // Element-wise
        return dL_dInput;
    }
};

class Linear : public Layer {
private:
    int in_features_, out_features_;
    bool use_bias_;
public:
    Linear(int in_f, int out_f, bool ub=true, std::mt19937* gen_nullable=nullptr, const std::string& n="linear")
            : Layer(n), in_features_(in_f), out_features_(out_f), use_bias_(ub) {
        std::mt19937* g = gen_nullable;
        std::mt19937 gen_storage; // Fallback if gen_nullable is null
        if (!g) {
            gen_storage.seed(std::random_device{}());
            g = &gen_storage;
        }
        register_parameter("weight", Tensor::glorot_uniform(in_features_, out_features_, *g));
        if(use_bias_) register_parameter("bias", Tensor::zeros(1, out_features_));
    }

    Tensor forward(const Tensor& in, GradientCache& cache, const std::string& output_tensor_name) override {
        cache.cache_tensor(get_cache_key("in_for_grad"), in); // Save input for backward

        Tensor W = parameters_.at("weight"); // (out_features, in_features)
        // Input 'in' is (batch_size, in_features)
        // in.transpose() is (in_features, batch_size)
        // W.matmul(in.transpose()) is (out_features, batch_size)
        Tensor out_intermediate = W.matmul(in.transpose());
        // out_intermediate.transpose() is (batch_size, out_features)
        Tensor out = out_intermediate.transpose();

        if(use_bias_){
            Tensor b = parameters_.at("bias"); // (1, out_features)
            out = out + b.broadcast_to(out.rows(), out.cols()); // Broadcasting (1,H) to (B,H)
        }
        return out;
    }

    Tensor backward_calc_and_store_grads(const Tensor& dL_dOutput, GradientCache& cache) override {
        // dL_dOutput is (batch_size, out_features)
        Tensor orig_in = cache.get_cached_tensor(get_cache_key("in_for_grad")); // (batch_size, in_features)
        Tensor W_p = parameters_.at("weight"); // (out_features, in_features)

        if(use_bias_){
            // dL_dOutput.sum(0) sums over batch: (1, out_features)
            gradients_.at("bias") += dL_dOutput.sum(0);
        }

        // dL_dW = dL_dOutput.T @ orig_in
        // dL_dOutput.transpose() is (out_features, batch_size)
        // orig_in is (batch_size, in_features)
        // Result is (out_features, in_features), matches W_p shape.
        gradients_.at("weight") += dL_dOutput.transpose().matmul(orig_in);

        // dL_dInput = dL_dOutput @ W_p
        // dL_dOutput is (batch_size, out_features)
        // W_p is (out_features, in_features)
        // Result is (batch_size, in_features), matches orig_in shape.
        Tensor dL_dInput = dL_dOutput.matmul(W_p);
        return dL_dInput;
    }
};

class LayerNorm : public Layer {
private:
    float eps_; bool affine_; int features_;
public:
    LayerNorm(int features, float eps=1e-5f, bool affine=true, const std::string& name="ln")
            : Layer(name), eps_(eps), affine_(affine), features_(features){
        if(affine_){
            register_parameter("weight", Tensor::ones(1, features_)); // Gamma
            register_parameter("bias", Tensor::zeros(1, features_));   // Beta
        }
    }

    Tensor forward(const Tensor& in, GradientCache& cache, const std::string& output_tensor_name) override {
        // in is (batch_size, features_)
        cache.cache_tensor(get_cache_key("in_for_grad"), in);

        // Mean over features (axis 1)
        Tensor mean_val = in.mean(1); // (batch_size, 1)
        Tensor centered_in = in - mean_val.broadcast_to(in.rows(), in.cols()); // (B,F) - (B,F broadcasted from B,1)

        // Variance over features (axis 1)
        Tensor var_val = Tensor(centered_in * centered_in).mean(1); // (B,1)
        Tensor std_val = (var_val + eps_).sqrt(); // (B,1)

        Tensor inv_std_val = Tensor::ones(std_val.rows(), std_val.cols()) / std_val; // (B,1), element-wise 1/std

        Tensor normalized_in = centered_in * inv_std_val.broadcast_to(in.rows(), in.cols()); // (B,F) * (B,F broadcasted from B,1)

        cache.cache_tensor(get_cache_key("centered_in"), centered_in);
        cache.cache_tensor(get_cache_key("inv_std_val"), inv_std_val); // inv_std_val is (B,1)
        cache.cache_tensor(get_cache_key("normalized_in"), normalized_in);

        Tensor out = normalized_in;
        if(affine_){
            Tensor w = parameters_.at("weight"); // (1, F)
            Tensor b = parameters_.at("bias");   // (1, F)
            cache.cache_tensor(get_cache_key("affine_w_for_grad"), w); // Save W if affine for backward

            out = (out * w.broadcast_to(out.rows(), out.cols())) + b.broadcast_to(out.rows(), out.cols());
        }
        return out;
    }

    Tensor backward_calc_and_store_grads(const Tensor& dL_dOutput, GradientCache& cache) override {
        // dL_dOutput is (B, F)
        Tensor orig_in = cache.get_cached_tensor(get_cache_key("in_for_grad"));       // (B, F)
        Tensor centered_in = cache.get_cached_tensor(get_cache_key("centered_in")); // (B, F)
        Tensor inv_std_val_B1 = cache.get_cached_tensor(get_cache_key("inv_std_val")); // (B, 1)
        Tensor normalized_in = cache.get_cached_tensor(get_cache_key("normalized_in")); // (B, F)

        Tensor dL_dNormalized = dL_dOutput;

        if(affine_){
            Tensor w_1F = cache.get_cached_tensor(get_cache_key("affine_w_for_grad")); // (1, F)
            // dL_dW = sum over batch of (dL_dOutput * normalized_in)
            gradients_.at("weight") += Tensor(dL_dOutput * normalized_in).sum(0); // (B,F)*(B,F)->(B,F) then sum(0)->(1,F)
            // dL_dB = sum over batch of dL_dOutput
            gradients_.at("bias") += dL_dOutput.sum(0); // (B,F) then sum(0)->(1,F)

            // Propagate gradient before affine transformation
            dL_dNormalized = dL_dOutput * w_1F.broadcast_to(dL_dOutput.rows(), dL_dOutput.cols()); // (B,F)*(B,F broadcasted from 1,F)
        }

        // Broadcast inv_std_val_B1 to (B,F) for element-wise operations
        Tensor inv_std_val_BF = inv_std_val_B1.broadcast_to(orig_in.rows(), orig_in.cols());

        // Intermediate terms for dL_dInput, following standard LayerNorm backward derivation
        // dL/dvar = sum_{i=1 to F} [ dL/dx_norm_i * (-1/2) * (x_centered_i) * (var + eps)^(-3/2) ]
        Tensor dL_dVar_term = dL_dNormalized * centered_in * inv_std_val_BF.pow(3.0f) * Tensor(-0.5f);
        Tensor dL_dVar_B1 = dL_dVar_term.sum(1); // Sum over features F -> (B,1)

        // dL/dmean = sum_{i=1 to F} [ dL/dx_norm_i * (-1/std) ] + dL/dvar * sum_{i=1 to F} [ (-2/F) * x_centered_i ]
        // Note: sum_{i=1 to F} [x_centered_i] = 0. So, second term for dL/dmean from variance path is:
        // dL/dvar * (-2/F) * sum(x_centered_i) where sum is over features. This term is 0.
        // Simplified: dLdmean = sum(dL/dx_norm_i * (-1/std_i)) over features
        Tensor dL_dMean_term1_BF = dL_dNormalized * (inv_std_val_BF * Tensor(-1.0f));
        Tensor dL_dMean_B1 = dL_dMean_term1_BF.sum(1); // Sum over features F -> (B,1)
        // dL_dMean also gets a contribution from dL_dVar because var depends on mean.
        // d(var)/d(mean) = (-2/F) * sum(x_i - mean) = (-2/F) * sum(x_centered_i) = 0
        // So dL_dMean doesn't get a direct contribution from dL_dVar using this formulation.

        // dL/dx_i = (dL/dx_norm_i * (1/std)) + (dL/dvar * (2/F)*(x_i-mean)) + (dL/dmean * (1/F))
        Tensor dL_dInput = (dL_dNormalized * inv_std_val_BF) +
                           (dL_dVar_B1.broadcast_to(orig_in.rows(), orig_in.cols()) * Tensor(2.0f / features_) * centered_in) +
                           (dL_dMean_B1.broadcast_to(orig_in.rows(), orig_in.cols()) * Tensor(1.0f / features_));
        return dL_dInput;
    }
};


class Dropout : public Layer {
private:
    float p_dropout_; // Probability of dropping out an element
    mutable std::mt19937 rng_;
    mutable std::uniform_real_distribution<float> dist_;
public:
    Dropout(float p=0.5f, const std::string& n="dropout")
            : Layer(n), p_dropout_(p), dist_(0.f,1.f) {
        if(p_dropout_<0.f || p_dropout_>1.f) throw std::invalid_argument("Dropout p must be in [0,1]");
        rng_.seed(std::random_device{}()); // Default seed
    }

    void set_seed(unsigned int seed){ rng_.seed(seed); }

    Tensor forward(const Tensor& in, GradientCache& cache, const std::string& output_tensor_name) override {
        if(!training_mode_ || p_dropout_ == 0.f){
            // During inference or if p=0, dropout is identity. No mask needed for backward.
            // However, to make backward_calc_and_store_grads consistent, we can cache an all-ones mask.
            cache.cache_tensor(get_cache_key("mask"), Tensor::ones(in.rows(), in.cols()));
            return in;
        }

        float scale = 1.0f / (1.0f - p_dropout_); // Inverted dropout scale factor
        MatrixXf mask_data(in.rows(), in.cols());
        for(int r=0; r < mask_data.rows(); ++r) {
            for(int c=0; c < mask_data.cols(); ++c) {
                mask_data(r,c) = (dist_(rng_) > p_dropout_) ? scale : 0.0f;
            }
        }
        Tensor mask_tensor(std::move(mask_data));
        cache.cache_tensor(get_cache_key("mask"), mask_tensor);

        return in * mask_tensor; // Element-wise product
    }

    Tensor backward_calc_and_store_grads(const Tensor& dL_dOutput, GradientCache& cache) override {
        Tensor mask = cache.get_cached_tensor(get_cache_key("mask"));
        // Parameters are not learned in dropout, so no gradient updates for parameters_.
        return dL_dOutput * mask; // Element-wise product
    }
};


struct TemporalConfig { // Simplified for now
    int batch_size=1, input_size=4, hidden_size=64, num_layers=2;
    bool use_exponential_decay=true;
    float softclip_threshold=3.f, min_log_gamma=-10.f; // For GRUD
    float dropout_rate=0.1f; bool use_layer_norm=true; // For TemporalPredictor
    float learning_rate=2e-3f;
    int tbptt_steps=20, seed=0;
};

namespace activations { // Stateless activation functions (can be used by GRUD directly)
    Tensor stateless_sigmoid(const Tensor& x) {
        MatrixXf clipped_input_data = x.data().array().cwiseMax(-80.0f).cwiseMin(80.0f);
        MatrixXf result = (1.0f / (1.0f + (-clipped_input_data.array()).exp())).matrix();
        return Tensor(std::move(result));
    }
    Tensor stateless_tanh(const Tensor& x) {
        return Tensor(x.data().array().tanh().matrix());
    }
    Tensor stateless_softplus(const Tensor& x) {
        MatrixXf x_clipped = x.data().array().cwiseMax(-20.0f).cwiseMin(20.0f).matrix();
        return Tensor(MatrixXf((1.0f + x_clipped.array().exp()).log()));
    }
    Tensor softclip(const Tensor& x, float threshold) {
        if (threshold <= 0) throw std::invalid_argument("Threshold must be positive for softclip");
        ArrayXXf x_arr = x.data().array();
        ArrayXXf abs_x = x_arr.abs();
        Eigen::Array<bool,Eigen::Dynamic,Eigen::Dynamic> condition = (abs_x <= threshold);
        // result = condition ? x : sign(x) * (threshold + log(abs(x) - threshold + 1))
        MatrixXf result_data = condition.select(x.data(), x_arr.sign() * (threshold + (abs_x - threshold + 1.0f).log())).matrix();
        return Tensor(result_data);
    }
    Tensor softclip_derivative_wrt_input(const Tensor& input_val, float threshold) {
        if (threshold <= 0) throw std::invalid_argument("Threshold must be positive for softclip derivative");
        ArrayXXf abs_x = input_val.data().array().abs();
        Eigen::Array<bool,Eigen::Dynamic,Eigen::Dynamic> condition = (abs_x <= threshold);
        // derivative is 1 if |x| <= threshold, else 1 / (1 + |x| - threshold)
        MatrixXf grad_data = condition.select(MatrixXf::Ones(input_val.rows(),input_val.cols()), (1.0f / (1.0f + abs_x - threshold)).matrix());
        return Tensor(grad_data);
    }
} // namespace activations


class GRUDCell : public Layer {
private:
    TemporalConfig config_;
    int current_cell_input_size_; // Actual input size for this cell (can be x_dim or h_dim)

    // Sub-layers
    std::unique_ptr<Linear> impute_linear_; // x_hat_t = Linear_impute(h_prev_t)
    std::unique_ptr<Linear> W_r_, U_r_, V_r_; // For reset gate r_t
    std::unique_ptr<Linear> W_z_, U_z_, V_z_; // For update gate z_t
    std::unique_ptr<Linear> W_h_, U_h_, V_h_; // For candidate state h_tilde_t

    // Activation layers (could be stateless if preferred, but stateful allows caching for derivative)
    std::unique_ptr<SigmoidLayer> r_gate_activation_;
    std::unique_ptr<SigmoidLayer> z_gate_activation_;
    std::unique_ptr<TanhLayer> h_candidate_activation_;
    std::unique_ptr<SoftplusLayer> decay_softplus_activation_; // For gamma_h parameter

    // To store names of intermediate tensors created during forward pass, for use in backward
    struct ForwardPassTensorNames {
        std::string x_hat_t, x_tilde_t;
        std::string gamma_h_t, log_gamma_raw_BH, log_gamma_BH, h_decay_t;
        std::string Wr_out, Ur_out, Vr_out, r_arg_t, r_t;
        std::string Wz_out, Uz_out, Vz_out, z_arg_t, z_t;
        std::string rh_prod_t;
        std::string Wh_out, Uh_out, Vh_out, h_cand_arg_t, h_cand_t;
        // If decay is used:
        std::string decay_h_param_val, clipped_decay_1H, softplus_decay_1H_out;
    };
    ForwardPassTensorNames tensor_names_; // One per GRUDCell instance for its step

public:
    GRUDCell(const TemporalConfig& c, int in_size, std::mt19937& gen, const std::string& name_prefix)
            : Layer(name_prefix), config_(c), current_cell_input_size_(in_size) {

        // Define unique names for sub-layers based on the GRUDCell's name
        impute_linear_ = std::make_unique<Linear>(c.hidden_size, current_cell_input_size_, true, &gen, name_ + "_impute");

        W_r_ = std::make_unique<Linear>(current_cell_input_size_, c.hidden_size, true, &gen, name_ + "_Wr");
        U_r_ = std::make_unique<Linear>(c.hidden_size, c.hidden_size, false, &gen, name_ + "_Ur"); // No bias for U matrices typically
        V_r_ = std::make_unique<Linear>(1, c.hidden_size, false, &gen, name_ + "_Vr");             // No bias for V matrices typically

        W_z_ = std::make_unique<Linear>(current_cell_input_size_, c.hidden_size, true, &gen, name_ + "_Wz");
        U_z_ = std::make_unique<Linear>(c.hidden_size, c.hidden_size, false, &gen, name_ + "_Uz");
        V_z_ = std::make_unique<Linear>(1, c.hidden_size, false, &gen, name_ + "_Vz");

        W_h_ = std::make_unique<Linear>(current_cell_input_size_, c.hidden_size, true, &gen, name_ + "_Wh");
        U_h_ = std::make_unique<Linear>(c.hidden_size, c.hidden_size, false, &gen, name_ + "_Uh");
        V_h_ = std::make_unique<Linear>(1, c.hidden_size, false, &gen, name_ + "_Vh");

        // Initialize update gate bias for better initial learning (common practice)
        W_z_->get_parameters_mut().at("bias").data().setConstant(-1.0f); // Encourages z_t to be small initially

        if(config_.use_exponential_decay){
            register_parameter("decay_h_param", Tensor::zeros(1, c.hidden_size)); // Learnable decay parameter per hidden unit
            decay_softplus_activation_ = std::make_unique<SoftplusLayer>(name_ + "_decay_sp_act");
        }

        r_gate_activation_ = std::make_unique<SigmoidLayer>(name_ + "_r_act");
        z_gate_activation_ = std::make_unique<SigmoidLayer>(name_ + "_z_act");
        h_candidate_activation_ = std::make_unique<TanhLayer>(name_ + "_h_cand_act");
    }

    struct GRUDInputData { Tensor x_t, h_prev_t, dt_t; const Tensor* mask_t; };
    Tensor forward_step(const GRUDInputData& grud_in, GradientCache& cache, const std::string& h_new_output_name_for_grad) {
        const auto& x_t = grud_in.x_t;
        const auto& h_prev_t = grud_in.h_prev_t;
        const auto& dt_t = grud_in.dt_t;
        const auto* mask_t = grud_in.mask_t;

        tensor_names_.x_hat_t = get_cache_key("x_hat_t");
        tensor_names_.x_tilde_t = get_cache_key("x_tilde_t");
        tensor_names_.gamma_h_t = get_cache_key("gamma_h_t");
        tensor_names_.log_gamma_raw_BH = get_cache_key("log_gamma_raw_BH");
        tensor_names_.log_gamma_BH = get_cache_key("log_gamma_BH");
        tensor_names_.h_decay_t = get_cache_key("h_decay_t");
        tensor_names_.Wr_out = get_cache_key("Wr_out");
        tensor_names_.Ur_out = get_cache_key("Ur_out");
        tensor_names_.Vr_out = get_cache_key("Vr_out");
        tensor_names_.r_arg_t = get_cache_key("r_arg_t");
        tensor_names_.r_t = get_cache_key("r_t");
        tensor_names_.Wz_out = get_cache_key("Wz_out");
        tensor_names_.Uz_out = get_cache_key("Uz_out");
        tensor_names_.Vz_out = get_cache_key("Vz_out");
        tensor_names_.z_arg_t = get_cache_key("z_arg_t");
        tensor_names_.z_t = get_cache_key("z_t");
        tensor_names_.rh_prod_t = get_cache_key("rh_prod_t");
        tensor_names_.Wh_out = get_cache_key("Wh_out");
        tensor_names_.Uh_out = get_cache_key("Uh_out");
        tensor_names_.Vh_out = get_cache_key("Vh_out");
        tensor_names_.h_cand_arg_t = get_cache_key("h_cand_arg_t");
        tensor_names_.h_cand_t = get_cache_key("h_cand_t");
        if (config_.use_exponential_decay) {
            tensor_names_.decay_h_param_val = get_cache_key("decay_h_param_val");
            tensor_names_.clipped_decay_1H = get_cache_key("clipped_decay_1H");
            tensor_names_.softplus_decay_1H_out = get_cache_key("softplus_decay_1H_out");
        }

        cache.cache_tensor(get_cache_key("x_t_in"), x_t);
        cache.cache_tensor(get_cache_key("h_prev_t_in"), h_prev_t);
        cache.cache_tensor(get_cache_key("dt_t_in"), dt_t);
        if (mask_t) cache.cache_tensor(get_cache_key("mask_t_in"), *mask_t);

        Tensor x_hat_t_val = impute_linear_->forward(h_prev_t, cache, tensor_names_.x_hat_t);
        cache.cache_tensor(tensor_names_.x_hat_t, x_hat_t_val);

        Tensor x_tilde_t_val = x_t;
        if (mask_t) {
            Tensor ones_mask = Tensor::ones(mask_t->rows(), mask_t->cols());
            x_tilde_t_val = ((*mask_t) * x_t) + (Tensor(ones_mask - (*mask_t)) * x_hat_t_val);
        }
        cache.cache_tensor(tensor_names_.x_tilde_t, x_tilde_t_val);

        Tensor h_decay_t_val = h_prev_t;
        Tensor gamma_h_t_tensor_val;
        if(config_.use_exponential_decay){
            Tensor decay_h_p = parameters_.at("decay_h_param");
            cache.cache_tensor(tensor_names_.decay_h_param_val, decay_h_p);
            Tensor clipped_decay_1H = activations::softclip(decay_h_p, config_.softclip_threshold);
            cache.cache_tensor(tensor_names_.clipped_decay_1H, clipped_decay_1H);
            Tensor softplus_decay_1H = decay_softplus_activation_->forward(clipped_decay_1H, cache, tensor_names_.softplus_decay_1H_out);
            cache.cache_tensor(tensor_names_.softplus_decay_1H_out, softplus_decay_1H);
            Tensor sp_times_neg_one = Tensor(softplus_decay_1H * Tensor(-1.0f));
            Tensor log_gamma_raw_BH_tensor_val = dt_t.matmul(sp_times_neg_one);
            cache.cache_tensor(tensor_names_.log_gamma_raw_BH, log_gamma_raw_BH_tensor_val);
            Tensor log_gamma_BH_tensor_val = log_gamma_raw_BH_tensor_val.cwiseMax(config_.min_log_gamma).cwiseMin(-1e-4f);
            cache.cache_tensor(tensor_names_.log_gamma_BH, log_gamma_BH_tensor_val);
            gamma_h_t_tensor_val = log_gamma_BH_tensor_val.exp();
            cache.cache_tensor(tensor_names_.gamma_h_t, gamma_h_t_tensor_val);
            h_decay_t_val = h_prev_t * gamma_h_t_tensor_val;
        }
        cache.cache_tensor(tensor_names_.h_decay_t, h_decay_t_val);

        Tensor Wr_out_tensor_val = W_r_->forward(x_tilde_t_val, cache, tensor_names_.Wr_out); cache.cache_tensor(tensor_names_.Wr_out, Wr_out_tensor_val);
        Tensor Ur_out_tensor_val = U_r_->forward(h_decay_t_val, cache, tensor_names_.Ur_out); cache.cache_tensor(tensor_names_.Ur_out, Ur_out_tensor_val);
        Tensor Vr_out_tensor_val = V_r_->forward(dt_t, cache, tensor_names_.Vr_out);       cache.cache_tensor(tensor_names_.Vr_out, Vr_out_tensor_val);
        Tensor r_arg_t_tensor_val = Wr_out_tensor_val + Ur_out_tensor_val + Vr_out_tensor_val;
        cache.cache_tensor(tensor_names_.r_arg_t, r_arg_t_tensor_val);
        Tensor r_t_tensor_val = r_gate_activation_->forward(r_arg_t_tensor_val, cache, tensor_names_.r_t);
        cache.cache_tensor(tensor_names_.r_t, r_t_tensor_val);

        Tensor Wz_out_tensor_val = W_z_->forward(x_tilde_t_val, cache, tensor_names_.Wz_out); cache.cache_tensor(tensor_names_.Wz_out, Wz_out_tensor_val);
        Tensor Uz_out_tensor_val = U_z_->forward(h_decay_t_val, cache, tensor_names_.Uz_out); cache.cache_tensor(tensor_names_.Uz_out, Uz_out_tensor_val);
        Tensor Vz_out_tensor_val = V_z_->forward(dt_t, cache, tensor_names_.Vz_out);       cache.cache_tensor(tensor_names_.Vz_out, Vz_out_tensor_val);
        Tensor z_arg_t_tensor_val = Wz_out_tensor_val + Uz_out_tensor_val + Vz_out_tensor_val;
        cache.cache_tensor(tensor_names_.z_arg_t, z_arg_t_tensor_val);
        Tensor z_t_tensor_val = z_gate_activation_->forward(z_arg_t_tensor_val, cache, tensor_names_.z_t);
        cache.cache_tensor(tensor_names_.z_t, z_t_tensor_val);

        Tensor rh_prod_t_tensor_val = r_t_tensor_val * h_decay_t_val;
        cache.cache_tensor(tensor_names_.rh_prod_t, rh_prod_t_tensor_val);

        Tensor Wh_out_tensor_val = W_h_->forward(x_tilde_t_val, cache, tensor_names_.Wh_out); cache.cache_tensor(tensor_names_.Wh_out, Wh_out_tensor_val);
        Tensor Uh_out_tensor_val = U_h_->forward(rh_prod_t_tensor_val, cache, tensor_names_.Uh_out); cache.cache_tensor(tensor_names_.Uh_out, Uh_out_tensor_val);
        Tensor Vh_out_tensor_val = V_h_->forward(dt_t, cache, tensor_names_.Vh_out);       cache.cache_tensor(tensor_names_.Vh_out, Vh_out_tensor_val);
        Tensor h_cand_arg_t_tensor_val = Wh_out_tensor_val + Uh_out_tensor_val + Vh_out_tensor_val;
        cache.cache_tensor(tensor_names_.h_cand_arg_t, h_cand_arg_t_tensor_val);
        Tensor h_cand_t_tensor_val = h_candidate_activation_->forward(h_cand_arg_t_tensor_val, cache, tensor_names_.h_cand_t);
        cache.cache_tensor(tensor_names_.h_cand_t, h_cand_t_tensor_val);

        Tensor ones_zt = Tensor::ones(z_t_tensor_val.rows(), z_t_tensor_val.cols());
        Tensor term1_hnew = Tensor(ones_zt - z_t_tensor_val) * h_decay_t_val;
        Tensor term2_hnew = z_t_tensor_val * h_cand_t_tensor_val;
        Tensor h_new_t = term1_hnew + term2_hnew;

        return h_new_t;
    }

    Tensor forward(const Tensor& input, GradientCache& cache, const std::string& output_tensor_name) override {
        throw std::logic_error("GRUDCell::forward should not be called directly. Use forward_step via TemporalPredictor.");
    }

    struct GRUDBackwardOutput {
        Tensor dL_dH_prev_t;
        Tensor dL_dX_t_input;
        Tensor dL_dDt_input;
    };

    GRUDBackwardOutput grud_backward_step(const Tensor& dL_dH_new, GradientCache& cache) {
        Tensor x_t_orig = cache.get_cached_tensor(get_cache_key("x_t_in"));
        Tensor h_prev_t_orig = cache.get_cached_tensor(get_cache_key("h_prev_t_in"));
        Tensor dt_t_orig = cache.get_cached_tensor(get_cache_key("dt_t_in"));
        const Tensor* mask_t_ptr = cache.has_cached_tensor(get_cache_key("mask_t_in")) ? &cache.get_cached_tensor(get_cache_key("mask_t_in")) : nullptr;

        Tensor x_hat_t = cache.get_cached_tensor(tensor_names_.x_hat_t);
        Tensor x_tilde_t = cache.get_cached_tensor(tensor_names_.x_tilde_t);
        Tensor h_decay_t = cache.get_cached_tensor(tensor_names_.h_decay_t);
        Tensor r_t = cache.get_cached_tensor(tensor_names_.r_t);
        Tensor z_t = cache.get_cached_tensor(tensor_names_.z_t);
        Tensor rh_prod_t = cache.get_cached_tensor(tensor_names_.rh_prod_t);
        Tensor h_cand_t = cache.get_cached_tensor(tensor_names_.h_cand_t);

        Tensor dL_dX_t_agg = Tensor::zeros(x_t_orig.rows(), x_t_orig.cols());
        Tensor dL_dH_prev_t_agg = Tensor::zeros(h_prev_t_orig.rows(), h_prev_t_orig.cols());
        Tensor dL_dDt_agg = Tensor::zeros(dt_t_orig.rows(), dt_t_orig.cols());

        Tensor dL_dZ_t_from_hnew = Tensor((h_cand_t - h_decay_t) * dL_dH_new);
        Tensor dL_dH_decay_t_from_hnew = Tensor(Tensor(Tensor::ones(z_t.rows(), z_t.cols()) - z_t) * dL_dH_new);
        Tensor dL_dH_cand_t_from_hnew = Tensor(z_t * dL_dH_new);

        Tensor dL_dX_tilde_agg = Tensor::zeros(x_tilde_t.rows(), x_tilde_t.cols());
        Tensor dL_dH_decay_t_agg = dL_dH_decay_t_from_hnew;

        Tensor dL_dH_cand_arg_t = h_candidate_activation_->backward_calc_and_store_grads(dL_dH_cand_t_from_hnew, cache);
        Tensor dL_dX_tilde_from_Wh = W_h_->backward_calc_and_store_grads(dL_dH_cand_arg_t, cache);
        Tensor dL_dRh_prod_t_from_Uh = U_h_->backward_calc_and_store_grads(dL_dH_cand_arg_t, cache);
        Tensor dL_dDt_from_Vh = V_h_->backward_calc_and_store_grads(dL_dH_cand_arg_t, cache);
        dL_dX_tilde_agg += dL_dX_tilde_from_Wh;
        dL_dDt_agg += dL_dDt_from_Vh;

        Tensor dL_dR_t_from_rh = Tensor(dL_dRh_prod_t_from_Uh * h_decay_t);
        dL_dH_decay_t_agg += Tensor(dL_dRh_prod_t_from_Uh * r_t); // Fixed: explicit Tensor construction

        Tensor dL_dZ_arg_t = z_gate_activation_->backward_calc_and_store_grads(dL_dZ_t_from_hnew, cache);
        Tensor dL_dX_tilde_from_Wz = W_z_->backward_calc_and_store_grads(dL_dZ_arg_t, cache);
        Tensor dL_dH_decay_t_from_Uz = U_z_->backward_calc_and_store_grads(dL_dZ_arg_t, cache);
        Tensor dL_dDt_from_Vz = V_z_->backward_calc_and_store_grads(dL_dZ_arg_t, cache);
        dL_dX_tilde_agg += dL_dX_tilde_from_Wz;
        dL_dH_decay_t_agg += dL_dH_decay_t_from_Uz;
        dL_dDt_agg += dL_dDt_from_Vz;

        Tensor dL_dR_arg_t = r_gate_activation_->backward_calc_and_store_grads(dL_dR_t_from_rh, cache);
        Tensor dL_dX_tilde_from_Wr = W_r_->backward_calc_and_store_grads(dL_dR_arg_t, cache);
        Tensor dL_dH_decay_t_from_Ur = U_r_->backward_calc_and_store_grads(dL_dR_arg_t, cache);
        Tensor dL_dDt_from_Vr = V_r_->backward_calc_and_store_grads(dL_dR_arg_t, cache);
        dL_dX_tilde_agg += dL_dX_tilde_from_Wr;
        dL_dH_decay_t_agg += dL_dH_decay_t_from_Ur;
        dL_dDt_agg += dL_dDt_from_Vr;

        if (config_.use_exponential_decay) {
            Tensor gamma_h_t = cache.get_cached_tensor(tensor_names_.gamma_h_t);
            dL_dH_prev_t_agg += Tensor(dL_dH_decay_t_agg * gamma_h_t); // Fixed: explicit Tensor construction
            Tensor dL_dGamma_h_t = Tensor(dL_dH_decay_t_agg * h_prev_t_orig);
            Tensor dL_dLog_gamma_BH = Tensor(dL_dGamma_h_t * gamma_h_t);
            Tensor log_gamma_raw_BH_val = cache.get_cached_tensor(tensor_names_.log_gamma_raw_BH);
            Tensor log_gamma_BH_val = cache.get_cached_tensor(tensor_names_.log_gamma_BH);
            MatrixXf clip_mask_data = (log_gamma_raw_BH_val.data().array() == log_gamma_BH_val.data().array()).template cast<float>();
            Tensor clip_mask = Tensor(clip_mask_data);
            Tensor dL_dLog_gamma_raw_BH = Tensor(dL_dLog_gamma_BH * clip_mask);
            Tensor softplus_decay_1H_out = cache.get_cached_tensor(tensor_names_.softplus_decay_1H_out);
            Tensor neg_sp_decay_T = Tensor(Tensor(softplus_decay_1H_out * Tensor(-1.0f))).transpose();
            dL_dDt_agg += dL_dLog_gamma_raw_BH.matmul(neg_sp_decay_T);
            Tensor dL_dNeg_Softplus_decay_1H = dt_t_orig.transpose().matmul(dL_dLog_gamma_raw_BH);
            Tensor dL_dSoftplus_decay_1H = Tensor(dL_dNeg_Softplus_decay_1H * Tensor(-1.0f));
            Tensor dL_dClipped_decay_1H = decay_softplus_activation_->backward_calc_and_store_grads(dL_dSoftplus_decay_1H, cache);
            Tensor decay_h_param_val = cache.get_cached_tensor(tensor_names_.decay_h_param_val);
            Tensor d_softclip_dx = activations::softclip_derivative_wrt_input(decay_h_param_val, config_.softclip_threshold);
            Tensor dL_dDecay_h_param = Tensor(dL_dClipped_decay_1H * d_softclip_dx);
            gradients_.at("decay_h_param") += dL_dDecay_h_param;
        } else {
            dL_dH_prev_t_agg += dL_dH_decay_t_agg;
        }

        Tensor dL_dX_hat_t;
        if (mask_t_ptr) {
            Tensor ones_mask = Tensor::ones(mask_t_ptr->rows(), mask_t_ptr->cols());
            dL_dX_t_agg += Tensor(dL_dX_tilde_agg * (*mask_t_ptr)); // Fixed: explicit Tensor construction
            dL_dX_hat_t = Tensor(dL_dX_tilde_agg * Tensor(ones_mask - (*mask_t_ptr)));
        } else {
            dL_dX_t_agg += dL_dX_tilde_agg;
            dL_dX_hat_t = Tensor::zeros(x_hat_t.rows(), x_hat_t.cols());
        }

        Tensor dL_dH_prev_t_from_impute = impute_linear_->backward_calc_and_store_grads(dL_dX_hat_t, cache);
        dL_dH_prev_t_agg += dL_dH_prev_t_from_impute;

        return {dL_dH_prev_t_agg, dL_dX_t_agg, dL_dDt_agg};
    }

    // Override of Layer's virtual backward, now calls the specific GRUD version
    Tensor backward_calc_and_store_grads(const Tensor& dL_dOutput, GradientCache& cache) override {
        // This dL_dOutput is dL_dH_new
        GRUDBackwardOutput grads = grud_backward_step(dL_dOutput, cache);
        // The calling lambda in TemporalPredictor will use all fields of GRUDBackwardOutput
        // to accumulate gradients for the GRUD cell's inputs (h_prev, x_t, dt_t).
        // For the Layer::backward_calc_and_store_grads signature, we return dL/d(primary_input),
        // which for a recurrent cell is typically dL/d(h_prev).
        return grads.dL_dH_prev_t;
    }


    std::vector<Layer*> get_all_sub_layers_for_optimizer(){
        std::vector<Layer*> sub_layers = {
                impute_linear_.get(), W_r_.get(), U_r_.get(), V_r_.get(),
                W_z_.get(), U_z_.get(), V_z_.get(), W_h_.get(), U_h_.get(), V_h_.get(),
                r_gate_activation_.get(), z_gate_activation_.get(), h_candidate_activation_.get()
        };
        if(config_.use_exponential_decay && decay_softplus_activation_){
            sub_layers.push_back(decay_softplus_activation_.get());
        }
        // Add GRUDCell itself if it has direct parameters like 'decay_h_param'
        // This is already handled by TemporalPredictor.get_all_trainable_layers()
        // if (!parameters_.empty()) {
        //    sub_layers.push_back(this);
        // }
        return sub_layers;
    }
    void zero_grad_all_sub_layers() {
        for(auto* layer : get_all_sub_layers_for_optimizer()){
            layer->zero_grad();
        }
        // Also zero grad for GRUDCell's own direct parameters (e.g. decay_h_param)
        this->zero_grad();
    }
    void set_training_mode_all_sub_layers(bool training) {
        for(auto* layer : get_all_sub_layers_for_optimizer()){
            layer->set_training_mode(training);
        }
        this->set_training_mode(training); // For GRUDCell's own direct parameters if any logic depends on it
    }
};


class Loss {
public:
    virtual ~Loss()=default;
    virtual float forward(const Tensor& prediction, const Tensor& target) = 0;
    // target_grad_name: The name under which dL/dPrediction should be stored in the cache.
    virtual void backward(const Tensor& prediction, const Tensor& target, GradientCache& cache, const std::string& prediction_grad_name) = 0;
};

class MSELoss : public Loss {
public:
    float forward(const Tensor& prediction, const Tensor& target) override {
        Tensor diff = prediction - target;
        Tensor diff_sq = Tensor(diff * diff); // Element-wise square
        // Mean of (0.5 * diff_sq). Taking 0.5 out of mean for clarity.
        return 0.5f * diff_sq.mean(-1).data()(0,0); // mean(-1) gives scalar tensor
    }

    void backward(const Tensor& prediction, const Tensor& target, GradientCache& cache, const std::string& prediction_grad_name) override {
        Tensor diff = prediction - target;
        // For N elements, loss is (1/N) * sum(0.5 * diff_i^2).
        // dLoss/dpred_i = (1/N) * diff_i.
        float N = static_cast<float>(prediction.rows() * prediction.cols());
        if (N == 0) throw std::runtime_error("Cannot compute MSELoss backward with zero elements.");
        Tensor grad = diff * (1.0f / N);
        cache.set_gradient(prediction_grad_name, grad);
    }
};