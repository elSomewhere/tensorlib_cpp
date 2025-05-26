// file: tensor.h
#pragma once

#include <eigen/Dense>
#include <vector>
#include <deque>
#include <memory>
#include <cmath>
#include <random>
#include <iostream>
#include <iomanip> // For std::fixed, std::setprecision in debug prints
#include <unordered_map>
#include <functional>
#include <type_traits>
#include <string>
#include <stdexcept>
#include <algorithm> // For std::max, std::transform, std::min
#include <numeric>   // For std::iota
#include <map>       // For AdamOptimizer states
#include <atomic>    // For debug flags


// --- Debug Flags (extern declaration) ---
// These might become less directly relevant for simple binary ops but could be used for Expression::eval() calls.
extern bool DEBUG_GRUD_Wz_GRAD;
extern bool IS_ANALYTICAL_PASS_DEBUG;

// --- Forward Declarations ---
class Tensor; // The main Tensor class

namespace TensorExpressions {

// --- Expression Base Class (CRTP) ---
    template<typename Derived>
    class Expression {
    public:
        // Access to the derived object
        const Derived& derived() const { return *static_cast<const Derived*>(this); }
        Derived& derived() { return *static_cast<Derived*>(this); }

        // Common interface (implemented by derived classes or via CRTP)
        // eval() and shape() are defined in the derived concrete expression types (Tensor, UnaryOpExpression, BinaryOpExpression)

        // Operator to implicitly convert to Tensor (triggers evaluation) - Use explicit construction for clarity
        // operator Tensor() const; // Defined after Tensor is fully defined (better to use Tensor(expr) constructor)

        // Common operations that return new Expression types
        template<typename E2>
        auto operator+(const Expression<E2>& other) const;
        template<typename E2>
        auto operator-(const Expression<E2>& other) const;
        template<typename E2>
        auto operator*(const Expression<E2>& other) const; // Element-wise
        template<typename E2>
        auto operator/(const Expression<E2>& other) const; // Element-wise

        auto operator-() const; // Unary minus

        auto exp() const;
        auto log() const;
        auto sqrt() const;
        auto abs() const;
        auto tanh() const;
        auto sigmoid() const;
        auto stable_softplus() const; // Numerically stable softplus
        auto pow(float exponent) const;
    };


// --- Operation Functors ---
    namespace Ops {
        struct Add {
            Eigen::MatrixXf operator()(const Eigen::MatrixXf& a, const Eigen::MatrixXf& b) const { return a + b; }
        };
        struct Subtract {
            Eigen::MatrixXf operator()(const Eigen::MatrixXf& a, const Eigen::MatrixXf& b) const { return a - b; }
        };
        struct MultiplyElementWise { // Hadamard product
            Eigen::MatrixXf operator()(const Eigen::MatrixXf& a, const Eigen::MatrixXf& b) const { return a.array() * b.array(); }
        };
        struct DivideElementWise {
            Eigen::MatrixXf operator()(const Eigen::MatrixXf& a, const Eigen::MatrixXf& b) const {
                // Add epsilon for stability if desired, or let user handle potential division by zero.
                // For robustness in general library, an epsilon is good.
                return a.array() / (b.array() + 1e-9f); // Added small epsilon
            }
        };
        struct Negate {
            Eigen::MatrixXf operator()(const Eigen::MatrixXf& a) const { return -a; }
        };
        struct Exp {
            Eigen::MatrixXf operator()(const Eigen::MatrixXf& a) const { return a.array().exp().matrix(); }
        };
        struct Log {
            Eigen::MatrixXf operator()(const Eigen::MatrixXf& a) const {
                // Add epsilon to prevent log(0) or log(negative if data is not positive)
                return (a.array().cwiseMax(1e-9f)).log().matrix();
            }
        };
        struct Sqrt {
            Eigen::MatrixXf operator()(const Eigen::MatrixXf& a) const {
                // Ensure non-negative for sqrt
                return (a.array().cwiseMax(0.0f)).sqrt().matrix();
            }
        };
        struct Abs {
            Eigen::MatrixXf operator()(const Eigen::MatrixXf& a) const { return a.array().abs().matrix(); }
        };
        struct Tanh {
            Eigen::MatrixXf operator()(const Eigen::MatrixXf& a) const { return a.array().tanh().matrix(); }
        };
        struct Sigmoid {
            Eigen::MatrixXf operator()(const Eigen::MatrixXf& a) const {
                // Standard sigmoid. Eigen's exp is generally robust.
                // Optional clipping for extreme inputs if they are truly expected and cause overflow with platform's float.
                // Eigen::MatrixXf clipped_a = a.array().cwiseMax(-70.0f).cwiseMin(70.0f).matrix(); // Max ~70 for exp(float)
                return (1.0f / (1.0f + (-a.array()).exp())).matrix();
            }
        };
        struct StableSoftplus {
            Eigen::MatrixXf operator()(const Eigen::MatrixXf& x) const {
                // max(0, x) + log(1 + exp(-abs(x)))
                Eigen::MatrixXf zeros = Eigen::MatrixXf::Zero(x.rows(), x.cols());
                Eigen::MatrixXf term1 = x.array().max(zeros.array()).matrix(); // max(0,x)
                Eigen::MatrixXf term2 = (1.0f + (-x.array().abs()).exp()).log().matrix(); // log(1+exp(-abs(x)))
                return term1 + term2;
            }
        };
        struct Pow {
            float exponent_;
            Pow(float exp) : exponent_(exp) {}
            Eigen::MatrixXf operator()(const Eigen::MatrixXf& a) const { return a.array().pow(exponent_).matrix(); }
        };
    } // namespace Ops


// --- Unary Operation Expression ---
    template<typename Op, typename Expr>
    class UnaryOpExpression : public Expression<UnaryOpExpression<Op, Expr>> {
        const Expr expr_internal_;
        Op op_fn_;
    public:
        UnaryOpExpression(const Expr& sub_expr, Op op = Op()) : expr_internal_(sub_expr), op_fn_(op) {}

        Eigen::MatrixXf eval() const {
            return op_fn_(expr_internal_.eval());
        }
        Eigen::Vector2i shape() const {
            return expr_internal_.shape();
        }
    };

// Forward declaration of internal_broadcast_helper
    Eigen::MatrixXf internal_broadcast_helper(const Eigen::MatrixXf& mat, int r, int c);

// --- Binary Operation Expression ---
    template<typename Op, typename LHSExpr, typename RHSExpr>
    class BinaryOpExpression : public Expression<BinaryOpExpression<Op, LHSExpr, RHSExpr>> {
        const LHSExpr lhs_internal_;
        const RHSExpr rhs_internal_;
        Op op_fn_;
    public:
        BinaryOpExpression(const LHSExpr& lhs, const RHSExpr& rhs, Op op = Op())
                : lhs_internal_(lhs), rhs_internal_(rhs), op_fn_(op) {}

        Eigen::MatrixXf eval() const {
            Eigen::MatrixXf lhs_val = lhs_internal_.eval();
            Eigen::MatrixXf rhs_val = rhs_internal_.eval();

            Eigen::Vector2i lhs_s = lhs_internal_.shape();
            Eigen::Vector2i rhs_s = rhs_internal_.shape();

            int out_rows = std::max(lhs_s[0], rhs_s[0]);
            int out_cols = std::max(lhs_s[1], rhs_s[1]);

            const Eigen::MatrixXf* lhs_b_ptr = &lhs_val;
            const Eigen::MatrixXf* rhs_b_ptr = &rhs_val;
            Eigen::MatrixXf lhs_b_storage, rhs_b_storage;

            if (lhs_s[0] != out_rows || lhs_s[1] != out_cols) {
                lhs_b_storage = internal_broadcast_helper(lhs_val, out_rows, out_cols);
                lhs_b_ptr = &lhs_b_storage;
            }
            if (rhs_s[0] != out_rows || rhs_s[1] != out_cols) {
                rhs_b_storage = internal_broadcast_helper(rhs_val, out_rows, out_cols);
                rhs_b_ptr = &rhs_b_storage;
            }

            return op_fn_(*lhs_b_ptr, *rhs_b_ptr);
        }

        Eigen::Vector2i shape() const {
            auto lhs_shape_val = lhs_internal_.shape();
            auto rhs_shape_val = rhs_internal_.shape();
            return Eigen::Vector2i(std::max(lhs_shape_val[0], rhs_shape_val[0]),
                                   std::max(lhs_shape_val[1], rhs_shape_val[1]));
        }
    };

// --- Implementation of Expression<Derived> operators ---
    template<typename Derived> template<typename E2>
    auto Expression<Derived>::operator+(const Expression<E2>& other) const {
        return BinaryOpExpression<Ops::Add, Derived, E2>(derived(), other.derived());
    }
    template<typename Derived> template<typename E2>
    auto Expression<Derived>::operator-(const Expression<E2>& other) const {
        return BinaryOpExpression<Ops::Subtract, Derived, E2>(derived(), other.derived());
    }
    template<typename Derived> template<typename E2>
    auto Expression<Derived>::operator*(const Expression<E2>& other) const { // Element-wise
        return BinaryOpExpression<Ops::MultiplyElementWise, Derived, E2>(derived(), other.derived());
    }
    template<typename Derived> template<typename E2>
    auto Expression<Derived>::operator/(const Expression<E2>& other) const { // Element-wise
        return BinaryOpExpression<Ops::DivideElementWise, Derived, E2>(derived(), other.derived());
    }
    template<typename Derived>
    auto Expression<Derived>::operator-() const {
        return UnaryOpExpression<Ops::Negate, Derived>(derived());
    }
    template<typename Derived>
    auto Expression<Derived>::exp() const {
        return UnaryOpExpression<Ops::Exp, Derived>(derived());
    }
    template<typename Derived>
    auto Expression<Derived>::log() const {
        return UnaryOpExpression<Ops::Log, Derived>(derived());
    }
    template<typename Derived>
    auto Expression<Derived>::sqrt() const {
        return UnaryOpExpression<Ops::Sqrt, Derived>(derived());
    }
    template<typename Derived>
    auto Expression<Derived>::abs() const {
        return UnaryOpExpression<Ops::Abs, Derived>(derived());
    }
    template<typename Derived>
    auto Expression<Derived>::tanh() const {
        return UnaryOpExpression<Ops::Tanh, Derived>(derived());
    }
    template<typename Derived>
    auto Expression<Derived>::sigmoid() const {
        return UnaryOpExpression<Ops::Sigmoid, Derived>(derived());
    }
    template<typename Derived>
    auto Expression<Derived>::stable_softplus() const {
        return UnaryOpExpression<Ops::StableSoftplus, Derived>(derived());
    }
    template<typename Derived>
    auto Expression<Derived>::pow(float exponent) const {
        return UnaryOpExpression<Ops::Pow, Derived>(derived(), Ops::Pow(exponent));
    }

} // namespace TensorExpressions


// --- Tensor Class Definition ---
class Tensor : public TensorExpressions::Expression<Tensor> {
private:
    Eigen::MatrixXf data_;

public:
    // --- Constructors ---
    Tensor() : data_(0,0) {}
    Tensor(int rows, int cols) : data_(rows, cols) { data_.setZero(); }
    explicit Tensor(Eigen::MatrixXf data_matrix) : data_(std::move(data_matrix)) {}
    explicit Tensor(float scalar);

    template<typename DerivedExpr>
    Tensor(const TensorExpressions::Expression<DerivedExpr>& expr) : data_(expr.derived().eval()) {}

    template<typename DerivedExpr>
    Tensor& operator=(const TensorExpressions::Expression<DerivedExpr>& expr) {
        data_ = expr.derived().eval();
        return *this;
    }

    Tensor(const Tensor& other) : data_(other.data_) {}
    Tensor(Tensor&& other) noexcept : data_(std::move(other.data_)) {}
    Tensor& operator=(const Tensor& other) {
        if (this != &other) data_ = other.data_;
        return *this;
    }
    Tensor& operator=(Tensor&& other) noexcept {
        if (this != &other) data_ = std::move(other.data_);
        return *this;
    }

    // --- Expression Interface Implementation for Tensor ---
    const Eigen::MatrixXf& eval() const { return data_; }
    Eigen::Vector2i shape() const { return Eigen::Vector2i(static_cast<int>(data_.rows()), static_cast<int>(data_.cols())); }

    // --- Direct Data Access ---
    const Eigen::MatrixXf& data() const { return data_; }
    Eigen::MatrixXf& data() { return data_; }
    int rows() const { return static_cast<int>(data_.rows()); }
    int cols() const { return static_cast<int>(data_.cols()); }
    bool empty() const { return data_.size() == 0; }

    // --- In-place Operations (Force Evaluation) ---
    template<typename DerivedExpr>
    Tensor& operator+=(const TensorExpressions::Expression<DerivedExpr>& expr) {
        Tensor result_tensor = TensorExpressions::BinaryOpExpression<TensorExpressions::Ops::Add, Tensor, DerivedExpr>(*this, expr.derived());
        data_ = result_tensor.eval();
        return *this;
    }
    template<typename DerivedExpr>
    Tensor& operator-=(const TensorExpressions::Expression<DerivedExpr>& expr) {
        Tensor result_tensor = TensorExpressions::BinaryOpExpression<TensorExpressions::Ops::Subtract, Tensor, DerivedExpr>(*this, expr.derived());
        data_ = result_tensor.eval();
        return *this;
    }
    template<typename DerivedExpr>
    Tensor& operator*=(const TensorExpressions::Expression<DerivedExpr>& expr) { // Element-wise
        Tensor result_tensor = TensorExpressions::BinaryOpExpression<TensorExpressions::Ops::MultiplyElementWise, Tensor, DerivedExpr>(*this, expr.derived());
        data_ = result_tensor.eval();
        return *this;
    }
    template<typename DerivedExpr>
    Tensor& operator/=(const TensorExpressions::Expression<DerivedExpr>& expr) { // Element-wise
        Tensor result_tensor = TensorExpressions::BinaryOpExpression<TensorExpressions::Ops::DivideElementWise, Tensor, DerivedExpr>(*this, expr.derived());
        data_ = result_tensor.eval();
        return *this;
    }


    // --- Static Factory Methods ---
    static Tensor zeros(int rows, int cols);
    static Tensor ones(int rows, int cols);
    static Tensor randn(int rows, int cols, std::mt19937& gen);
    static Tensor glorot_uniform(int in_features, int out_features, std::mt19937& gen);

    // --- Other Tensor-Specific Methods (Mostly Materializing) ---
    Tensor broadcast_to(int target_rows, int target_cols) const;
    Tensor transpose() const;
    Tensor sum(int axis = -1) const;
    Tensor mean(int axis = -1) const;
    Tensor matmul(const Tensor& other) const;
    Tensor cwiseMax(float val) const;
    Tensor cwiseMin(float val) const;
};

// --- Implementation for internal_broadcast_helper ---
namespace TensorExpressions {
    inline Eigen::MatrixXf internal_broadcast_helper(const Eigen::MatrixXf& mat, int target_rows, int target_cols) {
        if (mat.rows() == target_rows && mat.cols() == target_cols) return mat;
        Eigen::MatrixXf result(target_rows, target_cols);
        if (mat.rows() == 1 && mat.cols() == target_cols) {
            result = mat.replicate(target_rows, 1);
        } else if (mat.cols() == 1 && mat.rows() == target_rows) {
            result = mat.replicate(1, target_cols);
        } else if (mat.rows() == 1 && mat.cols() == 1) {
            result.setConstant(mat(0,0));
        } else if (mat.rows() == target_rows && mat.cols() == 1) {
            result = mat.replicate(1, target_cols);
        } else if (mat.cols() == target_cols && mat.rows() == 1) {
            result = mat.replicate(target_rows, 1);
        } else {
            throw std::invalid_argument("internal_broadcast_helper: Invalid broadcast dimensions for matrix (" +
                                        std::to_string(mat.rows()) + "," + std::to_string(mat.cols()) + ") to (" +
                                        std::to_string(target_rows) + "," + std::to_string(target_cols) + ")");
        }
        return result;
    }
}


// --- Tensor Method Implementations ---
inline Tensor::Tensor(float scalar) : data_(Eigen::MatrixXf::Constant(1, 1, scalar)) {}

inline Tensor Tensor::zeros(int rows, int cols) { return Tensor(Eigen::MatrixXf::Zero(rows, cols)); }
inline Tensor Tensor::ones(int rows, int cols) { return Tensor(Eigen::MatrixXf::Ones(rows, cols)); }
inline Tensor Tensor::randn(int rows, int cols, std::mt19937& gen) {
    Eigen::MatrixXf d(rows, cols); std::normal_distribution<float> dist(0.f,1.f);
    for (long long i=0; i<d.size(); ++i) d.data()[i] = dist(gen); return Tensor(std::move(d));
}
inline Tensor Tensor::glorot_uniform(int in_features, int out_features, std::mt19937& gen) {
    float limit = std::sqrt(6.f / (in_features + out_features));
    Eigen::MatrixXf d(out_features, in_features); // Standard convention: (out, in)
    std::uniform_real_distribution<float> dist(-limit, limit);
    for (long long i=0; i<d.size(); ++i) d.data()[i] = dist(gen);
    return Tensor(std::move(d));
}

inline Tensor Tensor::broadcast_to(int target_rows, int target_cols) const {
    return Tensor(TensorExpressions::internal_broadcast_helper(data_, target_rows, target_cols));
}
inline Tensor Tensor::transpose() const { return Tensor(data_.transpose()); }

inline Tensor Tensor::sum(int axis) const {
    if (data_.size() == 0 && (axis == -1 || rows() == 0 || cols() == 0) ) return Tensor(0.f);
    if (data_.size() == 0) throw std::runtime_error("Sum of empty tensor with non-trivial axes is ambiguous");

    if (axis == -1) return Tensor(Eigen::MatrixXf::Constant(1,1,data_.sum()));
    if (axis == 0) { // Sum along columns (reduce rows)
        if (rows() == 0) return Tensor::zeros(1, cols());
        return Tensor(data_.colwise().sum()); // Result is 1xCols
    }
    if (axis == 1) { // Sum along rows (reduce columns)
        if (cols() == 0) return Tensor::zeros(rows(), 1);
        return Tensor(data_.rowwise().sum()); // Result is Rowsx1
    }
    throw std::invalid_argument("Tensor::sum: invalid axis " + std::to_string(axis));
}
inline Tensor Tensor::mean(int axis) const {
    if (data_.size() == 0) throw std::runtime_error("Mean of empty tensor is undefined.");
    if (axis == -1) return Tensor(Eigen::MatrixXf::Constant(1,1,data_.mean()));
    if (axis == 0) return Tensor(data_.colwise().mean());
    if (axis == 1) return Tensor(data_.rowwise().mean());
    throw std::invalid_argument("Tensor::mean: invalid axis " + std::to_string(axis));
}
inline Tensor Tensor::matmul(const Tensor& other) const {
    if (this->cols() != other.rows()) {
        throw std::invalid_argument("Matrix dimensions mismatch for matmul: LHS (" +
                                    std::to_string(this->rows()) + "," + std::to_string(this->cols()) + "), RHS (" +
                                    std::to_string(other.rows()) + "," + std::to_string(other.cols()) + ")");
    }
    return Tensor(this->data_ * other.data());
}
inline Tensor Tensor::cwiseMax(float v) const { return Tensor(data_.array().cwiseMax(v).matrix());}
inline Tensor Tensor::cwiseMin(float v) const { return Tensor(data_.array().cwiseMin(v).matrix());}


// --- Free function operators for Tensor and scalar ---
// These will promote scalar to Tensor and then use Expression template operators
inline auto operator*(const Tensor& t, float s) { return t * Tensor(s); }
inline auto operator*(float s, const Tensor& t) { return Tensor(s) * t; }
inline auto operator+(const Tensor& t, float s) { return t + Tensor(s); }
inline auto operator+(float s, const Tensor& t) { return Tensor(s) + t; }
inline auto operator-(const Tensor& t, float s) { return t - Tensor(s); }
inline auto operator-(float s, const Tensor& t) { return Tensor(s) - t; }
inline auto operator/(const Tensor& t, float s) {
    if (s == 0.f) throw std::runtime_error("Division by zero scalar.");
    return t / Tensor(s); // Uses Tensor element-wise division (with epsilon)
}

// Overloads for an Expression and a scalar
template<typename Derived>
auto operator*(const TensorExpressions::Expression<Derived>& expr, float s) {
    return expr.derived() * Tensor(s);
}
template<typename Derived>
auto operator*(float s, const TensorExpressions::Expression<Derived>& expr) {
    return Tensor(s) * expr.derived();
}
template<typename Derived>
auto operator+(const TensorExpressions::Expression<Derived>& expr, float s) {
    return expr.derived() + Tensor(s);
}
template<typename Derived>
auto operator+(float s, const TensorExpressions::Expression<Derived>& expr) {
    return Tensor(s) + expr.derived();
}
template<typename Derived>
auto operator-(const TensorExpressions::Expression<Derived>& expr, float s) {
    return expr.derived() - Tensor(s);
}
template<typename Derived>
auto operator-(float s, const TensorExpressions::Expression<Derived>& expr) {
    return Tensor(s) - expr.derived();
}
template<typename Derived>
auto operator/(const TensorExpressions::Expression<Derived>& expr, float s) {
    if (s == 0.f) throw std::runtime_error("Division by zero scalar in expression.");
    return expr.derived() / Tensor(s);
}


// --- Debug Print Utility ---
inline void print_tensor_debug_Terrasse(const std::string& name, const Tensor& t, int r_max=1, int c_max=1, bool full_if_small=false) {
    if (!DEBUG_GRUD_Wz_GRAD) return;

    std::ios_base::fmtflags old_flags = std::cout.flags();
    std::streamsize old_precision = std::cout.precision();

    std::cout << std::fixed << std::setprecision(5);
    if (t.empty()) {
        std::cout << " Terrasse DEBUG: " << name << " is EMPTY." << std::endl;
        std::cout.flags(old_flags);
        std::cout.precision(old_precision);
        return;
    }
    std::cout << " Terrasse DEBUG: " << name << " (Shape: " << t.rows() << "x" << t.cols() << ")";
    if (t.rows() == 0 || t.cols() == 0) {
        std::cout << " (Zero rows or cols)" << std::endl;
        std::cout.flags(old_flags);
        std::cout.precision(old_precision);
        return;
    }

    bool should_print_full = full_if_small && t.rows() <= r_max && t.cols() <= c_max;
    if (!full_if_small && t.rows() <= 2 && t.cols() <=3) {
        should_print_full = true;
    }


    if (should_print_full) {
        std::cout << ":\n" << t.data() << std::endl;
    } else {
        std::cout << " Val[0,0]: " << t.data()(0,0);
        if (t.cols() > 1 && c_max > 1 && t.rows() > 0) std::cout << " Val[0,1]: " << t.data()(0,1);
        std::cout << std::endl;
    }
    std::cout.flags(old_flags);
    std::cout.precision(old_precision);
}


// --- Activations namespace (example of using stable_softplus) ---
namespace activations {
    inline Tensor stateless_sigmoid(const Tensor& x) {
        return x.sigmoid();
    }
    inline Tensor stateless_tanh(const Tensor& x) {
        return x.tanh();
    }
    inline Tensor stateless_softplus(const Tensor& x) {
        return x.stable_softplus();
    }
    Tensor softclip(const Tensor& x, float threshold) {
        if (threshold <= 0) throw std::invalid_argument("Threshold must be positive for softclip");
        Eigen::ArrayXXf x_arr = x.data().array();
        Eigen::ArrayXXf abs_x = x_arr.abs();
        Eigen::Array<bool,Eigen::Dynamic,Eigen::Dynamic> condition = (abs_x <= threshold);
        Eigen::MatrixXf result_data = condition.select(x.data(), x_arr.sign() * (threshold + (abs_x - threshold + 1.0f).log())).matrix();
        return Tensor(result_data);
    }

    Tensor softclip_derivative_wrt_input(const Tensor& input_val, float threshold) {
        if (threshold <= 0) throw std::invalid_argument("Threshold must be positive for softclip derivative");
        Eigen::ArrayXXf abs_x = input_val.data().array().abs();
        Eigen::Array<bool,Eigen::Dynamic,Eigen::Dynamic> condition = (abs_x <= threshold);
        Eigen::MatrixXf grad_data = condition.select(Eigen::MatrixXf::Ones(input_val.rows(),input_val.cols()), (1.0f / (1.0f + abs_x - threshold)).matrix());
        return Tensor(grad_data);
    }
} // namespace activations


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
        propagated_gradients[tensor_name] = std::move(grad);
    }

    void accumulate_gradient(const std::string& tensor_name, const Tensor& grad_contribution) {
        if (grad_contribution.empty()) {
            return;
        }
        auto it = propagated_gradients.find(tensor_name);
        if (it == propagated_gradients.end()) {
            propagated_gradients[tensor_name] = grad_contribution;
        } else {
            if (it->second.shape() != grad_contribution.shape()) {
                // Try to broadcast grad_contribution to it->second's shape
                // This covers cases like accumulating a (1,C) gradient into a (B,C) gradient sum,
                // or a (B,1) into (B,C), or (1,1) into (B,C).
                try {
                    it->second += grad_contribution.broadcast_to(it->second.rows(), it->second.cols());
                } catch (const std::invalid_argument& e) {
                    throw std::runtime_error("Gradient shape mismatch for accumulation on " + tensor_name +
                                             ": existing (" + std::to_string(it->second.rows()) + "," + std::to_string(it->second.cols()) +
                                             "), new (" + std::to_string(grad_contribution.rows()) + "," + std::to_string(grad_contribution.cols()) + ")" +
                                             " and broadcast failed. Original error: " + e.what());
                }
            } else {
                it->second += grad_contribution; // Uses Tensor::operator+=
            }
        }
    }


    Tensor get_gradient(const std::string& tensor_name) const {
        auto it = propagated_gradients.find(tensor_name);
        if (it == propagated_gradients.end()) {
            throw std::runtime_error("Gradient not found for tensor: " + tensor_name );
        }
        return it->second;
    }
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
    void clear_all_except_parameters() {
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
    std::string name_;
public:
    Layer(std::string name) : training_mode_(true), name_(std::move(name)) {}
    virtual ~Layer() = default;

    virtual Tensor forward(const Tensor& input, GradientCache& cache, const std::string& output_tensor_name) = 0;
    virtual Tensor backward_calc_and_store_grads(const Tensor& dL_dOutput, GradientCache& cache) = 0;

    void set_training_mode(bool training) { training_mode_ = training; }
    bool is_training() const { return training_mode_; }
    const std::unordered_map<std::string, Tensor>& parameters() const { return parameters_; }
    std::unordered_map<std::string, Tensor>& get_parameters_mut() { return parameters_; }
    const std::unordered_map<std::string, Tensor>& gradients() const { return gradients_; }
    void zero_grad() { for (auto& [k,v] : gradients_) v.data().setZero(); }
    const std::string& name() const { return name_; }

    std::string get_cache_key(const std::string& suffix) const { return name_ + "_" + suffix; }

protected:
    void register_parameter(const std::string& pname, Tensor p) {
        parameters_[pname] = std::move(p);
        gradients_[pname] = Tensor::zeros(parameters_[pname].rows(), parameters_[pname].cols());
    }
};

class TanhLayer : public Layer {
public:
    TanhLayer(const std::string& name) : Layer(name) {}

    Tensor forward(const Tensor& input, GradientCache& cache, const std::string& output_tensor_name) override {
        Tensor output = input.tanh();
        cache.cache_tensor(get_cache_key("output_for_grad"), output);
        return output;
    }

    Tensor backward_calc_and_store_grads(const Tensor& dL_dOutput, GradientCache& cache) override {
        Tensor original_output = cache.get_cached_tensor(get_cache_key("output_for_grad"));
        Tensor output_squared = original_output * original_output;
        Tensor tanhx_grad_val = Tensor(1.0f) - output_squared; // Tensor(1.0f) is Expression
        Tensor dL_dInput = dL_dOutput * tanhx_grad_val;
        return dL_dInput;
    }
};


class SigmoidLayer : public Layer {
public:
    SigmoidLayer(const std::string& name) : Layer(name) {}
    Tensor forward(const Tensor& input, GradientCache& cache, const std::string& output_tensor_name) override {
        Tensor output = input.sigmoid();
        cache.cache_tensor(get_cache_key("output_for_grad"), output);
        return output;
    }
    Tensor backward_calc_and_store_grads(const Tensor& dL_dOutput, GradientCache& cache) override {
        Tensor original_output = cache.get_cached_tensor(get_cache_key("output_for_grad"));
        Tensor term_in_paren = Tensor(1.0f) - original_output;
        Tensor sigx_grad_val = original_output * term_in_paren;
        Tensor dL_dInput = dL_dOutput * sigx_grad_val;
        return dL_dInput;
    }
};

class SoftplusLayer : public Layer {
public:
    SoftplusLayer(const std::string& name) : Layer(name) {}
    Tensor forward(const Tensor& input, GradientCache& cache, const std::string& output_tensor_name) override {
        Tensor output = input.stable_softplus();
        cache.cache_tensor(get_cache_key("input_for_grad"), input);
        return output;
    }
    Tensor backward_calc_and_store_grads(const Tensor& dL_dOutput, GradientCache& cache) override {
        Tensor original_input = cache.get_cached_tensor(get_cache_key("input_for_grad"));
        Tensor softplus_grad_val = original_input.sigmoid();
        Tensor dL_dInput = dL_dOutput * softplus_grad_val;
        return dL_dInput;
    }
};

class Linear : public Layer {
private:
    int in_features_, out_features_;
    bool use_bias_;
public:
    Linear(int in_features, int out_features, bool use_bias=true, std::mt19937* gen_nullable=nullptr, const std::string& name="linear")
            : Layer(name), in_features_(in_features), out_features_(out_features), use_bias_(use_bias) {
        std::mt19937* gen_ptr = gen_nullable;
        std::mt19937 local_gen_storage;
        if (!gen_ptr) {
            local_gen_storage.seed(std::random_device{}());
            gen_ptr = &local_gen_storage;
        }
        // Note: Glorot uniform expects (out, in) for weights if Wx, (in,out) if xW.
        // Our matmul is input.matmul(W.transpose()), so W should be (out, in)
        register_parameter("weight", Tensor::glorot_uniform(in_features_, out_features_, *gen_ptr));
        if(use_bias_) register_parameter("bias", Tensor::zeros(1, out_features_));
    }

    Tensor forward(const Tensor& input, GradientCache& cache, const std::string& output_tensor_name) override {
        cache.cache_tensor(get_cache_key("input_for_grad"), input);
        Tensor W = parameters_.at("weight"); // This is a Tensor (Expression<Tensor>)
        Tensor out_expr = input.matmul(W.transpose());
        if(use_bias_){
            Tensor b = parameters_.at("bias");
            // out_expr is already a Tensor from matmul.
            // Adding b (a Tensor) to out_expr (a Tensor) will use Expression::operator+
            // and then the result is evaluated into the 'out' Tensor.
            Tensor out = out_expr + b.broadcast_to(out_expr.rows(), out_expr.cols());
            return out;
        }
        return out_expr; // out_expr is already a Tensor here
    }

    Tensor backward_calc_and_store_grads(const Tensor& dL_dOutput, GradientCache& cache) override {
        Tensor original_input = cache.get_cached_tensor(get_cache_key("input_for_grad"));
        Tensor W_param = parameters_.at("weight");

        if (DEBUG_GRUD_Wz_GRAD && this->name() == "focused_tp_check_L0_gru_cell_Wr") {
            print_tensor_debug_Terrasse("Linear Layer " + this->name() + ": dL_dOutput (is dL_dR_arg_t for Wr)", dL_dOutput, 2, 3, true);
        }

        // dL_dW = dL_dOutput.transpose().matmul(original_input)
        Tensor dL_dW = dL_dOutput.transpose().matmul(original_input);
        gradients_.at("weight") += dL_dW; // Uses Tensor::operator+=

        if(use_bias_){
            Tensor dL_dB = dL_dOutput.sum(0); // sum returns a Tensor
            gradients_.at("bias") += dL_dB;
            if (DEBUG_GRUD_Wz_GRAD && this->name() == "focused_tp_check_L0_gru_cell_Wr") {
                print_tensor_debug_Terrasse("Linear Layer " + this->name() + ": dL_dB (from dL_dOutput.sum(0))", dL_dB, 1, 3, true);
            }
        }

        Tensor dL_dInput = dL_dOutput.matmul(W_param);
        return dL_dInput;
    }
};

class LayerNorm : public Layer {
private:
    float eps_; bool affine_; int features_;
public:
    LayerNorm(int features, float eps=1e-5f, bool affine=true, const std::string& name="layernorm")
            : Layer(name), eps_(eps), affine_(affine), features_(features){
        if(affine_){
            register_parameter("weight", Tensor::ones(1, features_));
            register_parameter("bias", Tensor::zeros(1, features_));
        }
    }

    Tensor forward(const Tensor& input, GradientCache& cache, const std::string& output_tensor_name) override {
        cache.cache_tensor(get_cache_key("input_for_grad"), input);
        Tensor mean_val = input.mean(1); // (B, 1)
        Tensor centered_input = input - mean_val.broadcast_to(input.rows(), input.cols()); // (B,F) - (B,F)
        Tensor var_val_unscaled = centered_input * centered_input; // (B,F) element-wise
        Tensor var_val = var_val_unscaled.mean(1); // (B,1)
        Tensor std_dev_val = (var_val + eps_).sqrt(); // (B,1)
        // inv_std_dev = 1.0 / std_dev_val
        Tensor inv_std_dev = Tensor(1.0f) / std_dev_val; // (B,1)
        Tensor normalized_input = centered_input * inv_std_dev.broadcast_to(input.rows(), input.cols()); // (B,F) * (B,F)

        cache.cache_tensor(get_cache_key("centered_input_for_grad"), centered_input);
        cache.cache_tensor(get_cache_key("inv_std_dev_for_grad"), inv_std_dev); // This is (B,1)
        cache.cache_tensor(get_cache_key("normalized_input_for_grad"), normalized_input);


        if (DEBUG_GRUD_Wz_GRAD && this->name().find("output_ln") != std::string::npos) {
            std::cout << "  LN Debug (" << this->name() << " - Forward):\n";
            print_tensor_debug_Terrasse("    Input to LN", input, input.rows(), std::min(5, input.cols()), true);
            print_tensor_debug_Terrasse("    Mean of Input (axis 1)", mean_val, mean_val.rows(), std::min(5, mean_val.cols()), true);
            print_tensor_debug_Terrasse("    StdDev of Input (axis 1, after eps)", std_dev_val, std_dev_val.rows(), std::min(5, std_dev_val.cols()), true);
            print_tensor_debug_Terrasse("    InvStdDev", inv_std_dev, inv_std_dev.rows(), std::min(5, inv_std_dev.cols()), true);
            if (input.rows() > 0 && std_dev_val.rows() > 0) {
                for(int r=0; r < std::min(input.rows(), 2) ; ++r) {
                    if (std_dev_val.data()(r,0) < 1e-6 && std_dev_val.data()(r,0) > 0) {
                        std::cout << "    WARNING: Very small std_dev for batch " << r << ": " << std_dev_val.data()(r,0) << std::endl;
                    }
                }
            }
            print_tensor_debug_Terrasse("    Normalized Input (x_hat)", normalized_input, normalized_input.rows(), std::min(5, normalized_input.cols()), true);
        }

        Tensor output = normalized_input;
        if(affine_){
            Tensor w = parameters_.at("weight"); // (1,F)
            Tensor b = parameters_.at("bias");   // (1,F)
            cache.cache_tensor(get_cache_key("affine_weight_for_grad"), w);
            // output = (output * w_broadcast) + b_broadcast
            output = (output * w.broadcast_to(output.rows(), output.cols())) + b.broadcast_to(output.rows(), output.cols());
        }
        return output;
    }

    Tensor backward_calc_and_store_grads(const Tensor& dL_dOutput, GradientCache& cache) override {
        Tensor original_input = cache.get_cached_tensor(get_cache_key("input_for_grad"));
        Tensor inv_std_dev_B1 = cache.get_cached_tensor(get_cache_key("inv_std_dev_for_grad")); // Shape (B,1)
        Tensor normalized_input_BF = cache.get_cached_tensor(get_cache_key("normalized_input_for_grad")); // Shape (B,F)

        int B = original_input.rows();
        int F_int = original_input.cols();
        if (F_int == 0) throw std::runtime_error("LayerNorm backward: Number of features (F) is zero.");
        float F_float = static_cast<float>(F_int);

        Tensor dL_dNormalized_input = dL_dOutput; // Shape (B,F)

        if(affine_){
            Tensor w_affine_1F = cache.get_cached_tensor(get_cache_key("affine_weight_for_grad")); // Shape (1,F)

            // dL/dw = sum_batch (dL/dY * x_hat)
            // (dL/dY * x_hat) is (B,F) element-wise. Sum over B axis. Result (1,F)
            gradients_.at("weight") += Tensor(dL_dOutput * normalized_input_BF).sum(0);
            // dL/db = sum_batch (dL/dY)
            gradients_.at("bias") += dL_dOutput.sum(0); // Result (1,F)

            // dL/dx_hat = dL/dY * w
            dL_dNormalized_input = dL_dOutput * w_affine_1F.broadcast_to(B, F_int);
        }

        Tensor inv_std_dev_BF = inv_std_dev_B1.broadcast_to(B, F_int); // Shape (B,F)

        // Intermediate terms for dL/dx (all shapes become (B,F) after broadcasting the (B,1) sums)
        // sum(dL/dx_hat) / F  -- sum over F axis, giving (B,1), then broadcast to (B,F)
        Tensor term1 = Tensor(dL_dNormalized_input.sum(1) / F_float).broadcast_to(B,F_int);

        // sum(dL/dx_hat * x_hat) / F -- (dL/dx_hat * x_hat) is (B,F), sum over F -> (B,1), then broadcast
        Tensor term2_sum_part = Tensor(dL_dNormalized_input * normalized_input_BF).sum(1) / F_float; // (B,1)
        Tensor term2 = term2_sum_part.broadcast_to(B,F_int); // (B,F)

        // dL/dx = (1/std) * [dL/dx_hat - term1 - x_hat * term2]
        // All ops inside [] are (B,F)
        Tensor dL_dInput = inv_std_dev_BF * (dL_dNormalized_input - term1 - (normalized_input_BF * term2));


        if (DEBUG_GRUD_Wz_GRAD && this->name().find("output_ln") != std::string::npos) {
            std::cout << "  LN Debug (" << this->name() << " - Backward):\n";
            print_tensor_debug_Terrasse("    dL_dOutput (to LN)", dL_dOutput, dL_dOutput.rows(), std::min(5, dL_dOutput.cols()), true);
            if (affine_) print_tensor_debug_Terrasse("    dL_dNormalized_input (after affine grad)", dL_dNormalized_input, dL_dNormalized_input.rows(), std::min(5, dL_dNormalized_input.cols()), true);
            print_tensor_debug_Terrasse("    dL_dInput (from LN)", dL_dInput, dL_dInput.rows(), std::min(5, dL_dInput.cols()), true);
        }
        return dL_dInput;
    }
};


class Dropout : public Layer {
private:
    float p_dropout_;
    mutable std::mt19937 rng_; // mutable for use in const forward if it were const
    mutable std::uniform_real_distribution<float> dist_;
public:
    Dropout(float p=0.5f, const std::string& name="dropout")
            : Layer(name), p_dropout_(p), dist_(0.f,1.f) {
        if(p_dropout_<0.f || p_dropout_>1.f) throw std::invalid_argument("Dropout p must be in [0,1]");
        rng_.seed(std::random_device{}()); // Default seed
    }

    void set_seed(unsigned int seed){ rng_.seed(seed); }

    Tensor forward(const Tensor& input, GradientCache& cache, const std::string& output_tensor_name) override {
        if(!training_mode_ || p_dropout_ == 0.f){
            // Cache a mask of ones for backward pass if needed, or handle this logic in backward.
            // For simplicity, assume backward knows if it was training mode.
            // However, caching the mask is cleaner.
            cache.cache_tensor(get_cache_key("dropout_mask_for_grad"), Tensor::ones(input.rows(), input.cols()));
            return input;
        }
        if (p_dropout_ == 1.f) { // All dropout
            cache.cache_tensor(get_cache_key("dropout_mask_for_grad"), Tensor::zeros(input.rows(), input.cols()));
            return Tensor::zeros(input.rows(), input.cols());
        }

        float scale_factor = 1.0f / (1.0f - p_dropout_);
        Eigen::MatrixXf mask_data(input.rows(), input.cols());
        for(int r=0; r < mask_data.rows(); ++r) {
            for(int c=0; c < mask_data.cols(); ++c) {
                mask_data(r,c) = (dist_(rng_) > p_dropout_) ? scale_factor : 0.0f;
            }
        }
        Tensor mask_tensor(std::move(mask_data));
        cache.cache_tensor(get_cache_key("dropout_mask_for_grad"), mask_tensor);
        return input * mask_tensor; // Element-wise product
    }

    Tensor backward_calc_and_store_grads(const Tensor& dL_dOutput, GradientCache& cache) override {
        Tensor mask = cache.get_cached_tensor(get_cache_key("dropout_mask_for_grad"));
        return dL_dOutput * mask; // Element-wise product
    }
};


struct TemporalConfig {
    int batch_size=1, input_size=4, hidden_size=64, num_layers=2;
    bool use_exponential_decay=true;
    float softclip_threshold=3.f, min_log_gamma=-10.f;
    float dropout_rate=0.1f; bool use_layer_norm=true;
    float learning_rate=2e-3f;
    int tbptt_steps=20, seed=0;
};

// Note: GRUDCell and MSELoss would need to be included here.
// Their internal logic would use the new Tensor operations.
// For example, in GRUDCell forward_step:
// Tensor r_arg_t_val = Wr_out_val + Ur_out_val + Vr_out_val; (all are Tensors, so this uses Expression ops)
// Tensor r_t_val = r_gate_activation_->forward(r_arg_t_val, cache, tensor_names_.r_t);
// This structure remains valid.

class GRUDCell : public Layer {
private:
    TemporalConfig config_;
    int current_cell_input_size_;

    std::unique_ptr<Linear> impute_linear_;
    std::unique_ptr<Linear> W_r_, U_r_, V_r_;
    std::unique_ptr<Linear> W_z_, U_z_, V_z_;
    std::unique_ptr<Linear> W_h_, U_h_, V_h_;

    std::unique_ptr<SigmoidLayer> r_gate_activation_;
    std::unique_ptr<SigmoidLayer> z_gate_activation_;
    std::unique_ptr<TanhLayer> h_candidate_activation_;
    std::unique_ptr<SoftplusLayer> decay_softplus_activation_; // Will use stable softplus internally

    struct ForwardPassTensorNames {
        std::string x_hat_t, x_tilde_t;
        std::string gamma_h_t, log_gamma_raw_BH, log_gamma_BH, h_decay_t;
        std::string Wr_out, Ur_out, Vr_out, r_arg_t, r_t;
        std::string Wz_out, Uz_out, Vz_out, z_arg_t, z_t;
        std::string rh_prod_t;
        std::string Wh_out, Uh_out, Vh_out, h_cand_arg_t, h_cand_t;
        std::string decay_h_param_val, clipped_decay_1H, softplus_decay_1H_out;
    };
    ForwardPassTensorNames tensor_names_; // Initialized in forward_step

public:
    GRUDCell(const TemporalConfig& c, int input_size_for_this_cell, std::mt19937& gen, const std::string& name_prefix)
            : Layer(name_prefix), config_(c), current_cell_input_size_(input_size_for_this_cell) {

        impute_linear_ = std::make_unique<Linear>(c.hidden_size, current_cell_input_size_, true, &gen, name_ + "_impute");
        W_r_ = std::make_unique<Linear>(current_cell_input_size_, c.hidden_size, true, &gen, name_ + "_Wr");
        U_r_ = std::make_unique<Linear>(c.hidden_size, c.hidden_size, false, &gen, name_ + "_Ur"); // No bias for U matrices
        V_r_ = std::make_unique<Linear>(1, c.hidden_size, false, &gen, name_ + "_Vr"); // No bias for V matrices

        W_z_ = std::make_unique<Linear>(current_cell_input_size_, c.hidden_size, true, &gen, name_ + "_Wz");
        U_z_ = std::make_unique<Linear>(c.hidden_size, c.hidden_size, false, &gen, name_ + "_Uz");
        V_z_ = std::make_unique<Linear>(1, c.hidden_size, false, &gen, name_ + "_Vz");
        if (W_z_->parameters().count("bias")) { // Initialize z-gate bias for faster convergence
            W_z_->get_parameters_mut().at("bias").data().setConstant(-1.0f);
        }

        W_h_ = std::make_unique<Linear>(current_cell_input_size_, c.hidden_size, true, &gen, name_ + "_Wh");
        U_h_ = std::make_unique<Linear>(c.hidden_size, c.hidden_size, false, &gen, name_ + "_Uh");
        V_h_ = std::make_unique<Linear>(1, c.hidden_size, false, &gen, name_ + "_Vh");

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
        // Initialize tensor_names_ struct (or do it in constructor if names are static)
        // This is just for caching keys, can be simplified.
#define TN(base, suffix) tensor_names_.suffix = get_cache_key(#suffix)
        TN(tensor_names_, x_hat_t); TN(tensor_names_, x_tilde_t); TN(tensor_names_, gamma_h_t);
        TN(tensor_names_, log_gamma_raw_BH); TN(tensor_names_, log_gamma_BH); TN(tensor_names_, h_decay_t);
        TN(tensor_names_, Wr_out); TN(tensor_names_, Ur_out); TN(tensor_names_, Vr_out);
        TN(tensor_names_, r_arg_t); TN(tensor_names_, r_t); TN(tensor_names_, Wz_out);
        TN(tensor_names_, Uz_out); TN(tensor_names_, Vz_out); TN(tensor_names_, z_arg_t);
        TN(tensor_names_, z_t); TN(tensor_names_, rh_prod_t); TN(tensor_names_, Wh_out);
        TN(tensor_names_, Uh_out); TN(tensor_names_, Vh_out); TN(tensor_names_, h_cand_arg_t);
        TN(tensor_names_, h_cand_t);
        if (config_.use_exponential_decay) {
            TN(tensor_names_, decay_h_param_val); TN(tensor_names_, clipped_decay_1H);
            TN(tensor_names_, softplus_decay_1H_out);
        }
#undef TN


        const Tensor& x_t = grud_in.x_t;
        const Tensor& h_prev_t = grud_in.h_prev_t;
        const Tensor& dt_t = grud_in.dt_t; // (B, 1)
        const Tensor* mask_t = grud_in.mask_t; // (B, input_size)

        cache.cache_tensor(get_cache_key("x_t_in"), x_t);
        cache.cache_tensor(get_cache_key("h_prev_t_in"), h_prev_t);
        cache.cache_tensor(get_cache_key("dt_t_in"), dt_t);
        if (mask_t) cache.cache_tensor(get_cache_key("mask_t_in"), *mask_t);

        // 1. Impute missing values (x_hat)
        Tensor x_hat_t_val = impute_linear_->forward(h_prev_t, cache, tensor_names_.x_hat_t);
        cache.cache_tensor(tensor_names_.x_hat_t, x_hat_t_val);

        // 2. Combine observed and imputed (x_tilde)
        Tensor x_tilde_t_val = x_t;
        if (mask_t) {
            Tensor ones_mask = Tensor::ones(mask_t->rows(), mask_t->cols());
            // x_tilde = m * x + (1-m) * x_hat
            x_tilde_t_val = ((*mask_t) * x_t) + ((ones_mask - (*mask_t)) * x_hat_t_val);
        }
        cache.cache_tensor(tensor_names_.x_tilde_t, x_tilde_t_val);

        // 3. Decay hidden state (h_decay)
        Tensor h_decay_t_val = h_prev_t; // Start with h_prev_t
        Tensor gamma_h_t_tensor_val; // Will be (B, H)
        if(config_.use_exponential_decay){
            Tensor decay_h_p_1H = parameters_.at("decay_h_param"); // (1, H)
            cache.cache_tensor(tensor_names_.decay_h_param_val, decay_h_p_1H);

            Tensor clipped_decay_1H = activations::softclip(decay_h_p_1H, config_.softclip_threshold);
            cache.cache_tensor(tensor_names_.clipped_decay_1H, clipped_decay_1H);

            // Softplus layer for decay_h_p. Output is (1,H)
            Tensor softplus_decay_1H_out_val = decay_softplus_activation_->forward(clipped_decay_1H, cache, tensor_names_.softplus_decay_1H_out);
            cache.cache_tensor(tensor_names_.softplus_decay_1H_out, softplus_decay_1H_out_val);

            // log_gamma = dt * (-softplus(decay_param))
            // dt_t (B,1), softplus_decay_1H_out_val (1,H) -> matmul -> (B,H)
            Tensor log_gamma_raw_BH_val = dt_t.matmul(softplus_decay_1H_out_val * -1.0f);
            cache.cache_tensor(tensor_names_.log_gamma_raw_BH, log_gamma_raw_BH_val);

            // Clip log_gamma to prevent extreme values in exp()
            Tensor log_gamma_BH_val = log_gamma_raw_BH_val.cwiseMax(config_.min_log_gamma).cwiseMin(-1e-4f); // Avoid exp(0)=1 exactly if not intended
            cache.cache_tensor(tensor_names_.log_gamma_BH, log_gamma_BH_val);

            gamma_h_t_tensor_val = log_gamma_BH_val.exp(); // (B,H)
            cache.cache_tensor(tensor_names_.gamma_h_t, gamma_h_t_tensor_val);

            h_decay_t_val = h_prev_t * gamma_h_t_tensor_val; // Element-wise (B,H) * (B,H)
        }
        cache.cache_tensor(tensor_names_.h_decay_t, h_decay_t_val);

        // 4. Gates (r_t, z_t)
        // Reset gate r_t
        Tensor Wr_out_val = W_r_->forward(x_tilde_t_val, cache, tensor_names_.Wr_out); // (B,H)
        Tensor Ur_out_val = U_r_->forward(h_decay_t_val, cache, tensor_names_.Ur_out); // (B,H)
        Tensor Vr_out_val = V_r_->forward(dt_t, cache, tensor_names_.Vr_out);       // (B,H) (dt_t (B,1) broadcasted by Linear layer's matmul)
        Tensor r_arg_t_val = Wr_out_val + Ur_out_val + Vr_out_val;
        cache.cache_tensor(tensor_names_.r_arg_t, r_arg_t_val);
        Tensor r_t_val = r_gate_activation_->forward(r_arg_t_val, cache, tensor_names_.r_t);
        cache.cache_tensor(tensor_names_.r_t, r_t_val);

        // Update gate z_t
        Tensor Wz_out_val = W_z_->forward(x_tilde_t_val, cache, tensor_names_.Wz_out);
        Tensor Uz_out_val = U_z_->forward(h_decay_t_val, cache, tensor_names_.Uz_out);
        Tensor Vz_out_val = V_z_->forward(dt_t, cache, tensor_names_.Vz_out);
        Tensor z_arg_t_val = Wz_out_val + Uz_out_val + Vz_out_val;
        cache.cache_tensor(tensor_names_.z_arg_t, z_arg_t_val);
        Tensor z_t_val = z_gate_activation_->forward(z_arg_t_val, cache, tensor_names_.z_t);
        cache.cache_tensor(tensor_names_.z_t, z_t_val);

        // 5. Candidate hidden state (h_cand_t)
        Tensor rh_prod_t_val = r_t_val * h_decay_t_val; // Element-wise (B,H)
        cache.cache_tensor(tensor_names_.rh_prod_t, rh_prod_t_val);

        Tensor Wh_out_val = W_h_->forward(x_tilde_t_val, cache, tensor_names_.Wh_out);
        Tensor Uh_out_val = U_h_->forward(rh_prod_t_val, cache, tensor_names_.Uh_out); // Uh uses (r_t * h_decay_t)
        Tensor Vh_out_val = V_h_->forward(dt_t, cache, tensor_names_.Vh_out);
        Tensor h_cand_arg_t_val = Wh_out_val + Uh_out_val + Vh_out_val;
        cache.cache_tensor(tensor_names_.h_cand_arg_t, h_cand_arg_t_val);
        Tensor h_cand_t_val = h_candidate_activation_->forward(h_cand_arg_t_val, cache, tensor_names_.h_cand_t);
        cache.cache_tensor(tensor_names_.h_cand_t, h_cand_t_val);

        // 6. New hidden state (h_new_t)
        // h_new = (1-z) * h_decay + z * h_cand
        Tensor ones_zt_shape = Tensor::ones(z_t_val.rows(), z_t_val.cols()); // To match shape of z_t
        Tensor term1_hnew = (ones_zt_shape - z_t_val) * h_decay_t_val;
        Tensor term2_hnew = z_t_val * h_cand_t_val;
        Tensor h_new_t = term1_hnew + term2_hnew;

        return h_new_t; // This will be named h_new_output_name_for_grad by the caller (TemporalPredictor)
    }

    // This is a dummy implementation for Layer interface, GRUDCell uses forward_step
    Tensor forward(const Tensor& input, GradientCache& cache, const std::string& output_tensor_name) override {
        throw std::logic_error("GRUDCell::forward should not be called. Use forward_step via TemporalPredictor.");
    }

    struct GRUDBackwardOutput {
        Tensor dL_dH_prev_t;
        Tensor dL_dX_t_input; // Gradient w.r.t. the original x_t input to GRUD cell for this step
        Tensor dL_dDt_input;  // Gradient w.r.t. dt_t input
    };

    GRUDBackwardOutput grud_backward_step(const Tensor& dL_dH_new, GradientCache& cache) {
        bool specific_debug = DEBUG_GRUD_Wz_GRAD && (this->name() == "focused_tp_check_L0_gru_cell");

        if (specific_debug) print_tensor_debug_Terrasse(this->name() + " dL_dH_new (input)", dL_dH_new, 2,3, true);

        // Retrieve cached tensors from forward pass
        Tensor x_t_orig = cache.get_cached_tensor(get_cache_key("x_t_in"));
        Tensor h_prev_t_orig = cache.get_cached_tensor(get_cache_key("h_prev_t_in"));
        Tensor dt_t_orig = cache.get_cached_tensor(get_cache_key("dt_t_in"));
        const Tensor* mask_t_ptr = cache.has_cached_tensor(get_cache_key("mask_t_in")) ? &cache.get_cached_tensor(get_cache_key("mask_t_in")) : nullptr;

        Tensor x_hat_t = cache.get_cached_tensor(tensor_names_.x_hat_t);
        Tensor x_tilde_t = cache.get_cached_tensor(tensor_names_.x_tilde_t);
        Tensor h_decay_t = cache.get_cached_tensor(tensor_names_.h_decay_t);
        Tensor r_t = cache.get_cached_tensor(tensor_names_.r_t);
        Tensor z_t = cache.get_cached_tensor(tensor_names_.z_t);
        Tensor h_cand_t = cache.get_cached_tensor(tensor_names_.h_cand_t);
        // rh_prod_t not directly needed for grads, but its components (r_t, h_decay_t) are.

        // Initialize aggregate gradients
        Tensor dL_dX_t_agg = Tensor::zeros(x_t_orig.rows(), x_t_orig.cols());
        Tensor dL_dH_prev_t_agg = Tensor::zeros(h_prev_t_orig.rows(), h_prev_t_orig.cols());
        Tensor dL_dDt_agg = Tensor::zeros(dt_t_orig.rows(), dt_t_orig.cols()); // dt is (B,1)

        // --- Backpropagate through h_new_t = (1-z_t)*h_decay_t + z_t*h_cand_t ---
        // dL/dz_t = dL/dh_new * (h_cand_t - h_decay_t)
        Tensor dL_dZ_t_from_hnew = (h_cand_t - h_decay_t) * dL_dH_new;
        // dL/dh_decay_t = dL/dh_new * (1 - z_t)
        Tensor dL_dH_decay_t_from_hnew = (Tensor::ones(z_t.rows(), z_t.cols()) - z_t) * dL_dH_new;
        // dL/dh_cand_t = dL/dh_new * z_t
        Tensor dL_dH_cand_t_from_hnew = z_t * dL_dH_new;

        if (specific_debug) { /* ... debug prints ... */ }

        // Aggregate gradients (start with these contributions)
        Tensor dL_dX_tilde_agg = Tensor::zeros(x_tilde_t.rows(), x_tilde_t.cols());
        Tensor dL_dH_decay_t_agg = dL_dH_decay_t_from_hnew; // Initialize accumulator

        // --- Backpropagate through h_cand_t = tanh(Wh*x_tilde + Uh*(r*h_decay) + Vh*dt) ---
        Tensor dL_dH_cand_arg_t = h_candidate_activation_->backward_calc_and_store_grads(dL_dH_cand_t_from_hnew, cache);
        if (specific_debug) print_tensor_debug_Terrasse(this->name() + " dL_dH_cand_arg_t", dL_dH_cand_arg_t, 2,3,true);

        // Grads from h_cand_arg_t to inputs of Wh, Uh, Vh
        dL_dX_tilde_agg += W_h_->backward_calc_and_store_grads(dL_dH_cand_arg_t, cache);
        Tensor dL_dRh_prod_t_from_Uh = U_h_->backward_calc_and_store_grads(dL_dH_cand_arg_t, cache);
        dL_dDt_agg += V_h_->backward_calc_and_store_grads(dL_dH_cand_arg_t, cache);

        // --- Backpropagate through rh_prod_t = r_t * h_decay_t (input to Uh) ---
        // dL/dr_t (from rh_prod) = dL/d(rh_prod) * h_decay_t
        Tensor dL_dR_t_from_rh = dL_dRh_prod_t_from_Uh * h_decay_t;
        // dL/dh_decay_t (from rh_prod) = dL/d(rh_prod) * r_t
        dL_dH_decay_t_agg += dL_dRh_prod_t_from_Uh * r_t;

        // --- Backpropagate through z_t = sigmoid(Wz*x_tilde + Uz*h_decay + Vz*dt) ---
        Tensor dL_dZ_arg_t = z_gate_activation_->backward_calc_and_store_grads(dL_dZ_t_from_hnew, cache);
        dL_dX_tilde_agg += W_z_->backward_calc_and_store_grads(dL_dZ_arg_t, cache);
        dL_dH_decay_t_agg += U_z_->backward_calc_and_store_grads(dL_dZ_arg_t, cache);
        dL_dDt_agg += V_z_->backward_calc_and_store_grads(dL_dZ_arg_t, cache);

        // --- Backpropagate through r_t = sigmoid(Wr*x_tilde + Ur*h_decay + Vr*dt) ---
        // dL_dR_t_from_rh is the gradient for r_t
        Tensor dL_dR_arg_t = r_gate_activation_->backward_calc_and_store_grads(dL_dR_t_from_rh, cache);
        dL_dX_tilde_agg += W_r_->backward_calc_and_store_grads(dL_dR_arg_t, cache);
        dL_dH_decay_t_agg += U_r_->backward_calc_and_store_grads(dL_dR_arg_t, cache);
        dL_dDt_agg += V_r_->backward_calc_and_store_grads(dL_dR_arg_t, cache);

        // --- Backpropagate through h_decay_t = h_prev_t * gamma_h_t (or just h_prev_t if no decay) ---
        if (config_.use_exponential_decay) {
            Tensor gamma_h_t = cache.get_cached_tensor(tensor_names_.gamma_h_t); // (B,H)
            // dL/dh_prev_t (from h_decay) = dL/dh_decay_t * gamma_h_t
            dL_dH_prev_t_agg += dL_dH_decay_t_agg * gamma_h_t;
            // dL/dgamma_h_t = dL/dh_decay_t * h_prev_t
            Tensor dL_dGamma_h_t = dL_dH_decay_t_agg * h_prev_t_orig; // (B,H)

            // Backprop through gamma_h_t = exp(log_gamma_BH)
            // dL/dlog_gamma_BH = dL/dgamma_h_t * gamma_h_t
            Tensor dL_dLog_gamma_BH = dL_dGamma_h_t * gamma_h_t; // (B,H)

            // Backprop through clipping of log_gamma_BH
            Tensor log_gamma_raw_BH_val = cache.get_cached_tensor(tensor_names_.log_gamma_raw_BH); // (B,H)
            Tensor log_gamma_BH_val = cache.get_cached_tensor(tensor_names_.log_gamma_BH); // (B,H), clipped version
            // Create a mask where clipping occurred. grad passes if not clipped.
            Eigen::MatrixXf clip_mask_data = (log_gamma_raw_BH_val.data().array() == log_gamma_BH_val.data().array()).template cast<float>();
            Tensor clip_mask = Tensor(clip_mask_data); // (B,H)
            Tensor dL_dLog_gamma_raw_BH = dL_dLog_gamma_BH * clip_mask; // (B,H)

            // Backprop through log_gamma_raw_BH = dt_t * (-softplus_decay_1H_out_val)
            // dt_t is (B,1), softplus_decay_1H_out_val is (1,H)
            // Let W_decay = (-softplus_decay_1H_out_val) which is (1,H)
            // log_gamma_raw_BH = dt_t.matmul(W_decay)
            // dL/ddt_t (from log_gamma) = dL/dlog_gamma_raw_BH .matmul (W_decay.transpose())
            // dL/dW_decay (from log_gamma) = dt_t.transpose() .matmul (dL/dlog_gamma_raw_BH)
            Tensor softplus_decay_1H_out_val = cache.get_cached_tensor(tensor_names_.softplus_decay_1H_out); // (1,H)
            Tensor W_decay = softplus_decay_1H_out_val * -1.0f; // (1,H)
            dL_dDt_agg += dL_dLog_gamma_raw_BH.matmul(W_decay.transpose()); // (B,H)x(H,1) -> (B,1)

            Tensor dL_dW_decay_1H = dt_t_orig.transpose().matmul(dL_dLog_gamma_raw_BH); // (1,B)x(B,H) -> (1,H)
            Tensor dL_dSoftplus_decay_1H = dL_dW_decay_1H * -1.0f; // (1,H)

            // Backprop through softplus_decay_1H_out_val = softplus_layer(clipped_decay_1H)
            Tensor dL_dClipped_decay_1H = decay_softplus_activation_->backward_calc_and_store_grads(dL_dSoftplus_decay_1H, cache); // (1,H)

            // Backprop through clipped_decay_1H = softclip(decay_h_param, thresh)
            Tensor decay_h_param_val = cache.get_cached_tensor(tensor_names_.decay_h_param_val); // (1,H)
            Tensor d_softclip_dx = activations::softclip_derivative_wrt_input(decay_h_param_val, config_.softclip_threshold); // (1,H)
            Tensor dL_dDecay_h_param = dL_dClipped_decay_1H * d_softclip_dx; // (1,H)
            gradients_.at("decay_h_param") += dL_dDecay_h_param;
        } else {
            // If no exponential decay, h_decay_t is just h_prev_t
            dL_dH_prev_t_agg += dL_dH_decay_t_agg;
        }

        // --- Backpropagate through x_tilde_t = m*x_t + (1-m)*x_hat_t ---
        Tensor dL_dX_hat_t; // Will be (B, input_size)
        if (mask_t_ptr) {
            dL_dX_t_agg += dL_dX_tilde_agg * (*mask_t_ptr);
            Tensor ones_mask = Tensor::ones(mask_t_ptr->rows(), mask_t_ptr->cols());
            dL_dX_hat_t = dL_dX_tilde_agg * (ones_mask - (*mask_t_ptr));
        } else {
            // If no mask, x_tilde_t is x_t, so dL_dX_tilde_agg is for x_t
            dL_dX_t_agg += dL_dX_tilde_agg;
            // And x_hat_t path receives no gradient from x_tilde if no mask was used.
            dL_dX_hat_t = Tensor::zeros(x_hat_t.rows(), x_hat_t.cols());
        }

        // --- Backpropagate through x_hat_t = impute_linear(h_prev_t) ---
        Tensor dL_dH_prev_t_from_impute = impute_linear_->backward_calc_and_store_grads(dL_dX_hat_t, cache);
        dL_dH_prev_t_agg += dL_dH_prev_t_from_impute;

        return {dL_dH_prev_t_agg, dL_dX_t_agg, dL_dDt_agg};
    }


    // This is called by TemporalPredictor's backward loop if GRUDCell itself is treated as one layer.
    // However, TemporalPredictor adds backward functions for GRUDCells more granularly.
    // This function ensures GRUDCell can still behave as a Layer if needed.
    Tensor backward_calc_and_store_grads(const Tensor& dL_dOutput, GradientCache& cache) override {
        // dL_dOutput here is dL_dH_new for this GRUD cell
        GRUDBackwardOutput grads = grud_backward_step(dL_dOutput, cache);
        // The main "output" gradient for the *previous* step's hidden state is dL_dH_prev_t.
        // Gradients for x_t and dt_t are handled by accumulating them in the cache
        // by the TemporalPredictor's backward function logic if needed.
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
        return sub_layers;
    }
    void zero_grad_all_sub_layers() {
        for(auto* layer : get_all_sub_layers_for_optimizer()){
            layer->zero_grad();
        }
        this->zero_grad(); // Also zero grad for params owned by GRUDCell itself (decay_h_param)
    }
    void set_training_mode_all_sub_layers(bool training) {
        for(auto* layer : get_all_sub_layers_for_optimizer()){
            layer->set_training_mode(training);
        }
        this->set_training_mode(training);
    }
};


class Loss {
public:
    virtual ~Loss()=default;
    virtual float forward(const Tensor& prediction, const Tensor& target) = 0;
    // prediction_grad_name is the key in the cache where dL/dPrediction should be stored
    virtual void backward(const Tensor& prediction, const Tensor& target, GradientCache& cache, const std::string& prediction_grad_name) = 0;
};

class MSELoss : public Loss {
public:
    float forward(const Tensor& prediction, const Tensor& target) override {
        if (prediction.shape() != target.shape()) {
            throw std::runtime_error("MSELoss: Prediction and target shape mismatch. Pred: (" +
                                     std::to_string(prediction.rows()) + "," + std::to_string(prediction.cols()) + "), Target: (" +
                                     std::to_string(target.rows()) + "," + std::to_string(target.cols()) + ")");
        }
        Tensor diff = prediction - target; // Expression
        Tensor diff_sq = diff * diff;    // Expression
        // .mean(-1) returns a Tensor, .data()(0,0) extracts the scalar float
        return 0.5f * diff_sq.mean(-1).data()(0,0);
    }

    void backward(const Tensor& prediction, const Tensor& target, GradientCache& cache, const std::string& prediction_grad_name) override {
        if (prediction.shape() != target.shape()) {
            throw std::runtime_error("MSELoss backward: Prediction and target shape mismatch.");
        }
        Tensor diff = prediction - target; // Expression
        float num_elements = static_cast<float>(prediction.rows() * prediction.cols());
        if (num_elements == 0) throw std::runtime_error("Cannot compute MSELoss backward with zero elements in prediction.");
        // grad = diff / num_elements
        Tensor grad = diff / num_elements; // Expression, result is Tensor
        cache.set_gradient(prediction_grad_name, grad);
    }
};