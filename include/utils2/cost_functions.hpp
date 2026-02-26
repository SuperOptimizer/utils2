#pragma once
#include <array>
#include <vector>
#include <cstddef>
#include <cmath>
#include <concepts>
#include <algorithm>
#include <numeric>
#include <functional>
#include <limits>
#include <optional>
#include <span>
#include <type_traits>

namespace utils2 {

// ---------------------------------------------------------------------------
// Dual number for forward-mode automatic differentiation
// ---------------------------------------------------------------------------

template<std::floating_point T, std::size_t N>
struct Dual {
    T value;
    std::array<T, N> derivatives;

    constexpr Dual() noexcept : value{}, derivatives{} {}

    constexpr Dual(T v) noexcept : value{v}, derivatives{} {}

    constexpr Dual(T v, std::size_t deriv_index) noexcept
        : value{v}, derivatives{} {
        if (deriv_index < N) {
            derivatives[deriv_index] = T(1);
        }
    }

    // Arithmetic operators

    constexpr Dual operator+(const Dual& o) const noexcept {
        Dual r;
        r.value = value + o.value;
        for (std::size_t i = 0; i < N; ++i)
            r.derivatives[i] = derivatives[i] + o.derivatives[i];
        return r;
    }

    constexpr Dual operator-(const Dual& o) const noexcept {
        Dual r;
        r.value = value - o.value;
        for (std::size_t i = 0; i < N; ++i)
            r.derivatives[i] = derivatives[i] - o.derivatives[i];
        return r;
    }

    constexpr Dual operator*(const Dual& o) const noexcept {
        Dual r;
        r.value = value * o.value;
        for (std::size_t i = 0; i < N; ++i)
            r.derivatives[i] = derivatives[i] * o.value + value * o.derivatives[i];
        return r;
    }

    constexpr Dual operator/(const Dual& o) const noexcept {
        Dual r;
        r.value = value / o.value;
        T denom = o.value * o.value;
        for (std::size_t i = 0; i < N; ++i)
            r.derivatives[i] = (derivatives[i] * o.value - value * o.derivatives[i]) / denom;
        return r;
    }

    constexpr Dual operator-() const noexcept {
        Dual r;
        r.value = -value;
        for (std::size_t i = 0; i < N; ++i)
            r.derivatives[i] = -derivatives[i];
        return r;
    }

    constexpr Dual& operator+=(const Dual& o) noexcept { return *this = *this + o; }
    constexpr Dual& operator-=(const Dual& o) noexcept { return *this = *this - o; }
    constexpr Dual& operator*=(const Dual& o) noexcept { return *this = *this * o; }
    constexpr Dual& operator/=(const Dual& o) noexcept { return *this = *this / o; }

    // Comparison (on value only)
    constexpr auto operator<=>(const Dual& o) const noexcept { return value <=> o.value; }
    constexpr bool operator==(const Dual& o) const noexcept { return value == o.value; }
};

// ---------------------------------------------------------------------------
// Math functions for Dual numbers
// ---------------------------------------------------------------------------

template<std::floating_point T, std::size_t N>
constexpr Dual<T, N> sqrt(const Dual<T, N>& x) noexcept {
    using std::sqrt;
    Dual<T, N> r;
    r.value = sqrt(x.value);
    T half_inv = T(0.5) / r.value;
    for (std::size_t i = 0; i < N; ++i)
        r.derivatives[i] = x.derivatives[i] * half_inv;
    return r;
}

template<std::floating_point T, std::size_t N>
constexpr Dual<T, N> sin(const Dual<T, N>& x) noexcept {
    using std::sin; using std::cos;
    Dual<T, N> r;
    r.value = sin(x.value);
    T c = cos(x.value);
    for (std::size_t i = 0; i < N; ++i)
        r.derivatives[i] = x.derivatives[i] * c;
    return r;
}

template<std::floating_point T, std::size_t N>
constexpr Dual<T, N> cos(const Dual<T, N>& x) noexcept {
    using std::sin; using std::cos;
    Dual<T, N> r;
    r.value = cos(x.value);
    T s = -sin(x.value);
    for (std::size_t i = 0; i < N; ++i)
        r.derivatives[i] = x.derivatives[i] * s;
    return r;
}

template<std::floating_point T, std::size_t N>
constexpr Dual<T, N> exp(const Dual<T, N>& x) noexcept {
    using std::exp;
    Dual<T, N> r;
    r.value = exp(x.value);
    for (std::size_t i = 0; i < N; ++i)
        r.derivatives[i] = x.derivatives[i] * r.value;
    return r;
}

template<std::floating_point T, std::size_t N>
constexpr Dual<T, N> log(const Dual<T, N>& x) noexcept {
    using std::log;
    Dual<T, N> r;
    r.value = log(x.value);
    T inv = T(1) / x.value;
    for (std::size_t i = 0; i < N; ++i)
        r.derivatives[i] = x.derivatives[i] * inv;
    return r;
}

template<std::floating_point T, std::size_t N>
constexpr Dual<T, N> abs(const Dual<T, N>& x) noexcept {
    using std::abs;
    if (x.value >= T(0)) return x;
    return -x;
}

template<std::floating_point T, std::size_t N>
constexpr Dual<T, N> pow(const Dual<T, N>& base, const Dual<T, N>& exponent) noexcept {
    // d/d[...] base^exp = base^exp * (exp' * ln(base) + exp * base'/base)
    using std::pow; using std::log;
    Dual<T, N> r;
    r.value = pow(base.value, exponent.value);
    T log_base = log(base.value);
    T exp_over_base = exponent.value / base.value;
    for (std::size_t i = 0; i < N; ++i)
        r.derivatives[i] = r.value * (exponent.derivatives[i] * log_base
                                      + exp_over_base * base.derivatives[i]);
    return r;
}

template<std::floating_point T, std::size_t N>
constexpr Dual<T, N> atan2(const Dual<T, N>& y, const Dual<T, N>& x) noexcept {
    using std::atan2;
    Dual<T, N> r;
    r.value = atan2(y.value, x.value);
    T denom = x.value * x.value + y.value * y.value;
    for (std::size_t i = 0; i < N; ++i)
        r.derivatives[i] = (x.value * y.derivatives[i] - y.value * x.derivatives[i]) / denom;
    return r;
}

// ---------------------------------------------------------------------------
// Auto-diff helpers
// ---------------------------------------------------------------------------

/// Compute function value and gradient via forward-mode AD.
template<std::size_t N, std::floating_point T = double, typename F>
[[nodiscard]] constexpr std::pair<T, std::array<T, N>>
autodiff(F&& func, const std::array<T, N>& params) {
    std::array<Dual<T, N>, N> dual_params;
    for (std::size_t i = 0; i < N; ++i)
        dual_params[i] = Dual<T, N>(params[i], i);

    Dual<T, N> result = std::invoke(std::forward<F>(func), dual_params);
    return {result.value, result.derivatives};
}

/// Compute the Jacobian of a vector-valued function.
/// func(dual_params) must return std::array<Dual<T,N>, NumResiduals>.
template<std::size_t NumParams, std::size_t NumResiduals,
         std::floating_point T = double, typename F>
[[nodiscard]] constexpr auto jacobian(F&& func, const std::array<T, NumParams>& params)
    -> std::array<std::array<T, NumParams>, NumResiduals>
{
    std::array<Dual<T, NumParams>, NumParams> dual_params;
    for (std::size_t i = 0; i < NumParams; ++i)
        dual_params[i] = Dual<T, NumParams>(params[i], i);

    auto residuals = std::invoke(std::forward<F>(func), dual_params);

    std::array<std::array<T, NumParams>, NumResiduals> J{};
    for (std::size_t r = 0; r < NumResiduals; ++r)
        J[r] = residuals[r].derivatives;
    return J;
}

// ---------------------------------------------------------------------------
// Dense linear solve (Gaussian elimination with partial pivoting)
// ---------------------------------------------------------------------------

namespace detail {

/// Solve A*x = b in-place. A is n-by-n, b is length n.
/// Returns false if singular.
template<std::floating_point T>
[[nodiscard]] bool dense_solve(std::vector<T>& A, std::vector<T>& b, std::size_t n) {
    for (std::size_t col = 0; col < n; ++col) {
        // Partial pivot
        std::size_t best = col;
        T best_val = std::abs(A[col * n + col]);
        for (std::size_t row = col + 1; row < n; ++row) {
            T v = std::abs(A[row * n + col]);
            if (v > best_val) { best_val = v; best = row; }
        }
        if (best_val < std::numeric_limits<T>::epsilon() * T(100))
            return false;

        if (best != col) {
            for (std::size_t j = 0; j < n; ++j)
                std::swap(A[col * n + j], A[best * n + j]);
            std::swap(b[col], b[best]);
        }

        // Eliminate below
        T pivot = A[col * n + col];
        for (std::size_t row = col + 1; row < n; ++row) {
            T factor = A[row * n + col] / pivot;
            for (std::size_t j = col; j < n; ++j)
                A[row * n + j] -= factor * A[col * n + j];
            b[row] -= factor * b[col];
        }
    }

    // Back-substitution
    for (std::size_t i = n; i-- > 0;) {
        for (std::size_t j = i + 1; j < n; ++j)
            b[i] -= A[i * n + j] * b[j];
        b[i] /= A[i * n + i];
    }
    return true;
}

/// Compute residuals and Jacobian using forward-mode AD for dynamic sizes.
/// residual_func: span<const T> -> vector<T> for values.
/// We evaluate with Dual numbers one parameter at a time (forward-mode).
template<std::floating_point T, typename F>
void compute_jacobian_dynamic(
    F&& residual_func,
    std::span<const T> params,
    std::size_t num_params,
    std::size_t num_residuals,
    std::vector<T>& residuals_out,
    std::vector<T>& jacobian_out) // row-major num_residuals x num_params
{
    // Evaluate with T to get residual values
    residuals_out.resize(num_residuals);
    {
        auto r = residual_func(params);
        for (std::size_t i = 0; i < num_residuals; ++i)
            residuals_out[i] = r[i];
    }

    // Numerical Jacobian via central differences for dynamic-size problems
    jacobian_out.assign(num_residuals * num_params, T(0));
    const T eps = std::sqrt(std::numeric_limits<T>::epsilon());

    std::vector<T> params_plus(params.begin(), params.end());
    std::vector<T> params_minus(params.begin(), params.end());

    for (std::size_t p = 0; p < num_params; ++p) {
        T h = eps * std::max(std::abs(params[p]), T(1));
        params_plus[p] = params[p] + h;
        params_minus[p] = params[p] - h;

        auto rp = residual_func(std::span<const T>(params_plus));
        auto rm = residual_func(std::span<const T>(params_minus));

        T inv_2h = T(1) / (T(2) * h);
        for (std::size_t r = 0; r < num_residuals; ++r)
            jacobian_out[r * num_params + p] = (rp[r] - rm[r]) * inv_2h;

        params_plus[p] = params[p];
        params_minus[p] = params[p];
    }
}

} // namespace detail

// ---------------------------------------------------------------------------
// Levenberg-Marquardt solver
// ---------------------------------------------------------------------------

template<std::floating_point T = double>
struct LMConfig {
    std::size_t max_iterations = 100;
    T tolerance = T(1e-8);
    T initial_lambda = T(1e-3);
    T lambda_factor = T(10);
};

template<std::floating_point T = double>
struct LMResult {
    std::vector<T> params;
    T final_cost;
    std::size_t iterations;
    bool converged;
};

/// Solve a nonlinear least-squares problem: minimize 0.5 * sum(r_i^2).
/// residual_func(span<const T>) -> vector<T>  returns the residual vector.
template<std::floating_point T = double, typename F>
[[nodiscard]] LMResult<T> levenberg_marquardt(
    F&& residual_func,
    std::vector<T> initial_params,
    std::size_t num_residuals,
    LMConfig<T> config = {})
{
    const std::size_t np = initial_params.size();
    const std::size_t nr = num_residuals;

    LMResult<T> result;
    result.params = std::move(initial_params);
    result.converged = false;
    result.iterations = 0;

    T lambda = config.initial_lambda;

    // Compute initial residuals and cost
    std::vector<T> residuals, jac;
    detail::compute_jacobian_dynamic(
        residual_func, std::span<const T>(result.params), np, nr, residuals, jac);

    auto compute_cost = [&](const std::vector<T>& r) {
        T c = T(0);
        for (auto v : r) c += v * v;
        return T(0.5) * c;
    };

    T cost = compute_cost(residuals);

    for (std::size_t iter = 0; iter < config.max_iterations; ++iter) {
        result.iterations = iter + 1;

        // Build J^T J and J^T r
        std::vector<T> JtJ(np * np, T(0));
        std::vector<T> Jtr(np, T(0));

        for (std::size_t r = 0; r < nr; ++r) {
            for (std::size_t i = 0; i < np; ++i) {
                T ji = jac[r * np + i];
                Jtr[i] += ji * residuals[r];
                for (std::size_t j = 0; j < np; ++j)
                    JtJ[i * np + j] += ji * jac[r * np + j];
            }
        }

        // Check gradient convergence
        T grad_norm = T(0);
        for (auto g : Jtr) grad_norm += g * g;
        if (std::sqrt(grad_norm) < config.tolerance) {
            result.converged = true;
            break;
        }

        // Damped system: (J^T J + lambda * diag(J^T J)) * step = -J^T r
        std::vector<T> A = JtJ;
        std::vector<T> rhs(np);
        for (std::size_t i = 0; i < np; ++i) {
            A[i * np + i] += lambda * std::max(JtJ[i * np + i], T(1e-6));
            rhs[i] = -Jtr[i];
        }

        if (!detail::dense_solve(A, rhs, np)) {
            lambda *= config.lambda_factor;
            continue;
        }

        // Trial step
        std::vector<T> trial_params(np);
        for (std::size_t i = 0; i < np; ++i)
            trial_params[i] = result.params[i] + rhs[i];

        std::vector<T> trial_residuals, trial_jac;
        detail::compute_jacobian_dynamic(
            residual_func, std::span<const T>(trial_params), np, nr, trial_residuals, trial_jac);

        T trial_cost = compute_cost(trial_residuals);

        if (trial_cost < cost) {
            // Accept step
            result.params = std::move(trial_params);
            residuals = std::move(trial_residuals);
            jac = std::move(trial_jac);

            T improvement = cost - trial_cost;
            cost = trial_cost;
            lambda /= config.lambda_factor;

            if (improvement < config.tolerance * cost) {
                result.converged = true;
                break;
            }
        } else {
            // Reject step, increase damping
            lambda *= config.lambda_factor;
        }
    }

    result.final_cost = cost;
    return result;
}

// ---------------------------------------------------------------------------
// Common cost functions
// ---------------------------------------------------------------------------

/// Distance cost: residual = point - target (per dimension).
template<std::floating_point T, std::size_t Dims>
struct DistanceCost {
    std::array<T, Dims> target;

    /// Evaluate: returns squared distance.
    /// params points to Dims values representing the point.
    template<typename U>
    constexpr U operator()(const U* params) const noexcept {
        U total{};
        for (std::size_t i = 0; i < Dims; ++i) {
            U diff = params[i] - U(target[i]);
            total = total + diff * diff;
        }
        return total;
    }
};

/// Regularization cost: ||params||^2 * weight.
template<std::floating_point T>
struct RegularizationCost {
    T weight;

    template<typename U>
    constexpr U operator()(const U* params, std::size_t n) const noexcept {
        U total{};
        for (std::size_t i = 0; i < n; ++i)
            total = total + params[i] * params[i];
        return total * U(weight);
    }
};

// ---------------------------------------------------------------------------
// Gradient descent
// ---------------------------------------------------------------------------

/// Simple gradient descent optimizer for fixed-size parameter vectors.
/// objective: std::array<Dual<T,N>, N> -> Dual<T,N>
template<std::size_t N, std::floating_point T = double, typename F>
[[nodiscard]] std::array<T, N> gradient_descent(
    F&& objective,
    std::array<T, N> initial,
    T learning_rate = T(0.01),
    std::size_t max_iterations = 1000,
    T tolerance = T(1e-8))
{
    auto params = initial;
    T prev_cost = std::numeric_limits<T>::max();

    for (std::size_t iter = 0; iter < max_iterations; ++iter) {
        auto [cost, grad] = autodiff<N, T>(std::forward<F>(objective), params);

        T grad_norm_sq = T(0);
        for (std::size_t i = 0; i < N; ++i)
            grad_norm_sq += grad[i] * grad[i];

        if (std::sqrt(grad_norm_sq) < tolerance)
            break;

        if (std::abs(prev_cost - cost) < tolerance && iter > 0)
            break;

        for (std::size_t i = 0; i < N; ++i)
            params[i] -= learning_rate * grad[i];

        prev_cost = cost;
    }
    return params;
}

} // namespace utils2
