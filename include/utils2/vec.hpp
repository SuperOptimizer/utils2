#pragma once
#include <array>
#include <cmath>
#include <concepts>
#include <numeric>
#include <algorithm>
#include <type_traits>
#include <cstddef>
#include <compare>

namespace utils2 {

// ============================================================================
// Concepts
// ============================================================================

template<typename T>
concept Arithmetic = std::is_arithmetic_v<T>;

template<typename T>
concept FloatingPoint = std::floating_point<T>;

// Forward declarations
template<Arithmetic T, std::size_t N>
struct Vec;

template<Arithmetic T, std::size_t Rows, std::size_t Cols>
struct Mat;

template<typename V, typename T, std::size_t N>
concept VecLike = requires(const V& v) {
    { v[std::size_t{0}] } -> std::convertible_to<T>;
    requires sizeof(V) >= sizeof(T) * N;
};

// ============================================================================
// Vec<T, N> - N-dimensional vector
// ============================================================================

template<Arithmetic T, std::size_t N>
struct Vec {
    std::array<T, N> data{};

    // Default constructors
    constexpr Vec() noexcept = default;

    // Construct from N scalar arguments
    template<typename... Args>
        requires (sizeof...(Args) == N && (std::convertible_to<Args, T> && ...))
    constexpr Vec(Args... args) noexcept : data{static_cast<T>(args)...} {}

    // Construct from std::array
    constexpr explicit Vec(const std::array<T, N>& arr) noexcept : data{arr} {}

    // Fill constructor
    constexpr explicit Vec(T val) noexcept {
        data.fill(val);
    }

    // Element access
    [[nodiscard]] constexpr T& operator[](std::size_t i) noexcept { return data[i]; }
    [[nodiscard]] constexpr const T& operator[](std::size_t i) const noexcept { return data[i]; }

    [[nodiscard]] static constexpr std::size_t size() noexcept { return N; }

    // Iterators
    [[nodiscard]] constexpr auto begin() noexcept { return data.begin(); }
    [[nodiscard]] constexpr auto end() noexcept { return data.end(); }
    [[nodiscard]] constexpr auto begin() const noexcept { return data.begin(); }
    [[nodiscard]] constexpr auto end() const noexcept { return data.end(); }

    // Arithmetic: Vec op Vec
    [[nodiscard]] constexpr Vec operator+(const Vec& rhs) const noexcept {
        Vec result;
        for (std::size_t i = 0; i < N; ++i)
            result.data[i] = data[i] + rhs.data[i];
        return result;
    }

    [[nodiscard]] constexpr Vec operator-(const Vec& rhs) const noexcept {
        Vec result;
        for (std::size_t i = 0; i < N; ++i)
            result.data[i] = data[i] - rhs.data[i];
        return result;
    }

    [[nodiscard]] constexpr Vec operator*(const Vec& rhs) const noexcept {
        Vec result;
        for (std::size_t i = 0; i < N; ++i)
            result.data[i] = data[i] * rhs.data[i];
        return result;
    }

    [[nodiscard]] constexpr Vec operator/(const Vec& rhs) const noexcept {
        Vec result;
        for (std::size_t i = 0; i < N; ++i)
            result.data[i] = data[i] / rhs.data[i];
        return result;
    }

    // Arithmetic: Vec op scalar
    [[nodiscard]] constexpr Vec operator+(T s) const noexcept {
        Vec result;
        for (std::size_t i = 0; i < N; ++i)
            result.data[i] = data[i] + s;
        return result;
    }

    [[nodiscard]] constexpr Vec operator-(T s) const noexcept {
        Vec result;
        for (std::size_t i = 0; i < N; ++i)
            result.data[i] = data[i] - s;
        return result;
    }

    [[nodiscard]] constexpr Vec operator*(T s) const noexcept {
        Vec result;
        for (std::size_t i = 0; i < N; ++i)
            result.data[i] = data[i] * s;
        return result;
    }

    [[nodiscard]] constexpr Vec operator/(T s) const noexcept {
        Vec result;
        for (std::size_t i = 0; i < N; ++i)
            result.data[i] = data[i] / s;
        return result;
    }

    // Unary minus
    [[nodiscard]] constexpr Vec operator-() const noexcept {
        Vec result;
        for (std::size_t i = 0; i < N; ++i)
            result.data[i] = -data[i];
        return result;
    }

    // Compound assignment: Vec op= Vec
    constexpr Vec& operator+=(const Vec& rhs) noexcept {
        for (std::size_t i = 0; i < N; ++i) data[i] += rhs.data[i];
        return *this;
    }

    constexpr Vec& operator-=(const Vec& rhs) noexcept {
        for (std::size_t i = 0; i < N; ++i) data[i] -= rhs.data[i];
        return *this;
    }

    constexpr Vec& operator*=(const Vec& rhs) noexcept {
        for (std::size_t i = 0; i < N; ++i) data[i] *= rhs.data[i];
        return *this;
    }

    constexpr Vec& operator/=(const Vec& rhs) noexcept {
        for (std::size_t i = 0; i < N; ++i) data[i] /= rhs.data[i];
        return *this;
    }

    // Compound assignment: Vec op= scalar
    constexpr Vec& operator+=(T s) noexcept {
        for (std::size_t i = 0; i < N; ++i) data[i] += s;
        return *this;
    }

    constexpr Vec& operator-=(T s) noexcept {
        for (std::size_t i = 0; i < N; ++i) data[i] -= s;
        return *this;
    }

    constexpr Vec& operator*=(T s) noexcept {
        for (std::size_t i = 0; i < N; ++i) data[i] *= s;
        return *this;
    }

    constexpr Vec& operator/=(T s) noexcept {
        for (std::size_t i = 0; i < N; ++i) data[i] /= s;
        return *this;
    }

    // Comparisons
    [[nodiscard]] constexpr bool operator==(const Vec& rhs) const noexcept = default;
    [[nodiscard]] constexpr auto operator<=>(const Vec& rhs) const noexcept = default;

    // Named accessors for convenience (available when N >= 1..4)
    [[nodiscard]] constexpr T& x() noexcept requires (N >= 1) { return data[0]; }
    [[nodiscard]] constexpr const T& x() const noexcept requires (N >= 1) { return data[0]; }
    [[nodiscard]] constexpr T& y() noexcept requires (N >= 2) { return data[1]; }
    [[nodiscard]] constexpr const T& y() const noexcept requires (N >= 2) { return data[1]; }
    [[nodiscard]] constexpr T& z() noexcept requires (N >= 3) { return data[2]; }
    [[nodiscard]] constexpr const T& z() const noexcept requires (N >= 3) { return data[2]; }
    [[nodiscard]] constexpr T& w() noexcept requires (N >= 4) { return data[3]; }
    [[nodiscard]] constexpr const T& w() const noexcept requires (N >= 4) { return data[3]; }

    // Structured bindings support
    template<std::size_t I>
        requires (I < N)
    [[nodiscard]] constexpr T& get() & noexcept { return data[I]; }

    template<std::size_t I>
        requires (I < N)
    [[nodiscard]] constexpr const T& get() const& noexcept { return data[I]; }

    template<std::size_t I>
        requires (I < N)
    [[nodiscard]] constexpr T&& get() && noexcept { return std::move(data[I]); }
};

// Scalar op Vec (commutative forms)
template<Arithmetic T, std::size_t N>
[[nodiscard]] constexpr Vec<T, N> operator*(T s, const Vec<T, N>& v) noexcept {
    return v * s;
}

template<Arithmetic T, std::size_t N>
[[nodiscard]] constexpr Vec<T, N> operator+(T s, const Vec<T, N>& v) noexcept {
    return v + s;
}

// ============================================================================
// Vec free functions
// ============================================================================

template<Arithmetic T, std::size_t N>
[[nodiscard]] constexpr T dot(const Vec<T, N>& a, const Vec<T, N>& b) noexcept {
    T result{};
    for (std::size_t i = 0; i < N; ++i)
        result += a[i] * b[i];
    return result;
}

template<Arithmetic T>
[[nodiscard]] constexpr Vec<T, 3> cross(const Vec<T, 3>& a, const Vec<T, 3>& b) noexcept {
    return {
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
    };
}

template<Arithmetic T, std::size_t N>
[[nodiscard]] constexpr T norm_sq(const Vec<T, N>& v) noexcept {
    return dot(v, v);
}

template<Arithmetic T, std::size_t N>
[[nodiscard]] constexpr T norm(const Vec<T, N>& v) noexcept {
    if consteval {
        // constexpr-friendly path: use __builtin_sqrt if available
        return static_cast<T>(__builtin_sqrt(static_cast<double>(norm_sq(v))));
    } else {
        return static_cast<T>(std::sqrt(static_cast<double>(norm_sq(v))));
    }
}

template<Arithmetic T, std::size_t N>
[[nodiscard]] constexpr T norm_l1(const Vec<T, N>& v) noexcept {
    T result{};
    for (std::size_t i = 0; i < N; ++i) {
        auto val = v[i];
        result += val < T{} ? -val : val;
    }
    return result;
}

template<Arithmetic T, std::size_t N>
[[nodiscard]] constexpr T norm_linf(const Vec<T, N>& v) noexcept {
    T result{};
    for (std::size_t i = 0; i < N; ++i) {
        auto val = v[i] < T{} ? -v[i] : v[i];
        if (val > result) result = val;
    }
    return result;
}

template<FloatingPoint T, std::size_t N>
[[nodiscard]] constexpr Vec<T, N> normalize(const Vec<T, N>& v) noexcept {
    auto len = norm(v);
    if (len == T{}) return v;
    return v / len;
}

template<Arithmetic T, std::size_t N>
[[nodiscard]] constexpr T distance_sq(const Vec<T, N>& a, const Vec<T, N>& b) noexcept {
    return norm_sq(a - b);
}

template<Arithmetic T, std::size_t N>
[[nodiscard]] constexpr T distance(const Vec<T, N>& a, const Vec<T, N>& b) noexcept {
    return norm(a - b);
}

template<FloatingPoint T, std::size_t N>
[[nodiscard]] constexpr Vec<T, N> lerp(const Vec<T, N>& a, const Vec<T, N>& b, T t) noexcept {
    return a + (b - a) * t;
}

template<Arithmetic T, std::size_t N>
[[nodiscard]] constexpr Vec<T, N> clamp(
    const Vec<T, N>& v, const Vec<T, N>& lo, const Vec<T, N>& hi) noexcept
{
    Vec<T, N> result;
    for (std::size_t i = 0; i < N; ++i) {
        result[i] = v[i] < lo[i] ? lo[i] : (v[i] > hi[i] ? hi[i] : v[i]);
    }
    return result;
}

template<Arithmetic T, std::size_t N>
[[nodiscard]] constexpr Vec<T, N> min(const Vec<T, N>& a, const Vec<T, N>& b) noexcept {
    Vec<T, N> result;
    for (std::size_t i = 0; i < N; ++i)
        result[i] = a[i] < b[i] ? a[i] : b[i];
    return result;
}

template<Arithmetic T, std::size_t N>
[[nodiscard]] constexpr Vec<T, N> max(const Vec<T, N>& a, const Vec<T, N>& b) noexcept {
    Vec<T, N> result;
    for (std::size_t i = 0; i < N; ++i)
        result[i] = a[i] > b[i] ? a[i] : b[i];
    return result;
}

template<FloatingPoint T, std::size_t N>
[[nodiscard]] constexpr T interior_angle(const Vec<T, N>& a, const Vec<T, N>& b) noexcept {
    auto d = dot(a, b) / (norm(a) * norm(b));
    // Clamp to [-1, 1] for numerical safety
    d = d < T{-1} ? T{-1} : (d > T{1} ? T{1} : d);
    if consteval {
        return static_cast<T>(__builtin_acos(static_cast<double>(d)));
    } else {
        return static_cast<T>(std::acos(static_cast<double>(d)));
    }
}

template<FloatingPoint T, std::size_t N>
[[nodiscard]] constexpr Vec<T, N> reflect(const Vec<T, N>& v, const Vec<T, N>& n) noexcept {
    return v - n * (T{2} * dot(v, n));
}

template<FloatingPoint T, std::size_t N>
[[nodiscard]] constexpr Vec<T, N> project(const Vec<T, N>& v, const Vec<T, N>& onto) noexcept {
    return onto * (dot(v, onto) / dot(onto, onto));
}

// ============================================================================
// Mat<T, Rows, Cols> - Column-major matrix
// ============================================================================

template<Arithmetic T, std::size_t Rows, std::size_t Cols>
struct Mat {
    // Column-major: array of column vectors
    std::array<Vec<T, Rows>, Cols> cols{};

    constexpr Mat() noexcept = default;

    // Construct from column vectors
    template<typename... Vecs>
        requires (sizeof...(Vecs) == Cols && (std::same_as<Vecs, Vec<T, Rows>> && ...))
    constexpr Mat(Vecs... columns) noexcept : cols{columns...} {}

    // Element access (row, col)
    [[nodiscard]] constexpr T& operator()(std::size_t row, std::size_t col) noexcept {
        return cols[col][row];
    }

    [[nodiscard]] constexpr const T& operator()(std::size_t row, std::size_t col) const noexcept {
        return cols[col][row];
    }

    // Column access
    [[nodiscard]] constexpr Vec<T, Rows>& col(std::size_t c) noexcept { return cols[c]; }
    [[nodiscard]] constexpr const Vec<T, Rows>& col(std::size_t c) const noexcept { return cols[c]; }

    // Row access (constructs a new vector)
    [[nodiscard]] constexpr Vec<T, Cols> row(std::size_t r) const noexcept {
        Vec<T, Cols> result;
        for (std::size_t c = 0; c < Cols; ++c)
            result[c] = cols[c][r];
        return result;
    }

    [[nodiscard]] static constexpr std::size_t rows() noexcept { return Rows; }
    [[nodiscard]] static constexpr std::size_t num_cols() noexcept { return Cols; }

    // Identity matrix (square only)
    [[nodiscard]] static constexpr Mat identity() noexcept requires (Rows == Cols) {
        Mat result;
        for (std::size_t i = 0; i < Rows; ++i)
            result(i, i) = T{1};
        return result;
    }

    // Transpose
    [[nodiscard]] constexpr Mat<T, Cols, Rows> transpose() const noexcept {
        Mat<T, Cols, Rows> result;
        for (std::size_t r = 0; r < Rows; ++r)
            for (std::size_t c = 0; c < Cols; ++c)
                result(c, r) = (*this)(r, c);
        return result;
    }

    // Mat + Mat
    [[nodiscard]] constexpr Mat operator+(const Mat& rhs) const noexcept {
        Mat result;
        for (std::size_t c = 0; c < Cols; ++c)
            result.cols[c] = cols[c] + rhs.cols[c];
        return result;
    }

    // Mat - Mat
    [[nodiscard]] constexpr Mat operator-(const Mat& rhs) const noexcept {
        Mat result;
        for (std::size_t c = 0; c < Cols; ++c)
            result.cols[c] = cols[c] - rhs.cols[c];
        return result;
    }

    // Mat * scalar
    [[nodiscard]] constexpr Mat operator*(T s) const noexcept {
        Mat result;
        for (std::size_t c = 0; c < Cols; ++c)
            result.cols[c] = cols[c] * s;
        return result;
    }

    // Mat / scalar
    [[nodiscard]] constexpr Mat operator/(T s) const noexcept {
        Mat result;
        for (std::size_t c = 0; c < Cols; ++c)
            result.cols[c] = cols[c] / s;
        return result;
    }

    constexpr Mat& operator+=(const Mat& rhs) noexcept {
        for (std::size_t c = 0; c < Cols; ++c) cols[c] += rhs.cols[c];
        return *this;
    }

    constexpr Mat& operator-=(const Mat& rhs) noexcept {
        for (std::size_t c = 0; c < Cols; ++c) cols[c] -= rhs.cols[c];
        return *this;
    }

    constexpr Mat& operator*=(T s) noexcept {
        for (std::size_t c = 0; c < Cols; ++c) cols[c] *= s;
        return *this;
    }

    constexpr Mat& operator/=(T s) noexcept {
        for (std::size_t c = 0; c < Cols; ++c) cols[c] /= s;
        return *this;
    }

    // Comparisons
    [[nodiscard]] constexpr bool operator==(const Mat& rhs) const noexcept = default;

    // Unary minus
    [[nodiscard]] constexpr Mat operator-() const noexcept {
        Mat result;
        for (std::size_t c = 0; c < Cols; ++c)
            result.cols[c] = -cols[c];
        return result;
    }

    // Determinant (square matrices 2x2, 3x3, 4x4)
    [[nodiscard]] constexpr T determinant() const noexcept requires (Rows == Cols && Rows >= 2 && Rows <= 4) {
        if constexpr (Rows == 2) {
            return (*this)(0,0) * (*this)(1,1) - (*this)(0,1) * (*this)(1,0);
        } else if constexpr (Rows == 3) {
            return (*this)(0,0) * ((*this)(1,1) * (*this)(2,2) - (*this)(1,2) * (*this)(2,1))
                 - (*this)(0,1) * ((*this)(1,0) * (*this)(2,2) - (*this)(1,2) * (*this)(2,0))
                 + (*this)(0,2) * ((*this)(1,0) * (*this)(2,1) - (*this)(1,1) * (*this)(2,0));
        } else { // 4x4
            const auto& m = *this;
            T s0 = m(0,0) * m(1,1) - m(1,0) * m(0,1);
            T s1 = m(0,0) * m(1,2) - m(1,0) * m(0,2);
            T s2 = m(0,0) * m(1,3) - m(1,0) * m(0,3);
            T s3 = m(0,1) * m(1,2) - m(1,1) * m(0,2);
            T s4 = m(0,1) * m(1,3) - m(1,1) * m(0,3);
            T s5 = m(0,2) * m(1,3) - m(1,2) * m(0,3);

            T c5 = m(2,2) * m(3,3) - m(3,2) * m(2,3);
            T c4 = m(2,1) * m(3,3) - m(3,1) * m(2,3);
            T c3 = m(2,1) * m(3,2) - m(3,1) * m(2,2);
            T c2 = m(2,0) * m(3,3) - m(3,0) * m(2,3);
            T c1 = m(2,0) * m(3,2) - m(3,0) * m(2,2);
            T c0 = m(2,0) * m(3,1) - m(3,0) * m(2,1);

            return s0 * c5 - s1 * c4 + s2 * c3 + s3 * c2 - s4 * c1 + s5 * c0;
        }
    }

    // Inverse (square matrices 2x2, 3x3, 4x4)
    [[nodiscard]] constexpr Mat inverse() const noexcept requires (Rows == Cols && Rows >= 2 && Rows <= 4) {
        if constexpr (Rows == 2) {
            T det = determinant();
            T inv_det = T{1} / det;
            Mat result;
            result(0,0) =  (*this)(1,1) * inv_det;
            result(0,1) = -(*this)(0,1) * inv_det;
            result(1,0) = -(*this)(1,0) * inv_det;
            result(1,1) =  (*this)(0,0) * inv_det;
            return result;
        } else if constexpr (Rows == 3) {
            const auto& m = *this;
            Mat result;
            T det = determinant();
            T inv_det = T{1} / det;

            result(0,0) = (m(1,1) * m(2,2) - m(1,2) * m(2,1)) * inv_det;
            result(0,1) = (m(0,2) * m(2,1) - m(0,1) * m(2,2)) * inv_det;
            result(0,2) = (m(0,1) * m(1,2) - m(0,2) * m(1,1)) * inv_det;
            result(1,0) = (m(1,2) * m(2,0) - m(1,0) * m(2,2)) * inv_det;
            result(1,1) = (m(0,0) * m(2,2) - m(0,2) * m(2,0)) * inv_det;
            result(1,2) = (m(0,2) * m(1,0) - m(0,0) * m(1,2)) * inv_det;
            result(2,0) = (m(1,0) * m(2,1) - m(1,1) * m(2,0)) * inv_det;
            result(2,1) = (m(0,1) * m(2,0) - m(0,0) * m(2,1)) * inv_det;
            result(2,2) = (m(0,0) * m(1,1) - m(0,1) * m(1,0)) * inv_det;
            return result;
        } else { // 4x4
            const auto& m = *this;
            T s0 = m(0,0) * m(1,1) - m(1,0) * m(0,1);
            T s1 = m(0,0) * m(1,2) - m(1,0) * m(0,2);
            T s2 = m(0,0) * m(1,3) - m(1,0) * m(0,3);
            T s3 = m(0,1) * m(1,2) - m(1,1) * m(0,2);
            T s4 = m(0,1) * m(1,3) - m(1,1) * m(0,3);
            T s5 = m(0,2) * m(1,3) - m(1,2) * m(0,3);

            T c5 = m(2,2) * m(3,3) - m(3,2) * m(2,3);
            T c4 = m(2,1) * m(3,3) - m(3,1) * m(2,3);
            T c3 = m(2,1) * m(3,2) - m(3,1) * m(2,2);
            T c2 = m(2,0) * m(3,3) - m(3,0) * m(2,3);
            T c1 = m(2,0) * m(3,2) - m(3,0) * m(2,2);
            T c0 = m(2,0) * m(3,1) - m(3,0) * m(2,1);

            T inv_det = T{1} / (s0 * c5 - s1 * c4 + s2 * c3 + s3 * c2 - s4 * c1 + s5 * c0);

            Mat result;
            result(0,0) = ( m(1,1) * c5 - m(1,2) * c4 + m(1,3) * c3) * inv_det;
            result(0,1) = (-m(0,1) * c5 + m(0,2) * c4 - m(0,3) * c3) * inv_det;
            result(0,2) = ( m(0,1) * s5 - m(0,2) * s4 + m(0,3) * s3) * inv_det;
            result(0,3) = (-m(0,1) * (m(1,2)*m(2,3) - m(1,3)*m(2,2))
                           + m(0,2) * (m(1,1)*m(2,3) - m(1,3)*m(2,1))
                           - m(0,3) * (m(1,1)*m(2,2) - m(1,2)*m(2,1))) * inv_det;

            result(1,0) = (-m(1,0) * c5 + m(1,2) * c2 - m(1,3) * c1) * inv_det;
            result(1,1) = ( m(0,0) * c5 - m(0,2) * c2 + m(0,3) * c1) * inv_det;
            result(1,2) = (-m(0,0) * s5 + m(0,2) * s2 - m(0,3) * s1) * inv_det;
            result(1,3) = ( m(0,0) * (m(1,2)*m(2,3) - m(1,3)*m(2,2))
                           - m(0,2) * (m(1,0)*m(2,3) - m(1,3)*m(2,0))
                           + m(0,3) * (m(1,0)*m(2,2) - m(1,2)*m(2,0))) * inv_det;

            result(2,0) = ( m(1,0) * c4 - m(1,1) * c2 + m(1,3) * c0) * inv_det;
            result(2,1) = (-m(0,0) * c4 + m(0,1) * c2 - m(0,3) * c0) * inv_det;
            result(2,2) = ( m(0,0) * s4 - m(0,1) * s2 + m(0,3) * s0) * inv_det;
            result(2,3) = (-m(0,0) * (m(1,1)*m(2,3) - m(1,3)*m(2,1))
                           + m(0,1) * (m(1,0)*m(2,3) - m(1,3)*m(2,0))
                           - m(0,3) * (m(1,0)*m(2,1) - m(1,1)*m(2,0))) * inv_det;

            result(3,0) = (-m(1,0) * c3 + m(1,1) * c1 - m(1,2) * c0) * inv_det;
            result(3,1) = ( m(0,0) * c3 - m(0,1) * c1 + m(0,2) * c0) * inv_det;
            result(3,2) = (-m(0,0) * s3 + m(0,1) * s1 - m(0,2) * s0) * inv_det;
            result(3,3) = ( m(0,0) * (m(1,1)*m(2,2) - m(1,2)*m(2,1))
                           - m(0,1) * (m(1,0)*m(2,2) - m(1,2)*m(2,0))
                           + m(0,2) * (m(1,0)*m(2,1) - m(1,1)*m(2,0))) * inv_det;
            return result;
        }
    }
};

// Scalar * Mat
template<Arithmetic T, std::size_t Rows, std::size_t Cols>
[[nodiscard]] constexpr Mat<T, Rows, Cols> operator*(T s, const Mat<T, Rows, Cols>& m) noexcept {
    return m * s;
}

// Mat * Vec
template<Arithmetic T, std::size_t Rows, std::size_t Cols>
[[nodiscard]] constexpr Vec<T, Rows> operator*(const Mat<T, Rows, Cols>& m, const Vec<T, Cols>& v) noexcept {
    Vec<T, Rows> result;
    for (std::size_t c = 0; c < Cols; ++c)
        result += m.cols[c] * v[c];
    return result;
}

// Mat * Mat
template<Arithmetic T, std::size_t M, std::size_t N, std::size_t P>
[[nodiscard]] constexpr Mat<T, M, P> operator*(const Mat<T, M, N>& a, const Mat<T, N, P>& b) noexcept {
    Mat<T, M, P> result;
    for (std::size_t c = 0; c < P; ++c)
        result.cols[c] = a * b.cols[c];
    return result;
}

// ============================================================================
// Type aliases
// ============================================================================

// Vec aliases
using Vec2f = Vec<float, 2>;
using Vec3f = Vec<float, 3>;
using Vec4f = Vec<float, 4>;

using Vec2d = Vec<double, 2>;
using Vec3d = Vec<double, 3>;
using Vec4d = Vec<double, 4>;

using Vec2i = Vec<int, 2>;
using Vec3i = Vec<int, 3>;
using Vec4i = Vec<int, 4>;

// Mat aliases
using Mat2f = Mat<float, 2, 2>;
using Mat3f = Mat<float, 3, 3>;
using Mat4f = Mat<float, 4, 4>;

using Mat2d = Mat<double, 2, 2>;
using Mat3d = Mat<double, 3, 3>;
using Mat4d = Mat<double, 4, 4>;

} // namespace utils2

// ============================================================================
// Structured bindings support (std namespace specializations)
// ============================================================================

template<typename T, std::size_t N>
    requires utils2::Arithmetic<T>
struct std::tuple_size<utils2::Vec<T, N>> : std::integral_constant<std::size_t, N> {};

template<std::size_t I, typename T, std::size_t N>
    requires (utils2::Arithmetic<T> && I < N)
struct std::tuple_element<I, utils2::Vec<T, N>> {
    using type = T;
};
