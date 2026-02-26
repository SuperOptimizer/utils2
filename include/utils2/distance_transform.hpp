#pragma once
#include "mdspan.hpp"
#include <vector>
#include <array>
#include <cstddef>
#include <cmath>
#include <concepts>
#include <algorithm>
#include <limits>
#include <numeric>
#include <span>

namespace utils2 {

namespace detail {

/// Felzenszwalb-Huttenlocher parabolic envelope on a 1D signal.
/// `f` contains the input values (0 for background, +inf for foreground),
/// and `output` receives the squared distances. `n` is the length, `sp2` is
/// spacing squared.
template<std::floating_point F>
void fh_envelope_1d(
    std::span<const F> f,
    std::span<F> output,
    std::size_t n,
    F sp2) noexcept
{
    if (n == 0) return;
    if (n == 1) {
        output[0] = f[0];
        return;
    }

    constexpr auto inf = std::numeric_limits<F>::infinity();

    // v: locations of parabolas in lower envelope
    // z: intersection points between consecutive parabolas
    std::vector<int> v(n);
    std::vector<F> z(n + 1);

    int k = 0;
    v[0] = 0;
    z[0] = -inf;
    z[1] = +inf;

    for (std::size_t i = 1; i < n; ++i) {
        const auto qi = static_cast<int>(i);
        auto compute_s = [&](int vk) -> F {
            const F fi = f[i];
            const F fv = f[vk];
            // When both are infinite, inf-inf=NaN would corrupt the envelope.
            // Both parabolas have the same (infinite) height, so the
            // intersection is at the midpoint between them.
            if (fi == inf && fv == inf)
                return static_cast<F>(qi + vk) / F(2);
            const F yi = fi + sp2 * static_cast<F>(qi) * static_cast<F>(qi);
            const F yv = fv + sp2 * static_cast<F>(vk) * static_cast<F>(vk);
            return (yi - yv) / (F(2) * sp2 * static_cast<F>(qi - vk));
        };

        F s = compute_s(v[k]);
        while (k > 0 && s <= z[k]) {
            --k;
            s = compute_s(v[k]);
        }
        ++k;
        v[k] = qi;
        z[k] = s;
        z[k + 1] = +inf;
    }

    k = 0;
    for (std::size_t i = 0; i < n; ++i) {
        while (z[k + 1] < static_cast<F>(i)) {
            ++k;
        }
        const F diff = static_cast<F>(i) - static_cast<F>(v[k]);
        output[i] = sp2 * diff * diff + f[v[k]];
    }
}

/// Apply 1D squared EDT along one axis of a flat buffer.
/// `size` is the total number of elements; `dim_len` is the extent along
/// the axis being processed; `stride` is the element stride along that axis;
/// `spacing` is the grid spacing for that axis.
template<std::floating_point F>
void apply_edt_along_axis(
    std::span<F> data,
    std::size_t size,
    std::size_t dim_len,
    std::size_t stride,
    F spacing) noexcept
{
    const F sp2 = spacing * spacing;
    const std::size_t num_lines = size / dim_len;

    std::vector<F> line_in(dim_len);
    std::vector<F> line_out(dim_len);

    // Iterate over every 1D line along this axis.
    // The outer index enumerates lines; we reconstruct the starting offset.
    for (std::size_t line = 0; line < num_lines; ++line) {
        // Determine the starting flat index for this line.
        // We need the multi-index with the processed-axis component set to 0.
        // A generic way: interpret `line` in the index space with the axis
        // dimension removed.
        //
        // For simplicity use offset arithmetic: step through all elements
        // whose axis-component is 0.
        std::size_t base = 0;
        {
            // Decompose `line` into the "rest" indices.
            // block = product of all dimensions whose stride < stride (faster dims)
            // The base offset = (line % block) + (line / block) * block * dim_len
            // where block = stride.
            std::size_t inner = line % stride;
            std::size_t outer = line / stride;
            base = inner + outer * stride * dim_len;
        }

        // Extract the line.
        for (std::size_t j = 0; j < dim_len; ++j) {
            line_in[j] = data[base + j * stride];
        }

        // Run the parabolic envelope.
        fh_envelope_1d<F>(
            std::span<const F>(line_in.data(), dim_len),
            std::span<F>(line_out.data(), dim_len),
            dim_len,
            sp2);

        // Write back.
        for (std::size_t j = 0; j < dim_len; ++j) {
            data[base + j * stride] = line_out[j];
        }
    }
}

/// Initialise a flat buffer from a binary image:
/// foreground (nonzero) -> +inf, background (zero) -> 0.
template<std::integral T, std::floating_point F>
void init_binary(const T* src, F* dst, std::size_t n) noexcept
{
    constexpr auto inf = std::numeric_limits<F>::infinity();
    for (std::size_t i = 0; i < n; ++i) {
        dst[i] = (src[i] != T(0)) ? inf : F(0);
    }
}

/// Initialise a flat buffer from a multi-label image:
/// For each pixel, 0 if any neighbor along any axis has a different label,
/// +inf otherwise.
template<std::integral T, std::floating_point F>
void init_multilabel_2d(
    std::mdspan<const T, std::dextents<std::size_t, 2>> image,
    F* dst) noexcept
{
    constexpr auto inf = std::numeric_limits<F>::infinity();
    const auto rows = image.extent(0);
    const auto cols = image.extent(1);

    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t c = 0; c < cols; ++c) {
            const T label = image[r, c];
            bool boundary = (label == T(0));
            if (!boundary && r > 0 && image[r - 1, c] != label) boundary = true;
            if (!boundary && r + 1 < rows && image[r + 1, c] != label) boundary = true;
            if (!boundary && c > 0 && image[r, c - 1] != label) boundary = true;
            if (!boundary && c + 1 < cols && image[r, c + 1] != label) boundary = true;
            dst[r * cols + c] = boundary ? F(0) : inf;
        }
    }
}

template<std::integral T, std::floating_point F>
void init_multilabel_3d(
    std::mdspan<const T, std::dextents<std::size_t, 3>> volume,
    F* dst) noexcept
{
    constexpr auto inf = std::numeric_limits<F>::infinity();
    const auto d0 = volume.extent(0);
    const auto d1 = volume.extent(1);
    const auto d2 = volume.extent(2);

    for (std::size_t z = 0; z < d0; ++z) {
        for (std::size_t y = 0; y < d1; ++y) {
            for (std::size_t x = 0; x < d2; ++x) {
                const T label = volume[z, y, x];
                bool boundary = (label == T(0));
                if (!boundary && z > 0 && volume[z - 1, y, x] != label) boundary = true;
                if (!boundary && z + 1 < d0 && volume[z + 1, y, x] != label) boundary = true;
                if (!boundary && y > 0 && volume[z, y - 1, x] != label) boundary = true;
                if (!boundary && y + 1 < d1 && volume[z, y + 1, x] != label) boundary = true;
                if (!boundary && x > 0 && volume[z, y, x - 1] != label) boundary = true;
                if (!boundary && x + 1 < d2 && volume[z, y, x + 1] != label) boundary = true;
                dst[(z * d1 + y) * d2 + x] = boundary ? F(0) : inf;
            }
        }
    }
}

} // namespace detail

// -------------------------------------------------------------------------
// 1D squared EDT
// -------------------------------------------------------------------------

/// Compute the squared Euclidean distance transform of a 1D signal.
/// Input values: 0 = background, nonzero = foreground.
/// Output: squared distance to nearest background pixel, scaled by `spacing`.
template<std::floating_point F = float>
void edt_1d_sq(
    std::span<const F> input,
    std::span<F> output,
    F spacing = F(1)) noexcept
{
    const auto n = input.size();
    constexpr auto inf = std::numeric_limits<F>::infinity();

    // Initialise: foreground -> inf, background -> 0
    std::vector<F> f(n);
    for (std::size_t i = 0; i < n; ++i) {
        f[i] = (input[i] != F(0)) ? inf : F(0);
    }

    detail::fh_envelope_1d<F>(
        std::span<const F>(f.data(), n), output, n, spacing * spacing);
}

// -------------------------------------------------------------------------
// 2D EDT
// -------------------------------------------------------------------------

/// Squared Euclidean distance transform of a 2D binary image.
/// Nonzero pixels are foreground; zero pixels are background.
template<std::integral T, std::floating_point F = float>
[[nodiscard]] std::vector<F> edt_2d_sq(
    std::mdspan<const T, std::dextents<std::size_t, 2>> image,
    std::array<F, 2> spacing = {F(1), F(1)})
{
    const auto rows = image.extent(0);
    const auto cols = image.extent(1);
    const auto n = rows * cols;

    std::vector<F> buf(n);
    detail::init_binary(image.data_handle(), buf.data(), n);

    // Apply 1D EDT along columns (axis 0): stride = cols, dim_len = rows
    detail::apply_edt_along_axis<F>(
        std::span<F>(buf), n, rows, cols, spacing[0]);

    // Apply 1D EDT along rows (axis 1): stride = 1, dim_len = cols
    detail::apply_edt_along_axis<F>(
        std::span<F>(buf), n, cols, std::size_t(1), spacing[1]);

    return buf;
}

/// Euclidean distance transform of a 2D binary image (actual distances).
template<std::integral T, std::floating_point F = float>
[[nodiscard]] std::vector<F> edt_2d(
    std::mdspan<const T, std::dextents<std::size_t, 2>> image,
    std::array<F, 2> spacing = {F(1), F(1)})
{
    auto buf = edt_2d_sq<T, F>(image, spacing);
    for (auto& v : buf) {
        v = std::sqrt(v);
    }
    return buf;
}

// -------------------------------------------------------------------------
// 3D EDT
// -------------------------------------------------------------------------

/// Squared Euclidean distance transform of a 3D binary volume.
template<std::integral T, std::floating_point F = float>
[[nodiscard]] std::vector<F> edt_3d_sq(
    std::mdspan<const T, std::dextents<std::size_t, 3>> volume,
    std::array<F, 3> spacing = {F(1), F(1), F(1)})
{
    const auto d0 = volume.extent(0);
    const auto d1 = volume.extent(1);
    const auto d2 = volume.extent(2);
    const auto n = d0 * d1 * d2;

    std::vector<F> buf(n);
    detail::init_binary(volume.data_handle(), buf.data(), n);

    // Axis 0 (z): dim_len = d0, stride = d1 * d2
    detail::apply_edt_along_axis<F>(
        std::span<F>(buf), n, d0, d1 * d2, spacing[0]);

    // Axis 1 (y): dim_len = d1, stride = d2
    detail::apply_edt_along_axis<F>(
        std::span<F>(buf), n, d1, d2, spacing[1]);

    // Axis 2 (x): dim_len = d2, stride = 1
    detail::apply_edt_along_axis<F>(
        std::span<F>(buf), n, d2, std::size_t(1), spacing[2]);

    return buf;
}

/// Euclidean distance transform of a 3D binary volume (actual distances).
template<std::integral T, std::floating_point F = float>
[[nodiscard]] std::vector<F> edt_3d(
    std::mdspan<const T, std::dextents<std::size_t, 3>> volume,
    std::array<F, 3> spacing = {F(1), F(1), F(1)})
{
    auto buf = edt_3d_sq<T, F>(volume, spacing);
    for (auto& v : buf) {
        v = std::sqrt(v);
    }
    return buf;
}

// -------------------------------------------------------------------------
// Multi-label variants
// -------------------------------------------------------------------------

/// Euclidean distance transform of a 2D multi-label image.
/// Each pixel is assigned the distance to the nearest pixel with a different
/// label (or with label 0, which is always treated as boundary).
template<std::integral T, std::floating_point F = float>
[[nodiscard]] std::vector<F> edt_2d_multilabel(
    std::mdspan<const T, std::dextents<std::size_t, 2>> image,
    std::array<F, 2> spacing = {F(1), F(1)})
{
    const auto rows = image.extent(0);
    const auto cols = image.extent(1);
    const auto n = rows * cols;

    std::vector<F> buf(n);
    detail::init_multilabel_2d<T, F>(image, buf.data());

    detail::apply_edt_along_axis<F>(
        std::span<F>(buf), n, rows, cols, spacing[0]);
    detail::apply_edt_along_axis<F>(
        std::span<F>(buf), n, cols, std::size_t(1), spacing[1]);

    for (auto& v : buf) {
        v = std::sqrt(v);
    }
    return buf;
}

/// Euclidean distance transform of a 3D multi-label volume.
template<std::integral T, std::floating_point F = float>
[[nodiscard]] std::vector<F> edt_3d_multilabel(
    std::mdspan<const T, std::dextents<std::size_t, 3>> volume,
    std::array<F, 3> spacing = {F(1), F(1), F(1)})
{
    const auto d0 = volume.extent(0);
    const auto d1 = volume.extent(1);
    const auto d2 = volume.extent(2);
    const auto n = d0 * d1 * d2;

    std::vector<F> buf(n);
    detail::init_multilabel_3d<T, F>(volume, buf.data());

    detail::apply_edt_along_axis<F>(
        std::span<F>(buf), n, d0, d1 * d2, spacing[0]);
    detail::apply_edt_along_axis<F>(
        std::span<F>(buf), n, d1, d2, spacing[1]);
    detail::apply_edt_along_axis<F>(
        std::span<F>(buf), n, d2, std::size_t(1), spacing[2]);

    for (auto& v : buf) {
        v = std::sqrt(v);
    }
    return buf;
}

} // namespace utils2
