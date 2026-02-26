#pragma once
#include "mdspan.hpp"
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <concepts>
#include <algorithm>
#include <type_traits>

namespace utils2 {

// ---------------------------------------------------------------------------
// Interpolation method selector
// ---------------------------------------------------------------------------
enum class Interpolation : std::uint8_t { nearest, linear, cubic };

// ---------------------------------------------------------------------------
// 1-D building blocks
// ---------------------------------------------------------------------------

/// Linear interpolation between two values.
template<std::floating_point T>
[[nodiscard]] constexpr T lerp(T a, T b, T t) noexcept {
    return a + t * (b - a);
}

/// Catmull-Rom cubic interpolation over four equally-spaced points.
/// p0--p1--p2--p3, with t in [0,1] between p1 and p2.
template<std::floating_point T>
[[nodiscard]] constexpr T cubic(T p0, T p1, T p2, T p3, T t) noexcept {
    return p1 + T(0.5) * t * (p2 - p0
        + t * (T(2) * p0 - T(5) * p1 + T(4) * p2 - p3
            + t * (T(3) * (p1 - p2) + p3 - p0)));
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------
namespace detail {

/// Clamp a floating-point coordinate to [0, max] and decompose into integer
/// index and fractional part.
[[nodiscard]] inline constexpr auto decompose(double coord, std::size_t extent) noexcept
    -> std::pair<std::size_t, double>
{
    double const max = static_cast<double>(extent - 1);
    double const c   = std::clamp(coord, 0.0, max);
    double       i{};
    double const f = std::modf(c, &i);
    auto const   idx = static_cast<std::size_t>(i);
    // If we landed exactly on the last pixel, keep index in-bounds.
    if (idx >= extent - 1 && extent >= 2)
        return {extent - 2, c - static_cast<double>(extent - 2)};
    return {idx, f};
}

/// Clamp an index to [0, extent-1].
[[nodiscard]] inline constexpr std::size_t clamp_idx(
    std::ptrdiff_t i, std::size_t extent) noexcept
{
    if (i < 0) return 0;
    if (static_cast<std::size_t>(i) >= extent) return extent - 1;
    return static_cast<std::size_t>(i);
}

} // namespace detail

// ---------------------------------------------------------------------------
// 2-D bilinear interpolation
// ---------------------------------------------------------------------------

/// Bilinear interpolation from four corner values.
///   v00 = f(y0, x0)   v10 = f(y1, x0)
///   v01 = f(y0, x1)   v11 = f(y1, x1)
template<std::floating_point T>
[[nodiscard]] constexpr T bilinear(
    T v00, T v10, T v01, T v11,
    T ty, T tx) noexcept
{
    return lerp(lerp(v00, v01, tx),
                lerp(v10, v11, tx),
                ty);
}

/// Bilinear interpolation on a 2-D mdspan grid.
/// Coordinates are (y, x) -- row-major order.
template<typename T, typename Extents>
[[nodiscard]] T bilinear(
    std::mdspan<const T, Extents> grid,
    double y, double x) noexcept
{
    auto const [iy, fy] = detail::decompose(y, grid.extent(0));
    auto const [ix, fx] = detail::decompose(x, grid.extent(1));

    std::size_t const iy1 = std::min(iy + 1, grid.extent(0) - 1);
    std::size_t const ix1 = std::min(ix + 1, grid.extent(1) - 1);

    auto const ty = static_cast<T>(fy);
    auto const tx = static_cast<T>(fx);

    return bilinear(
        grid[iy,  ix],  grid[iy1, ix],
        grid[iy,  ix1], grid[iy1, ix1],
        ty, tx);
}

// ---------------------------------------------------------------------------
// 3-D trilinear interpolation
// ---------------------------------------------------------------------------

/// Trilinear interpolation from eight corner values.
///   Naming: vZYX  (0 = low, 1 = high along that axis).
template<std::floating_point T>
[[nodiscard]] constexpr T trilinear(
    T v000, T v100, T v010, T v110,
    T v001, T v101, T v011, T v111,
    T tz, T ty, T tx) noexcept
{
    T const c00 = lerp(v000, v001, tx);
    T const c10 = lerp(v010, v011, tx);
    T const c01 = lerp(v100, v101, tx);
    T const c11 = lerp(v110, v111, tx);

    T const c0 = lerp(c00, c10, ty);
    T const c1 = lerp(c01, c11, ty);

    return lerp(c0, c1, tz);
}

/// Trilinear interpolation on a 3-D mdspan volume.
/// Coordinates are (z, y, x).
template<typename T, typename Extents>
[[nodiscard]] T trilinear(
    std::mdspan<const T, Extents> volume,
    double z, double y, double x) noexcept
{
    auto const [iz, fz] = detail::decompose(z, volume.extent(0));
    auto const [iy, fy] = detail::decompose(y, volume.extent(1));
    auto const [ix, fx] = detail::decompose(x, volume.extent(2));

    std::size_t const iz1 = std::min(iz + 1, volume.extent(0) - 1);
    std::size_t const iy1 = std::min(iy + 1, volume.extent(1) - 1);
    std::size_t const ix1 = std::min(ix + 1, volume.extent(2) - 1);

    auto const tz = static_cast<T>(fz);
    auto const ty = static_cast<T>(fy);
    auto const tx = static_cast<T>(fx);

    return trilinear(
        volume[iz,  iy,  ix],  volume[iz1, iy,  ix],
        volume[iz,  iy1, ix],  volume[iz1, iy1, ix],
        volume[iz,  iy,  ix1], volume[iz1, iy,  ix1],
        volume[iz,  iy1, ix1], volume[iz1, iy1, ix1],
        tz, ty, tx);
}

// ---------------------------------------------------------------------------
// 3-D tricubic interpolation (separable Catmull-Rom)
// ---------------------------------------------------------------------------

/// Tricubic interpolation on a 3-D mdspan volume.
/// Uses a 4x4x4 neighbourhood centred on the sample point.
/// Coordinates are (z, y, x).
template<typename T, typename Extents>
[[nodiscard]] T tricubic(
    std::mdspan<const T, Extents> volume,
    double z, double y, double x) noexcept
{
    auto const [iz, fz] = detail::decompose(z, volume.extent(0));
    auto const [iy, fy] = detail::decompose(y, volume.extent(1));
    auto const [ix, fx] = detail::decompose(x, volume.extent(2));

    auto const tz = static_cast<T>(fz);
    auto const ty = static_cast<T>(fy);
    auto const tx = static_cast<T>(fx);

    auto const nz = volume.extent(0);
    auto const ny = volume.extent(1);
    auto const nx = volume.extent(2);

    // Fetch clamped voxel value.
    auto fetch = [&](std::ptrdiff_t dz, std::ptrdiff_t dy, std::ptrdiff_t dx) -> T {
        auto const cz = detail::clamp_idx(static_cast<std::ptrdiff_t>(iz) + dz, nz);
        auto const cy = detail::clamp_idx(static_cast<std::ptrdiff_t>(iy) + dy, ny);
        auto const cx = detail::clamp_idx(static_cast<std::ptrdiff_t>(ix) + dx, nx);
        return volume[cz, cy, cx];
    };

    // Interpolate along x for each (dz, dy) pair, then along y, then z.
    std::array<T, 4> z_vals{};
    for (std::ptrdiff_t dz = -1; dz <= 2; ++dz) {
        std::array<T, 4> y_vals{};
        for (std::ptrdiff_t dy = -1; dy <= 2; ++dy) {
            y_vals[static_cast<std::size_t>(dy + 1)] = cubic(
                fetch(dz, dy, -1), fetch(dz, dy, 0),
                fetch(dz, dy,  1), fetch(dz, dy, 2), tx);
        }
        z_vals[static_cast<std::size_t>(dz + 1)] = cubic(
            y_vals[0], y_vals[1], y_vals[2], y_vals[3], ty);
    }
    return cubic(z_vals[0], z_vals[1], z_vals[2], z_vals[3], tz);
}

// ---------------------------------------------------------------------------
// Nearest-neighbour sampling (any dimensionality via mdspan)
// ---------------------------------------------------------------------------
namespace detail {

/// Round each coordinate to the nearest integer and index into data.
/// Works for 1-D, 2-D, 3-D, etc.
template<typename T, typename Extents, std::size_t... Is>
[[nodiscard]] T nearest_impl(
    std::mdspan<const T, Extents> data,
    std::index_sequence<Is...>,
    auto... coords) noexcept
{
    static_assert(sizeof...(coords) == Extents::rank());
    double const raw[] = {static_cast<double>(coords)...};
    auto idx = [&]<std::size_t I>() -> std::size_t {
        double const max = static_cast<double>(data.extent(I) - 1);
        double const c   = std::clamp(raw[I], 0.0, max);
        return static_cast<std::size_t>(std::round(c));
    };
    return data[idx.template operator()<Is>()...];
}

} // namespace detail

template<typename T, typename Extents>
[[nodiscard]] T nearest(
    std::mdspan<const T, Extents> data,
    auto... coords) noexcept
{
    static_assert(sizeof...(coords) == Extents::rank());
    return detail::nearest_impl(
        data,
        std::make_index_sequence<Extents::rank()>{},
        coords...);
}

// ---------------------------------------------------------------------------
// Generic N-D sample dispatcher
// ---------------------------------------------------------------------------

/// Dispatch to nearest / linear / cubic based on the method enum.
///
/// - nearest: works for any rank.
/// - linear : supports rank 2 (bilinear) and rank 3 (trilinear).
/// - cubic  : supports rank 3 (tricubic).
template<typename T, typename Extents>
[[nodiscard]] T sample(
    std::mdspan<const T, Extents> data,
    Interpolation method,
    auto... coords)
{
    static_assert(sizeof...(coords) == Extents::rank());

    if (method == Interpolation::nearest) {
        return nearest(data, coords...);
    }

    if constexpr (Extents::rank() == 2) {
        double const c[] = {static_cast<double>(coords)...};
        if (method == Interpolation::linear || method == Interpolation::cubic)
            return bilinear(data, c[0], c[1]);
    } else if constexpr (Extents::rank() == 3) {
        double const c[] = {static_cast<double>(coords)...};
        if (method == Interpolation::linear)
            return trilinear(data, c[0], c[1], c[2]);
        if (method == Interpolation::cubic)
            return tricubic(data, c[0], c[1], c[2]);
    }

    // Fallback for unsupported rank / method combinations: nearest.
    return nearest(data, coords...);
}

// ---------------------------------------------------------------------------
// Gradient via central differences
// ---------------------------------------------------------------------------

/// Compute the gradient at the given coordinates using central differences
/// with linear interpolation.  Returns one partial derivative per dimension.
///
/// Coords order must match the mdspan extents (e.g. z, y, x for rank 3).
template<typename T, typename Extents, std::size_t Dims>
[[nodiscard]] std::array<T, Dims> gradient(
    std::mdspan<const T, Extents> data,
    auto... coords)
{
    static_assert(sizeof...(coords) == Dims);
    static_assert(Dims == Extents::rank());

    constexpr double kHalfStep = 0.5;
    double const c[] = {static_cast<double>(coords)...};
    std::array<T, Dims> grad{};

    // For each dimension, perturb +/- half-step and use linear sample.
    auto sample_at = [&](double const* pos) -> T {
        if constexpr (Dims == 2)
            return bilinear(data, pos[0], pos[1]);
        else if constexpr (Dims == 3)
            return trilinear(data, pos[0], pos[1], pos[2]);
        else
            return nearest(data, pos[0]); // 1-D fallback
    };

    for (std::size_t d = 0; d < Dims; ++d) {
        double fwd[Dims];
        double bwd[Dims];
        for (std::size_t i = 0; i < Dims; ++i) {
            fwd[i] = c[i];
            bwd[i] = c[i];
        }
        double const max_d = static_cast<double>(data.extent(d) - 1);
        fwd[d] = std::clamp(c[d] + kHalfStep, 0.0, max_d);
        bwd[d] = std::clamp(c[d] - kHalfStep, 0.0, max_d);

        double const actual_step = fwd[d] - bwd[d];
        T const val_fwd = sample_at(fwd);
        T const val_bwd = sample_at(bwd);

        grad[d] = (actual_step > 0.0)
            ? static_cast<T>((val_fwd - val_bwd) / static_cast<T>(actual_step))
            : T{0};
    }
    return grad;
}

} // namespace utils2
