#pragma once
#include "mdspan.hpp"
#include <vector>
#include <array>
#include <span>
#include <cstddef>
#include <cstdint>
#include <concepts>
#include <algorithm>

#include "connectivity.hpp"
#include "disjoint_set.hpp"

namespace utils2 {

// --- Erosion ----------------------------------------------------------------

template<std::integral T>
[[nodiscard]] std::vector<T> erode_2d(
    std::mdspan<const T, std::dextents<std::size_t, 2>> image,
    StructuringElement se = StructuringElement::square)
{
    const auto rows = image.extent(0);
    const auto cols = image.extent(1);
    std::vector<T> out(rows * cols);

    detail::dispatch_se(se, [&](const auto& offsets) {
        for (std::size_t r = 0; r < rows; ++r) {
            for (std::size_t c = 0; c < cols; ++c) {
                T val = std::numeric_limits<T>::max();
                detail::for_each_neighbor(image, r, c, offsets,
                    [&](T v) noexcept { val = std::min(val, v); });
                out[r * cols + c] = val;
            }
        }
    });
    return out;
}

// --- Dilation ---------------------------------------------------------------

template<std::integral T>
[[nodiscard]] std::vector<T> dilate_2d(
    std::mdspan<const T, std::dextents<std::size_t, 2>> image,
    StructuringElement se = StructuringElement::square)
{
    const auto rows = image.extent(0);
    const auto cols = image.extent(1);
    std::vector<T> out(rows * cols);

    detail::dispatch_se(se, [&](const auto& offsets) {
        for (std::size_t r = 0; r < rows; ++r) {
            for (std::size_t c = 0; c < cols; ++c) {
                T val = std::numeric_limits<T>::lowest();
                detail::for_each_neighbor(image, r, c, offsets,
                    [&](T v) noexcept { val = std::max(val, v); });
                out[r * cols + c] = val;
            }
        }
    });
    return out;
}

// --- Opening (erode then dilate) -------------------------------------------

template<std::integral T>
[[nodiscard]] std::vector<T> open_2d(
    std::mdspan<const T, std::dextents<std::size_t, 2>> image,
    StructuringElement se = StructuringElement::square)
{
    const auto rows = image.extent(0);
    const auto cols = image.extent(1);
    auto eroded = erode_2d(image, se);
    std::mdspan<const T, std::dextents<std::size_t, 2>> eroded_view(
        eroded.data(), rows, cols);
    return dilate_2d(eroded_view, se);
}

// --- Closing (dilate then erode) -------------------------------------------

template<std::integral T>
[[nodiscard]] std::vector<T> close_2d(
    std::mdspan<const T, std::dextents<std::size_t, 2>> image,
    StructuringElement se = StructuringElement::square)
{
    const auto rows = image.extent(0);
    const auto cols = image.extent(1);
    auto dilated = dilate_2d(image, se);
    std::mdspan<const T, std::dextents<std::size_t, 2>> dilated_view(
        dilated.data(), rows, cols);
    return erode_2d(dilated_view, se);
}

// --- Threshold --------------------------------------------------------------

template<typename T, std::integral Out = std::uint8_t>
[[nodiscard]] std::vector<Out> threshold_2d(
    std::mdspan<const T, std::dextents<std::size_t, 2>> image,
    T threshold_value,
    Out fg = Out(255), Out bg = Out(0))
{
    const auto rows = image.extent(0);
    const auto cols = image.extent(1);
    std::vector<Out> out(rows * cols);
    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t c = 0; c < cols; ++c) {
            out[r * cols + c] = (image[r, c] >= threshold_value) ? fg : bg;
        }
    }
    return out;
}

// --- Zhang-Suen thinning ----------------------------------------------------

namespace detail {

// 8-neighbors clockwise from north: P2, P3, P4, P5, P6, P7, P8, P9
//
//   P9  P2  P3
//   P8  P1  P4
//   P7  P6  P5
//
inline constexpr std::array<std::array<int, 2>, 8> zs_neighbors{{
    {-1,  0},  // P2
    {-1,  1},  // P3
    { 0,  1},  // P4
    { 1,  1},  // P5
    { 1,  0},  // P6
    { 1, -1},  // P7
    { 0, -1},  // P8
    {-1, -1},  // P9
}};

// Fetch P2..P9 as booleans (nonzero = true). Out-of-bounds treated as 0.
template<std::integral T>
std::array<bool, 8> get_zs_neighbors(
    std::mdspan<const T, std::dextents<std::size_t, 2>> img,
    std::size_t r, std::size_t c) noexcept
{
    std::array<bool, 8> p{};
    const auto rows = img.extent(0);
    const auto cols = img.extent(1);
    for (int i = 0; i < 8; ++i) {
        auto nr = static_cast<std::ptrdiff_t>(r) + zs_neighbors[i][0];
        auto nc = static_cast<std::ptrdiff_t>(c) + zs_neighbors[i][1];
        if (nr >= 0 && nc >= 0 &&
            static_cast<std::size_t>(nr) < rows &&
            static_cast<std::size_t>(nc) < cols) {
            p[i] = (img[static_cast<std::size_t>(nr),
                        static_cast<std::size_t>(nc)] != T(0));
        }
    }
    return p;
}

// B(P1): number of nonzero neighbors among P2..P9.
inline int zs_B(const std::array<bool, 8>& p) noexcept
{
    int count = 0;
    for (bool v : p) count += v;
    return count;
}

// A(P1): number of 0->1 transitions in the ordered sequence P2..P9..P2.
inline int zs_A(const std::array<bool, 8>& p) noexcept
{
    int count = 0;
    for (int i = 0; i < 8; ++i) {
        if (!p[i] && p[(i + 1) % 8]) ++count;
    }
    return count;
}

} // namespace detail

template<std::integral T>
[[nodiscard]] std::vector<T> thin_2d(
    std::mdspan<const T, std::dextents<std::size_t, 2>> image)
{
    const auto rows = image.extent(0);
    const auto cols = image.extent(1);

    // Working copy.
    std::vector<T> buf(rows * cols);
    for (std::size_t r = 0; r < rows; ++r)
        for (std::size_t c = 0; c < cols; ++c)
            buf[r * cols + c] = image[r, c];

    std::vector<std::size_t> to_remove;
    bool changed = true;

    while (changed) {
        changed = false;

        // -- Sub-iteration 1 --
        {
            std::mdspan<const T, std::dextents<std::size_t, 2>> view(
                buf.data(), rows, cols);
            to_remove.clear();
            for (std::size_t r = 1; r + 1 < rows; ++r) {
                for (std::size_t c = 1; c + 1 < cols; ++c) {
                    if (buf[r * cols + c] == T(0)) continue;
                    auto p = detail::get_zs_neighbors(view, r, c);
                    int B = detail::zs_B(p);
                    if (B < 2 || B > 6) continue;
                    if (detail::zs_A(p) != 1) continue;
                    // P2*P4*P6 == 0
                    if (p[0] && p[2] && p[4]) continue;
                    // P4*P6*P8 == 0
                    if (p[2] && p[4] && p[6]) continue;
                    to_remove.push_back(r * cols + c);
                }
            }
            for (auto idx : to_remove) { buf[idx] = T(0); changed = true; }
        }

        // -- Sub-iteration 2 --
        {
            std::mdspan<const T, std::dextents<std::size_t, 2>> view(
                buf.data(), rows, cols);
            to_remove.clear();
            for (std::size_t r = 1; r + 1 < rows; ++r) {
                for (std::size_t c = 1; c + 1 < cols; ++c) {
                    if (buf[r * cols + c] == T(0)) continue;
                    auto p = detail::get_zs_neighbors(view, r, c);
                    int B = detail::zs_B(p);
                    if (B < 2 || B > 6) continue;
                    if (detail::zs_A(p) != 1) continue;
                    // P2*P4*P8 == 0
                    if (p[0] && p[2] && p[6]) continue;
                    // P2*P6*P8 == 0
                    if (p[0] && p[4] && p[6]) continue;
                    to_remove.push_back(r * cols + c);
                }
            }
            for (auto idx : to_remove) { buf[idx] = T(0); changed = true; }
        }
    }

    return buf;
}

// --- Flood fill -------------------------------------------------------------

template<std::integral T>
void flood_fill_2d(
    std::mdspan<T, std::dextents<std::size_t, 2>> image,
    std::array<std::size_t, 2> seed,
    T new_value)
{
    const auto rows = image.extent(0);
    const auto cols = image.extent(1);
    if (seed[0] >= rows || seed[1] >= cols) return;

    const T old_value = image[seed[0], seed[1]];
    if (old_value == new_value) return;

    // Simple stack-based flood fill (4-connected).
    std::vector<std::array<std::size_t, 2>> stack;
    stack.push_back(seed);
    image[seed[0], seed[1]] = new_value;

    while (!stack.empty()) {
        auto [r, c] = stack.back();
        stack.pop_back();

        constexpr std::array<std::array<int, 2>, 4> dirs{{
            {-1, 0}, {1, 0}, {0, -1}, {0, 1}
        }};
        for (const auto& [dr, dc] : dirs) {
            auto nr = static_cast<std::ptrdiff_t>(r) + dr;
            auto nc = static_cast<std::ptrdiff_t>(c) + dc;
            if (nr < 0 || nc < 0) continue;
            auto ur = static_cast<std::size_t>(nr);
            auto uc = static_cast<std::size_t>(nc);
            if (ur >= rows || uc >= cols) continue;
            if (image[ur, uc] == old_value) {
                image[ur, uc] = new_value;
                stack.push_back({ur, uc});
            }
        }
    }
}

// --- Small component removal ------------------------------------------------

template<std::integral T>
[[nodiscard]] std::vector<T> remove_small_components_2d(
    std::mdspan<const T, std::dextents<std::size_t, 2>> image,
    std::size_t min_size)
{
    const auto rows = image.extent(0);
    const auto cols = image.extent(1);
    const auto n = rows * cols;

    // Build union-find over foreground pixels (8-connected).
    DisjointSet<std::size_t> uf(n);

    auto idx = [cols](std::size_t r, std::size_t c) noexcept {
        return r * cols + c;
    };

    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t c = 0; c < cols; ++c) {
            if (image[r, c] == T(0)) continue;
            // Check right and three lower neighbors for 8-connectivity.
            if (c + 1 < cols && image[r, c + 1] != T(0))
                uf.unite(idx(r, c), idx(r, c + 1));
            if (r + 1 < rows) {
                if (image[r + 1, c] != T(0))
                    uf.unite(idx(r, c), idx(r + 1, c));
                if (c + 1 < cols && image[r + 1, c + 1] != T(0))
                    uf.unite(idx(r, c), idx(r + 1, c + 1));
                if (c > 0 && image[r + 1, c - 1] != T(0))
                    uf.unite(idx(r, c), idx(r + 1, c - 1));
            }
        }
    }

    // Count component sizes.
    std::vector<std::size_t> comp_size(n, 0);
    for (std::size_t r = 0; r < rows; ++r)
        for (std::size_t c = 0; c < cols; ++c)
            if (image[r, c] != T(0))
                ++comp_size[uf.find(idx(r, c))];

    // Build output, zeroing pixels in small components.
    std::vector<T> out(n);
    for (std::size_t r = 0; r < rows; ++r)
        for (std::size_t c = 0; c < cols; ++c) {
            auto i = idx(r, c);
            out[i] = (image[r, c] != T(0) && comp_size[uf.find(i)] >= min_size)
                         ? image[r, c]
                         : T(0);
        }

    return out;
}

} // namespace utils2
