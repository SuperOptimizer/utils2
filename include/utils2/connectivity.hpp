#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <concepts>
#include "mdspan.hpp"
#include <utility>

namespace utils2 {

// ---------------------------------------------------------------------------
// GridConnectivity -- shared connectivity enum for 2D and 3D grids
// ---------------------------------------------------------------------------

enum class GridConnectivity : std::uint8_t {
    four       = 4,   // 2D: cardinal only
    eight      = 8,   // 2D: cardinal + diagonal
    six        = 6,   // 3D: face-adjacent
    eighteen   = 18,  // 3D: face + edge-adjacent
    twenty_six = 26   // 3D: face + edge + corner-adjacent
};

// ---------------------------------------------------------------------------
// StructuringElement -- morphological structuring elements
// ---------------------------------------------------------------------------

enum class StructuringElement : std::uint8_t {
    cross,   // 3x3 cross  (4-connected)
    square,  // 3x3 square (8-connected)
    disk_3,  // 3x3 disk approximation (same as square for 3x3)
    disk_5,  // 5x5 disk approximation
};

// ---------------------------------------------------------------------------
// Full neighbor offset structs (with Euclidean distance for pathfinding)
// ---------------------------------------------------------------------------

struct Offset2D {
    int dy, dx;
    double dist;
};

struct Offset3D {
    int dz, dy, dx;
    double dist;
};

// ---------------------------------------------------------------------------
// Predecessor-only offset structs (for connected components forward pass)
// These only include neighbors that precede the current pixel in raster order
// ---------------------------------------------------------------------------

struct PredOffset2D {
    int dy, dx;
};

struct PredOffset3D {
    int dz, dy, dx;
};

// ---------------------------------------------------------------------------
// Full neighbor offset tables (all directions, with distances)
// ---------------------------------------------------------------------------

[[nodiscard]] inline auto offsets_2d(GridConnectivity conn) noexcept
    -> std::pair<const Offset2D*, std::size_t>
{
    static constexpr Offset2D table_4[] = {
        { -1,  0, 1.0 },
        {  1,  0, 1.0 },
        {  0, -1, 1.0 },
        {  0,  1, 1.0 },
    };

    static constexpr double sqrt2 = 1.4142135623730951;
    static constexpr Offset2D table_8[] = {
        { -1,  0, 1.0 },
        {  1,  0, 1.0 },
        {  0, -1, 1.0 },
        {  0,  1, 1.0 },
        { -1, -1, sqrt2 },
        { -1,  1, sqrt2 },
        {  1, -1, sqrt2 },
        {  1,  1, sqrt2 },
    };

    switch (conn) {
        case GridConnectivity::four:
            return {table_4, 4};
        case GridConnectivity::eight:
        default:
            return {table_8, 8};
    }
}

[[nodiscard]] inline auto offsets_3d(GridConnectivity conn) noexcept
    -> std::pair<const Offset3D*, std::size_t>
{
    static constexpr double sqrt2 = 1.4142135623730951;
    static constexpr double sqrt3 = 1.7320508075688772;

    // 6-connectivity: face neighbors
    static constexpr Offset3D table_6[] = {
        { -1,  0,  0, 1.0 },
        {  1,  0,  0, 1.0 },
        {  0, -1,  0, 1.0 },
        {  0,  1,  0, 1.0 },
        {  0,  0, -1, 1.0 },
        {  0,  0,  1, 1.0 },
    };

    // 18-connectivity: face + edge neighbors
    static constexpr Offset3D table_18[] = {
        // face (6)
        { -1,  0,  0, 1.0 },
        {  1,  0,  0, 1.0 },
        {  0, -1,  0, 1.0 },
        {  0,  1,  0, 1.0 },
        {  0,  0, -1, 1.0 },
        {  0,  0,  1, 1.0 },
        // edge (12)
        { -1, -1,  0, sqrt2 },
        { -1,  1,  0, sqrt2 },
        {  1, -1,  0, sqrt2 },
        {  1,  1,  0, sqrt2 },
        { -1,  0, -1, sqrt2 },
        { -1,  0,  1, sqrt2 },
        {  1,  0, -1, sqrt2 },
        {  1,  0,  1, sqrt2 },
        {  0, -1, -1, sqrt2 },
        {  0, -1,  1, sqrt2 },
        {  0,  1, -1, sqrt2 },
        {  0,  1,  1, sqrt2 },
    };

    // 26-connectivity: face + edge + corner neighbors
    static constexpr Offset3D table_26[] = {
        // face (6)
        { -1,  0,  0, 1.0 },
        {  1,  0,  0, 1.0 },
        {  0, -1,  0, 1.0 },
        {  0,  1,  0, 1.0 },
        {  0,  0, -1, 1.0 },
        {  0,  0,  1, 1.0 },
        // edge (12)
        { -1, -1,  0, sqrt2 },
        { -1,  1,  0, sqrt2 },
        {  1, -1,  0, sqrt2 },
        {  1,  1,  0, sqrt2 },
        { -1,  0, -1, sqrt2 },
        { -1,  0,  1, sqrt2 },
        {  1,  0, -1, sqrt2 },
        {  1,  0,  1, sqrt2 },
        {  0, -1, -1, sqrt2 },
        {  0, -1,  1, sqrt2 },
        {  0,  1, -1, sqrt2 },
        {  0,  1,  1, sqrt2 },
        // corner (8)
        { -1, -1, -1, sqrt3 },
        { -1, -1,  1, sqrt3 },
        { -1,  1, -1, sqrt3 },
        { -1,  1,  1, sqrt3 },
        {  1, -1, -1, sqrt3 },
        {  1, -1,  1, sqrt3 },
        {  1,  1, -1, sqrt3 },
        {  1,  1,  1, sqrt3 },
    };

    switch (conn) {
        case GridConnectivity::six:
            return {table_6, 6};
        case GridConnectivity::eighteen:
            return {table_18, 18};
        case GridConnectivity::twenty_six:
        default:
            return {table_26, 26};
    }
}

// ---------------------------------------------------------------------------
// Predecessor-only offset tables (for connected components forward pass)
// Only neighbors preceding the current pixel in raster order
// (y decreasing, then x decreasing for 2D; z/y/x decreasing for 3D)
// ---------------------------------------------------------------------------

[[nodiscard]] inline auto predecessors_2d(GridConnectivity conn) noexcept
    -> std::pair<const PredOffset2D*, std::size_t>
{
    // 4-connectivity: cardinal predecessors only
    static constexpr PredOffset2D table_4[] = {
        {-1, 0}, {0, -1}
    };

    // 8-connectivity: cardinal + diagonal predecessors
    static constexpr PredOffset2D table_8[] = {
        {-1, -1}, {-1, 0}, {-1, 1}, {0, -1}
    };

    switch (conn) {
        case GridConnectivity::four:
            return {table_4, 2};
        case GridConnectivity::eight:
        default:
            return {table_8, 4};
    }
}

[[nodiscard]] inline auto predecessors_3d(GridConnectivity conn) noexcept
    -> std::pair<const PredOffset3D*, std::size_t>
{
    // 6-connectivity: face-adjacent predecessors in raster order
    static constexpr PredOffset3D table_6[] = {
        {-1, 0, 0}, {0, -1, 0}, {0, 0, -1}
    };

    // 18-connectivity: face + edge-adjacent predecessors
    static constexpr PredOffset3D table_18[] = {
        {-1,  0,  0}, {-1, -1,  0}, {-1,  1,  0}, {-1,  0, -1}, {-1,  0,  1},
        { 0, -1,  0}, { 0, -1, -1}, { 0, -1,  1}, { 0,  0, -1}
    };

    // 26-connectivity: face + edge + corner-adjacent predecessors
    static constexpr PredOffset3D table_26[] = {
        {-1, -1, -1}, {-1, -1,  0}, {-1, -1,  1},
        {-1,  0, -1}, {-1,  0,  0}, {-1,  0,  1},
        {-1,  1, -1}, {-1,  1,  0}, {-1,  1,  1},
        { 0, -1, -1}, { 0, -1,  0}, { 0, -1,  1},
        { 0,  0, -1}
    };

    switch (conn) {
        case GridConnectivity::six:
            return {table_6, 3};
        case GridConnectivity::eighteen:
            return {table_18, 9};
        case GridConnectivity::twenty_six:
        default:
            return {table_26, 13};
    }
}

// ---------------------------------------------------------------------------
// Structuring element offset tables (morphology)
// ---------------------------------------------------------------------------

namespace detail {

// 3x3 cross (4-connected): N, S, E, W + center.
inline constexpr std::array<std::array<int, 2>, 5> cross_offsets{{
    {0, 0}, {-1, 0}, {1, 0}, {0, -1}, {0, 1}
}};

// 3x3 square (8-connected): all 9 cells.
inline constexpr std::array<std::array<int, 2>, 9> square_offsets{{
    {-1, -1}, {-1, 0}, {-1, 1},
    { 0, -1}, { 0, 0}, { 0, 1},
    { 1, -1}, { 1, 0}, { 1, 1}
}};

// disk_3 is the same footprint as square for a 3x3 kernel.
inline constexpr auto disk3_offsets = square_offsets;

// 5x5 disk approximation (diamond + filled circle-ish, 21 cells).
inline constexpr std::array<std::array<int, 2>, 21> disk5_offsets{{
    {-2, 0},
    {-1,-1}, {-1, 0}, {-1, 1},
    { 0,-2}, { 0,-1}, { 0, 0}, { 0, 1}, { 0, 2},
    { 1,-1}, { 1, 0}, { 1, 1},
    { 2, 0},
    // additional ring pixels for a rounder shape
    {-2,-1}, {-2, 1},
    { 2,-1}, { 2, 1},
    {-1,-2}, {-1, 2},
    { 1,-2}, { 1, 2}
}};

// ---------------------------------------------------------------------------
// Morphology helpers
// ---------------------------------------------------------------------------

/// Calls fn(neighbor_value) for every valid neighbor of (r, c) according
/// to the given offset table.
template<std::integral T, typename Offsets, typename Fn>
void for_each_neighbor(
    std::mdspan<const T, std::dextents<std::size_t, 2>> image,
    std::size_t r, std::size_t c,
    const Offsets& offsets, Fn&& fn) noexcept
{
    const auto rows = image.extent(0);
    const auto cols = image.extent(1);
    for (const auto& [dr, dc] : offsets) {
        const auto nr = static_cast<std::ptrdiff_t>(r) + dr;
        const auto nc = static_cast<std::ptrdiff_t>(c) + dc;
        if (nr >= 0 && nc >= 0 &&
            static_cast<std::size_t>(nr) < rows &&
            static_cast<std::size_t>(nc) < cols) {
            fn(image[static_cast<std::size_t>(nr), static_cast<std::size_t>(nc)]);
        }
    }
}

/// Dispatch structuring element to the correct offset table and call body.
template<typename Fn>
decltype(auto) dispatch_se(StructuringElement se, Fn&& body)
{
    switch (se) {
        case StructuringElement::cross:  return body(cross_offsets);
        case StructuringElement::square: return body(square_offsets);
        case StructuringElement::disk_3: return body(disk3_offsets);
        case StructuringElement::disk_5: return body(disk5_offsets);
    }
    // unreachable -- satisfy the compiler
    return body(square_offsets);
}

} // namespace detail

} // namespace utils2
