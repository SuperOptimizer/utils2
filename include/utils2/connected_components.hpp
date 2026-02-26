#pragma once
#include "mdspan.hpp"
#include <vector>
#include <array>
#include <cstddef>
#include <cstdint>
#include <concepts>
#include <algorithm>
#include <numeric>
#include <span>

#include "connectivity.hpp"
#include "disjoint_set.hpp"

namespace utils2 {

// ---------------------------------------------------------------------------
// Implementation details
// ---------------------------------------------------------------------------

namespace detail {

/// Two-pass connected components for 2D images.
/// @tparam MultiLabel  if true, same-value adjacency; if false, binary (nonzero = foreground).
template <bool MultiLabel, std::integral T, typename Extents>
[[nodiscard]] std::pair<std::vector<std::uint32_t>, std::uint32_t>
cc_2d_impl(std::mdspan<const T, Extents> image, GridConnectivity conn) noexcept
{
    const auto rows = image.extent(0);
    const auto cols = image.extent(1);
    const auto voxels = rows * cols;

    std::vector<std::uint32_t> labels(voxels, 0);
    // Pre-allocate union-find with generous capacity; label 0 is background.
    // Worst case: every other pixel is a new label, so voxels/2+1 is plenty.
    // We grow if needed (see below).
    DisjointSet<std::uint32_t> uf(std::max<std::size_t>(voxels / 2 + 1, 256));
    std::uint32_t next_label = 1;

    const auto [preds, npreds] = predecessors_2d(conn);

    // --- Pass 1: forward raster scan ---
    for (std::size_t y = 0; y < rows; ++y) {
        for (std::size_t x = 0; x < cols; ++x) {
            const auto val = image[y, x];

            // Background check
            if constexpr (!MultiLabel) {
                if (val == T{0}) continue;
            } else {
                if (val == T{0}) continue;
            }

            const std::size_t idx = y * cols + x;
            std::uint32_t min_label = 0;

            // Lambda to process one neighbor
            auto check_neighbor = [&](int dy, int dx) noexcept {
                const auto ny = static_cast<std::ptrdiff_t>(y) + dy;
                const auto nx = static_cast<std::ptrdiff_t>(x) + dx;
                if (ny < 0 || nx < 0 ||
                    static_cast<std::size_t>(ny) >= rows ||
                    static_cast<std::size_t>(nx) >= cols)
                    return;

                const auto nval = image[static_cast<std::size_t>(ny),
                                        static_cast<std::size_t>(nx)];

                // Connectivity test
                if constexpr (MultiLabel) {
                    if (nval != val) return;
                } else {
                    if (nval == T{0}) return;
                }

                const std::size_t nidx = static_cast<std::size_t>(ny) * cols +
                                          static_cast<std::size_t>(nx);
                const auto nlabel = labels[nidx];
                if (nlabel == 0) return;

                if (min_label == 0) {
                    min_label = nlabel;
                } else {
                    // Union the two labels
                    min_label = uf.unite(min_label, nlabel);
                }
            };

            for (std::size_t i = 0; i < npreds; ++i) {
                check_neighbor(preds[i].dy, preds[i].dx);
            }

            if (min_label == 0) {
                // New component -- grow the DisjointSet if at capacity.
                if (next_label >= uf.size()) {
                    uf.grow(uf.size() * 2);
                }
                labels[idx] = next_label;
                ++next_label;
            } else {
                labels[idx] = uf.find(min_label);
            }
        }
    }

    // --- Pass 2: flatten labels to roots ---
    // Build a relabeling map so output labels are consecutive 1..N.
    std::vector<std::uint32_t> remap(next_label, 0);
    std::uint32_t num_components = 0;
    for (std::uint32_t l = 1; l < next_label; ++l) {
        auto root = uf.find(l);
        if (remap[root] == 0) {
            remap[root] = ++num_components;
        }
        remap[l] = remap[root];
    }

    for (auto& l : labels) {
        if (l != 0) {
            l = remap[l];
        }
    }

    return {std::move(labels), num_components};
}

/// Two-pass connected components for 3D volumes.
template <bool MultiLabel, std::integral T, typename Extents>
[[nodiscard]] std::pair<std::vector<std::uint32_t>, std::uint32_t>
cc_3d_impl(std::mdspan<const T, Extents> volume, GridConnectivity conn) noexcept
{
    const auto depth = volume.extent(0);
    const auto rows  = volume.extent(1);
    const auto cols  = volume.extent(2);
    const auto voxels = depth * rows * cols;
    const auto slice_stride = rows * cols;

    std::vector<std::uint32_t> labels(voxels, 0);
    DisjointSet<std::uint32_t> uf(std::max<std::size_t>(voxels / 2 + 1, 256));
    std::uint32_t next_label = 1;

    const auto [preds, npreds] = predecessors_3d(conn);

    // --- Pass 1: forward raster scan ---
    for (std::size_t z = 0; z < depth; ++z) {
        for (std::size_t y = 0; y < rows; ++y) {
            for (std::size_t x = 0; x < cols; ++x) {
                const auto val = volume[z, y, x];
                if (val == T{0}) continue;

                const std::size_t idx = z * slice_stride + y * cols + x;
                std::uint32_t min_label = 0;

                auto check_neighbor = [&](int dz, int dy, int dx) noexcept {
                    const auto nz = static_cast<std::ptrdiff_t>(z) + dz;
                    const auto ny = static_cast<std::ptrdiff_t>(y) + dy;
                    const auto nx = static_cast<std::ptrdiff_t>(x) + dx;
                    if (nz < 0 || ny < 0 || nx < 0 ||
                        static_cast<std::size_t>(nz) >= depth ||
                        static_cast<std::size_t>(ny) >= rows ||
                        static_cast<std::size_t>(nx) >= cols)
                        return;

                    const auto nval = volume[static_cast<std::size_t>(nz),
                                             static_cast<std::size_t>(ny),
                                             static_cast<std::size_t>(nx)];

                    if constexpr (MultiLabel) {
                        if (nval != val) return;
                    } else {
                        if (nval == T{0}) return;
                    }

                    const std::size_t nidx =
                        static_cast<std::size_t>(nz) * slice_stride +
                        static_cast<std::size_t>(ny) * cols +
                        static_cast<std::size_t>(nx);
                    const auto nlabel = labels[nidx];
                    if (nlabel == 0) return;

                    if (min_label == 0) {
                        min_label = nlabel;
                    } else {
                        min_label = uf.unite(min_label, nlabel);
                    }
                };

                for (std::size_t i = 0; i < npreds; ++i) {
                    check_neighbor(preds[i].dz, preds[i].dy, preds[i].dx);
                }

                if (min_label == 0) {
                    if (next_label >= uf.size()) {
                        uf.grow(uf.size() * 2);
                    }
                    labels[idx] = next_label;
                    ++next_label;
                } else {
                    labels[idx] = uf.find(min_label);
                }
            }
        }
    }

    // --- Pass 2: consecutive relabeling ---
    std::vector<std::uint32_t> remap(next_label, 0);
    std::uint32_t num_components = 0;
    for (std::uint32_t l = 1; l < next_label; ++l) {
        auto root = uf.find(l);
        if (remap[root] == 0) {
            remap[root] = ++num_components;
        }
        remap[l] = remap[root];
    }

    for (auto& l : labels) {
        if (l != 0) {
            l = remap[l];
        }
    }

    return {std::move(labels), num_components};
}

} // namespace detail

// ---------------------------------------------------------------------------
// 2D connected components -- binary
// ---------------------------------------------------------------------------

/// Label connected components in a 2D binary image (nonzero = foreground).
/// Returns {flat label vector, number of components}. Label 0 = background.
template <std::integral T, typename Extents>
[[nodiscard]] std::pair<std::vector<std::uint32_t>, std::uint32_t>
connected_components_2d(
    std::mdspan<const T, Extents> image,
    GridConnectivity conn = GridConnectivity::eight) noexcept
{
    return detail::cc_2d_impl<false>(image, conn);
}

// ---------------------------------------------------------------------------
// 2D connected components -- multi-label
// ---------------------------------------------------------------------------

/// Label connected components in a 2D multi-label image.
/// Pixels with the same nonzero value that are adjacent form a component.
template <std::integral T, typename Extents>
[[nodiscard]] std::pair<std::vector<std::uint32_t>, std::uint32_t>
connected_components_2d_multilabel(
    std::mdspan<const T, Extents> image,
    GridConnectivity conn = GridConnectivity::eight) noexcept
{
    return detail::cc_2d_impl<true>(image, conn);
}

// ---------------------------------------------------------------------------
// 3D connected components -- binary
// ---------------------------------------------------------------------------

/// Label connected components in a 3D binary volume (nonzero = foreground).
/// Returns {flat label vector, number of components}. Label 0 = background.
template <std::integral T, typename Extents>
[[nodiscard]] std::pair<std::vector<std::uint32_t>, std::uint32_t>
connected_components_3d(
    std::mdspan<const T, Extents> volume,
    GridConnectivity conn = GridConnectivity::twenty_six) noexcept
{
    return detail::cc_3d_impl<false>(volume, conn);
}

// ---------------------------------------------------------------------------
// 3D connected components -- multi-label
// ---------------------------------------------------------------------------

/// Label connected components in a 3D multi-label volume.
/// Voxels with the same nonzero value that are adjacent form a component.
template <std::integral T, typename Extents>
[[nodiscard]] std::pair<std::vector<std::uint32_t>, std::uint32_t>
connected_components_3d_multilabel(
    std::mdspan<const T, Extents> volume,
    GridConnectivity conn = GridConnectivity::twenty_six) noexcept
{
    return detail::cc_3d_impl<true>(volume, conn);
}

// ---------------------------------------------------------------------------
// Utility: count pixels per label
// ---------------------------------------------------------------------------

/// Return a vector of size num_labels+1 where result[i] is the number of
/// pixels with label i. Index 0 counts background pixels.
template <std::integral T>
[[nodiscard]] std::vector<std::size_t> label_counts(
    std::span<const T> labels, std::size_t num_labels) noexcept
{
    std::vector<std::size_t> counts(num_labels + 1, 0);
    for (auto l : labels) {
        auto ul = static_cast<std::size_t>(l);
        if (ul <= num_labels) {
            ++counts[ul];
        }
    }
    return counts;
}

// ---------------------------------------------------------------------------
// Utility: remove small components
// ---------------------------------------------------------------------------

/// Set to 0 any label whose component has fewer than @p min_size pixels.
/// Returns the number of components removed.
template <std::integral T>
std::size_t remove_small_components(
    std::span<T> labels, std::size_t min_size) noexcept
{
    // First find the max label to size the counts vector.
    T max_label{0};
    for (auto l : labels) {
        if (l > max_label) max_label = l;
    }
    if (max_label == T{0}) return 0;

    const auto nl = static_cast<std::size_t>(max_label);
    std::vector<std::size_t> counts(nl + 1, 0);
    for (auto l : labels) {
        ++counts[static_cast<std::size_t>(l)];
    }

    // Mark which labels to remove.
    std::vector<bool> remove(nl + 1, false);
    std::size_t removed = 0;
    for (std::size_t i = 1; i <= nl; ++i) {
        if (counts[i] > 0 && counts[i] < min_size) {
            remove[i] = true;
            ++removed;
        }
    }

    if (removed > 0) {
        for (auto& l : labels) {
            if (l != T{0} && remove[static_cast<std::size_t>(l)]) {
                l = T{0};
            }
        }
    }

    return removed;
}

// ---------------------------------------------------------------------------
// Utility: keep largest component only
// ---------------------------------------------------------------------------

/// Zero out every label except the one with the most pixels.
template <std::integral T>
void keep_largest(std::span<T> labels, std::size_t num_labels) noexcept
{
    if (num_labels == 0) return;

    auto counts = label_counts(std::span<const T>{labels}, num_labels);

    // Find the label with the largest count (skip index 0 = background).
    std::size_t best_label = 1;
    std::size_t best_count = counts[1];
    for (std::size_t i = 2; i <= num_labels; ++i) {
        if (counts[i] > best_count) {
            best_count = counts[i];
            best_label = i;
        }
    }

    const auto keep = static_cast<T>(best_label);
    for (auto& l : labels) {
        if (l != T{0} && l != keep) {
            l = T{0};
        }
    }
}

} // namespace utils2
