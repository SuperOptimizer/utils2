#pragma once
#include "mdspan.hpp"
#include <vector>
#include <array>
#include <queue>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <concepts>
#include <algorithm>
#include <limits>
#include <functional>
#include <optional>

#include "connectivity.hpp"

namespace utils2 {

// ---------------------------------------------------------------------------
// PathResult -- returned by all pathfinding functions
// ---------------------------------------------------------------------------

template <std::floating_point T>
struct PathResult {
    std::vector<std::array<std::size_t, 2>> path;
    T total_cost{};
    bool found{false};
};

// Partial specialization is not allowed for structs, so use a 3D variant via
// a separate alias-friendly struct.
template <std::floating_point T>
struct PathResult3D {
    std::vector<std::array<std::size_t, 3>> path;
    T total_cost{};
    bool found{false};
};

template <std::floating_point T, std::size_t Dims>
struct PathResultND {
    std::vector<std::array<std::size_t, Dims>> path;
    T total_cost{};
    bool found{false};
};

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

namespace detail {

struct HeapNode {
    double cost;
    std::size_t index;

    [[nodiscard]] constexpr bool operator>(const HeapNode& o) const noexcept {
        return cost > o.cost;
    }
};

using MinHeap = std::priority_queue<HeapNode, std::vector<HeapNode>,
                                    std::greater<HeapNode>>;

// -- Path backtracking --------------------------------------------------------

template <std::size_t Dims>
[[nodiscard]] inline auto backtrack(
    const std::vector<std::size_t>& parent,
    std::size_t goal_lin,
    const std::array<std::size_t, Dims>& strides,
    const std::array<std::size_t, Dims>& shape) noexcept
    -> std::vector<std::array<std::size_t, Dims>>
{
    std::vector<std::array<std::size_t, Dims>> path;
    for (std::size_t cur = goal_lin; cur != std::numeric_limits<std::size_t>::max();
         cur = parent[cur])
    {
        std::array<std::size_t, Dims> coord{};
        auto tmp = cur;
        for (std::size_t d = 0; d < Dims; ++d) {
            coord[d] = tmp / strides[d];
            tmp %= strides[d];
        }
        path.push_back(coord);
    }
    std::ranges::reverse(path);
    return path;
}

// Compute row-major strides from shape.
template <std::size_t Dims>
[[nodiscard]] constexpr auto strides_from_shape(
    const std::array<std::size_t, Dims>& shape) noexcept
    -> std::array<std::size_t, Dims>
{
    std::array<std::size_t, Dims> s{};
    if constexpr (Dims > 0) {
        s[Dims - 1] = 1;
        for (std::size_t i = Dims - 1; i > 0; --i) {
            s[i - 1] = s[i] * shape[i];
        }
    }
    return s;
}

} // namespace detail

// ---------------------------------------------------------------------------
// dijkstra_2d
// ---------------------------------------------------------------------------

template <typename T, typename Extents>
[[nodiscard]] PathResult<double> dijkstra_2d(
    std::mdspan<const T, Extents> cost_field,
    std::array<std::size_t, 2> start,
    std::array<std::size_t, 2> goal,
    GridConnectivity conn = GridConnectivity::eight)
{
    const auto H = cost_field.extent(0);
    const auto W = cost_field.extent(1);
    const std::size_t total = H * W;
    const std::array<std::size_t, 2> shape{H, W};
    const auto strides = detail::strides_from_shape(shape);

    const std::size_t start_lin = start[0] * W + start[1];
    const std::size_t goal_lin  = goal[0]  * W + goal[1];

    std::vector<double> dist(total, std::numeric_limits<double>::infinity());
    std::vector<std::size_t> parent(total, std::numeric_limits<std::size_t>::max());
    dist[start_lin] = 0.0;

    detail::MinHeap heap;
    heap.push({0.0, start_lin});

    const auto [offsets, noff] = offsets_2d(conn);

    while (!heap.empty()) {
        auto [cur_cost, cur] = heap.top();
        heap.pop();

        if (cur == goal_lin) {
            auto path = detail::backtrack<2>(parent, goal_lin, strides, shape);
            return {std::move(path), cur_cost, true};
        }

        if (cur_cost > dist[cur]) continue;

        const std::size_t cy = cur / W;
        const std::size_t cx = cur % W;

        for (std::size_t i = 0; i < noff; ++i) {
            const auto ny = static_cast<std::ptrdiff_t>(cy) + offsets[i].dy;
            const auto nx = static_cast<std::ptrdiff_t>(cx) + offsets[i].dx;
            if (ny < 0 || nx < 0 ||
                static_cast<std::size_t>(ny) >= H ||
                static_cast<std::size_t>(nx) >= W)
                continue;

            const auto uny = static_cast<std::size_t>(ny);
            const auto unx = static_cast<std::size_t>(nx);
            const std::size_t nlin = uny * W + unx;

            const double edge = (static_cast<double>(cost_field[cy, cx]) +
                                 static_cast<double>(cost_field[uny, unx])) *
                                0.5 * offsets[i].dist;
            const double new_cost = dist[cur] + edge;
            if (new_cost < dist[nlin]) {
                dist[nlin] = new_cost;
                parent[nlin] = cur;
                heap.push({new_cost, nlin});
            }
        }
    }

    return {{}, std::numeric_limits<double>::infinity(), false};
}

// ---------------------------------------------------------------------------
// dijkstra_3d
// ---------------------------------------------------------------------------

template <typename T, typename Extents>
[[nodiscard]] PathResult3D<double> dijkstra_3d(
    std::mdspan<const T, Extents> cost_field,
    std::array<std::size_t, 3> start,
    std::array<std::size_t, 3> goal,
    GridConnectivity conn = GridConnectivity::twenty_six)
{
    const auto D = cost_field.extent(0);
    const auto H = cost_field.extent(1);
    const auto W = cost_field.extent(2);
    const std::size_t total = D * H * W;
    const std::array<std::size_t, 3> shape{D, H, W};
    const auto strides = detail::strides_from_shape(shape);

    auto lin = [&](std::size_t z, std::size_t y, std::size_t x) noexcept {
        return z * strides[0] + y * strides[1] + x;
    };

    const std::size_t start_lin = lin(start[0], start[1], start[2]);
    const std::size_t goal_lin  = lin(goal[0],  goal[1],  goal[2]);

    std::vector<double> dist(total, std::numeric_limits<double>::infinity());
    std::vector<std::size_t> parent(total, std::numeric_limits<std::size_t>::max());
    dist[start_lin] = 0.0;

    detail::MinHeap heap;
    heap.push({0.0, start_lin});

    const auto [offsets, noff] = offsets_3d(conn);

    while (!heap.empty()) {
        auto [cur_cost, cur] = heap.top();
        heap.pop();

        if (cur == goal_lin) {
            auto path = detail::backtrack<3>(parent, goal_lin, strides, shape);
            return {std::move(path), cur_cost, true};
        }

        if (cur_cost > dist[cur]) continue;

        const std::size_t cz = cur / strides[0];
        const std::size_t cy = (cur % strides[0]) / strides[1];
        const std::size_t cx = cur % strides[1];

        for (std::size_t i = 0; i < noff; ++i) {
            const auto nz = static_cast<std::ptrdiff_t>(cz) + offsets[i].dz;
            const auto ny = static_cast<std::ptrdiff_t>(cy) + offsets[i].dy;
            const auto nx = static_cast<std::ptrdiff_t>(cx) + offsets[i].dx;
            if (nz < 0 || ny < 0 || nx < 0 ||
                static_cast<std::size_t>(nz) >= D ||
                static_cast<std::size_t>(ny) >= H ||
                static_cast<std::size_t>(nx) >= W)
                continue;

            const auto unz = static_cast<std::size_t>(nz);
            const auto uny = static_cast<std::size_t>(ny);
            const auto unx = static_cast<std::size_t>(nx);
            const std::size_t nlin = lin(unz, uny, unx);

            const double edge =
                (static_cast<double>(cost_field[cz, cy, cx]) +
                 static_cast<double>(cost_field[unz, uny, unx])) *
                0.5 * offsets[i].dist;
            const double new_cost = dist[cur] + edge;
            if (new_cost < dist[nlin]) {
                dist[nlin] = new_cost;
                parent[nlin] = cur;
                heap.push({new_cost, nlin});
            }
        }
    }

    return {{}, std::numeric_limits<double>::infinity(), false};
}

// ---------------------------------------------------------------------------
// distance_field_2d -- Dijkstra from a single source to all reachable cells
// ---------------------------------------------------------------------------

template <typename T, typename Extents>
[[nodiscard]] std::vector<double> distance_field_2d(
    std::mdspan<const T, Extents> cost_field,
    std::array<std::size_t, 2> source,
    GridConnectivity conn = GridConnectivity::eight)
{
    const auto H = cost_field.extent(0);
    const auto W = cost_field.extent(1);
    const std::size_t total = H * W;

    std::vector<double> dist(total, std::numeric_limits<double>::infinity());
    const std::size_t src_lin = source[0] * W + source[1];
    dist[src_lin] = 0.0;

    detail::MinHeap heap;
    heap.push({0.0, src_lin});

    const auto [offsets, noff] = offsets_2d(conn);

    while (!heap.empty()) {
        auto [cur_cost, cur] = heap.top();
        heap.pop();

        if (cur_cost > dist[cur]) continue;

        const std::size_t cy = cur / W;
        const std::size_t cx = cur % W;

        for (std::size_t i = 0; i < noff; ++i) {
            const auto ny = static_cast<std::ptrdiff_t>(cy) + offsets[i].dy;
            const auto nx = static_cast<std::ptrdiff_t>(cx) + offsets[i].dx;
            if (ny < 0 || nx < 0 ||
                static_cast<std::size_t>(ny) >= H ||
                static_cast<std::size_t>(nx) >= W)
                continue;

            const auto uny = static_cast<std::size_t>(ny);
            const auto unx = static_cast<std::size_t>(nx);
            const std::size_t nlin = uny * W + unx;

            const double edge = (static_cast<double>(cost_field[cy, cx]) +
                                 static_cast<double>(cost_field[uny, unx])) *
                                0.5 * offsets[i].dist;
            const double new_cost = dist[cur] + edge;
            if (new_cost < dist[nlin]) {
                dist[nlin] = new_cost;
                heap.push({new_cost, nlin});
            }
        }
    }

    return dist;
}

// ---------------------------------------------------------------------------
// distance_field_3d -- Dijkstra from a single source to all reachable cells
// ---------------------------------------------------------------------------

template <typename T, typename Extents>
[[nodiscard]] std::vector<double> distance_field_3d(
    std::mdspan<const T, Extents> cost_field,
    std::array<std::size_t, 3> source,
    GridConnectivity conn = GridConnectivity::twenty_six)
{
    const auto D = cost_field.extent(0);
    const auto H = cost_field.extent(1);
    const auto W = cost_field.extent(2);
    const std::size_t total = D * H * W;
    const std::array<std::size_t, 3> shape{D, H, W};
    const auto strides = detail::strides_from_shape(shape);

    auto lin = [&](std::size_t z, std::size_t y, std::size_t x) noexcept {
        return z * strides[0] + y * strides[1] + x;
    };

    std::vector<double> dist(total, std::numeric_limits<double>::infinity());
    dist[lin(source[0], source[1], source[2])] = 0.0;

    detail::MinHeap heap;
    heap.push({0.0, lin(source[0], source[1], source[2])});

    const auto [offsets, noff] = offsets_3d(conn);

    while (!heap.empty()) {
        auto [cur_cost, cur] = heap.top();
        heap.pop();

        if (cur_cost > dist[cur]) continue;

        const std::size_t cz = cur / strides[0];
        const std::size_t cy = (cur % strides[0]) / strides[1];
        const std::size_t cx = cur % strides[1];

        for (std::size_t i = 0; i < noff; ++i) {
            const auto nz = static_cast<std::ptrdiff_t>(cz) + offsets[i].dz;
            const auto ny = static_cast<std::ptrdiff_t>(cy) + offsets[i].dy;
            const auto nx = static_cast<std::ptrdiff_t>(cx) + offsets[i].dx;
            if (nz < 0 || ny < 0 || nx < 0 ||
                static_cast<std::size_t>(nz) >= D ||
                static_cast<std::size_t>(ny) >= H ||
                static_cast<std::size_t>(nx) >= W)
                continue;

            const auto unz = static_cast<std::size_t>(nz);
            const auto uny = static_cast<std::size_t>(ny);
            const auto unx = static_cast<std::size_t>(nx);
            const std::size_t nlin = lin(unz, uny, unx);

            const double edge =
                (static_cast<double>(cost_field[cz, cy, cx]) +
                 static_cast<double>(cost_field[unz, uny, unx])) *
                0.5 * offsets[i].dist;
            const double new_cost = dist[cur] + edge;
            if (new_cost < dist[nlin]) {
                dist[nlin] = new_cost;
                heap.push({new_cost, nlin});
            }
        }
    }

    return dist;
}

// ---------------------------------------------------------------------------
// bidirectional_dijkstra_3d -- search from both ends, meet in the middle
// ---------------------------------------------------------------------------

template <typename T, typename Extents>
[[nodiscard]] PathResult3D<double> bidirectional_dijkstra_3d(
    std::mdspan<const T, Extents> cost_field,
    std::array<std::size_t, 3> start,
    std::array<std::size_t, 3> goal,
    GridConnectivity conn = GridConnectivity::twenty_six)
{
    const auto D = cost_field.extent(0);
    const auto H = cost_field.extent(1);
    const auto W = cost_field.extent(2);
    const std::size_t total = D * H * W;
    const std::array<std::size_t, 3> shape{D, H, W};
    const auto strides = detail::strides_from_shape(shape);

    auto lin = [&](std::size_t z, std::size_t y, std::size_t x) noexcept {
        return z * strides[0] + y * strides[1] + x;
    };

    const std::size_t start_lin = lin(start[0], start[1], start[2]);
    const std::size_t goal_lin  = lin(goal[0],  goal[1],  goal[2]);

    if (start_lin == goal_lin)
        return {.path = {start}, .total_cost = 0.0, .found = true};

    constexpr auto INF = std::numeric_limits<double>::infinity();

    std::vector<double> dist_f(total, INF);
    std::vector<double> dist_b(total, INF);
    std::vector<std::size_t> parent_f(total, std::numeric_limits<std::size_t>::max());
    std::vector<std::size_t> parent_b(total, std::numeric_limits<std::size_t>::max());

    dist_f[start_lin] = 0.0;
    dist_b[goal_lin]  = 0.0;

    detail::MinHeap heap_f, heap_b;
    heap_f.push({0.0, start_lin});
    heap_b.push({0.0, goal_lin});

    const auto [offsets, noff] = offsets_3d(conn);

    double best_cost = INF;
    std::size_t meeting = std::numeric_limits<std::size_t>::max();

    // Helper: expand one side
    auto expand = [&](detail::MinHeap& heap,
                      std::vector<double>& dist_mine,
                      std::vector<std::size_t>& par_mine,
                      const std::vector<double>& dist_other) -> bool {
        if (heap.empty()) return false;

        auto [cur_cost, cur] = heap.top();
        heap.pop();

        // If this node's cost exceeds the best known meeting cost, prune.
        if (cur_cost > best_cost) return false;
        if (cur_cost > dist_mine[cur]) return true;

        const std::size_t cz = cur / strides[0];
        const std::size_t cy = (cur % strides[0]) / strides[1];
        const std::size_t cx = cur % strides[1];

        for (std::size_t i = 0; i < noff; ++i) {
            const auto nz = static_cast<std::ptrdiff_t>(cz) + offsets[i].dz;
            const auto ny = static_cast<std::ptrdiff_t>(cy) + offsets[i].dy;
            const auto nx = static_cast<std::ptrdiff_t>(cx) + offsets[i].dx;
            if (nz < 0 || ny < 0 || nx < 0 ||
                static_cast<std::size_t>(nz) >= D ||
                static_cast<std::size_t>(ny) >= H ||
                static_cast<std::size_t>(nx) >= W)
                continue;

            const auto unz = static_cast<std::size_t>(nz);
            const auto uny = static_cast<std::size_t>(ny);
            const auto unx = static_cast<std::size_t>(nx);
            const std::size_t nlin = lin(unz, uny, unx);

            const double edge =
                (static_cast<double>(cost_field[cz, cy, cx]) +
                 static_cast<double>(cost_field[unz, uny, unx])) *
                0.5 * offsets[i].dist;
            const double new_cost = dist_mine[cur] + edge;
            if (new_cost < dist_mine[nlin]) {
                dist_mine[nlin] = new_cost;
                par_mine[nlin] = cur;
                heap.push({new_cost, nlin});

                // Check if the other side has reached this node.
                const double through = new_cost + dist_other[nlin];
                if (through < best_cost) {
                    best_cost = through;
                    meeting = nlin;
                }
            }
        }
        return true;
    };

    // Alternate forward and backward expansion.
    while (!heap_f.empty() || !heap_b.empty()) {
        bool progress = false;
        if (!heap_f.empty()) {
            progress |= expand(heap_f, dist_f, parent_f, dist_b);
        }
        if (!heap_b.empty()) {
            progress |= expand(heap_b, dist_b, parent_b, dist_f);
        }
        if (!progress) break;
    }

    if (meeting == std::numeric_limits<std::size_t>::max()) {
        return {{}, INF, false};
    }

    // Reconstruct: forward path to meeting, backward path from meeting.
    auto fwd = detail::backtrack<3>(parent_f, meeting, strides, shape);
    auto bwd = detail::backtrack<3>(parent_b, meeting, strides, shape);

    // bwd goes from goal to meeting; reverse it so it goes meeting -> goal,
    // then skip the first element (meeting is already in fwd).
    std::ranges::reverse(bwd);
    if (!bwd.empty()) bwd.erase(bwd.begin());

    fwd.insert(fwd.end(), bwd.begin(), bwd.end());

    return {std::move(fwd), best_cost, true};
}

// ---------------------------------------------------------------------------
// dijkstra_anisotropic -- per-axis weighted Dijkstra (2D or 3D)
// ---------------------------------------------------------------------------

namespace detail {

template <std::size_t Dims>
struct OffsetND {
    std::array<int, Dims> delta;
    double base_dist; // unweighted euclidean distance
};

// Build the offset table for anisotropic search at runtime based on
// connectivity. Returns a vector because the table size depends on conn.
template <std::size_t Dims>
[[nodiscard]] inline auto offsets_nd(GridConnectivity conn)
    -> std::vector<OffsetND<Dims>>
{
    std::vector<OffsetND<Dims>> result;

    if constexpr (Dims == 2) {
        const auto [tbl, n] = offsets_2d(conn);
        result.reserve(n);
        for (std::size_t i = 0; i < n; ++i) {
            OffsetND<2> o;
            o.delta = {tbl[i].dy, tbl[i].dx};
            o.base_dist = tbl[i].dist;
            result.push_back(o);
        }
    } else if constexpr (Dims == 3) {
        const auto [tbl, n] = offsets_3d(conn);
        result.reserve(n);
        for (std::size_t i = 0; i < n; ++i) {
            OffsetND<3> o;
            o.delta = {tbl[i].dz, tbl[i].dy, tbl[i].dx};
            o.base_dist = tbl[i].dist;
            result.push_back(o);
        }
    }

    return result;
}

// Compute weighted euclidean distance for an offset given axis weights.
template <std::size_t Dims>
[[nodiscard]] inline double weighted_dist(
    const std::array<int, Dims>& delta,
    const std::array<double, Dims>& weights) noexcept
{
    double sum = 0.0;
    for (std::size_t d = 0; d < Dims; ++d) {
        const double w = weights[d] * static_cast<double>(delta[d]);
        sum += w * w;
    }
    return std::sqrt(sum);
}

} // namespace detail

template <typename T, typename Extents, std::size_t Dims>
[[nodiscard]] PathResultND<double, Dims> dijkstra_anisotropic(
    std::mdspan<const T, Extents> cost_field,
    std::array<std::size_t, Dims> start,
    std::array<std::size_t, Dims> goal,
    std::array<double, Dims> axis_weights,
    GridConnectivity conn)
{
    static_assert(Dims == 2 || Dims == 3,
                  "dijkstra_anisotropic supports 2D and 3D only");

    // Collect shape.
    std::array<std::size_t, Dims> shape{};
    for (std::size_t d = 0; d < Dims; ++d) {
        shape[d] = cost_field.extent(d);
    }

    const auto strides = detail::strides_from_shape(shape);

    auto to_lin = [&](const std::array<std::size_t, Dims>& idx) noexcept {
        std::size_t l = 0;
        for (std::size_t d = 0; d < Dims; ++d) l += strides[d] * idx[d];
        return l;
    };

    std::size_t total = 1;
    for (std::size_t d = 0; d < Dims; ++d) total *= shape[d];

    const std::size_t start_lin = to_lin(start);
    const std::size_t goal_lin  = to_lin(goal);

    std::vector<double> dist(total, std::numeric_limits<double>::infinity());
    std::vector<std::size_t> parent(total, std::numeric_limits<std::size_t>::max());
    dist[start_lin] = 0.0;

    detail::MinHeap heap;
    heap.push({0.0, start_lin});

    const auto offsets = detail::offsets_nd<Dims>(conn);

    // Decompose linear index to ND coordinate.
    auto from_lin = [&](std::size_t l) noexcept {
        std::array<std::size_t, Dims> coord{};
        auto tmp = l;
        for (std::size_t d = 0; d < Dims; ++d) {
            coord[d] = tmp / strides[d];
            tmp %= strides[d];
        }
        return coord;
    };

    // Access cost field element by coordinate array.
    auto cost_at = [&](const std::array<std::size_t, Dims>& c) -> double {
        if constexpr (Dims == 2) {
            return static_cast<double>(cost_field[c[0], c[1]]);
        } else {
            return static_cast<double>(cost_field[c[0], c[1], c[2]]);
        }
    };

    while (!heap.empty()) {
        auto [cur_cost, cur] = heap.top();
        heap.pop();

        if (cur == goal_lin) {
            auto path = detail::backtrack<Dims>(parent, goal_lin, strides, shape);
            return {std::move(path), cur_cost, true};
        }

        if (cur_cost > dist[cur]) continue;

        const auto cur_coord = from_lin(cur);

        for (const auto& off : offsets) {
            // Compute neighbor coordinate with bounds checking.
            std::array<std::size_t, Dims> nc{};
            bool valid = true;
            for (std::size_t d = 0; d < Dims; ++d) {
                const auto v =
                    static_cast<std::ptrdiff_t>(cur_coord[d]) + off.delta[d];
                if (v < 0 || static_cast<std::size_t>(v) >= shape[d]) {
                    valid = false;
                    break;
                }
                nc[d] = static_cast<std::size_t>(v);
            }
            if (!valid) continue;

            const std::size_t nlin = to_lin(nc);
            const double step_dist =
                detail::weighted_dist<Dims>(off.delta, axis_weights);
            const double edge =
                (cost_at(cur_coord) + cost_at(nc)) * 0.5 * step_dist;
            const double new_cost = dist[cur] + edge;
            if (new_cost < dist[nlin]) {
                dist[nlin] = new_cost;
                parent[nlin] = cur;
                heap.push({new_cost, nlin});
            }
        }
    }

    return {{}, std::numeric_limits<double>::infinity(), false};
}

} // namespace utils2
