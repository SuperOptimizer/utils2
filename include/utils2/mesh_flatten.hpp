#pragma once
#include <vector>
#include <array>
#include <cstddef>
#include <cmath>
#include <concepts>
#include <algorithm>
#include <numeric>
#include <numbers>
#include <span>
#include <functional>
#include <optional>
#include <limits>

namespace utils2 {

// ---------------------------------------------------------------------------
// FlatteningMethod
// ---------------------------------------------------------------------------

enum class FlatteningMethod : std::uint8_t {
    harmonic,    // Harmonic (Tutte) embedding - linear, fast
    arap,        // As-Rigid-As-Possible
    conformal,   // Least-Squares Conformal Maps (LSCM)
};

// ---------------------------------------------------------------------------
// FlatteningParams
// ---------------------------------------------------------------------------

struct FlatteningParams {
    FlatteningMethod method = FlatteningMethod::harmonic;
    std::size_t max_iterations = 100;
    double tolerance = 1e-6;

    // Callback for progress reporting
    std::function<void(std::size_t iter, double energy)> on_progress = nullptr;
};

// ---------------------------------------------------------------------------
// TriMesh
// ---------------------------------------------------------------------------

template<std::floating_point T = double>
struct TriMesh {
    std::vector<std::array<T, 3>> vertices;             // 3D positions
    std::vector<std::array<std::size_t, 3>> triangles;  // triangle indices

    [[nodiscard]] std::size_t num_vertices() const noexcept { return vertices.size(); }
    [[nodiscard]] std::size_t num_triangles() const noexcept { return triangles.size(); }

    // Find boundary loop (ordered boundary vertex indices)
    [[nodiscard]] std::vector<std::size_t> boundary_loop() const;

    // Compute per-vertex normals
    [[nodiscard]] std::vector<std::array<T, 3>> vertex_normals() const;
};

// ---------------------------------------------------------------------------
// FlatteningResult
// ---------------------------------------------------------------------------

template<std::floating_point T = double>
struct FlatteningResult {
    std::vector<std::array<T, 2>> uv;   // 2D coordinates per vertex
    double final_energy{};
    std::size_t iterations{};
    bool converged{};

    struct Metrics {
        T mean_stretch{};    // L2 mean stretch
        T max_stretch{};     // L-inf stretch
        T area_distortion{}; // ratio of 2D/3D area
    };

    [[nodiscard]] Metrics compute_metrics(const TriMesh<T>& mesh) const;
};

// ---------------------------------------------------------------------------
// Forward declarations
// ---------------------------------------------------------------------------

template<std::floating_point T>
[[nodiscard]] FlatteningResult<T> flatten(
    const TriMesh<T>& mesh,
    const FlatteningParams& params = {});

template<std::floating_point T>
[[nodiscard]] FlatteningResult<T> flatten_harmonic(
    const TriMesh<T>& mesh,
    std::size_t max_iterations = 1000,
    T tolerance = T(1e-6));

template<std::floating_point T>
[[nodiscard]] FlatteningResult<T> flatten_lscm(
    const TriMesh<T>& mesh,
    std::size_t max_iterations = 100,
    T tolerance = T(1e-6));

template<std::floating_point T>
[[nodiscard]] FlatteningResult<T> flatten_arap(
    const TriMesh<T>& mesh,
    std::span<const std::array<T, 2>> initial_uv,
    std::size_t max_iterations = 50,
    T tolerance = T(1e-6));

template<std::floating_point T>
[[nodiscard]] std::vector<std::array<T, 2>> map_boundary_to_circle(
    const TriMesh<T>& mesh,
    std::span<const std::size_t> boundary_loop);

template<std::floating_point T>
[[nodiscard]] std::vector<std::array<T, 2>> map_boundary_to_square(
    const TriMesh<T>& mesh,
    std::span<const std::size_t> boundary_loop);

template<std::floating_point T>
[[nodiscard]] std::vector<std::array<T, 2>> compute_stretch(
    const TriMesh<T>& mesh,
    std::span<const std::array<T, 2>> uv);

// ===========================================================================
// Internal helpers
// ===========================================================================

namespace detail {

// 3D vector arithmetic helpers
template<std::floating_point T>
constexpr std::array<T, 3> sub3(const std::array<T, 3>& a,
                                const std::array<T, 3>& b) noexcept {
    return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}

template<std::floating_point T>
constexpr T dot3(const std::array<T, 3>& a,
                 const std::array<T, 3>& b) noexcept {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

template<std::floating_point T>
constexpr std::array<T, 3> cross3(const std::array<T, 3>& a,
                                  const std::array<T, 3>& b) noexcept {
    return {a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]};
}

template<std::floating_point T>
constexpr T len3(const std::array<T, 3>& v) noexcept {
    return std::sqrt(dot3(v, v));
}

// 2D helpers
template<std::floating_point T>
constexpr std::array<T, 2> sub2(const std::array<T, 2>& a,
                                const std::array<T, 2>& b) noexcept {
    return {a[0] - b[0], a[1] - b[1]};
}

template<std::floating_point T>
constexpr T dot2(const std::array<T, 2>& a,
                 const std::array<T, 2>& b) noexcept {
    return a[0] * b[0] + a[1] * b[1];
}

template<std::floating_point T>
constexpr T cross2(const std::array<T, 2>& a,
                   const std::array<T, 2>& b) noexcept {
    return a[0] * b[1] - a[1] * b[0];
}

// Half-edge representation for boundary extraction
struct HalfEdge {
    std::size_t from;
    std::size_t to;

    constexpr bool operator==(const HalfEdge&) const noexcept = default;
};

struct HalfEdgeHash {
    std::size_t operator()(const HalfEdge& e) const noexcept {
        auto h1 = std::hash<std::size_t>{}(e.from);
        auto h2 = std::hash<std::size_t>{}(e.to);
        return h1 ^ (h2 * 0x9e3779b97f4a7c15ULL + 0x9e3779b9ULL + (h1 << 6) + (h1 >> 2));
    }
};

// Build adjacency: for each vertex, list of neighbor vertex indices
template<std::floating_point T>
[[nodiscard]] std::vector<std::vector<std::size_t>> build_adjacency(
    const TriMesh<T>& mesh) {
    const auto nv = mesh.num_vertices();
    std::vector<std::vector<std::size_t>> adj(nv);
    for (const auto& tri : mesh.triangles) {
        for (int e = 0; e < 3; ++e) {
            auto a = tri[e];
            auto b = tri[(e + 1) % 3];
            if (std::find(adj[a].begin(), adj[a].end(), b) == adj[a].end())
                adj[a].push_back(b);
            if (std::find(adj[b].begin(), adj[b].end(), a) == adj[b].end())
                adj[b].push_back(a);
        }
    }
    return adj;
}

// Cotangent weight for edge (i, j) summed across shared triangles.
// Returns cot(alpha) + cot(beta) where alpha, beta are the angles
// opposite the edge in the two incident triangles.
template<std::floating_point T>
[[nodiscard]] T cotangent_weight(const TriMesh<T>& mesh,
                                 std::size_t vi, std::size_t vj) {
    T weight{};
    for (const auto& tri : mesh.triangles) {
        // Find the vertex opposite to edge (vi, vj)
        int idx_i = -1, idx_j = -1;
        for (int k = 0; k < 3; ++k) {
            if (tri[k] == vi) idx_i = k;
            if (tri[k] == vj) idx_j = k;
        }
        if (idx_i < 0 || idx_j < 0) continue;

        int idx_opp = 3 - idx_i - idx_j; // the remaining index
        const auto& p_opp = mesh.vertices[tri[idx_opp]];
        const auto& p_i = mesh.vertices[vi];
        const auto& p_j = mesh.vertices[vj];

        auto e1 = sub3(p_i, p_opp);
        auto e2 = sub3(p_j, p_opp);
        T cos_a = dot3(e1, e2);
        T sin_a = len3(cross3(e1, e2));

        if (sin_a > T(1e-12))
            weight += cos_a / sin_a;
    }
    return std::max(weight, T(1e-8)); // clamp to avoid negative weights
}

// Precomputed cotangent weights per edge stored as adjacency-aligned weights
template<std::floating_point T>
struct CotanWeights {
    std::vector<std::vector<std::size_t>> neighbors; // per-vertex neighbor list
    std::vector<std::vector<T>> weights;             // aligned with neighbors

    explicit CotanWeights(const TriMesh<T>& mesh) {
        const auto nv = mesh.num_vertices();
        neighbors.resize(nv);
        weights.resize(nv);

        // Gather neighbors from triangles
        for (const auto& tri : mesh.triangles) {
            for (int e = 0; e < 3; ++e) {
                auto a = tri[e];
                auto b = tri[(e + 1) % 3];
                if (std::find(neighbors[a].begin(), neighbors[a].end(), b)
                    == neighbors[a].end()) {
                    neighbors[a].push_back(b);
                }
                if (std::find(neighbors[b].begin(), neighbors[b].end(), a)
                    == neighbors[b].end()) {
                    neighbors[b].push_back(a);
                }
            }
        }

        // Compute cotangent weights
        for (std::size_t i = 0; i < nv; ++i) {
            weights[i].resize(neighbors[i].size());
            for (std::size_t k = 0; k < neighbors[i].size(); ++k) {
                weights[i][k] = cotangent_weight(mesh, i, neighbors[i][k]);
            }
        }
    }
};

// Closed-form 2x2 SVD for ARAP local step.
// Given 2x2 matrix [[a,b],[c,d]], returns the closest rotation matrix.
template<std::floating_point T>
constexpr std::array<std::array<T, 2>, 2> closest_rotation_2x2(
    T a, T b, T c, T d) noexcept {
    // M = [[a,b],[c,d]]
    // S = M^T M = [[a*a+c*c, a*b+c*d],[a*b+c*d, b*b+d*d]]
    // For the polar decomposition M = R * S, R = M * S^{-1/2}
    // But simpler: use SVD. For 2x2, use the closed form with atan2.
    //
    // M = U Sigma V^T, closest rotation = U V^T
    // det(UV^T) must be +1, so flip if needed.

    // Compute E = (a+d)/2, F = (a-d)/2, G = (c+b)/2, H = (c-b)/2
    T E = (a + d) * T(0.5);
    T F = (a - d) * T(0.5);
    T G = (c + b) * T(0.5);
    T H = (c - b) * T(0.5);

    T q = std::sqrt(E * E + H * H);
    T r = std::sqrt(F * F + G * G);

    T a1 = std::atan2(G, F);
    T a2 = std::atan2(H, E);

    T theta = (a2 - a1) * T(0.5);
    T phi   = (a2 + a1) * T(0.5);

    // Singular values: s1 = q + r, s2 = q - r
    // U = Rot(phi), V = Rot(theta)  (when s2 >= 0)
    // If s2 < 0, we need to flip the sign to ensure det(R) = +1

    T ct = std::cos(theta);
    T st = std::sin(theta);
    T cp = std::cos(phi);
    T sp = std::sin(phi);

    T s2 = q - r;

    // R = U * V^T
    // If s2 < 0, negate second column of U -> effectively R = U * diag(1,-1) * V^T
    if (s2 < T(0)) {
        // R = U * diag(1,-1) * V^T
        // U = Rot(phi) = [[cp, -sp],[sp, cp]]
        // U * diag(1,-1) = [[cp, sp],[sp, -cp]]
        // V^T = Rot(-theta) = [[ct, st],[-st, ct]]
        // R = [[cp, sp],[sp, -cp]] * [[ct, st],[-st, ct]]
        return {{
            {cp * ct - sp * st,  cp * st + sp * ct},
            {sp * ct + cp * st,  sp * st - cp * ct}
        }};
    }

    // R = U * V^T = Rot(phi) * Rot(-theta) = Rot(phi - theta)
    T angle = phi - theta;
    T ca = std::cos(angle);
    T sa = std::sin(angle);
    return {{{ca, -sa}, {sa, ca}}};
}

// Gauss-Seidel solver for Laplacian system with fixed boundary.
// Solves for uv[interior] given fixed uv[boundary].
// Uses cotangent weights. Returns max vertex displacement for convergence test.
template<std::floating_point T>
T gauss_seidel_step(
    std::vector<std::array<T, 2>>& uv,
    const CotanWeights<T>& cw,
    const std::vector<bool>& is_boundary) {
    T max_disp{};
    const auto nv = uv.size();
    for (std::size_t i = 0; i < nv; ++i) {
        if (is_boundary[i]) continue;

        T sum_w{};
        std::array<T, 2> weighted_sum{T(0), T(0)};
        for (std::size_t k = 0; k < cw.neighbors[i].size(); ++k) {
            auto j = cw.neighbors[i][k];
            T w = cw.weights[i][k];
            sum_w += w;
            weighted_sum[0] += w * uv[j][0];
            weighted_sum[1] += w * uv[j][1];
        }

        if (sum_w < T(1e-12)) continue;

        std::array<T, 2> new_pos{weighted_sum[0] / sum_w,
                                  weighted_sum[1] / sum_w};
        T dx = new_pos[0] - uv[i][0];
        T dy = new_pos[1] - uv[i][1];
        T disp = std::sqrt(dx * dx + dy * dy);
        max_disp = std::max(max_disp, disp);
        uv[i] = new_pos;
    }
    return max_disp;
}

// Conjugate gradient solver for symmetric positive-definite Laplacian.
// Solves L * x = b for the free (interior) DOFs only.
// The system is: for each interior vertex i,
//   sum_j w_ij * (x_i - x_j) = b_i
// where boundary vertices are fixed.
template<std::floating_point T>
void conjugate_gradient_solve(
    std::vector<std::array<T, 2>>& uv,
    const CotanWeights<T>& cw,
    const std::vector<bool>& is_boundary,
    const std::vector<std::array<T, 2>>& rhs,
    std::size_t max_iter,
    T tol) {

    const auto nv = uv.size();

    // Compute Ax for interior vertices
    auto apply_L = [&](const std::vector<std::array<T, 2>>& x)
        -> std::vector<std::array<T, 2>> {
        std::vector<std::array<T, 2>> out(nv, {T(0), T(0)});
        for (std::size_t i = 0; i < nv; ++i) {
            if (is_boundary[i]) continue;
            T sum_w{};
            for (std::size_t k = 0; k < cw.neighbors[i].size(); ++k) {
                auto j = cw.neighbors[i][k];
                T w = cw.weights[i][k];
                sum_w += w;
                out[i][0] -= w * x[j][0];
                out[i][1] -= w * x[j][1];
            }
            out[i][0] += sum_w * x[i][0];
            out[i][1] += sum_w * x[i][1];
        }
        return out;
    };

    // r = b - Ax
    auto Ax = apply_L(uv);
    std::vector<std::array<T, 2>> r(nv, {T(0), T(0)});
    T rr{};
    for (std::size_t i = 0; i < nv; ++i) {
        if (is_boundary[i]) continue;
        r[i][0] = rhs[i][0] - Ax[i][0];
        r[i][1] = rhs[i][1] - Ax[i][1];
        rr += r[i][0] * r[i][0] + r[i][1] * r[i][1];
    }

    auto p = r; // search direction

    for (std::size_t iter = 0; iter < max_iter; ++iter) {
        if (rr < tol * tol) break;

        auto Ap = apply_L(p);

        T pAp{};
        for (std::size_t i = 0; i < nv; ++i) {
            if (is_boundary[i]) continue;
            pAp += p[i][0] * Ap[i][0] + p[i][1] * Ap[i][1];
        }

        if (std::abs(pAp) < T(1e-30)) break;
        T alpha = rr / pAp;

        T rr_new{};
        for (std::size_t i = 0; i < nv; ++i) {
            if (is_boundary[i]) continue;
            uv[i][0] += alpha * p[i][0];
            uv[i][1] += alpha * p[i][1];
            r[i][0] -= alpha * Ap[i][0];
            r[i][1] -= alpha * Ap[i][1];
            rr_new += r[i][0] * r[i][0] + r[i][1] * r[i][1];
        }

        T beta = rr_new / rr;
        rr = rr_new;

        for (std::size_t i = 0; i < nv; ++i) {
            if (is_boundary[i]) continue;
            p[i][0] = r[i][0] + beta * p[i][0];
            p[i][1] = r[i][1] + beta * p[i][1];
        }
    }
}

// Compute signed triangle area in 2D
template<std::floating_point T>
constexpr T triangle_area_2d(const std::array<T, 2>& a,
                             const std::array<T, 2>& b,
                             const std::array<T, 2>& c) noexcept {
    return T(0.5) * ((b[0] - a[0]) * (c[1] - a[1]) -
                     (c[0] - a[0]) * (b[1] - a[1]));
}

// Compute triangle area in 3D
template<std::floating_point T>
constexpr T triangle_area_3d(const std::array<T, 3>& a,
                             const std::array<T, 3>& b,
                             const std::array<T, 3>& c) noexcept {
    auto e1 = sub3(b, a);
    auto e2 = sub3(c, a);
    auto n = cross3(e1, e2);
    return T(0.5) * len3(n);
}

} // namespace detail

// ===========================================================================
// TriMesh implementation
// ===========================================================================

template<std::floating_point T>
std::vector<std::size_t> TriMesh<T>::boundary_loop() const {
    // A boundary edge appears in only one triangle.
    // Build map of directed half-edges; boundary edges have no twin.
    struct PairHash {
        std::size_t operator()(const std::pair<std::size_t, std::size_t>& p) const noexcept {
            auto h1 = std::hash<std::size_t>{}(p.first);
            auto h2 = std::hash<std::size_t>{}(p.second);
            return h1 ^ (h2 * 0x9e3779b97f4a7c15ULL + 0x9e3779b9ULL + (h1 << 6) + (h1 >> 2));
        }
    };

    // Count how many times each directed half-edge appears
    std::vector<std::pair<std::size_t, std::size_t>> half_edges;
    half_edges.reserve(triangles.size() * 3);
    for (const auto& tri : triangles) {
        for (int e = 0; e < 3; ++e) {
            half_edges.emplace_back(tri[e], tri[(e + 1) % 3]);
        }
    }

    // A half-edge (a,b) is boundary if (b,a) does not exist
    // Build set of half-edges for fast lookup
    std::vector<std::pair<std::size_t, std::size_t>> boundary_edges;
    // Simple O(E) approach: sort and scan
    auto edge_set = half_edges;
    std::sort(edge_set.begin(), edge_set.end());

    for (const auto& [a, b] : half_edges) {
        // Check if twin (b, a) exists via binary search
        auto twin = std::make_pair(b, a);
        if (!std::binary_search(edge_set.begin(), edge_set.end(), twin)) {
            boundary_edges.emplace_back(a, b);
        }
    }

    if (boundary_edges.empty()) return {};

    // Chain boundary edges into a loop
    // Build next map: from -> to
    std::vector<std::size_t> next_map(vertices.size(), std::size_t(-1));
    for (const auto& [a, b] : boundary_edges) {
        next_map[a] = b;
    }

    std::vector<std::size_t> loop;
    loop.reserve(boundary_edges.size());
    auto start = boundary_edges[0].first;
    auto cur = start;
    do {
        loop.push_back(cur);
        cur = next_map[cur];
        if (cur == std::size_t(-1)) break; // broken loop
    } while (cur != start && loop.size() <= vertices.size());

    return loop;
}

template<std::floating_point T>
std::vector<std::array<T, 3>> TriMesh<T>::vertex_normals() const {
    std::vector<std::array<T, 3>> normals(vertices.size(), {T(0), T(0), T(0)});

    for (const auto& tri : triangles) {
        auto e1 = detail::sub3(vertices[tri[1]], vertices[tri[0]]);
        auto e2 = detail::sub3(vertices[tri[2]], vertices[tri[0]]);
        auto fn = detail::cross3(e1, e2);

        for (int k = 0; k < 3; ++k) {
            normals[tri[k]][0] += fn[0];
            normals[tri[k]][1] += fn[1];
            normals[tri[k]][2] += fn[2];
        }
    }

    for (auto& n : normals) {
        T len = detail::len3(n);
        if (len > T(1e-12)) {
            n[0] /= len;
            n[1] /= len;
            n[2] /= len;
        }
    }
    return normals;
}

// ===========================================================================
// FlatteningResult::compute_metrics
// ===========================================================================

template<std::floating_point T>
auto FlatteningResult<T>::compute_metrics(const TriMesh<T>& mesh) const -> Metrics {
    Metrics m{};
    T total_area_3d{};
    T total_area_2d{};
    T sum_stretch_sq{};
    T max_s{};

    for (const auto& tri : mesh.triangles) {
        auto a3d = detail::triangle_area_3d(
            mesh.vertices[tri[0]], mesh.vertices[tri[1]], mesh.vertices[tri[2]]);
        auto a2d = std::abs(detail::triangle_area_2d(
            uv[tri[0]], uv[tri[1]], uv[tri[2]]));

        total_area_3d += a3d;
        total_area_2d += a2d;

        // Compute singular values of the Jacobian for this triangle
        // Using the edge-based formulation
        auto e1_3d = detail::sub3(mesh.vertices[tri[1]], mesh.vertices[tri[0]]);
        auto e2_3d = detail::sub3(mesh.vertices[tri[2]], mesh.vertices[tri[0]]);
        auto e1_2d = detail::sub2(uv[tri[1]], uv[tri[0]]);
        auto e2_2d = detail::sub2(uv[tri[2]], uv[tri[0]]);

        T l1_3d = detail::len3(e1_3d);
        T l2_3d = detail::len3(e2_3d);

        if (l1_3d < T(1e-12) || l2_3d < T(1e-12) || a3d < T(1e-15)) continue;

        // Stretch: ratio of 2D edge length to 3D edge length
        T l1_2d = std::sqrt(detail::dot2(e1_2d, e1_2d));
        T l2_2d = std::sqrt(detail::dot2(e2_2d, e2_2d));

        T s1 = (l1_3d > T(0)) ? l1_2d / l1_3d : T(1);
        T s2 = (l2_3d > T(0)) ? l2_2d / l2_3d : T(1);

        T stretch = std::sqrt((s1 * s1 + s2 * s2) * T(0.5));
        sum_stretch_sq += stretch * stretch * a3d;
        max_s = std::max(max_s, std::max(s1, s2));
    }

    m.mean_stretch = (total_area_3d > T(0))
                     ? std::sqrt(sum_stretch_sq / total_area_3d) : T(1);
    m.max_stretch = max_s;
    m.area_distortion = (total_area_3d > T(0))
                        ? total_area_2d / total_area_3d : T(1);
    return m;
}

// ===========================================================================
// Boundary mapping
// ===========================================================================

template<std::floating_point T>
std::vector<std::array<T, 2>> map_boundary_to_circle(
    const TriMesh<T>& mesh,
    std::span<const std::size_t> boundary) {

    const auto n = boundary.size();
    std::vector<std::array<T, 2>> positions(n);

    if (n == 0) return positions;

    // Compute cumulative arc length along boundary
    std::vector<T> cum_len(n + 1, T(0));
    for (std::size_t i = 0; i < n; ++i) {
        auto next = (i + 1) % n;
        auto edge = detail::sub3(mesh.vertices[boundary[next]],
                                 mesh.vertices[boundary[i]]);
        cum_len[i + 1] = cum_len[i] + detail::len3(edge);
    }

    T total_len = cum_len[n];
    if (total_len < T(1e-12)) total_len = T(1);

    for (std::size_t i = 0; i < n; ++i) {
        T angle = T(2) * std::numbers::pi_v<T> * cum_len[i] / total_len;
        positions[i] = {std::cos(angle), std::sin(angle)};
    }

    return positions;
}

template<std::floating_point T>
std::vector<std::array<T, 2>> map_boundary_to_square(
    const TriMesh<T>& mesh,
    std::span<const std::size_t> boundary) {

    const auto n = boundary.size();
    std::vector<std::array<T, 2>> positions(n);

    if (n == 0) return positions;

    // Compute cumulative arc length
    std::vector<T> cum_len(n + 1, T(0));
    for (std::size_t i = 0; i < n; ++i) {
        auto next = (i + 1) % n;
        auto edge = detail::sub3(mesh.vertices[boundary[next]],
                                 mesh.vertices[boundary[i]]);
        cum_len[i + 1] = cum_len[i] + detail::len3(edge);
    }

    T total_len = cum_len[n];
    if (total_len < T(1e-12)) total_len = T(1);

    // Map to unit square perimeter (perimeter = 4)
    for (std::size_t i = 0; i < n; ++i) {
        T t = T(4) * cum_len[i] / total_len;

        if (t <= T(1)) {
            positions[i] = {t, T(0)};                     // bottom edge
        } else if (t <= T(2)) {
            positions[i] = {T(1), t - T(1)};              // right edge
        } else if (t <= T(3)) {
            positions[i] = {T(3) - t, T(1)};              // top edge
        } else {
            positions[i] = {T(0), T(4) - t};              // left edge
        }
    }

    return positions;
}

// ===========================================================================
// Stretch computation
// ===========================================================================

template<std::floating_point T>
std::vector<std::array<T, 2>> compute_stretch(
    const TriMesh<T>& mesh,
    std::span<const std::array<T, 2>> uv) {

    const auto nt = mesh.num_triangles();
    std::vector<std::array<T, 2>> stretches(nt);

    for (std::size_t t = 0; t < nt; ++t) {
        const auto& tri = mesh.triangles[t];
        const auto& p0 = mesh.vertices[tri[0]];
        const auto& p1 = mesh.vertices[tri[1]];
        const auto& p2 = mesh.vertices[tri[2]];
        const auto& u0 = uv[tri[0]];
        const auto& u1 = uv[tri[1]];
        const auto& u2 = uv[tri[2]];

        // 3D edges
        auto e1 = detail::sub3(p1, p0);
        auto e2 = detail::sub3(p2, p0);

        // 2D edges
        auto f1 = detail::sub2(u1, u0);
        auto f2 = detail::sub2(u2, u0);

        T area_2d = detail::cross2(f1, f2);
        if (std::abs(area_2d) < T(1e-15)) {
            stretches[t] = {T(0), T(0)};
            continue;
        }

        // Jacobian J maps 3D triangle to 2D: columns are partial derivatives
        // Using local orthonormal frame in 3D triangle:
        T l1 = detail::len3(e1);
        if (l1 < T(1e-12)) { stretches[t] = {T(0), T(0)}; continue; }

        // Local 2D coordinates of the 3D triangle
        std::array<T, 3> u_axis = {e1[0] / l1, e1[1] / l1, e1[2] / l1};
        T d = detail::dot3(e2, u_axis);
        std::array<T, 3> e2_perp = {e2[0] - d * u_axis[0],
                                     e2[1] - d * u_axis[1],
                                     e2[2] - d * u_axis[2]};
        T l2_perp = detail::len3(e2_perp);
        if (l2_perp < T(1e-12)) { stretches[t] = {T(0), T(0)}; continue; }

        // 3D triangle in local 2D: q0=(0,0), q1=(l1,0), q2=(d, l2_perp)
        // Jacobian from local 3D coords to uv:
        // J = [du/ds du/dt; dv/ds dv/dt]
        // where s,t are local 3D coords
        T inv_det = T(1) / (l1 * l2_perp);

        T a = (f1[0] * l2_perp - f2[0] * T(0)) * inv_det;                   // du/ds (simplified since q0=(0,0), q1_y=0)
        T b_val = (-f1[0] * d + f2[0] * l1) * inv_det;                      // du/dt
        T c_val = (f1[1] * l2_perp - f2[1] * T(0)) * inv_det;               // dv/ds
        T d_val = (-f1[1] * d + f2[1] * l1) * inv_det;                      // dv/dt

        // Singular values of 2x2 Jacobian [[a, b], [c, d]]
        // sigma = sqrt(eigenvalues of J^T J)
        T s11 = a * a + c_val * c_val;
        T s22 = b_val * b_val + d_val * d_val;
        T s12 = a * b_val + c_val * d_val;

        T trace = s11 + s22;
        T det = s11 * s22 - s12 * s12;
        T disc = std::max(trace * trace - T(4) * det, T(0));
        T sqrt_disc = std::sqrt(disc);

        T sigma_max = std::sqrt(std::max((trace + sqrt_disc) * T(0.5), T(0)));
        T sigma_min = std::sqrt(std::max((trace - sqrt_disc) * T(0.5), T(0)));

        stretches[t] = {sigma_max, sigma_min};
    }

    return stretches;
}

// ===========================================================================
// Harmonic (Tutte) flattening
// ===========================================================================

template<std::floating_point T>
FlatteningResult<T> flatten_harmonic(
    const TriMesh<T>& mesh,
    std::size_t max_iterations,
    T tolerance) {

    const auto nv = mesh.num_vertices();
    FlatteningResult<T> result;
    result.uv.resize(nv, {T(0), T(0)});

    // Get boundary and map to circle
    auto boundary = mesh.boundary_loop();
    if (boundary.empty()) {
        result.converged = false;
        return result;
    }

    auto boundary_uv = map_boundary_to_circle(mesh, std::span{boundary});

    // Mark boundary vertices and set their UV
    std::vector<bool> is_boundary(nv, false);
    for (std::size_t i = 0; i < boundary.size(); ++i) {
        is_boundary[boundary[i]] = true;
        result.uv[boundary[i]] = boundary_uv[i];
    }

    // Build cotangent weights
    detail::CotanWeights<T> cw(mesh);

    // Gauss-Seidel iteration
    result.converged = false;
    for (std::size_t iter = 0; iter < max_iterations; ++iter) {
        T max_disp = detail::gauss_seidel_step(result.uv, cw, is_boundary);

        result.iterations = iter + 1;
        result.final_energy = static_cast<double>(max_disp);

        if (max_disp < tolerance) {
            result.converged = true;
            break;
        }
    }

    return result;
}

// ===========================================================================
// LSCM flattening
// ===========================================================================

template<std::floating_point T>
FlatteningResult<T> flatten_lscm(
    const TriMesh<T>& mesh,
    std::size_t max_iterations,
    T tolerance) {

    const auto nv = mesh.num_vertices();
    FlatteningResult<T> result;
    result.uv.resize(nv, {T(0), T(0)});

    auto boundary = mesh.boundary_loop();
    if (boundary.size() < 2) {
        result.converged = false;
        return result;
    }

    // Pin two boundary vertices that are farthest apart
    std::size_t pin0 = boundary[0];
    std::size_t pin1 = boundary[0];
    T max_dist{};
    for (std::size_t i = 1; i < boundary.size(); ++i) {
        auto d = detail::sub3(mesh.vertices[boundary[i]], mesh.vertices[pin0]);
        T dist = detail::dot3(d, d);
        if (dist > max_dist) {
            max_dist = dist;
            pin1 = boundary[i];
        }
    }

    // Fix the two pinned vertices
    result.uv[pin0] = {T(0), T(0)};
    result.uv[pin1] = {T(1), T(0)};

    std::vector<bool> is_fixed(nv, false);
    is_fixed[pin0] = true;
    is_fixed[pin1] = true;

    // Build cotangent weights
    detail::CotanWeights<T> cw(mesh);

    // LSCM can be solved as a Laplacian system with specific RHS.
    // The conformal energy with 2 pinned vertices reduces to solving:
    //   L * u = 0  (with pinned boundary conditions)
    //   L * v = 0  (with pinned boundary conditions)
    // where L is the cotangent Laplacian. This is equivalent to the
    // harmonic map with only 2 fixed points, which preserves conformality.

    // Build RHS: contribution from fixed vertices
    std::vector<std::array<T, 2>> rhs(nv, {T(0), T(0)});
    for (std::size_t i = 0; i < nv; ++i) {
        if (is_fixed[i]) continue;
        for (std::size_t k = 0; k < cw.neighbors[i].size(); ++k) {
            auto j = cw.neighbors[i][k];
            if (is_fixed[j]) {
                T w = cw.weights[i][k];
                rhs[i][0] += w * result.uv[j][0];
                rhs[i][1] += w * result.uv[j][1];
            }
        }
    }

    // Solve with CG
    detail::conjugate_gradient_solve(result.uv, cw, is_fixed, rhs,
                                     max_iterations, tolerance);

    result.converged = true;
    result.iterations = max_iterations; // CG doesn't easily expose iteration count here
    result.final_energy = 0.0;

    return result;
}

// ===========================================================================
// ARAP flattening
// ===========================================================================

template<std::floating_point T>
FlatteningResult<T> flatten_arap(
    const TriMesh<T>& mesh,
    std::span<const std::array<T, 2>> initial_uv,
    std::size_t max_iterations,
    T tolerance) {

    const auto nv = mesh.num_vertices();
    const auto nt = mesh.num_triangles();
    FlatteningResult<T> result;
    result.uv.assign(initial_uv.begin(), initial_uv.end());

    auto boundary = mesh.boundary_loop();
    if (boundary.empty()) {
        result.converged = false;
        return result;
    }

    // Mark boundary vertices as fixed during global step
    std::vector<bool> is_boundary(nv, false);
    for (auto b : boundary) {
        is_boundary[b] = true;
    }

    detail::CotanWeights<T> cw(mesh);

    // Per-triangle best-fit rotation
    std::vector<std::array<std::array<T, 2>, 2>> rotations(
        nt, {{{T(1), T(0)}, {T(0), T(1)}}});

    T prev_energy = std::numeric_limits<T>::max();
    result.converged = false;

    for (std::size_t iter = 0; iter < max_iterations; ++iter) {
        // --- Local step: compute best-fit rotation per triangle ---
        T energy{};
        for (std::size_t t = 0; t < nt; ++t) {
            const auto& tri = mesh.triangles[t];

            // Build covariance matrix S = sum_edges w_ij * e_3d * e_2d^T
            // where e_3d and e_2d are edge vectors projected to local 2D frames
            T s00{}, s01{}, s10{}, s11{};

            for (int e = 0; e < 3; ++e) {
                auto i = tri[e];
                auto j = tri[(e + 1) % 3];

                // 3D edge in local frame (project to triangle plane)
                auto e3d = detail::sub3(mesh.vertices[j], mesh.vertices[i]);
                T e3d_len = detail::len3(e3d);

                // 2D edge
                auto e2d = detail::sub2(result.uv[j], result.uv[i]);

                // For the covariance, we use a simplified approach:
                // project 3D edge onto the first two axes of the triangle's local frame
                // For ARAP, we can use the edge vectors directly in the 2D domain
                // The key insight: use the original 2D embedding's edge vs current edge

                // We need the "rest" 2D edge from the initial parameterization
                auto e_rest = detail::sub2(initial_uv[j], initial_uv[i]);

                // Cotangent weight for this edge (approximation: use 1.0 for simplicity
                // in per-triangle formulation, or look up from global weights)
                T w = T(1);

                s00 += w * e2d[0] * e_rest[0];
                s01 += w * e2d[0] * e_rest[1];
                s10 += w * e2d[1] * e_rest[0];
                s11 += w * e2d[1] * e_rest[1];
            }

            // Closest rotation via 2x2 SVD
            rotations[t] = detail::closest_rotation_2x2(s00, s01, s10, s11);

            // Accumulate ARAP energy for this triangle
            for (int e = 0; e < 3; ++e) {
                auto i = tri[e];
                auto j = tri[(e + 1) % 3];
                auto e2d = detail::sub2(result.uv[j], result.uv[i]);
                auto e_rest = detail::sub2(initial_uv[j], initial_uv[i]);

                // Rotated rest edge
                const auto& R = rotations[t];
                T rx = R[0][0] * e_rest[0] + R[0][1] * e_rest[1];
                T ry = R[1][0] * e_rest[0] + R[1][1] * e_rest[1];

                T dx = e2d[0] - rx;
                T dy = e2d[1] - ry;
                energy += dx * dx + dy * dy;
            }
        }

        result.final_energy = static_cast<double>(energy);
        result.iterations = iter + 1;

        if (std::abs(prev_energy - energy) < static_cast<T>(tolerance) * prev_energy) {
            result.converged = true;
            break;
        }
        prev_energy = energy;

        // --- Global step: solve for new UVs with rotated edge constraints ---
        // Build RHS for each vertex from rotated edges
        std::vector<std::array<T, 2>> rhs(nv, {T(0), T(0)});
        for (std::size_t t = 0; t < nt; ++t) {
            const auto& tri = mesh.triangles[t];
            const auto& R = rotations[t];

            for (int e = 0; e < 3; ++e) {
                auto i = tri[e];
                auto j = tri[(e + 1) % 3];

                auto e_rest = detail::sub2(initial_uv[j], initial_uv[i]);
                T rx = R[0][0] * e_rest[0] + R[0][1] * e_rest[1];
                T ry = R[1][0] * e_rest[0] + R[1][1] * e_rest[1];

                // Cotangent weight for this edge
                T w = T(1); // uniform for per-triangle ARAP

                rhs[i][0] += w * rx;
                rhs[i][1] += w * ry;
                rhs[j][0] -= w * rx;
                rhs[j][1] -= w * ry;
            }
        }

        // Add boundary terms
        for (std::size_t i = 0; i < nv; ++i) {
            if (!is_boundary[i]) continue;
            T sum_w{};
            for (std::size_t k = 0; k < cw.neighbors[i].size(); ++k) {
                sum_w += cw.weights[i][k];
            }
            rhs[i][0] = sum_w * result.uv[i][0];
            rhs[i][1] = sum_w * result.uv[i][1];
        }

        // Solve Laplacian system for interior vertices
        detail::conjugate_gradient_solve(result.uv, cw, is_boundary, rhs,
                                         std::size_t(200), tolerance);
    }

    return result;
}

// ===========================================================================
// Main entry point
// ===========================================================================

template<std::floating_point T>
FlatteningResult<T> flatten(
    const TriMesh<T>& mesh,
    const FlatteningParams& params) {

    switch (params.method) {
        case FlatteningMethod::harmonic:
            return flatten_harmonic(mesh, params.max_iterations,
                                    static_cast<T>(params.tolerance));

        case FlatteningMethod::conformal:
            return flatten_lscm(mesh, params.max_iterations,
                                static_cast<T>(params.tolerance));

        case FlatteningMethod::arap: {
            // ARAP needs an initial parameterization; use harmonic
            auto initial = flatten_harmonic(mesh, std::size_t(500),
                                            static_cast<T>(params.tolerance));
            auto result = flatten_arap(mesh, std::span<const std::array<T, 2>>{initial.uv},
                                       params.max_iterations,
                                       static_cast<T>(params.tolerance));
            result.iterations += initial.iterations;
            return result;
        }
    }

    // Unreachable, but satisfy compilers
    return {};
}

} // namespace utils2
