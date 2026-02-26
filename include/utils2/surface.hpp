#pragma once
#include "vec.hpp"
#include <vector>
#include <cstddef>
#include <cmath>
#include <concepts>
#include <algorithm>
#include <numeric>
#include <optional>
#include <functional>
#include <numbers>
#include <span>

namespace utils2 {

// ---------------------------------------------------------------------------
// Backward-compatible type aliases
// ---------------------------------------------------------------------------

template<std::floating_point T = double>
using Point3 = Vec<T, 3>;

template<std::floating_point T = double>
using Point2 = Vec<T, 2>;

// ---------------------------------------------------------------------------
// Ray-triangle intersection result
// ---------------------------------------------------------------------------

template<std::floating_point T>
struct RayHit {
    T t;
    T u, v;
    bool hit;
};

// ---------------------------------------------------------------------------
// Basic geometry primitives
// ---------------------------------------------------------------------------

/// Triangle area via half the magnitude of the cross product of two edges.
template<std::floating_point T>
[[nodiscard]] constexpr T triangle_area(const Vec<T, 3>& a,
                                        const Vec<T, 3>& b,
                                        const Vec<T, 3>& c) noexcept {
    const auto ab = b - a;
    const auto ac = c - a;
    return norm(cross(ab, ac)) * T{0.5};
}

/// Quad area as the sum of two triangle areas (split along diagonal a-c).
template<std::floating_point T>
[[nodiscard]] constexpr T quad_area(const Vec<T, 3>& a,
                                    const Vec<T, 3>& b,
                                    const Vec<T, 3>& c,
                                    const Vec<T, 3>& d) noexcept {
    return triangle_area(a, b, c) + triangle_area(a, c, d);
}

/// Triangle normal (unit-length) via normalized cross product of two edges.
template<std::floating_point T>
[[nodiscard]] constexpr Vec<T, 3> triangle_normal(const Vec<T, 3>& a,
                                                   const Vec<T, 3>& b,
                                                   const Vec<T, 3>& c) noexcept {
    const auto ab = b - a;
    const auto ac = c - a;
    return normalize(cross(ab, ac));
}

/// Barycentric coordinates of point p with respect to triangle (a, b, c).
/// Returns {u, v, w} such that p ~ u*a + v*b + w*c.
template<std::floating_point T>
[[nodiscard]] constexpr Vec<T, 3> barycentric(const Vec<T, 3>& p,
                                               const Vec<T, 3>& a,
                                               const Vec<T, 3>& b,
                                               const Vec<T, 3>& c) noexcept {
    const auto v0 = b - a;
    const auto v1 = c - a;
    const auto v2 = p - a;

    const T d00 = dot(v0, v0);
    const T d01 = dot(v0, v1);
    const T d11 = dot(v1, v1);
    const T d20 = dot(v2, v0);
    const T d21 = dot(v2, v1);

    const T denom = d00 * d11 - d01 * d01;
    if (std::abs(denom) < std::numeric_limits<T>::epsilon()) {
        return {T{0}, T{0}, T{0}};
    }

    const T inv = T{1} / denom;
    const T v = (d11 * d20 - d01 * d21) * inv;
    const T w = (d00 * d21 - d01 * d20) * inv;
    const T u = T{1} - v - w;
    return {u, v, w};
}

/// Point-in-triangle test in 2D using sign-of-cross-product method.
template<std::floating_point T>
[[nodiscard]] constexpr bool point_in_triangle(const Vec<T, 2>& p,
                                               const Vec<T, 2>& a,
                                               const Vec<T, 2>& b,
                                               const Vec<T, 2>& c) noexcept {
    auto sign = [](const Vec<T, 2>& p1, const Vec<T, 2>& p2,
                   const Vec<T, 2>& p3) -> T {
        return (p1[0] - p3[0]) * (p2[1] - p3[1])
             - (p2[0] - p3[0]) * (p1[1] - p3[1]);
    };

    const T d1 = sign(p, a, b);
    const T d2 = sign(p, b, c);
    const T d3 = sign(p, c, a);

    const bool has_neg = (d1 < T{0}) || (d2 < T{0}) || (d3 < T{0});
    const bool has_pos = (d1 > T{0}) || (d2 > T{0}) || (d3 > T{0});
    return !(has_neg && has_pos);
}

/// Closest point on triangle (a, b, c) to query point p in 3D.
/// Uses Voronoi region projection.
template<std::floating_point T>
[[nodiscard]] constexpr Vec<T, 3> closest_point_on_triangle(
    const Vec<T, 3>& p,
    const Vec<T, 3>& a,
    const Vec<T, 3>& b,
    const Vec<T, 3>& c) noexcept {

    const auto ab = b - a;
    const auto ac = c - a;
    const auto ap = p - a;

    const T d1 = dot(ab, ap);
    const T d2 = dot(ac, ap);
    if (d1 <= T{0} && d2 <= T{0}) return a;

    const auto bp = p - b;
    const T d3 = dot(ab, bp);
    const T d4 = dot(ac, bp);
    if (d3 >= T{0} && d4 <= d3) return b;

    const T vc = d1 * d4 - d3 * d2;
    if (vc <= T{0} && d1 >= T{0} && d3 <= T{0}) {
        const T v = d1 / (d1 - d3);
        return a + ab * v;
    }

    const auto cp = p - c;
    const T d5 = dot(ab, cp);
    const T d6 = dot(ac, cp);
    if (d6 >= T{0} && d5 <= d6) return c;

    const T vb = d5 * d2 - d1 * d6;
    if (vb <= T{0} && d2 >= T{0} && d6 <= T{0}) {
        const T w = d2 / (d2 - d6);
        return a + ac * w;
    }

    const T va = d3 * d6 - d5 * d4;
    if (va <= T{0} && (d4 - d3) >= T{0} && (d5 - d6) >= T{0}) {
        const T w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return b + (c - b) * w;
    }

    const T denom = T{1} / (va + vb + vc);
    const T v = vb * denom;
    const T w = vc * denom;
    return a + ab * v + ac * w;
}

/// Ray-triangle intersection using the Moller-Trumbore algorithm.
template<std::floating_point T>
[[nodiscard]] constexpr RayHit<T> ray_triangle(
    const Vec<T, 3>& origin,
    const Vec<T, 3>& direction,
    const Vec<T, 3>& a,
    const Vec<T, 3>& b,
    const Vec<T, 3>& c) noexcept {

    constexpr T eps = std::numeric_limits<T>::epsilon();
    const auto e1 = b - a;
    const auto e2 = c - a;
    const auto h = cross(direction, e2);
    const T det = dot(e1, h);

    if (std::abs(det) < eps) {
        return {T{0}, T{0}, T{0}, false};
    }

    const T inv_det = T{1} / det;
    const auto s = origin - a;
    const T u = dot(s, h) * inv_det;
    if (u < T{0} || u > T{1}) {
        return {T{0}, T{0}, T{0}, false};
    }

    const auto q = cross(s, e1);
    const T v = dot(direction, q) * inv_det;
    if (v < T{0} || u + v > T{1}) {
        return {T{0}, T{0}, T{0}, false};
    }

    const T t = dot(e2, q) * inv_det;
    if (t < eps) {
        return {T{0}, T{0}, T{0}, false};
    }

    return {t, u, v, true};
}

// ---------------------------------------------------------------------------
// ParametricSurface (abstract base)
// ---------------------------------------------------------------------------

template<std::floating_point T = double>
class ParametricSurface {
public:
    virtual ~ParametricSurface() = default;

    /// Evaluate the surface position at parameter (u, v).
    [[nodiscard]] virtual Vec<T, 3> evaluate(T u, T v) const = 0;

    /// Surface normal at parameter (u, v).
    [[nodiscard]] virtual Vec<T, 3> normal(T u, T v) const = 0;

    /// Minimum parameter bounds (u_min, v_min).
    [[nodiscard]] virtual Vec<T, 2> param_min() const noexcept = 0;

    /// Maximum parameter bounds (u_max, v_max).
    [[nodiscard]] virtual Vec<T, 2> param_max() const noexcept = 0;
};

// ---------------------------------------------------------------------------
// PlaneSurface
// ---------------------------------------------------------------------------

template<std::floating_point T = double>
class PlaneSurface final : public ParametricSurface<T> {
    Vec<T, 3> origin_;
    Vec<T, 3> u_axis_;
    Vec<T, 3> v_axis_;
    Vec<T, 3> normal_;

public:
    PlaneSurface(Vec<T, 3> origin, Vec<T, 3> u_axis, Vec<T, 3> v_axis)
        : origin_(origin)
        , u_axis_(u_axis)
        , v_axis_(v_axis)
        , normal_(normalize(cross(u_axis, v_axis))) {}

    [[nodiscard]] Vec<T, 3> evaluate(T u, T v) const override {
        return origin_ + u_axis_ * u + v_axis_ * v;
    }

    [[nodiscard]] Vec<T, 3> normal([[maybe_unused]] T u,
                                    [[maybe_unused]] T v) const override {
        return normal_;
    }

    [[nodiscard]] Vec<T, 2> param_min() const noexcept override {
        return {T{0}, T{0}};
    }

    [[nodiscard]] Vec<T, 2> param_max() const noexcept override {
        return {T{1}, T{1}};
    }

    [[nodiscard]] const Vec<T, 3>& origin() const noexcept { return origin_; }
    [[nodiscard]] const Vec<T, 3>& u_axis() const noexcept { return u_axis_; }
    [[nodiscard]] const Vec<T, 3>& v_axis() const noexcept { return v_axis_; }
};

// ---------------------------------------------------------------------------
// Mesh surface area utilities
// ---------------------------------------------------------------------------

/// Total surface area of a triangle mesh.
template<std::floating_point T>
[[nodiscard]] T mesh_surface_area(
    std::span<const Vec<T, 3>> vertices,
    std::span<const std::array<std::size_t, 3>> triangles) noexcept {

    T total{0};
    for (const auto& tri : triangles) {
        total += triangle_area(vertices[tri[0]],
                               vertices[tri[1]],
                               vertices[tri[2]]);
    }
    return total;
}

/// Per-triangle areas for a triangle mesh.
template<std::floating_point T>
[[nodiscard]] std::vector<T> triangle_areas(
    std::span<const Vec<T, 3>> vertices,
    std::span<const std::array<std::size_t, 3>> triangles) {

    std::vector<T> areas;
    areas.reserve(triangles.size());
    for (const auto& tri : triangles) {
        areas.push_back(triangle_area(vertices[tri[0]],
                                      vertices[tri[1]],
                                      vertices[tri[2]]));
    }
    return areas;
}

// ---------------------------------------------------------------------------
// Coordinate grid generation
// ---------------------------------------------------------------------------

/// Generate a width x height grid of 3D points.
/// coords(r, c) = origin + u_axis * c + v_axis * r
/// Points are stored in row-major order.
template<std::floating_point T>
[[nodiscard]] std::vector<Vec<T, 3>> make_coord_grid(
    const Vec<T, 3>& origin,
    const Vec<T, 3>& u_axis,
    const Vec<T, 3>& v_axis,
    std::size_t width,
    std::size_t height) {

    std::vector<Vec<T, 3>> grid;
    grid.reserve(width * height);
    for (std::size_t r = 0; r < height; ++r) {
        const auto row_offset = v_axis * static_cast<T>(r);
        for (std::size_t c = 0; c < width; ++c) {
            const auto col_offset = u_axis * static_cast<T>(c);
            grid.push_back(origin + row_offset + col_offset);
        }
    }
    return grid;
}

// ---------------------------------------------------------------------------
// Generalized winding number
// ---------------------------------------------------------------------------

/// Generalized winding number of a query point with respect to a closed
/// triangle mesh.  Values near +/-1 indicate the point is inside; values
/// near 0 indicate outside.
template<std::floating_point T>
[[nodiscard]] T generalized_winding_number(
    const Vec<T, 3>& query,
    std::span<const Vec<T, 3>> vertices,
    std::span<const std::array<std::size_t, 3>> triangles) {

    // Accumulate the signed solid angle subtended by each triangle as
    // seen from the query point.  The sum divided by 4*pi gives the
    // generalized winding number.

    T total_solid_angle{0};

    for (const auto& tri : triangles) {
        const auto a = vertices[tri[0]] - query;
        const auto b = vertices[tri[1]] - query;
        const auto c = vertices[tri[2]] - query;

        const T la = norm(a);
        const T lb = norm(b);
        const T lc = norm(c);

        // Degenerate: query is at a vertex
        if (la < std::numeric_limits<T>::epsilon() ||
            lb < std::numeric_limits<T>::epsilon() ||
            lc < std::numeric_limits<T>::epsilon()) {
            continue;
        }

        // Solid angle via the formula:
        //   tan(omega/2) = (a . (b x c)) /
        //       (la*lb*lc + (a.b)*lc + (b.c)*la + (c.a)*lb)
        const T numerator = dot(a, cross(b, c));
        const T denominator = la * lb * lc
                            + dot(a, b) * lc
                            + dot(b, c) * la
                            + dot(c, a) * lb;

        total_solid_angle += T{2} * std::atan2(numerator, denominator);
    }

    return total_solid_angle / (T{4} * std::numbers::pi_v<T>);
}

} // namespace utils2
