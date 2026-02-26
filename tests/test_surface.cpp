#include <utils2/test.hpp>
#include <utils2/surface.hpp>
#include <cmath>
#include <vector>
#include <numbers>

using P3 = utils2::Point3<double>;
using P2 = utils2::Point2<double>;

TEST_CASE("triangle_area") {
    P3 a{0.0, 0.0, 0.0};
    P3 b{1.0, 0.0, 0.0};
    P3 c{0.0, 1.0, 0.0};

    double area = utils2::triangle_area(a, b, c);
    REQUIRE_NEAR(area, 0.5, 1e-12);
}

TEST_CASE("triangle_area degenerate") {
    P3 a{0.0, 0.0, 0.0};
    P3 b{1.0, 0.0, 0.0};
    P3 c{2.0, 0.0, 0.0}; // collinear

    double area = utils2::triangle_area(a, b, c);
    REQUIRE_NEAR(area, 0.0, 1e-12);
}

TEST_CASE("triangle_normal") {
    P3 a{0.0, 0.0, 0.0};
    P3 b{1.0, 0.0, 0.0};
    P3 c{0.0, 1.0, 0.0};

    auto n = utils2::triangle_normal(a, b, c);
    REQUIRE_NEAR(n[0], 0.0, 1e-12);
    REQUIRE_NEAR(n[1], 0.0, 1e-12);
    REQUIRE_NEAR(n[2], 1.0, 1e-12);
}

TEST_CASE("barycentric coordinates") {
    P3 a{0.0, 0.0, 0.0};
    P3 b{1.0, 0.0, 0.0};
    P3 c{0.0, 1.0, 0.0};

    // Centroid
    P3 centroid{1.0 / 3.0, 1.0 / 3.0, 0.0};
    auto bary = utils2::barycentric(centroid, a, b, c);
    REQUIRE_NEAR(bary[0], 1.0 / 3.0, 1e-10);
    REQUIRE_NEAR(bary[1], 1.0 / 3.0, 1e-10);
    REQUIRE_NEAR(bary[2], 1.0 / 3.0, 1e-10);

    // Vertex a
    auto bary_a = utils2::barycentric(a, a, b, c);
    REQUIRE_NEAR(bary_a[0], 1.0, 1e-10);
    REQUIRE_NEAR(bary_a[1], 0.0, 1e-10);
    REQUIRE_NEAR(bary_a[2], 0.0, 1e-10);
}

TEST_CASE("point_in_triangle 2D") {
    P2 a{0.0, 0.0};
    P2 b{1.0, 0.0};
    P2 c{0.0, 1.0};

    REQUIRE(utils2::point_in_triangle(P2{0.1, 0.1}, a, b, c));
    REQUIRE(utils2::point_in_triangle(P2{0.0, 0.0}, a, b, c));  // vertex
    REQUIRE(!utils2::point_in_triangle(P2{1.0, 1.0}, a, b, c)); // outside
    REQUIRE(!utils2::point_in_triangle(P2{-0.1, 0.5}, a, b, c));
}

TEST_CASE("closest_point_on_triangle") {
    P3 a{0.0, 0.0, 0.0};
    P3 b{1.0, 0.0, 0.0};
    P3 c{0.0, 1.0, 0.0};

    // Point directly above centroid.
    P3 query{1.0 / 3.0, 1.0 / 3.0, 5.0};
    auto cp = utils2::closest_point_on_triangle(query, a, b, c);
    REQUIRE_NEAR(cp[0], 1.0 / 3.0, 1e-10);
    REQUIRE_NEAR(cp[1], 1.0 / 3.0, 1e-10);
    REQUIRE_NEAR(cp[2], 0.0, 1e-10);

    // Point near vertex a.
    P3 near_a{-0.5, -0.5, 0.0};
    auto cp_a = utils2::closest_point_on_triangle(near_a, a, b, c);
    REQUIRE_NEAR(cp_a[0], 0.0, 1e-10);
    REQUIRE_NEAR(cp_a[1], 0.0, 1e-10);
    REQUIRE_NEAR(cp_a[2], 0.0, 1e-10);
}

TEST_CASE("ray_triangle intersection hit") {
    P3 a{0.0, 0.0, 0.0};
    P3 b{1.0, 0.0, 0.0};
    P3 c{0.0, 1.0, 0.0};

    P3 origin{0.25, 0.25, -1.0};
    P3 dir{0.0, 0.0, 1.0};

    auto hit = utils2::ray_triangle(origin, dir, a, b, c);
    REQUIRE(hit.hit);
    REQUIRE_NEAR(hit.t, 1.0, 1e-10);
    REQUIRE_GE(hit.u, 0.0);
    REQUIRE_GE(hit.v, 0.0);
    REQUIRE_LE(hit.u + hit.v, 1.0 + 1e-10);
}

TEST_CASE("ray_triangle intersection miss") {
    P3 a{0.0, 0.0, 0.0};
    P3 b{1.0, 0.0, 0.0};
    P3 c{0.0, 1.0, 0.0};

    // Ray pointing away from triangle.
    P3 origin{0.25, 0.25, 1.0};
    P3 dir{0.0, 0.0, 1.0};

    auto hit = utils2::ray_triangle(origin, dir, a, b, c);
    REQUIRE(!hit.hit);
}

TEST_CASE("ray_triangle parallel miss") {
    P3 a{0.0, 0.0, 0.0};
    P3 b{1.0, 0.0, 0.0};
    P3 c{0.0, 1.0, 0.0};

    P3 origin{0.0, 0.0, 1.0};
    P3 dir{1.0, 0.0, 0.0}; // parallel to triangle plane

    auto hit = utils2::ray_triangle(origin, dir, a, b, c);
    REQUIRE(!hit.hit);
}

TEST_CASE("PlaneSurface evaluate and normal") {
    P3 origin{0.0, 0.0, 0.0};
    P3 u_axis{1.0, 0.0, 0.0};
    P3 v_axis{0.0, 1.0, 0.0};

    utils2::PlaneSurface<double> plane(origin, u_axis, v_axis);

    auto pt = plane.evaluate(0.5, 0.3);
    REQUIRE_NEAR(pt[0], 0.5, 1e-12);
    REQUIRE_NEAR(pt[1], 0.3, 1e-12);
    REQUIRE_NEAR(pt[2], 0.0, 1e-12);

    auto n = plane.normal(0.0, 0.0);
    REQUIRE_NEAR(n[0], 0.0, 1e-12);
    REQUIRE_NEAR(n[1], 0.0, 1e-12);
    REQUIRE_NEAR(n[2], 1.0, 1e-12);

    auto pmin = plane.param_min();
    auto pmax = plane.param_max();
    REQUIRE_NEAR(pmin[0], 0.0, 1e-12);
    REQUIRE_NEAR(pmax[0], 1.0, 1e-12);
}

TEST_CASE("mesh_surface_area") {
    // Unit square as two triangles.
    std::vector<P3> verts = {
        {0.0, 0.0, 0.0},
        {1.0, 0.0, 0.0},
        {1.0, 1.0, 0.0},
        {0.0, 1.0, 0.0}
    };
    std::vector<std::array<std::size_t, 3>> tris = {
        {0, 1, 2},
        {0, 2, 3}
    };

    double area = utils2::mesh_surface_area<double>(verts, tris);
    REQUIRE_NEAR(area, 1.0, 1e-12);
}

TEST_CASE("make_coord_grid") {
    P3 origin{0.0, 0.0, 0.0};
    P3 u_axis{1.0, 0.0, 0.0};
    P3 v_axis{0.0, 1.0, 0.0};

    auto grid = utils2::make_coord_grid(origin, u_axis, v_axis,
                                         std::size_t(3), std::size_t(2));

    REQUIRE_EQ(grid.size(), std::size_t(6));

    // (r=0, c=0)
    REQUIRE_NEAR(grid[0][0], 0.0, 1e-12);
    REQUIRE_NEAR(grid[0][1], 0.0, 1e-12);

    // (r=0, c=2)
    REQUIRE_NEAR(grid[2][0], 2.0, 1e-12);
    REQUIRE_NEAR(grid[2][1], 0.0, 1e-12);

    // (r=1, c=0)
    REQUIRE_NEAR(grid[3][0], 0.0, 1e-12);
    REQUIRE_NEAR(grid[3][1], 1.0, 1e-12);
}

TEST_CASE("quad_area") {
    P3 a{0.0, 0.0, 0.0};
    P3 b{1.0, 0.0, 0.0};
    P3 c{1.0, 1.0, 0.0};
    P3 d{0.0, 1.0, 0.0};

    double area = utils2::quad_area(a, b, c, d);
    REQUIRE_NEAR(area, 1.0, 1e-12);
}

TEST_CASE("triangle_area non-planar in 3D") {
    // A triangle not in the XY plane
    P3 a{0.0, 0.0, 0.0};
    P3 b{1.0, 0.0, 0.0};
    P3 c{0.0, 0.0, 1.0};

    double area = utils2::triangle_area(a, b, c);
    REQUIRE_NEAR(area, 0.5, 1e-12);
}

TEST_CASE("triangle_normal non-trivial") {
    // Triangle in XZ plane: normal should be (0, -1, 0) or (0, 1, 0)
    P3 a{0.0, 0.0, 0.0};
    P3 b{1.0, 0.0, 0.0};
    P3 c{0.0, 0.0, 1.0};

    auto n = utils2::triangle_normal(a, b, c);
    // (b-a) = (1,0,0), (c-a) = (0,0,1)
    // cross = (0*1 - 0*0, 0*0 - 1*1, 1*0 - 0*0) = (0, -1, 0)
    REQUIRE_NEAR(n[0], 0.0, 1e-12);
    REQUIRE_NEAR(n[1], -1.0, 1e-12);
    REQUIRE_NEAR(n[2], 0.0, 1e-12);
}

TEST_CASE("barycentric at edge midpoint") {
    P3 a{0.0, 0.0, 0.0};
    P3 b{1.0, 0.0, 0.0};
    P3 c{0.0, 1.0, 0.0};

    // Midpoint of edge AB
    P3 mid_ab{0.5, 0.0, 0.0};
    auto bary = utils2::barycentric(mid_ab, a, b, c);
    REQUIRE_NEAR(bary[0], 0.5, 1e-10);
    REQUIRE_NEAR(bary[1], 0.5, 1e-10);
    REQUIRE_NEAR(bary[2], 0.0, 1e-10);
}

TEST_CASE("closest_point_on_triangle near edge") {
    P3 a{0.0, 0.0, 0.0};
    P3 b{1.0, 0.0, 0.0};
    P3 c{0.0, 1.0, 0.0};

    // Point near edge AB, outside triangle
    P3 query{0.5, -1.0, 0.0};
    auto cp = utils2::closest_point_on_triangle(query, a, b, c);
    // Should project onto edge AB
    REQUIRE_NEAR(cp[0], 0.5, 1e-10);
    REQUIRE_NEAR(cp[1], 0.0, 1e-10);
    REQUIRE_NEAR(cp[2], 0.0, 1e-10);
}

TEST_CASE("closest_point_on_triangle near edge BC") {
    P3 a{0.0, 0.0, 0.0};
    P3 b{1.0, 0.0, 0.0};
    P3 c{0.0, 1.0, 0.0};

    // Point near edge BC, outside triangle
    P3 query{1.0, 1.0, 0.0};
    auto cp = utils2::closest_point_on_triangle(query, a, b, c);
    // On edge BC, midpoint: b + 0.5*(c-b) = (0.5, 0.5, 0)
    REQUIRE_NEAR(cp[0], 0.5, 1e-10);
    REQUIRE_NEAR(cp[1], 0.5, 1e-10);
    REQUIRE_NEAR(cp[2], 0.0, 1e-10);
}

TEST_CASE("closest_point_on_triangle inside") {
    P3 a{0.0, 0.0, 0.0};
    P3 b{2.0, 0.0, 0.0};
    P3 c{0.0, 2.0, 0.0};

    // Point inside the triangle at z=3
    P3 query{0.5, 0.5, 3.0};
    auto cp = utils2::closest_point_on_triangle(query, a, b, c);
    REQUIRE_NEAR(cp[0], 0.5, 1e-10);
    REQUIRE_NEAR(cp[1], 0.5, 1e-10);
    REQUIRE_NEAR(cp[2], 0.0, 1e-10);
}

TEST_CASE("ray_triangle oblique hit") {
    P3 a{0.0, 0.0, 0.0};
    P3 b{2.0, 0.0, 0.0};
    P3 c{0.0, 2.0, 0.0};

    P3 origin{0.5, 0.5, -2.0};
    P3 dir{0.0, 0.0, 1.0};

    auto hit = utils2::ray_triangle(origin, dir, a, b, c);
    REQUIRE(hit.hit);
    REQUIRE_NEAR(hit.t, 2.0, 1e-10);
}

TEST_CASE("ray_triangle outside hit") {
    P3 a{0.0, 0.0, 0.0};
    P3 b{1.0, 0.0, 0.0};
    P3 c{0.0, 1.0, 0.0};

    // Ray aimed outside the triangle
    P3 origin{5.0, 5.0, -1.0};
    P3 dir{0.0, 0.0, 1.0};

    auto hit = utils2::ray_triangle(origin, dir, a, b, c);
    REQUIRE(!hit.hit);
}

TEST_CASE("triangle_areas per-triangle") {
    std::vector<P3> verts = {
        {0.0, 0.0, 0.0},
        {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0},
        {1.0, 1.0, 0.0}
    };
    std::vector<std::array<std::size_t, 3>> tris = {
        {0, 1, 2},
        {1, 3, 2}
    };

    auto areas = utils2::triangle_areas<double>(verts, tris);
    REQUIRE_EQ(areas.size(), std::size_t(2));
    REQUIRE_NEAR(areas[0], 0.5, 1e-12);
    REQUIRE_NEAR(areas[1], 0.5, 1e-12);
}

TEST_CASE("PlaneSurface tilted plane") {
    P3 origin{1.0, 2.0, 3.0};
    P3 u_axis{1.0, 0.0, 1.0};
    P3 v_axis{0.0, 1.0, 0.0};

    utils2::PlaneSurface<double> plane(origin, u_axis, v_axis);

    auto pt = plane.evaluate(1.0, 1.0);
    REQUIRE_NEAR(pt[0], 2.0, 1e-12);
    REQUIRE_NEAR(pt[1], 3.0, 1e-12);
    REQUIRE_NEAR(pt[2], 4.0, 1e-12);

    // normal = normalize(cross((1,0,1), (0,1,0)))
    // cross = (0*0-1*1, 1*0-1*0, 1*1-0*0) = (-1, 0, 1)
    auto n = plane.normal(0.0, 0.0);
    double len = std::sqrt(2.0);
    REQUIRE_NEAR(n[0], -1.0 / len, 1e-10);
    REQUIRE_NEAR(n[1], 0.0, 1e-10);
    REQUIRE_NEAR(n[2], 1.0 / len, 1e-10);
}

TEST_CASE("PlaneSurface accessors") {
    P3 origin{1.0, 2.0, 3.0};
    P3 u_axis{1.0, 0.0, 0.0};
    P3 v_axis{0.0, 1.0, 0.0};

    utils2::PlaneSurface<double> plane(origin, u_axis, v_axis);

    REQUIRE_NEAR(plane.origin()[0], 1.0, 1e-12);
    REQUIRE_NEAR(plane.u_axis()[0], 1.0, 1e-12);
    REQUIRE_NEAR(plane.v_axis()[1], 1.0, 1e-12);
}

TEST_CASE("generalized_winding_number inside cube") {
    // Build a simple closed mesh: unit cube centered at origin
    // 8 vertices, 12 triangles
    std::vector<P3> verts = {
        {-1, -1, -1}, {1, -1, -1}, {1, 1, -1}, {-1, 1, -1},
        {-1, -1,  1}, {1, -1,  1}, {1, 1,  1}, {-1, 1,  1}
    };
    std::vector<std::array<std::size_t, 3>> tris = {
        // -Z face
        {0, 2, 1}, {0, 3, 2},
        // +Z face
        {4, 5, 6}, {4, 6, 7},
        // -X face
        {0, 4, 7}, {0, 7, 3},
        // +X face
        {1, 2, 6}, {1, 6, 5},
        // -Y face
        {0, 1, 5}, {0, 5, 4},
        // +Y face
        {2, 3, 7}, {2, 7, 6},
    };

    // Inside the cube
    P3 inside{0.0, 0.0, 0.0};
    double wn_inside = utils2::generalized_winding_number<double>(
        inside, std::span<const P3>(verts),
        std::span<const std::array<std::size_t, 3>>(tris));
    REQUIRE_NEAR(std::abs(wn_inside), 1.0, 0.01);

    // Outside the cube
    P3 outside{10.0, 0.0, 0.0};
    double wn_outside = utils2::generalized_winding_number<double>(
        outside, std::span<const P3>(verts),
        std::span<const std::array<std::size_t, 3>>(tris));
    REQUIRE_NEAR(wn_outside, 0.0, 0.01);
}

TEST_CASE("make_coord_grid 1x1") {
    P3 origin{0.0, 0.0, 0.0};
    P3 u_axis{1.0, 0.0, 0.0};
    P3 v_axis{0.0, 1.0, 0.0};

    auto grid = utils2::make_coord_grid(origin, u_axis, v_axis,
                                         std::size_t(1), std::size_t(1));
    REQUIRE_EQ(grid.size(), std::size_t(1));
    REQUIRE_NEAR(grid[0][0], 0.0, 1e-12);
    REQUIRE_NEAR(grid[0][1], 0.0, 1e-12);
}

TEST_CASE("point_in_triangle midpoint of edge") {
    P2 a{0.0, 0.0};
    P2 b{1.0, 0.0};
    P2 c{0.0, 1.0};

    // Midpoint of hypotenuse (b-c edge)
    P2 mid{0.5, 0.5};
    REQUIRE(utils2::point_in_triangle(mid, a, b, c));
}

UTILS2_TEST_MAIN()
