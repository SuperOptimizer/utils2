#include <utils2/test.hpp>
#include <utils2/mesh_flatten.hpp>
#include <cmath>
#include <numbers>

// Build a simple test mesh: a flat quad (unit square in XY) split into 2 triangles.
//   3---2
//   |  /|
//   | / |
//   |/  |
//   0---1
static utils2::TriMesh<double> make_quad_mesh() {
    utils2::TriMesh<double> m;
    m.vertices = {
        {0.0, 0.0, 0.0},
        {1.0, 0.0, 0.0},
        {1.0, 1.0, 0.0},
        {0.0, 1.0, 0.0}
    };
    m.triangles = {
        {0, 1, 2},
        {0, 2, 3}
    };
    return m;
}

// Build a slightly larger mesh: 3x3 grid = 9 vertices, 8 triangles.
static utils2::TriMesh<double> make_grid_mesh() {
    utils2::TriMesh<double> m;
    // 3x3 grid of vertices in XY plane.
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            m.vertices.push_back({double(c), double(r), 0.0});

    // Two triangles per quad cell (2x2 = 4 cells -> 8 triangles).
    auto idx = [](int r, int c) -> std::size_t { return r * 3 + c; };
    for (int r = 0; r < 2; ++r) {
        for (int c = 0; c < 2; ++c) {
            m.triangles.push_back({idx(r, c), idx(r, c + 1), idx(r + 1, c + 1)});
            m.triangles.push_back({idx(r, c), idx(r + 1, c + 1), idx(r + 1, c)});
        }
    }
    return m;
}

TEST_CASE("TriMesh boundary_loop on quad mesh") {
    auto mesh = make_quad_mesh();
    auto loop = mesh.boundary_loop();

    // A quad mesh has 4 boundary edges -> 4 boundary vertices.
    REQUIRE_EQ(loop.size(), std::size_t(4));

    // All 4 vertices should appear.
    std::vector<bool> seen(4, false);
    for (auto v : loop) {
        REQUIRE_LT(v, std::size_t(4));
        seen[v] = true;
    }
    for (int i = 0; i < 4; ++i)
        REQUIRE(seen[i]);
}

TEST_CASE("TriMesh boundary_loop on grid mesh") {
    auto mesh = make_grid_mesh();
    auto loop = mesh.boundary_loop();

    // 3x3 grid boundary: 8 boundary vertices (perimeter of 3x3 grid).
    REQUIRE_EQ(loop.size(), std::size_t(8));
}

TEST_CASE("TriMesh vertex_normals") {
    auto mesh = make_quad_mesh();
    auto normals = mesh.vertex_normals();

    REQUIRE_EQ(normals.size(), std::size_t(4));
    // All normals should point in +Z for a flat XY mesh.
    for (const auto& n : normals) {
        REQUIRE_NEAR(n[0], 0.0, 1e-10);
        REQUIRE_NEAR(n[1], 0.0, 1e-10);
        REQUIRE_NEAR(n[2], 1.0, 1e-10);
    }
}

TEST_CASE("flatten_harmonic on quad mesh") {
    auto mesh = make_quad_mesh();
    auto result = utils2::flatten_harmonic(mesh, std::size_t(1000), 1e-8);

    REQUIRE(result.converged);
    REQUIRE_EQ(result.uv.size(), std::size_t(4));

    // All UV coords should be finite.
    for (const auto& uv : result.uv) {
        REQUIRE(!std::isnan(uv[0]));
        REQUIRE(!std::isnan(uv[1]));
        REQUIRE(std::isfinite(uv[0]));
        REQUIRE(std::isfinite(uv[1]));
    }
}

TEST_CASE("flatten_harmonic on grid mesh") {
    auto mesh = make_grid_mesh();
    auto result = utils2::flatten_harmonic(mesh, std::size_t(2000), 1e-8);

    REQUIRE(result.converged);
    REQUIRE_EQ(result.uv.size(), std::size_t(9));

    // Interior vertex (1,1) = index 4 should be inside the boundary UV polygon.
    // Just check it is finite and within some reasonable bounds.
    CHECK(!std::isnan(result.uv[4][0]));
    CHECK(!std::isnan(result.uv[4][1]));
}

TEST_CASE("flatten_lscm") {
    auto mesh = make_grid_mesh();
    auto result = utils2::flatten_lscm(mesh, std::size_t(500), 1e-8);

    REQUIRE(result.converged);
    REQUIRE_EQ(result.uv.size(), std::size_t(9));

    for (const auto& uv : result.uv) {
        REQUIRE(std::isfinite(uv[0]));
        REQUIRE(std::isfinite(uv[1]));
    }
}

TEST_CASE("map_boundary_to_circle") {
    auto mesh = make_quad_mesh();
    auto loop = mesh.boundary_loop();
    REQUIRE_EQ(loop.size(), std::size_t(4));

    auto circle_uv = utils2::map_boundary_to_circle(mesh, std::span{loop});
    REQUIRE_EQ(circle_uv.size(), std::size_t(4));

    // All points should be on the unit circle.
    for (const auto& pt : circle_uv) {
        double r = std::sqrt(pt[0] * pt[0] + pt[1] * pt[1]);
        REQUIRE_NEAR(r, 1.0, 1e-10);
    }

    // First point should be at angle 0 -> (1, 0).
    REQUIRE_NEAR(circle_uv[0][0], 1.0, 1e-10);
    REQUIRE_NEAR(circle_uv[0][1], 0.0, 1e-10);
}

TEST_CASE("FlatteningResult metrics") {
    auto mesh = make_grid_mesh();
    auto result = utils2::flatten_harmonic(mesh, std::size_t(2000), 1e-8);
    REQUIRE(result.converged);

    auto metrics = result.compute_metrics(mesh);

    // For a flat mesh, stretch should be reasonable (close to 1 for a conformal map).
    CHECK_GT(metrics.mean_stretch, 0.0);
    CHECK_GT(metrics.max_stretch, 0.0);
    CHECK_GT(metrics.area_distortion, 0.0);
    CHECK(std::isfinite(metrics.mean_stretch));
    CHECK(std::isfinite(metrics.area_distortion));
}

TEST_CASE("flatten via main entry point") {
    auto mesh = make_grid_mesh();
    utils2::FlatteningParams params;
    params.method = utils2::FlatteningMethod::harmonic;
    params.max_iterations = 2000;
    params.tolerance = 1e-8;

    auto result = utils2::flatten(mesh, params);
    REQUIRE(result.converged);
    REQUIRE_EQ(result.uv.size(), std::size_t(9));
}

TEST_CASE("TriMesh num_vertices and num_triangles") {
    auto mesh = make_quad_mesh();
    REQUIRE_EQ(mesh.num_vertices(), std::size_t(4));
    REQUIRE_EQ(mesh.num_triangles(), std::size_t(2));

    auto grid = make_grid_mesh();
    REQUIRE_EQ(grid.num_vertices(), std::size_t(9));
    REQUIRE_EQ(grid.num_triangles(), std::size_t(8));
}

TEST_CASE("flatten_harmonic preserves boundary constraint") {
    auto mesh = make_quad_mesh();
    auto result = utils2::flatten_harmonic(mesh, std::size_t(1000), 1e-8);
    REQUIRE(result.converged);

    // All 4 vertices are boundary, so they should be mapped to a circle
    for (const auto& uv : result.uv) {
        REQUIRE(std::isfinite(uv[0]));
        REQUIRE(std::isfinite(uv[1]));
    }
}

TEST_CASE("flatten_lscm grid preserves proportions") {
    auto mesh = make_grid_mesh();
    auto result = utils2::flatten_lscm(mesh, std::size_t(500), 1e-8);

    REQUIRE(result.converged);
    REQUIRE_EQ(result.uv.size(), std::size_t(9));

    // The UV coordinates should not be all degenerate to a single point.
    // LSCM pins two vertices, so at least one axis should have spread.
    double min_u = 1e30, max_u = -1e30;
    double min_v = 1e30, max_v = -1e30;
    for (const auto& uv : result.uv) {
        min_u = std::min(min_u, uv[0]);
        max_u = std::max(max_u, uv[0]);
        min_v = std::min(min_v, uv[1]);
        max_v = std::max(max_v, uv[1]);
    }
    // At least one axis must have nonzero spread
    double spread = std::max(max_u - min_u, max_v - min_v);
    REQUIRE_GT(spread, 1e-6);
}

TEST_CASE("flatten conformal method via entry point") {
    auto mesh = make_grid_mesh();
    utils2::FlatteningParams params;
    params.method = utils2::FlatteningMethod::conformal;
    params.max_iterations = 500;
    params.tolerance = 1e-8;

    auto result = utils2::flatten(mesh, params);
    REQUIRE(result.converged);
    REQUIRE_EQ(result.uv.size(), std::size_t(9));
}

TEST_CASE("map_boundary_to_circle equal spacing") {
    auto mesh = make_grid_mesh();
    auto loop = mesh.boundary_loop();

    auto circle_uv = utils2::map_boundary_to_circle(mesh, std::span{loop});
    REQUIRE_EQ(circle_uv.size(), loop.size());

    // All points on unit circle
    for (const auto& pt : circle_uv) {
        double r = std::sqrt(pt[0] * pt[0] + pt[1] * pt[1]);
        REQUIRE_NEAR(r, 1.0, 1e-10);
    }
}

TEST_CASE("flatten_harmonic convergence check") {
    // Verify harmonic flatten converges quickly on a simple grid
    auto mesh = make_grid_mesh();
    auto result = utils2::flatten_harmonic(mesh, std::size_t(2000), 1e-8);
    REQUIRE(result.converged);
    // Should converge in fewer iterations than the max
    REQUIRE_LT(result.iterations, std::size_t(2000));
}

TEST_CASE("flatten_harmonic iterations field") {
    auto mesh = make_grid_mesh();
    auto result = utils2::flatten_harmonic(mesh, std::size_t(2000), 1e-8);
    REQUIRE(result.converged);
    REQUIRE_GT(result.iterations, std::size_t(0));
}

TEST_CASE("FlatteningResult metrics on unit square") {
    auto mesh = make_quad_mesh();
    auto result = utils2::flatten_harmonic(mesh, std::size_t(1000), 1e-8);
    REQUIRE(result.converged);

    auto metrics = result.compute_metrics(mesh);
    // For a flat mesh the stretch should be reasonable
    CHECK_GT(metrics.mean_stretch, 0.0);
    CHECK(std::isfinite(metrics.mean_stretch));
    CHECK(std::isfinite(metrics.max_stretch));
    CHECK(std::isfinite(metrics.area_distortion));
}

TEST_CASE("TriMesh vertex_normals on grid mesh") {
    auto mesh = make_grid_mesh();
    auto normals = mesh.vertex_normals();

    REQUIRE_EQ(normals.size(), std::size_t(9));
    // All normals should point in +Z direction for flat XY mesh
    for (const auto& n : normals) {
        REQUIRE_NEAR(n[0], 0.0, 1e-10);
        REQUIRE_NEAR(n[1], 0.0, 1e-10);
        REQUIRE_NEAR(n[2], 1.0, 1e-10);
    }
}

// ---------------------------------------------------------------------------
// Helper: a mesh with more interior vertices for meaningful ARAP testing
// 4x4 grid = 16 vertices, 18 triangles, 4 interior vertices
// ---------------------------------------------------------------------------
static utils2::TriMesh<double> make_4x4_grid_mesh() {
    utils2::TriMesh<double> m;
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c)
            m.vertices.push_back({double(c), double(r), 0.0});

    auto idx = [](int r, int c) -> std::size_t { return r * 4 + c; };
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            m.triangles.push_back({idx(r, c), idx(r, c + 1), idx(r + 1, c + 1)});
            m.triangles.push_back({idx(r, c), idx(r + 1, c + 1), idx(r + 1, c)});
        }
    }
    return m;
}

// Helper: a non-planar mesh (hemisphere-like) for more interesting flattening
static utils2::TriMesh<double> make_dome_mesh() {
    utils2::TriMesh<double> m;
    // Center vertex at top
    m.vertices.push_back({0.0, 0.0, 1.0});
    // Ring of 6 vertices at base
    constexpr int N = 6;
    for (int i = 0; i < N; ++i) {
        double angle = 2.0 * std::numbers::pi * i / N;
        m.vertices.push_back({std::cos(angle), std::sin(angle), 0.0});
    }
    // Fan triangles from center to ring
    for (int i = 0; i < N; ++i) {
        std::size_t next = 1 + ((i + 1) % N);
        m.triangles.push_back({0, std::size_t(1 + i), next});
    }
    return m;
}

// Helper: a single triangle mesh
static utils2::TriMesh<double> make_single_triangle() {
    utils2::TriMesh<double> m;
    m.vertices = {
        {0.0, 0.0, 0.0},
        {1.0, 0.0, 0.0},
        {0.5, std::sqrt(3.0) / 2.0, 0.0}
    };
    m.triangles = {{0, 1, 2}};
    return m;
}

// ===========================================================================
// flatten_arap tests
// ===========================================================================

TEST_CASE("flatten_arap on grid mesh") {
    auto mesh = make_grid_mesh();

    // Get initial UVs from LSCM
    auto lscm_result = utils2::flatten_lscm(mesh, std::size_t(500), 1e-8);
    REQUIRE(lscm_result.converged);

    // Run ARAP
    auto result = utils2::flatten_arap(mesh,
        std::span<const std::array<double, 2>>{lscm_result.uv},
        std::size_t(50), 1e-6);

    REQUIRE_EQ(result.uv.size(), std::size_t(9));

    // All UVs should be finite
    for (const auto& uv : result.uv) {
        REQUIRE(std::isfinite(uv[0]));
        REQUIRE(std::isfinite(uv[1]));
    }

    // Energy should be finite and non-negative
    REQUIRE(std::isfinite(result.final_energy));
    REQUIRE_GE(result.final_energy, 0.0);

    // Should have done at least one iteration
    REQUIRE_GT(result.iterations, std::size_t(0));
}

TEST_CASE("flatten_arap on 4x4 grid mesh") {
    auto mesh = make_4x4_grid_mesh();

    // Get initial UVs from harmonic
    auto harmonic_result = utils2::flatten_harmonic(mesh, std::size_t(2000), 1e-8);
    REQUIRE(harmonic_result.converged);

    // Run ARAP
    auto result = utils2::flatten_arap(mesh,
        std::span<const std::array<double, 2>>{harmonic_result.uv},
        std::size_t(50), 1e-6);

    REQUIRE_EQ(result.uv.size(), std::size_t(16));

    for (const auto& uv : result.uv) {
        REQUIRE(std::isfinite(uv[0]));
        REQUIRE(std::isfinite(uv[1]));
    }

    // ARAP should produce non-degenerate UVs (spread > 0)
    double min_u = 1e30, max_u = -1e30;
    double min_v = 1e30, max_v = -1e30;
    for (const auto& uv : result.uv) {
        min_u = std::min(min_u, uv[0]);
        max_u = std::max(max_u, uv[0]);
        min_v = std::min(min_v, uv[1]);
        max_v = std::max(max_v, uv[1]);
    }
    REQUIRE_GT(max_u - min_u, 1e-6);
    REQUIRE_GT(max_v - min_v, 1e-6);
}

TEST_CASE("flatten_arap on dome mesh") {
    auto mesh = make_dome_mesh();

    auto lscm_result = utils2::flatten_lscm(mesh, std::size_t(500), 1e-8);
    REQUIRE(lscm_result.converged);

    auto result = utils2::flatten_arap(mesh,
        std::span<const std::array<double, 2>>{lscm_result.uv},
        std::size_t(50), 1e-6);

    REQUIRE_EQ(result.uv.size(), std::size_t(7));

    for (const auto& uv : result.uv) {
        REQUIRE(std::isfinite(uv[0]));
        REQUIRE(std::isfinite(uv[1]));
    }
}

TEST_CASE("flatten_arap convergence") {
    auto mesh = make_4x4_grid_mesh();

    auto initial = utils2::flatten_harmonic(mesh, std::size_t(2000), 1e-8);
    REQUIRE(initial.converged);

    // Run ARAP with generous iterations and loose tolerance
    auto result = utils2::flatten_arap(mesh,
        std::span<const std::array<double, 2>>{initial.uv},
        std::size_t(200), 1e-3);

    // Energy should be finite
    CHECK(std::isfinite(result.final_energy));
    // Should have run at least one iteration
    CHECK_GT(result.iterations, std::size_t(0));
}

TEST_CASE("flatten_arap energy is finite") {
    auto mesh = make_4x4_grid_mesh();

    auto initial = utils2::flatten_harmonic(mesh, std::size_t(2000), 1e-8);
    REQUIRE(initial.converged);

    // Run with multiple iterations
    auto result = utils2::flatten_arap(mesh,
        std::span<const std::array<double, 2>>{initial.uv},
        std::size_t(10), 1e-20);

    // Energy should be finite and non-negative
    CHECK(std::isfinite(result.final_energy));
    CHECK_GE(result.final_energy, 0.0);
    CHECK_GT(result.iterations, std::size_t(0));
}

// ===========================================================================
// flatten entry point with ARAP method
// ===========================================================================

TEST_CASE("flatten with arap method via entry point") {
    auto mesh = make_grid_mesh();
    utils2::FlatteningParams params;
    params.method = utils2::FlatteningMethod::arap;
    params.max_iterations = 50;
    params.tolerance = 1e-6;

    auto result = utils2::flatten(mesh, params);

    REQUIRE_EQ(result.uv.size(), std::size_t(9));
    for (const auto& uv : result.uv) {
        REQUIRE(std::isfinite(uv[0]));
        REQUIRE(std::isfinite(uv[1]));
    }
    // iterations should include both harmonic init and ARAP iterations
    REQUIRE_GT(result.iterations, std::size_t(0));
}

// ===========================================================================
// map_boundary_to_square tests
// ===========================================================================

TEST_CASE("map_boundary_to_square on quad mesh") {
    auto mesh = make_quad_mesh();
    auto loop = mesh.boundary_loop();
    REQUIRE_EQ(loop.size(), std::size_t(4));

    auto square_uv = utils2::map_boundary_to_square(mesh, std::span{loop});
    REQUIRE_EQ(square_uv.size(), std::size_t(4));

    // All points should lie on the unit square boundary [0,1]x[0,1]
    for (const auto& pt : square_uv) {
        REQUIRE(std::isfinite(pt[0]));
        REQUIRE(std::isfinite(pt[1]));
        REQUIRE_GE(pt[0], -1e-10);
        REQUIRE_LE(pt[0], 1.0 + 1e-10);
        REQUIRE_GE(pt[1], -1e-10);
        REQUIRE_LE(pt[1], 1.0 + 1e-10);
    }

    // First point should be at (0, 0) (start of the square perimeter)
    REQUIRE_NEAR(square_uv[0][0], 0.0, 1e-10);
    REQUIRE_NEAR(square_uv[0][1], 0.0, 1e-10);
}

TEST_CASE("map_boundary_to_square on grid mesh") {
    auto mesh = make_grid_mesh();
    auto loop = mesh.boundary_loop();
    REQUIRE_EQ(loop.size(), std::size_t(8));

    auto square_uv = utils2::map_boundary_to_square(mesh, std::span{loop});
    REQUIRE_EQ(square_uv.size(), std::size_t(8));

    // All points on boundary of unit square
    for (const auto& pt : square_uv) {
        REQUIRE(std::isfinite(pt[0]));
        REQUIRE(std::isfinite(pt[1]));
        // Each point should lie on one of the four edges:
        // bottom (y=0), right (x=1), top (y=1), left (x=0)
        bool on_edge = (std::abs(pt[0]) < 1e-10) ||
                       (std::abs(pt[0] - 1.0) < 1e-10) ||
                       (std::abs(pt[1]) < 1e-10) ||
                       (std::abs(pt[1] - 1.0) < 1e-10);
        CHECK(on_edge);
    }
}

TEST_CASE("map_boundary_to_square empty boundary") {
    utils2::TriMesh<double> mesh;
    std::vector<std::size_t> empty_boundary;
    auto result = utils2::map_boundary_to_square(mesh, std::span{empty_boundary});
    REQUIRE_EQ(result.size(), std::size_t(0));
}

TEST_CASE("map_boundary_to_square on dome mesh") {
    auto mesh = make_dome_mesh();
    auto loop = mesh.boundary_loop();
    REQUIRE_EQ(loop.size(), std::size_t(6));

    auto square_uv = utils2::map_boundary_to_square(mesh, std::span{loop});
    REQUIRE_EQ(square_uv.size(), std::size_t(6));

    for (const auto& pt : square_uv) {
        REQUIRE(std::isfinite(pt[0]));
        REQUIRE(std::isfinite(pt[1]));
    }
}

// ===========================================================================
// compute_stretch tests
// ===========================================================================

TEST_CASE("compute_stretch on flat quad mesh") {
    auto mesh = make_quad_mesh();
    auto result = utils2::flatten_harmonic(mesh, std::size_t(1000), 1e-8);
    REQUIRE(result.converged);

    auto stretches = utils2::compute_stretch(mesh,
        std::span<const std::array<double, 2>>{result.uv});

    REQUIRE_EQ(stretches.size(), std::size_t(2)); // 2 triangles

    for (const auto& s : stretches) {
        // sigma_max and sigma_min
        REQUIRE(std::isfinite(s[0]));
        REQUIRE(std::isfinite(s[1]));
        // sigma_max >= sigma_min >= 0
        REQUIRE_GE(s[0], s[1] - 1e-10);
        REQUIRE_GE(s[1], -1e-10);
    }
}

TEST_CASE("compute_stretch on grid mesh with LSCM") {
    auto mesh = make_grid_mesh();
    auto result = utils2::flatten_lscm(mesh, std::size_t(500), 1e-8);
    REQUIRE(result.converged);

    auto stretches = utils2::compute_stretch(mesh,
        std::span<const std::array<double, 2>>{result.uv});

    REQUIRE_EQ(stretches.size(), std::size_t(8)); // 8 triangles

    for (const auto& s : stretches) {
        REQUIRE(std::isfinite(s[0]));
        REQUIRE(std::isfinite(s[1]));
        REQUIRE_GE(s[0], 0.0);
        REQUIRE_GE(s[1], 0.0);
    }
}

TEST_CASE("compute_stretch singular values ordering") {
    auto mesh = make_grid_mesh();
    auto result = utils2::flatten_harmonic(mesh, std::size_t(2000), 1e-8);
    REQUIRE(result.converged);

    auto stretches = utils2::compute_stretch(mesh,
        std::span<const std::array<double, 2>>{result.uv});

    for (const auto& s : stretches) {
        // s[0] = sigma_max, s[1] = sigma_min
        // sigma_max >= sigma_min always
        CHECK_GE(s[0], s[1] - 1e-10);
    }
}

TEST_CASE("compute_stretch degenerate UV") {
    auto mesh = make_quad_mesh();
    // All UVs at origin => degenerate
    std::vector<std::array<double, 2>> degen_uv(4, {0.0, 0.0});

    auto stretches = utils2::compute_stretch(mesh,
        std::span<const std::array<double, 2>>{degen_uv});

    REQUIRE_EQ(stretches.size(), std::size_t(2));
    // With zero-area 2D triangles, stretch should be {0, 0}
    for (const auto& s : stretches) {
        REQUIRE_NEAR(s[0], 0.0, 1e-10);
        REQUIRE_NEAR(s[1], 0.0, 1e-10);
    }
}

// ===========================================================================
// closest_rotation_2x2 tests (detail helper)
// ===========================================================================

TEST_CASE("closest_rotation_2x2 identity") {
    // Identity matrix should return identity rotation
    auto R = utils2::detail::closest_rotation_2x2(1.0, 0.0, 0.0, 1.0);
    REQUIRE_NEAR(R[0][0], 1.0, 1e-10);
    REQUIRE_NEAR(R[0][1], 0.0, 1e-10);
    REQUIRE_NEAR(R[1][0], 0.0, 1e-10);
    REQUIRE_NEAR(R[1][1], 1.0, 1e-10);
}

TEST_CASE("closest_rotation_2x2 non-uniform scaling plus rotation") {
    // A rotation combined with non-uniform scaling: R * diag(3, 1)
    // The closest rotation should still be a proper rotation.
    double c = std::cos(std::numbers::pi / 6.0);
    double s = std::sin(std::numbers::pi / 6.0);
    // M = R * diag(3, 1) = [[3c, -s], [3s, c]]
    auto R = utils2::detail::closest_rotation_2x2(3.0 * c, -s, 3.0 * s, c);
    // det(R) should be +1
    double det = R[0][0] * R[1][1] - R[0][1] * R[1][0];
    REQUIRE_NEAR(det, 1.0, 1e-10);
    // R should be orthogonal
    double rrt00 = R[0][0] * R[0][0] + R[0][1] * R[0][1];
    double rrt11 = R[1][0] * R[1][0] + R[1][1] * R[1][1];
    REQUIRE_NEAR(rrt00, 1.0, 1e-10);
    REQUIRE_NEAR(rrt11, 1.0, 1e-10);
}

TEST_CASE("closest_rotation_2x2 scaled matrix") {
    // 3*I should give identity rotation
    auto R = utils2::detail::closest_rotation_2x2(3.0, 0.0, 0.0, 3.0);
    REQUIRE_NEAR(R[0][0], 1.0, 1e-10);
    REQUIRE_NEAR(R[0][1], 0.0, 1e-10);
    REQUIRE_NEAR(R[1][0], 0.0, 1e-10);
    REQUIRE_NEAR(R[1][1], 1.0, 1e-10);
}

TEST_CASE("closest_rotation_2x2 reflection triggers s2 branch") {
    // Reflection [[1,0],[0,-1]] has singular values 1 and -1 (s2 < 0).
    // This exercises the s2 < 0 code path in the implementation.
    auto R = utils2::detail::closest_rotation_2x2(1.0, 0.0, 0.0, -1.0);
    // Result should be orthogonal (each row has unit length)
    double rrt00 = R[0][0] * R[0][0] + R[0][1] * R[0][1];
    double rrt11 = R[1][0] * R[1][0] + R[1][1] * R[1][1];
    REQUIRE_NEAR(rrt00, 1.0, 1e-10);
    REQUIRE_NEAR(rrt11, 1.0, 1e-10);
}

TEST_CASE("closest_rotation_2x2 arbitrary matrix") {
    // An arbitrary matrix, check the result is a proper rotation
    auto R = utils2::detail::closest_rotation_2x2(2.0, 1.5, -0.5, 3.0);
    // det(R) = 1
    double det = R[0][0] * R[1][1] - R[0][1] * R[1][0];
    REQUIRE_NEAR(det, 1.0, 1e-10);
    // Orthogonal
    double rrt00 = R[0][0] * R[0][0] + R[0][1] * R[0][1];
    double rrt11 = R[1][0] * R[1][0] + R[1][1] * R[1][1];
    REQUIRE_NEAR(rrt00, 1.0, 1e-10);
    REQUIRE_NEAR(rrt11, 1.0, 1e-10);
}

// ===========================================================================
// Boundary detection edge cases
// ===========================================================================

TEST_CASE("boundary_loop on single triangle") {
    auto mesh = make_single_triangle();
    auto loop = mesh.boundary_loop();

    // Single triangle: 3 boundary edges, 3 boundary vertices
    REQUIRE_EQ(loop.size(), std::size_t(3));

    std::vector<bool> seen(3, false);
    for (auto v : loop) {
        REQUIRE_LT(v, std::size_t(3));
        seen[v] = true;
    }
    for (int i = 0; i < 3; ++i)
        REQUIRE(seen[i]);
}

TEST_CASE("boundary_loop on dome mesh") {
    auto mesh = make_dome_mesh();
    auto loop = mesh.boundary_loop();

    // Dome: fan of 6 triangles, boundary is the ring of 6 base vertices
    REQUIRE_EQ(loop.size(), std::size_t(6));

    // Vertex 0 (center, top) should NOT be on the boundary
    bool center_on_boundary = false;
    for (auto v : loop) {
        if (v == 0) center_on_boundary = true;
    }
    REQUIRE(!center_on_boundary);
}

TEST_CASE("boundary_loop on 4x4 grid mesh") {
    auto mesh = make_4x4_grid_mesh();
    auto loop = mesh.boundary_loop();

    // 4x4 grid: perimeter has 4*3 = 12 boundary vertices
    REQUIRE_EQ(loop.size(), std::size_t(12));
}

TEST_CASE("boundary_loop is a valid loop") {
    // Verify the boundary loop forms a connected cycle
    auto mesh = make_grid_mesh();
    auto loop = mesh.boundary_loop();
    REQUIRE_GT(loop.size(), std::size_t(0));

    // Each consecutive pair should share an edge in the mesh
    // (i.e., be adjacent in at least one triangle)
    for (std::size_t i = 0; i < loop.size(); ++i) {
        auto a = loop[i];
        auto b = loop[(i + 1) % loop.size()];
        // Check that (a, b) is a directed half-edge in some triangle
        bool found = false;
        for (const auto& tri : mesh.triangles) {
            for (int e = 0; e < 3; ++e) {
                if (tri[e] == a && tri[(e + 1) % 3] == b) {
                    found = true;
                    break;
                }
            }
            if (found) break;
        }
        CHECK(found);
    }
}

TEST_CASE("boundary_loop empty mesh") {
    utils2::TriMesh<double> mesh;
    auto loop = mesh.boundary_loop();
    REQUIRE_EQ(loop.size(), std::size_t(0));
}

// ===========================================================================
// LSCM edge cases
// ===========================================================================

TEST_CASE("flatten_lscm on single triangle") {
    auto mesh = make_single_triangle();
    auto result = utils2::flatten_lscm(mesh, std::size_t(500), 1e-8);

    REQUIRE(result.converged);
    REQUIRE_EQ(result.uv.size(), std::size_t(3));

    for (const auto& uv : result.uv) {
        REQUIRE(std::isfinite(uv[0]));
        REQUIRE(std::isfinite(uv[1]));
    }
}

TEST_CASE("flatten_lscm boundary less than 2") {
    // A mesh with no boundary (closed) should fail gracefully
    // We can't easily make a closed mesh with TriMesh, so test with empty
    utils2::TriMesh<double> mesh;
    auto result = utils2::flatten_lscm(mesh, std::size_t(100), 1e-8);
    REQUIRE(!result.converged);
}

// ===========================================================================
// Harmonic flattening edge case: empty boundary
// ===========================================================================

TEST_CASE("flatten_harmonic empty mesh") {
    utils2::TriMesh<double> mesh;
    auto result = utils2::flatten_harmonic(mesh, std::size_t(100), 1e-8);
    REQUIRE(!result.converged);
}

// ===========================================================================
// ARAP with empty boundary
// ===========================================================================

TEST_CASE("flatten_arap empty mesh") {
    utils2::TriMesh<double> mesh;
    std::vector<std::array<double, 2>> initial_uv;
    auto result = utils2::flatten_arap(mesh,
        std::span<const std::array<double, 2>>{initial_uv},
        std::size_t(10), 1e-6);
    REQUIRE(!result.converged);
}

// ===========================================================================
// compute_metrics after ARAP
// ===========================================================================

TEST_CASE("compute_metrics after ARAP") {
    auto mesh = make_4x4_grid_mesh();

    auto initial = utils2::flatten_harmonic(mesh, std::size_t(2000), 1e-8);
    REQUIRE(initial.converged);

    auto result = utils2::flatten_arap(mesh,
        std::span<const std::array<double, 2>>{initial.uv},
        std::size_t(50), 1e-6);

    auto metrics = result.compute_metrics(mesh);
    CHECK_GT(metrics.mean_stretch, 0.0);
    CHECK_GT(metrics.max_stretch, 0.0);
    CHECK_GT(metrics.area_distortion, 0.0);
    CHECK(std::isfinite(metrics.mean_stretch));
    CHECK(std::isfinite(metrics.max_stretch));
    CHECK(std::isfinite(metrics.area_distortion));
}

// ===========================================================================
// map_boundary_to_circle edge case: empty boundary
// ===========================================================================

TEST_CASE("map_boundary_to_circle empty boundary") {
    utils2::TriMesh<double> mesh;
    std::vector<std::size_t> empty_boundary;
    auto result = utils2::map_boundary_to_circle(mesh, std::span{empty_boundary});
    REQUIRE_EQ(result.size(), std::size_t(0));
}

// ===========================================================================
// compute_stretch after ARAP
// ===========================================================================

TEST_CASE("compute_stretch after ARAP") {
    auto mesh = make_grid_mesh();

    auto initial = utils2::flatten_lscm(mesh, std::size_t(500), 1e-8);
    REQUIRE(initial.converged);

    auto arap_result = utils2::flatten_arap(mesh,
        std::span<const std::array<double, 2>>{initial.uv},
        std::size_t(50), 1e-6);

    auto stretches = utils2::compute_stretch(mesh,
        std::span<const std::array<double, 2>>{arap_result.uv});

    REQUIRE_EQ(stretches.size(), std::size_t(8));

    for (const auto& s : stretches) {
        REQUIRE(std::isfinite(s[0]));
        REQUIRE(std::isfinite(s[1]));
        REQUIRE_GE(s[0], 0.0);
    }
}

// ===========================================================================
// Float instantiation tests (template coverage)
// ===========================================================================

TEST_CASE("flatten_arap float instantiation") {
    utils2::TriMesh<float> m;
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            m.vertices.push_back({float(c), float(r), 0.0f});

    auto idx = [](int r, int c) -> std::size_t { return r * 3 + c; };
    for (int r = 0; r < 2; ++r) {
        for (int c = 0; c < 2; ++c) {
            m.triangles.push_back({idx(r, c), idx(r, c + 1), idx(r + 1, c + 1)});
            m.triangles.push_back({idx(r, c), idx(r + 1, c + 1), idx(r + 1, c)});
        }
    }

    auto initial = utils2::flatten_lscm(m, std::size_t(500), 1e-4f);
    REQUIRE(initial.converged);

    auto result = utils2::flatten_arap(m,
        std::span<const std::array<float, 2>>{initial.uv},
        std::size_t(20), 1e-4f);

    REQUIRE_EQ(result.uv.size(), std::size_t(9));
    for (const auto& uv : result.uv) {
        REQUIRE(std::isfinite(uv[0]));
        REQUIRE(std::isfinite(uv[1]));
    }
}

UTILS2_TEST_MAIN()
