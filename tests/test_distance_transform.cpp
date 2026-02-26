#include <utils2/test.hpp>
#include <utils2/distance_transform.hpp>
#include <vector>
#include <utils2/mdspan.hpp>
#include <cmath>

using namespace utils2;

// ---- 1D EDT ----------------------------------------------------------------

TEST_CASE("edt 1D single fg pixel") {
    // bg fg bg
    std::vector<float> input = {0, 1, 0};
    std::vector<float> output(3);
    edt_1d_sq<float>(input, output, 1.0f);

    CHECK_NEAR(output[0], 0.0f, 1e-6);
    CHECK_NEAR(output[2], 0.0f, 1e-6);
    CHECK_NEAR(output[1], 1.0f, 1e-6);
}

TEST_CASE("edt 1D all background") {
    std::vector<float> input = {0, 0, 0};
    std::vector<float> output(3);
    edt_1d_sq<float>(input, output);

    for (int i = 0; i < 3; ++i)
        CHECK_NEAR(output[i], 0.0f, 1e-6);
}

TEST_CASE("edt 1D all foreground") {
    std::vector<float> input = {1, 1, 1};
    std::vector<float> output(3);
    edt_1d_sq<float>(input, output);

    // All foreground, no background to measure distance to -> all infinite
    for (int i = 0; i < 3; ++i)
        CHECK_GT(output[i], 1e30f);
}

TEST_CASE("edt 1D fg at start only") {
    std::vector<float> input = {0, 1, 1};
    std::vector<float> output(3);
    edt_1d_sq<float>(input, output, 1.0f);

    CHECK_NEAR(output[0], 0.0f, 1e-6);
    CHECK_NEAR(output[1], 1.0f, 1e-6);
    CHECK_NEAR(output[2], 4.0f, 1e-6);
}

TEST_CASE("edt 1D with spacing") {
    std::vector<float> input = {0, 1, 0};
    std::vector<float> output(3);
    edt_1d_sq<float>(input, output, 2.0f);

    // spacing=2, so squared distance from fg at index 1 to bg at 0 or 2 is (2*1)^2 = 4
    CHECK_NEAR(output[1], 4.0f, 1e-6);
}

// ---- 2D EDT ----------------------------------------------------------------

TEST_CASE("edt 2D single foreground pixel") {
    std::vector<int> img = {
        0, 0, 0,
        0, 1, 0,
        0, 0, 0
    };
    std::mdspan<const int, std::dextents<std::size_t, 2>> view(img.data(), 3, 3);
    auto dist = edt_2d<int, float>(view);

    CHECK_NEAR(dist[0], 0.0f, 1e-6);
    CHECK_NEAR(dist[4], 1.0f, 1e-6);
}

TEST_CASE("edt 2D all background") {
    std::vector<int> img(9, 0);
    std::mdspan<const int, std::dextents<std::size_t, 2>> view(img.data(), 3, 3);
    auto dist = edt_2d<int, float>(view);

    for (auto d : dist)
        CHECK_NEAR(d, 0.0f, 1e-6);
}

TEST_CASE("edt 2D fg row with bg borders") {
    std::vector<int> img = {
        0, 0, 0,
        1, 1, 1,
        0, 0, 0
    };
    std::mdspan<const int, std::dextents<std::size_t, 2>> view(img.data(), 3, 3);
    auto dist = edt_2d<int, float>(view);

    CHECK_NEAR(dist[3], 1.0f, 1e-5);
    CHECK_NEAR(dist[4], 1.0f, 1e-5);
    CHECK_NEAR(dist[5], 1.0f, 1e-5);
}

TEST_CASE("edt 2D column") {
    // Single column (3 rows, 1 col)
    std::vector<int> img = {0, 1, 0};
    std::mdspan<const int, std::dextents<std::size_t, 2>> view(img.data(), 3, 1);
    auto dist = edt_2d<int, float>(view);

    CHECK_NEAR(dist[0], 0.0f, 1e-5);
    CHECK_NEAR(dist[1], 1.0f, 1e-5);
    CHECK_NEAR(dist[2], 0.0f, 1e-5);
}

// ---- 3D EDT ----------------------------------------------------------------

TEST_CASE("edt 3D single voxel") {
    std::vector<int> vol(27, 0);
    vol[13] = 1;  // center = (1,1,1)
    std::mdspan<const int, std::dextents<std::size_t, 3>> view(vol.data(), 3, 3, 3);
    auto dist = edt_3d<int, float>(view);

    CHECK_NEAR(dist[13], 1.0f, 1e-5);
    CHECK_NEAR(dist[0], 0.0f, 1e-6);
}

TEST_CASE("edt 3D all bg") {
    std::vector<int> vol(8, 0);
    std::mdspan<const int, std::dextents<std::size_t, 3>> view(vol.data(), 2, 2, 2);
    auto dist = edt_3d<int, float>(view);

    for (auto d : dist)
        CHECK_NEAR(d, 0.0f, 1e-6);
}

TEST_CASE("edt 3D single fg layer") {
    std::vector<int> vol = {0, 1, 0};
    std::mdspan<const int, std::dextents<std::size_t, 3>> view(vol.data(), 3, 1, 1);
    auto dist = edt_3d<int, float>(view);

    CHECK_NEAR(dist[0], 0.0f, 1e-6);
    CHECK_NEAR(dist[1], 1.0f, 1e-5);
    CHECK_NEAR(dist[2], 0.0f, 1e-6);
}

// ---- Anisotropic spacing ---------------------------------------------------

TEST_CASE("edt 2D anisotropic spacing") {
    std::vector<int> img = {0, 1, 0};
    std::mdspan<const int, std::dextents<std::size_t, 2>> view(img.data(), 3, 1);
    auto dist = edt_2d<int, float>(view, {2.0f, 1.0f});

    CHECK_NEAR(dist[1], 2.0f, 1e-5);
}

TEST_CASE("edt 3D anisotropic spacing") {
    std::vector<int> vol = {0, 1, 0};
    std::mdspan<const int, std::dextents<std::size_t, 3>> view(vol.data(), 3, 1, 1);
    auto dist = edt_3d<int, float>(view, {3.0f, 1.0f, 1.0f});

    CHECK_NEAR(dist[1], 3.0f, 1e-5);
}

// ---- Multi-label EDT -------------------------------------------------------

TEST_CASE("edt 2D multilabel vertical boundary") {
    // Two columns with different labels: all pixels are on the boundary
    std::vector<int> img = {
        1, 2,
        1, 2,
        1, 2
    };
    std::mdspan<const int, std::dextents<std::size_t, 2>> view(img.data(), 3, 2);
    auto dist = edt_2d_multilabel<int, float>(view);

    for (auto d : dist)
        CHECK_NEAR(d, 0.0f, 1e-5);
}

// ---- Verify known properties -----------------------------------------------

TEST_CASE("edt 2D squared distances monotonic from bg") {
    // 1 column, bg at top, increasing fg
    std::vector<int> img = {0, 1, 1, 1};
    std::mdspan<const int, std::dextents<std::size_t, 2>> view(img.data(), 4, 1);
    auto dist_sq = edt_2d_sq<int, float>(view);

    CHECK_NEAR(dist_sq[0], 0.0f, 1e-6);
    CHECK_LT(dist_sq[1], dist_sq[2]);
    CHECK_LT(dist_sq[2], dist_sq[3]);
}

TEST_CASE("edt 2D bg pixels always zero") {
    // bg pixels should always have distance 0
    std::vector<int> img = {
        0, 0, 0,
        0, 1, 0,
        0, 0, 0
    };
    std::mdspan<const int, std::dextents<std::size_t, 2>> view(img.data(), 3, 3);
    auto dist = edt_2d<int, float>(view);

    // All bg pixels have distance 0
    CHECK_NEAR(dist[0], 0.0f, 1e-6);
    CHECK_NEAR(dist[1], 0.0f, 1e-6);
    CHECK_NEAR(dist[2], 0.0f, 1e-6);
    CHECK_NEAR(dist[3], 0.0f, 1e-6);
    // fg pixel should have positive distance
    CHECK_GT(dist[4], 0.0f);
}

// ---- Bug hunt: single foreground pixel, verify exact Euclidean distances ----

TEST_CASE("edt 2D single fg pixel exact distances") {
    // 7x7 image with single fg pixel at center (3,3).
    // All bg pixels should have distance 0.
    // The fg pixel should have distance = min distance to any bg pixel.
    // Since all surrounding pixels are bg, dist = 1.0.
    // But also verify that each fg pixel (there's only one) has correct distance.
    const std::size_t N = 7;
    std::vector<int> img(N * N, 0);
    img[3 * N + 3] = 1;

    std::mdspan<const int, std::dextents<std::size_t, 2>> view(img.data(), N, N);
    auto dist = edt_2d<int, float>(view);

    // Background pixels have distance 0
    for (std::size_t r = 0; r < N; ++r)
        for (std::size_t c = 0; c < N; ++c)
            if (!(r == 3 && c == 3))
                CHECK_NEAR(dist[r * N + c], 0.0f, 1e-5);

    // Foreground pixel at (3,3): nearest bg is at distance 1 (adjacent)
    CHECK_NEAR(dist[3 * N + 3], 1.0f, 1e-5);
}

TEST_CASE("edt 2D large foreground block exact distances") {
    // 9x9 image with a 5x5 foreground block in the center (rows 2-6, cols 2-6).
    const std::size_t N = 9;
    std::vector<int> img(N * N, 0);
    for (std::size_t r = 2; r <= 6; ++r)
        for (std::size_t c = 2; c <= 6; ++c)
            img[r * N + c] = 1;

    std::mdspan<const int, std::dextents<std::size_t, 2>> view(img.data(), N, N);
    auto dist = edt_2d<int, float>(view);

    // Edge of block (2,4): nearest bg at (1,4), dist=1
    CHECK_NEAR(dist[2 * N + 4], 1.0f, 1e-4);
    // Corner of block (2,2): nearest bg at (1,2) or (2,1), dist=1
    CHECK_NEAR(dist[2 * N + 2], 1.0f, 1e-4);
    // Interior pixel (4,4): nearest bg at (4,1) or (1,4) etc, dist=3
    CHECK_NEAR(dist[4 * N + 4], 3.0f, 1e-4);
}

TEST_CASE("edt 2D anisotropic single fg pixel") {
    // 5x5 with fg at (2,2), spacing = {2.0, 1.0}
    // Distance to nearest bg along row (axis 1, spacing 1.0) = 1.
    // Distance to nearest bg along col (axis 0, spacing 2.0) = 2.
    // Closest bg is at (2,1) or (2,3), dist = 1*1.0 = 1.0
    // Or (1,2) or (3,2), dist = 1*2.0 = 2.0.
    // So minimum distance = 1.0.

    const std::size_t N = 5;
    std::vector<int> img(N * N, 0);
    img[2 * N + 2] = 1;

    std::mdspan<const int, std::dextents<std::size_t, 2>> view(img.data(), N, N);
    auto dist = edt_2d<int, float>(view, {2.0f, 1.0f});

    // fg pixel at (2,2): nearest bg at (2,1) or (2,3) with x-spacing 1.0 -> dist 1.0
    CHECK_NEAR(dist[2 * N + 2], 1.0f, 1e-4);
}

TEST_CASE("edt 2D anisotropic Nx1 column") {
    // 5x1 image (5 rows, 1 col), fg at rows 1,2,3 (bg at row 0 and row 4)
    // Spacing = {3.0, 1.0}. Axis 0 is rows with spacing 3.0.
    std::vector<int> img = {0, 1, 1, 1, 0};
    std::mdspan<const int, std::dextents<std::size_t, 2>> view(img.data(), 5, 1);
    auto dist = edt_2d<int, float>(view, {3.0f, 1.0f});

    CHECK_NEAR(dist[0], 0.0f, 1e-5);  // bg
    CHECK_NEAR(dist[1], 3.0f, 1e-4);  // 1 step from bg at row 0
    CHECK_NEAR(dist[2], 6.0f, 1e-4);  // 2 steps from bg (equidistant from both sides)
    CHECK_NEAR(dist[3], 3.0f, 1e-4);  // 1 step from bg at row 4
    CHECK_NEAR(dist[4], 0.0f, 1e-5);  // bg
}

TEST_CASE("edt 2D anisotropic 1xN column-axis") {
    // 1x5 image (1 row, 5 cols), fg at cols 1,2,3 (bg at col 0 and 4)
    // Spacing = {1.0, 3.0}. Axis 1 is cols with spacing 3.0.
    std::vector<int> img = {0, 1, 1, 1, 0};
    std::mdspan<const int, std::dextents<std::size_t, 2>> view(img.data(), 1, 5);
    auto dist_sq = edt_2d_sq<int, float>(view, {1.0f, 3.0f});

    CHECK_NEAR(dist_sq[0], 0.0f, 1e-5);
    CHECK_NEAR(dist_sq[1], 9.0f, 1e-3);
    CHECK_NEAR(dist_sq[2], 36.0f, 1e-3);
    CHECK_NEAR(dist_sq[3], 9.0f, 1e-3);   // 1 step from bg at col 4
    CHECK_NEAR(dist_sq[4], 0.0f, 1e-5);   // bg pixel
}

TEST_CASE("edt 2D brute force verification") {
    // 7x7 image with a solid 3x3 block of foreground in the center.
    const std::size_t N = 7;
    std::vector<int> img(N * N, 0);
    for (std::size_t r = 2; r <= 4; ++r)
        for (std::size_t c = 2; c <= 4; ++c)
            img[r * N + c] = 1;

    std::mdspan<const int, std::dextents<std::size_t, 2>> view(img.data(), N, N);
    auto dist = edt_2d<int, float>(view);

    // Brute-force: for each pixel, compute min Euclidean distance to a bg pixel
    for (std::size_t r = 0; r < N; ++r) {
        for (std::size_t c = 0; c < N; ++c) {
            float min_dist = 1e30f;
            for (std::size_t br = 0; br < N; ++br) {
                for (std::size_t bc = 0; bc < N; ++bc) {
                    if (img[br * N + bc] == 0) {
                        float dr = static_cast<float>(r) - static_cast<float>(br);
                        float dc = static_cast<float>(c) - static_cast<float>(bc);
                        float d = std::sqrt(dr * dr + dc * dc);
                        min_dist = std::min(min_dist, d);
                    }
                }
            }
            CHECK_NEAR(dist[r * N + c], min_dist, 1e-3);
        }
    }
}

TEST_CASE("edt 2D consecutive fg at start of column") {
    // Previously crashed: column 0 is [fg,fg,bg] -> [inf,inf,0].
    // compute_s returns -inf when comparing finite vs infinite parabola,
    // and -inf <= z[0]=-inf caused k to underflow. Fixed with k>0 guard.
    std::vector<int> img = {1, 0, 1, 0, 0, 0};
    std::mdspan<const int, std::dextents<std::size_t, 2>> view(img.data(), 3, 2);
    auto dist = edt_2d<int, float>(view);

    // fg at (0,0): nearest bg at (0,1), dist=1
    CHECK_NEAR(dist[0], 1.0f, 1e-4);
    // fg at (1,0): nearest bg at (1,1) or (2,0), dist=1
    CHECK_NEAR(dist[2], 1.0f, 1e-4);
    // bg pixels should be 0
    CHECK_NEAR(dist[1], 0.0f, 1e-5);
    CHECK_NEAR(dist[3], 0.0f, 1e-5);
    CHECK_NEAR(dist[4], 0.0f, 1e-5);
    CHECK_NEAR(dist[5], 0.0f, 1e-5);
}

UTILS2_TEST_MAIN()
