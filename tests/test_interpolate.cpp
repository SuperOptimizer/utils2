#include <utils2/test.hpp>
#include <utils2/interpolate.hpp>
#include <vector>
#include <utils2/mdspan.hpp>

using namespace utils2;

// ---- 1D lerp ---------------------------------------------------------------

TEST_CASE("lerp basics") {
    CHECK_NEAR(lerp(0.0, 1.0, 0.0),  0.0, 1e-12);
    CHECK_NEAR(lerp(0.0, 1.0, 1.0),  1.0, 1e-12);
    CHECK_NEAR(lerp(0.0, 1.0, 0.5),  0.5, 1e-12);
    CHECK_NEAR(lerp(2.0, 8.0, 0.25), 3.5, 1e-12);
}

// ---- 1D cubic (Catmull-Rom) ------------------------------------------------

TEST_CASE("cubic at endpoints") {
    // t=0 should give p1, t=1 should give p2
    CHECK_NEAR(cubic(0.0, 1.0, 2.0, 3.0, 0.0), 1.0, 1e-12);
    CHECK_NEAR(cubic(0.0, 1.0, 2.0, 3.0, 1.0), 2.0, 1e-12);
}

TEST_CASE("cubic midpoint on linear data") {
    // On linear data, cubic should reproduce linear interpolation
    CHECK_NEAR(cubic(0.0, 1.0, 2.0, 3.0, 0.5), 1.5, 1e-12);
}

// ---- 2D bilinear from values -----------------------------------------------

TEST_CASE("bilinear from values") {
    // corners all same -> same result
    CHECK_NEAR(bilinear(5.0, 5.0, 5.0, 5.0, 0.5, 0.5), 5.0, 1e-12);
    // check at corners
    CHECK_NEAR(bilinear(0.0, 1.0, 2.0, 3.0, 0.0, 0.0), 0.0, 1e-12);
    CHECK_NEAR(bilinear(0.0, 1.0, 2.0, 3.0, 1.0, 0.0), 1.0, 1e-12);
    CHECK_NEAR(bilinear(0.0, 1.0, 2.0, 3.0, 0.0, 1.0), 2.0, 1e-12);
    CHECK_NEAR(bilinear(0.0, 1.0, 2.0, 3.0, 1.0, 1.0), 3.0, 1e-12);
}

// ---- 2D bilinear from mdspan -----------------------------------------------

TEST_CASE("bilinear mdspan") {
    // 3x3 grid with values = row + col
    std::vector<double> data = {
        0, 1, 2,
        1, 2, 3,
        2, 3, 4
    };
    std::mdspan grid(data.data(), 3, 3);
    std::mdspan<const double, std::dextents<std::size_t, 2>> cgrid(data.data(), 3, 3);

    // On-grid sample
    CHECK_NEAR(bilinear(cgrid, 0.0, 0.0), 0.0, 1e-12);
    CHECK_NEAR(bilinear(cgrid, 1.0, 1.0), 2.0, 1e-12);
    // Mid-point between (0,0) and (1,1)
    CHECK_NEAR(bilinear(cgrid, 0.5, 0.5), 1.0, 1e-12);
    // Clamped beyond boundary
    CHECK_NEAR(bilinear(cgrid, -1.0, 0.0), 0.0, 1e-12);
    CHECK_NEAR(bilinear(cgrid, 2.0, 2.0), 4.0, 1e-12);
}

// ---- 3D trilinear from values ----------------------------------------------

TEST_CASE("trilinear from values") {
    // All same -> same
    CHECK_NEAR(trilinear(1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0, 0.5,0.5,0.5), 1.0, 1e-12);
    // Corners
    CHECK_NEAR(trilinear(0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0, 0.0,0.0,0.0), 0.0, 1e-12);
    CHECK_NEAR(trilinear(0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0, 1.0,1.0,1.0), 7.0, 1e-12);
}

// ---- 3D trilinear from mdspan ----------------------------------------------

TEST_CASE("trilinear mdspan") {
    // 2x2x2 volume
    std::vector<double> vol = {0,1, 2,3,  4,5, 6,7};
    std::mdspan<const double, std::dextents<std::size_t, 3>> v(vol.data(), 2, 2, 2);

    CHECK_NEAR(trilinear(v, 0.0, 0.0, 0.0), 0.0, 1e-12);
    CHECK_NEAR(trilinear(v, 1.0, 1.0, 1.0), 7.0, 1e-12);
    CHECK_NEAR(trilinear(v, 0.5, 0.5, 0.5), 3.5, 1e-12);
}

// ---- 3D tricubic -----------------------------------------------------------

TEST_CASE("tricubic on linear data") {
    // 4x4x4 volume where value = z + y + x
    std::vector<double> vol(4*4*4);
    for (int z = 0; z < 4; ++z)
        for (int y = 0; y < 4; ++y)
            for (int x = 0; x < 4; ++x)
                vol[z*16 + y*4 + x] = static_cast<double>(z + y + x);

    std::mdspan<const double, std::dextents<std::size_t, 3>> v(vol.data(), 4, 4, 4);

    // On linear data, tricubic should match the linear value
    CHECK_NEAR(tricubic(v, 1.5, 1.5, 1.5), 4.5, 1e-6);
    CHECK_NEAR(tricubic(v, 1.0, 2.0, 1.0), 4.0, 1e-6);
}

// ---- Nearest neighbor ------------------------------------------------------

TEST_CASE("nearest 2D") {
    std::vector<float> data = {10, 20, 30, 40};
    std::mdspan<const float, std::dextents<std::size_t, 2>> grid(data.data(), 2, 2);

    CHECK_NEAR(nearest(grid, 0.0, 0.0), 10.0f, 1e-6);
    CHECK_NEAR(nearest(grid, 0.4, 0.4), 10.0f, 1e-6);
    CHECK_NEAR(nearest(grid, 0.6, 0.6), 40.0f, 1e-6);
    CHECK_NEAR(nearest(grid, 1.0, 0.0), 30.0f, 1e-6);
}

TEST_CASE("nearest 3D clamping") {
    std::vector<float> vol = {1, 2, 3, 4, 5, 6, 7, 8};
    std::mdspan<const float, std::dextents<std::size_t, 3>> v(vol.data(), 2, 2, 2);

    // Beyond bounds should clamp
    CHECK_NEAR(nearest(v, -5.0, -5.0, -5.0), 1.0f, 1e-6);
    CHECK_NEAR(nearest(v, 99.0, 99.0, 99.0), 8.0f, 1e-6);
}

// ---- sample() dispatcher ---------------------------------------------------

TEST_CASE("sample dispatcher 2D") {
    std::vector<double> data = {0, 1, 2, 3};
    std::mdspan<const double, std::dextents<std::size_t, 2>> grid(data.data(), 2, 2);

    // nearest
    CHECK_NEAR(sample(grid, Interpolation::nearest, 0.4, 0.4), 0.0, 1e-12);
    // linear -> bilinear
    CHECK_NEAR(sample(grid, Interpolation::linear, 0.5, 0.5), 1.5, 1e-12);
    // cubic -> bilinear for 2D
    CHECK_NEAR(sample(grid, Interpolation::cubic, 0.5, 0.5), 1.5, 1e-12);
}

TEST_CASE("sample dispatcher 3D") {
    std::vector<double> vol = {0,1, 2,3,  4,5, 6,7};
    std::mdspan<const double, std::dextents<std::size_t, 3>> v(vol.data(), 2, 2, 2);

    CHECK_NEAR(sample(v, Interpolation::linear, 0.5, 0.5, 0.5), 3.5, 1e-12);
    CHECK_NEAR(sample(v, Interpolation::nearest, 0.0, 0.0, 0.0), 0.0, 1e-12);
}

// ---- Gradient via central differences --------------------------------------

TEST_CASE("gradient 2D on linear ramp") {
    // 5x5 grid where value = x coordinate
    std::vector<double> data(25);
    for (int r = 0; r < 5; ++r)
        for (int c = 0; c < 5; ++c)
            data[r*5 + c] = static_cast<double>(c);

    std::mdspan<const double, std::dextents<std::size_t, 2>> grid(data.data(), 5, 5);

    auto g = gradient<double, std::dextents<std::size_t, 2>, 2>(grid, 2.0, 2.0);
    // dy should be ~0, dx should be ~1
    CHECK_NEAR(g[0], 0.0, 1e-6);
    CHECK_NEAR(g[1], 1.0, 1e-6);
}

TEST_CASE("gradient 3D on z-ramp") {
    // 4x4x4 where value = z
    std::vector<double> vol(64);
    for (int z = 0; z < 4; ++z)
        for (int i = 0; i < 16; ++i)
            vol[z*16 + i] = static_cast<double>(z);

    std::mdspan<const double, std::dextents<std::size_t, 3>> v(vol.data(), 4, 4, 4);

    auto g = gradient<double, std::dextents<std::size_t, 3>, 3>(v, 2.0, 2.0, 2.0);
    CHECK_NEAR(g[0], 1.0, 1e-6);  // dz ~ 1
    CHECK_NEAR(g[1], 0.0, 1e-6);  // dy ~ 0
    CHECK_NEAR(g[2], 0.0, 1e-6);  // dx ~ 0
}

// ---- Edge/boundary behavior ------------------------------------------------

TEST_CASE("bilinear on 1x1 grid") {
    std::vector<double> data = {42.0};
    std::mdspan<const double, std::dextents<std::size_t, 2>> grid(data.data(), 1, 1);
    CHECK_NEAR(bilinear(grid, 0.0, 0.0), 42.0, 1e-12);
}

TEST_CASE("trilinear clamped coordinates") {
    std::vector<double> vol = {0,1, 2,3,  4,5, 6,7};
    std::mdspan<const double, std::dextents<std::size_t, 3>> v(vol.data(), 2, 2, 2);

    // Negative coordinates clamp to 0
    CHECK_NEAR(trilinear(v, -1.0, 0.0, 0.0), 0.0, 1e-12);
    // Beyond max clamp to last
    CHECK_NEAR(trilinear(v, 5.0, 5.0, 5.0), 7.0, 1e-12);
}

// ---- Bug hunt: bilinear on 2x2 image with known values --------------------

TEST_CASE("bilinear 2x2 exact pixel corners") {
    // 2x2 grid: v(0,0)=10, v(0,1)=20, v(1,0)=30, v(1,1)=40
    std::vector<double> data = {10.0, 20.0, 30.0, 40.0};
    std::mdspan<const double, std::dextents<std::size_t, 2>> grid(data.data(), 2, 2);

    // Exact pixel locations
    CHECK_NEAR(bilinear(grid, 0.0, 0.0), 10.0, 1e-12);
    CHECK_NEAR(bilinear(grid, 0.0, 1.0), 20.0, 1e-12);
    CHECK_NEAR(bilinear(grid, 1.0, 0.0), 30.0, 1e-12);
    CHECK_NEAR(bilinear(grid, 1.0, 1.0), 40.0, 1e-12);

    // Midpoint: (0.5, 0.5) should be average of all four = 25.0
    CHECK_NEAR(bilinear(grid, 0.5, 0.5), 25.0, 1e-12);

    // Along edges
    CHECK_NEAR(bilinear(grid, 0.0, 0.5), 15.0, 1e-12);  // avg(10,20)
    CHECK_NEAR(bilinear(grid, 1.0, 0.5), 35.0, 1e-12);  // avg(30,40)
    CHECK_NEAR(bilinear(grid, 0.5, 0.0), 20.0, 1e-12);  // avg(10,30)
    CHECK_NEAR(bilinear(grid, 0.5, 1.0), 30.0, 1e-12);  // avg(20,40)
}

TEST_CASE("bilinear at image edges and boundaries") {
    // 3x3 grid with known values
    std::vector<double> data = {
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0
    };
    std::mdspan<const double, std::dextents<std::size_t, 2>> grid(data.data(), 3, 3);

    // At (0,0) - top left corner
    CHECK_NEAR(bilinear(grid, 0.0, 0.0), 1.0, 1e-12);
    // At (2,2) - bottom right corner (rows-1, cols-1)
    CHECK_NEAR(bilinear(grid, 2.0, 2.0), 9.0, 1e-12);
    // At (2,0) - bottom left
    CHECK_NEAR(bilinear(grid, 2.0, 0.0), 7.0, 1e-12);
    // At (0,2) - top right
    CHECK_NEAR(bilinear(grid, 0.0, 2.0), 3.0, 1e-12);

    // Between two pixels on top edge: (0, 0.5) -> avg(1,2) = 1.5
    CHECK_NEAR(bilinear(grid, 0.0, 0.5), 1.5, 1e-12);
    // Between two pixels on left edge: (0.5, 0) -> avg(1,4) = 2.5
    CHECK_NEAR(bilinear(grid, 0.5, 0.0), 2.5, 1e-12);
}

TEST_CASE("bilinear consistency: linear data reproduces exactly") {
    // For f(y,x) = 2*y + 3*x + 1, bilinear should be exact
    std::vector<double> data(16);
    for (int y = 0; y < 4; ++y)
        for (int x = 0; x < 4; ++x)
            data[y * 4 + x] = 2.0 * y + 3.0 * x + 1.0;

    std::mdspan<const double, std::dextents<std::size_t, 2>> grid(data.data(), 4, 4);

    // Sample at various fractional positions
    CHECK_NEAR(bilinear(grid, 1.3, 2.7), 2.0 * 1.3 + 3.0 * 2.7 + 1.0, 1e-10);
    CHECK_NEAR(bilinear(grid, 0.5, 0.5), 2.0 * 0.5 + 3.0 * 0.5 + 1.0, 1e-10);
    CHECK_NEAR(bilinear(grid, 2.9, 0.1), 2.0 * 2.9 + 3.0 * 0.1 + 1.0, 1e-10);
}

TEST_CASE("bilinear clamping at extreme coordinates") {
    std::vector<double> data = {10.0, 20.0, 30.0, 40.0};
    std::mdspan<const double, std::dextents<std::size_t, 2>> grid(data.data(), 2, 2);

    // Negative coordinates should clamp to edge
    CHECK_NEAR(bilinear(grid, -10.0, -10.0), 10.0, 1e-12);
    // Way beyond should clamp to far corner
    CHECK_NEAR(bilinear(grid, 100.0, 100.0), 40.0, 1e-12);
    // Mixed
    CHECK_NEAR(bilinear(grid, -5.0, 100.0), 20.0, 1e-12);
    CHECK_NEAR(bilinear(grid, 100.0, -5.0), 30.0, 1e-12);
}

// ---- Bug hunt: trilinear on 2x2x2 with known formula ----------------------

TEST_CASE("trilinear 2x2x2 known values") {
    // vZYX naming: v000=0, v100=1, v010=2, v110=3, v001=4, v101=5, v011=6, v111=7
    // Standard layout: z varies slowest.
    // Index: z*4 + y*2 + x
    // z=0: v000=0, v010=2, v001=4, v011=6
    // z=1: v100=1, v110=3, v101=5, v111=7
    std::vector<double> vol = {0.0, 4.0, 2.0, 6.0,  1.0, 5.0, 3.0, 7.0};
    std::mdspan<const double, std::dextents<std::size_t, 3>> v(vol.data(), 2, 2, 2);

    // At exact corners
    CHECK_NEAR(trilinear(v, 0.0, 0.0, 0.0), 0.0, 1e-12);
    CHECK_NEAR(trilinear(v, 1.0, 1.0, 1.0), 7.0, 1e-12);
    CHECK_NEAR(trilinear(v, 0.0, 0.0, 1.0), 4.0, 1e-12);
    CHECK_NEAR(trilinear(v, 1.0, 0.0, 0.0), 1.0, 1e-12);

    // Center: average of all 8 = 28/8 = 3.5
    CHECK_NEAR(trilinear(v, 0.5, 0.5, 0.5), 3.5, 1e-12);
}

UTILS2_TEST_MAIN()
