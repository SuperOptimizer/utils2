#include <utils2/test.hpp>
#include <utils2/morphology.hpp>
#include <vector>
#include <utils2/mdspan.hpp>
#include <numeric>

using namespace utils2;

// ---- erode_2d --------------------------------------------------------------

TEST_CASE("erode_2d removes thin protrusion") {
    // 5x5 with a single pixel protrusion
    std::vector<uint8_t> img = {
        0, 0, 0, 0, 0,
        0, 1, 1, 1, 0,
        0, 1, 1, 1, 0,
        0, 1, 1, 1, 0,
        0, 0, 1, 0, 0   // protrusion at (4,2)
    };
    std::mdspan<const uint8_t, std::dextents<std::size_t, 2>> view(img.data(), 5, 5);
    auto result = erode_2d(view, StructuringElement::square);

    // After erosion with 3x3 square, only interior pixel (2,2) might survive
    // The protrusion and border pixels should be eroded
    CHECK_EQ(result[4*5+2], 0);  // protrusion eroded
    CHECK_EQ(result[2*5+2], 1);  // center survives (all 8 neighbors are 1)
}

TEST_CASE("erode_2d cross element") {
    // 3x3 all foreground, erode with cross
    std::vector<uint8_t> img(9, 1);
    std::mdspan<const uint8_t, std::dextents<std::size_t, 2>> view(img.data(), 3, 3);
    auto result = erode_2d(view, StructuringElement::cross);

    // Center pixel: all cardinal neighbors are fg -> survives
    CHECK_EQ(result[4], 1);
    // Corner (0,0): north/west are out of bounds (border = clamped to pixel value)
    // Actually for_each_neighbor only visits valid pixels, erosion takes min of all visited
    // So edges might survive since they only see in-bounds neighbors
}

// ---- dilate_2d -------------------------------------------------------------

TEST_CASE("dilate_2d expands") {
    // Single pixel in center of 5x5
    std::vector<uint8_t> img(25, 0);
    img[2*5+2] = 1;
    std::mdspan<const uint8_t, std::dextents<std::size_t, 2>> view(img.data(), 5, 5);

    auto result = dilate_2d(view, StructuringElement::square);
    // Center and all 8 neighbors should be 1
    CHECK_EQ(result[2*5+2], 1);
    CHECK_EQ(result[1*5+1], 1);
    CHECK_EQ(result[1*5+2], 1);
    CHECK_EQ(result[1*5+3], 1);
    CHECK_EQ(result[3*5+2], 1);
    // Pixels 2 away should still be 0
    CHECK_EQ(result[0*5+0], 0);
    CHECK_EQ(result[4*5+4], 0);
}

TEST_CASE("dilate_2d cross element") {
    std::vector<uint8_t> img(25, 0);
    img[2*5+2] = 1;
    std::mdspan<const uint8_t, std::dextents<std::size_t, 2>> view(img.data(), 5, 5);

    auto result = dilate_2d(view, StructuringElement::cross);
    // Cardinal neighbors dilated
    CHECK_EQ(result[1*5+2], 1);
    CHECK_EQ(result[3*5+2], 1);
    CHECK_EQ(result[2*5+1], 1);
    CHECK_EQ(result[2*5+3], 1);
    // Diagonals NOT dilated by cross
    CHECK_EQ(result[1*5+1], 0);
    CHECK_EQ(result[3*5+3], 0);
}

// ---- open_2d / close_2d ---------------------------------------------------

TEST_CASE("open_2d removes isolated pixel") {
    // Opening removes small bright features
    std::vector<uint8_t> img(25, 0);
    img[2*5+2] = 1;  // isolated pixel
    std::mdspan<const uint8_t, std::dextents<std::size_t, 2>> view(img.data(), 5, 5);

    auto result = open_2d(view, StructuringElement::square);
    // Isolated pixel should be removed by opening
    CHECK_EQ(result[2*5+2], 0);
}

TEST_CASE("close_2d fills small hole") {
    // All fg except one interior pixel
    std::vector<uint8_t> img(25, 1);
    img[2*5+2] = 0;  // hole
    std::mdspan<const uint8_t, std::dextents<std::size_t, 2>> view(img.data(), 5, 5);

    auto result = close_2d(view, StructuringElement::square);
    // Hole should be filled by closing
    CHECK_EQ(result[2*5+2], 1);
}

// ---- thin_2d (Zhang-Suen) --------------------------------------------------

TEST_CASE("thin_2d horizontal line") {
    // 5x7 with a thick horizontal bar (3 pixels wide)
    std::vector<uint8_t> img = {
        0, 0, 0, 0, 0, 0, 0,
        0, 1, 1, 1, 1, 1, 0,
        0, 1, 1, 1, 1, 1, 0,
        0, 1, 1, 1, 1, 1, 0,
        0, 0, 0, 0, 0, 0, 0
    };
    std::mdspan<const uint8_t, std::dextents<std::size_t, 2>> view(img.data(), 5, 7);
    auto skel = thin_2d(view);

    // After thinning, the skeleton should be thin: sum of fg pixels should be
    // less than original (15) but more than 0
    int total = 0;
    for (auto v : skel) total += (v != 0);
    CHECK_GT(total, 0);
    CHECK_LT(total, 15);
}

TEST_CASE("thin_2d preserves topology") {
    // Single-pixel-wide line should not be removed
    std::vector<uint8_t> img = {
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 1, 1, 1, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0
    };
    std::mdspan<const uint8_t, std::dextents<std::size_t, 2>> view(img.data(), 5, 5);
    auto skel = thin_2d(view);

    // The middle row should still have foreground pixels
    int mid_row_fg = 0;
    for (int c = 0; c < 5; ++c)
        mid_row_fg += (skel[2*5+c] != 0);
    CHECK_GT(mid_row_fg, 0);
}

// ---- flood_fill_2d ---------------------------------------------------------

TEST_CASE("flood_fill_2d basic") {
    std::vector<int> img = {
        0, 0, 1,
        0, 0, 1,
        1, 1, 1
    };
    std::mdspan<int, std::dextents<std::size_t, 2>> view(img.data(), 3, 3);

    flood_fill_2d(view, {0u, 0u}, 5);
    // All connected 0-pixels from (0,0) should become 5
    CHECK_EQ(img[0], 5);
    CHECK_EQ(img[1], 5);
    CHECK_EQ(img[3], 5);
    CHECK_EQ(img[4], 5);
    // Foreground pixels unchanged
    CHECK_EQ(img[2], 1);
    CHECK_EQ(img[5], 1);
}

TEST_CASE("flood_fill_2d same value noop") {
    std::vector<int> img = {1, 1, 1, 1};
    std::mdspan<int, std::dextents<std::size_t, 2>> view(img.data(), 2, 2);
    flood_fill_2d(view, {0u, 0u}, 1);  // same value, should not infinite loop
    CHECK_EQ(img[0], 1);
}

TEST_CASE("flood_fill_2d out of bounds seed") {
    std::vector<int> img = {0, 0, 0, 0};
    std::mdspan<int, std::dextents<std::size_t, 2>> view(img.data(), 2, 2);
    flood_fill_2d(view, {10u, 10u}, 5);  // out of bounds, should be noop
    CHECK_EQ(img[0], 0);
}

// ---- remove_small_components_2d --------------------------------------------

TEST_CASE("remove_small_components_2d") {
    std::vector<uint8_t> img = {
        1, 1, 0, 0, 1,
        1, 1, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 1, 0,
        0, 0, 0, 0, 0
    };
    std::mdspan<const uint8_t, std::dextents<std::size_t, 2>> view(img.data(), 5, 5);

    auto result = remove_small_components_2d(view, 3u);
    // Component at top-left (4 pixels) survives, isolated pixels removed
    CHECK_NE(result[0], 0);
    CHECK_NE(result[1], 0);
    CHECK_EQ(result[4], 0);   // single pixel at (0,4)
    CHECK_EQ(result[3*5+3], 0); // single pixel at (3,3)
}

// ---- threshold_2d ----------------------------------------------------------

TEST_CASE("threshold_2d") {
    std::vector<int> img = {
        10, 20, 30,
        40, 50, 60
    };
    std::mdspan<const int, std::dextents<std::size_t, 2>> view(img.data(), 2, 3);

    auto result = threshold_2d(view, 30);
    CHECK_EQ(result[0], 0);    // 10 < 30
    CHECK_EQ(result[1], 0);    // 20 < 30
    CHECK_EQ(result[2], 255);  // 30 >= 30
    CHECK_EQ(result[3], 255);  // 40 >= 30
    CHECK_EQ(result[4], 255);  // 50 >= 30
    CHECK_EQ(result[5], 255);  // 60 >= 30
}

TEST_CASE("threshold_2d custom values") {
    std::vector<int> img = {5, 15};
    std::mdspan<const int, std::dextents<std::size_t, 2>> view(img.data(), 1, 2);

    auto result = threshold_2d<int, uint8_t>(view, 10, 1, 0);
    CHECK_EQ(result[0], 0);
    CHECK_EQ(result[1], 1);
}

// ---- Bug hunt: remove_small_components_2d stress test ----------------------

TEST_CASE("remove_small_components_2d scattered single pixels") {
    // 20x20 image with single scattered foreground pixels (no 8-connected neighbors)
    const std::size_t rows = 20, cols = 20;
    std::vector<uint8_t> img(rows * cols, 0);
    // Place pixels at every other position in both axes (checkerboard, even positions)
    std::size_t fg_count = 0;
    for (std::size_t r = 0; r < rows; r += 2) {
        for (std::size_t c = 0; c < cols; c += 2) {
            img[r * cols + c] = 1;
            ++fg_count;
        }
    }
    // Under 8-connectivity, each pixel at (r,c) with r,c even has no 8-connected
    // fg neighbor (nearest fg is at distance 2). So each is its own component.

    std::mdspan<const uint8_t, std::dextents<std::size_t, 2>> view(img.data(), rows, cols);
    auto result = remove_small_components_2d(view, 2u);

    // All single-pixel components should be removed
    for (std::size_t i = 0; i < result.size(); ++i) {
        CHECK_EQ(result[i], 0);
    }
}

TEST_CASE("remove_small_components_2d preserves large component") {
    // Large connected blob + small isolated pixels
    const std::size_t rows = 10, cols = 10;
    std::vector<uint8_t> img(rows * cols, 0);
    // Fill a 5x5 block in the top-left
    for (std::size_t r = 0; r < 5; ++r)
        for (std::size_t c = 0; c < 5; ++c)
            img[r * cols + c] = 1;
    // Place isolated pixels far away
    img[9 * cols + 9] = 1;
    img[7 * cols + 9] = 1;

    std::mdspan<const uint8_t, std::dextents<std::size_t, 2>> view(img.data(), rows, cols);
    auto result = remove_small_components_2d(view, 3u);

    // Large block (25 pixels) should survive
    for (std::size_t r = 0; r < 5; ++r)
        for (std::size_t c = 0; c < 5; ++c)
            CHECK_NE(result[r * cols + c], 0);

    // Isolated pixels should be removed
    CHECK_EQ(result[9 * cols + 9], 0);
    CHECK_EQ(result[7 * cols + 9], 0);
}

TEST_CASE("remove_small_components_2d all background") {
    std::vector<uint8_t> img(25, 0);
    std::mdspan<const uint8_t, std::dextents<std::size_t, 2>> view(img.data(), 5, 5);
    auto result = remove_small_components_2d(view, 1u);
    for (auto v : result) CHECK_EQ(v, 0);
}

// ---- disk_3 structuring element (exercises dispatch_se disk_3 branch) ------

TEST_CASE("erode_2d disk_3 same as square") {
    std::vector<uint8_t> img(9, 1);
    std::mdspan<const uint8_t, std::dextents<std::size_t, 2>> view(img.data(), 3, 3);

    auto result_disk3 = erode_2d(view, StructuringElement::disk_3);
    auto result_square = erode_2d(view, StructuringElement::square);

    // disk_3 is defined as an alias for square, so results should be identical
    REQUIRE_EQ(result_disk3.size(), result_square.size());
    for (std::size_t i = 0; i < result_disk3.size(); ++i) {
        CHECK_EQ(result_disk3[i], result_square[i]);
    }
}

TEST_CASE("dilate_2d disk_3") {
    std::vector<uint8_t> img(25, 0);
    img[2*5+2] = 1;
    std::mdspan<const uint8_t, std::dextents<std::size_t, 2>> view(img.data(), 5, 5);

    auto result = dilate_2d(view, StructuringElement::disk_3);
    // Same as square: center and all 8 neighbors should be 1
    CHECK_EQ(result[2*5+2], 1);
    CHECK_EQ(result[1*5+1], 1);
    CHECK_EQ(result[1*5+2], 1);
    CHECK_EQ(result[1*5+3], 1);
}

// ---- disk_5 structuring element (exercises dispatch_se disk_5 branch) ------

TEST_CASE("erode_2d disk_5") {
    // 7x7 all foreground
    std::vector<uint8_t> img(49, 1);
    std::mdspan<const uint8_t, std::dextents<std::size_t, 2>> view(img.data(), 7, 7);

    auto result = erode_2d(view, StructuringElement::disk_5);
    // Center (3,3) has all disk_5 neighbors in-bounds and all fg, should survive
    CHECK_EQ(result[3*7+3], 1);
    // Corner (0,0) doesn't have all disk_5 neighbors -- but since erosion takes min
    // of only valid neighbors, and all valid are 1, it survives too with all-fg image
    // The key is: this exercises the disk_5 dispatch path
}

TEST_CASE("dilate_2d disk_5") {
    // Single pixel in center of 7x7
    std::vector<uint8_t> img(49, 0);
    img[3*7+3] = 1;
    std::mdspan<const uint8_t, std::dextents<std::size_t, 2>> view(img.data(), 7, 7);

    auto result = dilate_2d(view, StructuringElement::disk_5);
    // Center should be dilated
    CHECK_EQ(result[3*7+3], 1);
    // Cardinal neighbors at distance 1 should be dilated
    CHECK_EQ(result[2*7+3], 1);
    CHECK_EQ(result[4*7+3], 1);
    CHECK_EQ(result[3*7+2], 1);
    CHECK_EQ(result[3*7+4], 1);
    // Pixels at distance 2 along cardinal should be dilated by disk_5
    CHECK_EQ(result[1*7+3], 1);
    CHECK_EQ(result[5*7+3], 1);
    // Far corners should NOT be dilated (distance > disk radius)
    CHECK_EQ(result[0*7+0], 0);
    CHECK_EQ(result[6*7+6], 0);
}

TEST_CASE("open_2d with disk_5 removes small features") {
    // 9x9 with a single isolated pixel
    std::vector<uint8_t> img(81, 0);
    img[4*9+4] = 1;
    std::mdspan<const uint8_t, std::dextents<std::size_t, 2>> view(img.data(), 9, 9);

    auto result = open_2d(view, StructuringElement::disk_5);
    // Isolated pixel removed by opening
    CHECK_EQ(result[4*9+4], 0);
}

TEST_CASE("close_2d with disk_5 fills small holes") {
    // 9x9 all fg except center
    std::vector<uint8_t> img(81, 1);
    img[4*9+4] = 0;
    std::mdspan<const uint8_t, std::dextents<std::size_t, 2>> view(img.data(), 9, 9);

    auto result = close_2d(view, StructuringElement::disk_5);
    // Hole should be filled by closing
    CHECK_EQ(result[4*9+4], 1);
}

UTILS2_TEST_MAIN()
