#include <utils2/test.hpp>
#include <utils2/tiff.hpp>
#include <filesystem>
#include <vector>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <cmath>

namespace fs = std::filesystem;

static fs::path make_temp_dir(const char* suffix) {
    auto p = fs::temp_directory_path() / ("utils2_test_tiff_" + std::string(suffix));
    fs::create_directories(p);
    return p;
}

// ============================================================================
// TiffImageInfo
// ============================================================================

TEST_CASE("TiffImageInfo default values") {
    utils2::TiffImageInfo info;
    REQUIRE_EQ(info.width, std::uint32_t(0));
    REQUIRE_EQ(info.height, std::uint32_t(0));
    REQUIRE_EQ(info.bits_per_sample, std::uint16_t(8));
    REQUIRE_EQ(info.samples_per_pixel, std::uint16_t(1));
    REQUIRE_EQ(info.compression, std::uint16_t(1));
    REQUIRE_EQ(info.photometric, std::uint16_t(1));
    REQUIRE_EQ(info.sample_format, std::uint16_t(1));
    REQUIRE_EQ(info.planar_config, std::uint16_t(1));
    REQUIRE_EQ(info.rows_per_strip, std::uint32_t(0));
}

TEST_CASE("TiffImageInfo pixel_bytes and row_bytes") {
    utils2::TiffImageInfo info;
    info.width = 100;
    info.height = 50;
    info.bits_per_sample = 16;
    info.samples_per_pixel = 3;

    REQUIRE_EQ(info.pixel_bytes(), std::size_t(6));
    REQUIRE_EQ(info.row_bytes(), std::size_t(600));
    REQUIRE_EQ(info.image_bytes(), std::size_t(30000));
}

TEST_CASE("TiffImageInfo single pixel calculations") {
    SECTION("8-bit grayscale") {
        utils2::TiffImageInfo info;
        info.width = 1; info.height = 1;
        info.bits_per_sample = 8; info.samples_per_pixel = 1;
        REQUIRE_EQ(info.pixel_bytes(), std::size_t(1));
        REQUIRE_EQ(info.image_bytes(), std::size_t(1));
    }
    SECTION("32-bit RGBA") {
        utils2::TiffImageInfo info;
        info.width = 1; info.height = 1;
        info.bits_per_sample = 32; info.samples_per_pixel = 4;
        REQUIRE_EQ(info.pixel_bytes(), std::size_t(16));
        REQUIRE_EQ(info.image_bytes(), std::size_t(16));
    }
    SECTION("16-bit RGB") {
        utils2::TiffImageInfo info;
        info.width = 1; info.height = 1;
        info.bits_per_sample = 16; info.samples_per_pixel = 3;
        REQUIRE_EQ(info.pixel_bytes(), std::size_t(6));
        REQUIRE_EQ(info.image_bytes(), std::size_t(6));
    }
}

// ============================================================================
// 8-bit grayscale write/read
// ============================================================================

TEST_CASE("TIFF write and read back 8-bit grayscale") {
    auto dir = make_temp_dir("gray8");
    auto path = dir / "gray8.tif";

    constexpr std::uint32_t W = 16, H = 8;
    utils2::TiffImageInfo info;
    info.width = W;
    info.height = H;
    info.bits_per_sample = 8;
    info.samples_per_pixel = 1;
    info.photometric = 1; // BlackIsZero

    std::vector<std::uint8_t> pixels(W * H);
    std::iota(pixels.begin(), pixels.end(), std::uint8_t(0));

    utils2::write_tiff(path, info, pixels);

    utils2::TiffReader reader(path);
    REQUIRE_EQ(reader.num_pages(), std::size_t(1));

    auto ri = reader.info(0);
    REQUIRE_EQ(ri.width, W);
    REQUIRE_EQ(ri.height, H);
    REQUIRE_EQ(ri.bits_per_sample, std::uint16_t(8));
    REQUIRE_EQ(ri.samples_per_pixel, std::uint16_t(1));

    auto data = reader.read(0);
    REQUIRE_EQ(data.size(), pixels.size());

    for (std::size_t i = 0; i < pixels.size(); ++i)
        REQUIRE_EQ(data[i], pixels[i]);

    fs::remove_all(dir);
}

// ============================================================================
// 16-bit grayscale write/read
// ============================================================================

TEST_CASE("TIFF write and read back 16-bit grayscale") {
    auto dir = make_temp_dir("gray16");
    auto path = dir / "gray16.tif";

    constexpr std::uint32_t W = 10, H = 10;
    utils2::TiffImageInfo info;
    info.width = W;
    info.height = H;
    info.bits_per_sample = 16;
    info.samples_per_pixel = 1;
    info.photometric = 1;

    std::vector<std::uint16_t> pixels(W * H);
    for (std::size_t i = 0; i < pixels.size(); ++i)
        pixels[i] = static_cast<std::uint16_t>(i * 100);

    utils2::write_tiff_u16(path, info, pixels);

    utils2::TiffReader reader(path);
    auto ri = reader.info(0);
    REQUIRE_EQ(ri.bits_per_sample, std::uint16_t(16));

    auto data = reader.read_u16(0);
    REQUIRE_EQ(data.size(), pixels.size());

    for (std::size_t i = 0; i < pixels.size(); ++i)
        REQUIRE_EQ(data[i], pixels[i]);

    fs::remove_all(dir);
}

// ============================================================================
// RGB write/read
// ============================================================================

TEST_CASE("TIFF write and read back RGB") {
    auto dir = make_temp_dir("rgb");
    auto path = dir / "rgb.tif";

    constexpr std::uint32_t W = 4, H = 4;
    utils2::TiffImageInfo info;
    info.width = W;
    info.height = H;
    info.bits_per_sample = 8;
    info.samples_per_pixel = 3;
    info.photometric = 2; // RGB

    std::vector<std::uint8_t> pixels(W * H * 3);
    for (std::size_t i = 0; i < pixels.size(); i += 3) {
        pixels[i + 0] = 255; // R
        pixels[i + 1] = 0;   // G
        pixels[i + 2] = 128; // B
    }

    utils2::write_tiff(path, info, pixels);

    utils2::TiffReader reader(path);
    auto ri = reader.info(0);
    REQUIRE_EQ(ri.samples_per_pixel, std::uint16_t(3));
    REQUIRE_EQ(ri.photometric, std::uint16_t(2));

    auto data = reader.read(0);
    REQUIRE_EQ(data.size(), pixels.size());

    // Verify first pixel.
    REQUIRE_EQ(data[0], std::uint8_t(255));
    REQUIRE_EQ(data[1], std::uint8_t(0));
    REQUIRE_EQ(data[2], std::uint8_t(128));

    fs::remove_all(dir);
}

// ============================================================================
// RGBA write/read
// ============================================================================

TEST_CASE("TIFF write and read back RGBA 8-bit") {
    auto dir = make_temp_dir("rgba8");
    auto path = dir / "rgba8.tif";

    constexpr std::uint32_t W = 8, H = 6;
    utils2::TiffImageInfo info;
    info.width = W;
    info.height = H;
    info.bits_per_sample = 8;
    info.samples_per_pixel = 4;
    info.photometric = 2; // RGB (with extra alpha sample)

    std::vector<std::uint8_t> pixels(W * H * 4);
    for (std::size_t i = 0; i < pixels.size(); i += 4) {
        pixels[i + 0] = 10;  // R
        pixels[i + 1] = 20;  // G
        pixels[i + 2] = 30;  // B
        pixels[i + 3] = 200; // A
    }

    utils2::write_tiff(path, info, pixels);

    utils2::TiffReader reader(path);
    auto ri = reader.info(0);
    REQUIRE_EQ(ri.width, W);
    REQUIRE_EQ(ri.height, H);
    REQUIRE_EQ(ri.samples_per_pixel, std::uint16_t(4));
    REQUIRE_EQ(ri.bits_per_sample, std::uint16_t(8));

    auto data = reader.read(0);
    REQUIRE_EQ(data.size(), pixels.size());

    // Check first and last pixel
    REQUIRE_EQ(data[0], std::uint8_t(10));
    REQUIRE_EQ(data[1], std::uint8_t(20));
    REQUIRE_EQ(data[2], std::uint8_t(30));
    REQUIRE_EQ(data[3], std::uint8_t(200));

    auto last = data.size() - 4;
    REQUIRE_EQ(data[last + 0], std::uint8_t(10));
    REQUIRE_EQ(data[last + 3], std::uint8_t(200));

    fs::remove_all(dir);
}

// ============================================================================
// 16-bit RGB write/read
// ============================================================================

TEST_CASE("TIFF write and read back 16-bit RGB") {
    auto dir = make_temp_dir("rgb16");
    auto path = dir / "rgb16.tif";

    constexpr std::uint32_t W = 4, H = 3;
    utils2::TiffImageInfo info;
    info.width = W;
    info.height = H;
    info.bits_per_sample = 16;
    info.samples_per_pixel = 3;
    info.photometric = 2;

    std::vector<std::uint16_t> pixels(W * H * 3);
    for (std::size_t i = 0; i < pixels.size(); i += 3) {
        pixels[i + 0] = 1000;
        pixels[i + 1] = 2000;
        pixels[i + 2] = 3000;
    }

    utils2::write_tiff_u16(path, info, pixels);

    utils2::TiffReader reader(path);
    auto ri = reader.info(0);
    REQUIRE_EQ(ri.bits_per_sample, std::uint16_t(16));
    REQUIRE_EQ(ri.samples_per_pixel, std::uint16_t(3));

    auto data = reader.read_u16(0);
    REQUIRE_EQ(data.size(), pixels.size());
    REQUIRE_EQ(data[0], std::uint16_t(1000));
    REQUIRE_EQ(data[1], std::uint16_t(2000));
    REQUIRE_EQ(data[2], std::uint16_t(3000));

    fs::remove_all(dir);
}

// ============================================================================
// Float32 write/read (sample_format=3)
// ============================================================================

TEST_CASE("TIFF write and read back float32 grayscale") {
    auto dir = make_temp_dir("float32");
    auto path = dir / "float32.tif";

    constexpr std::uint32_t W = 5, H = 4;
    utils2::TiffImageInfo info;
    info.width = W;
    info.height = H;
    info.bits_per_sample = 32;
    info.samples_per_pixel = 1;
    info.photometric = 1;
    info.sample_format = 3; // float

    // Build float data and write as raw bytes
    std::vector<float> float_pixels(W * H);
    for (std::size_t i = 0; i < float_pixels.size(); ++i)
        float_pixels[i] = static_cast<float>(i) * 0.1f;

    std::vector<std::uint8_t> bytes(float_pixels.size() * sizeof(float));
    std::memcpy(bytes.data(), float_pixels.data(), bytes.size());

    utils2::write_tiff(path, info, bytes);

    utils2::TiffReader reader(path);
    auto ri = reader.info(0);
    REQUIRE_EQ(ri.bits_per_sample, std::uint16_t(32));
    REQUIRE_EQ(ri.sample_format, std::uint16_t(3));
    REQUIRE_EQ(ri.pixel_bytes(), std::size_t(4));

    auto data = reader.read(0);
    REQUIRE_EQ(data.size(), bytes.size());

    // Read back as floats and verify
    std::vector<float> read_floats(W * H);
    std::memcpy(read_floats.data(), data.data(), data.size());

    for (std::size_t i = 0; i < float_pixels.size(); ++i)
        REQUIRE_NEAR(read_floats[i], float_pixels[i], 1e-6);

    fs::remove_all(dir);
}

// ============================================================================
// Multi-strip TIFF (rows_per_strip < height)
// ============================================================================

TEST_CASE("TIFF multi-strip write and read") {
    auto dir = make_temp_dir("multistrip");
    auto path = dir / "multistrip.tif";

    constexpr std::uint32_t W = 10, H = 20;
    utils2::TiffImageInfo info;
    info.width = W;
    info.height = H;
    info.bits_per_sample = 8;
    info.samples_per_pixel = 1;
    info.photometric = 1;
    info.rows_per_strip = 4; // 5 strips of 4 rows each

    std::vector<std::uint8_t> pixels(W * H);
    for (std::size_t i = 0; i < pixels.size(); ++i)
        pixels[i] = static_cast<std::uint8_t>(i % 256);

    utils2::write_tiff(path, info, pixels);

    utils2::TiffReader reader(path);
    auto ri = reader.info(0);
    REQUIRE_EQ(ri.width, W);
    REQUIRE_EQ(ri.height, H);
    REQUIRE_EQ(ri.rows_per_strip, std::uint32_t(4));

    auto data = reader.read(0);
    REQUIRE_EQ(data.size(), pixels.size());

    for (std::size_t i = 0; i < pixels.size(); ++i)
        REQUIRE_EQ(data[i], pixels[i]);

    fs::remove_all(dir);
}

TEST_CASE("TIFF multi-strip with uneven last strip") {
    auto dir = make_temp_dir("multistrip_uneven");
    auto path = dir / "uneven.tif";

    constexpr std::uint32_t W = 8, H = 10;
    utils2::TiffImageInfo info;
    info.width = W;
    info.height = H;
    info.bits_per_sample = 8;
    info.samples_per_pixel = 1;
    info.photometric = 1;
    info.rows_per_strip = 3; // 4 strips: 3,3,3,1 rows

    std::vector<std::uint8_t> pixels(W * H);
    std::iota(pixels.begin(), pixels.end(), std::uint8_t(0));

    utils2::write_tiff(path, info, pixels);

    utils2::TiffReader reader(path);
    auto data = reader.read(0);
    REQUIRE_EQ(data.size(), pixels.size());

    for (std::size_t i = 0; i < pixels.size(); ++i)
        REQUIRE_EQ(data[i], pixels[i]);

    fs::remove_all(dir);
}

TEST_CASE("TIFF multi-strip RGB") {
    auto dir = make_temp_dir("multistrip_rgb");
    auto path = dir / "multistrip_rgb.tif";

    constexpr std::uint32_t W = 6, H = 12;
    utils2::TiffImageInfo info;
    info.width = W;
    info.height = H;
    info.bits_per_sample = 8;
    info.samples_per_pixel = 3;
    info.photometric = 2;
    info.rows_per_strip = 5; // 3 strips: 5,5,2 rows

    std::vector<std::uint8_t> pixels(W * H * 3);
    for (std::size_t i = 0; i < pixels.size(); ++i)
        pixels[i] = static_cast<std::uint8_t>(i % 256);

    utils2::write_tiff(path, info, pixels);

    utils2::TiffReader reader(path);
    auto data = reader.read(0);
    REQUIRE_EQ(data.size(), pixels.size());

    for (std::size_t i = 0; i < pixels.size(); ++i)
        REQUIRE_EQ(data[i], pixels[i]);

    fs::remove_all(dir);
}

// ============================================================================
// read_strip individual strip reading
// ============================================================================

TEST_CASE("TIFF read individual strips") {
    auto dir = make_temp_dir("read_strip");
    auto path = dir / "strips.tif";

    constexpr std::uint32_t W = 4, H = 8;
    utils2::TiffImageInfo info;
    info.width = W;
    info.height = H;
    info.bits_per_sample = 8;
    info.samples_per_pixel = 1;
    info.photometric = 1;
    info.rows_per_strip = 2; // 4 strips of 2 rows

    std::vector<std::uint8_t> pixels(W * H);
    for (std::size_t i = 0; i < pixels.size(); ++i)
        pixels[i] = static_cast<std::uint8_t>(i);

    utils2::write_tiff(path, info, pixels);

    utils2::TiffReader reader(path);

    // Read strip 0 (rows 0-1)
    auto strip0 = reader.read_strip(0, 0);
    REQUIRE_EQ(strip0.size(), std::size_t(W * 2));
    REQUIRE_EQ(strip0[0], std::uint8_t(0));
    REQUIRE_EQ(strip0[W * 2 - 1], std::uint8_t(W * 2 - 1));

    // Read strip 3 (rows 6-7, last strip)
    auto strip3 = reader.read_strip(0, 3);
    REQUIRE_EQ(strip3.size(), std::size_t(W * 2));
    REQUIRE_EQ(strip3[0], static_cast<std::uint8_t>(W * 6));

    fs::remove_all(dir);
}

// ============================================================================
// Multi-page TIFF
// ============================================================================

TEST_CASE("TIFF multi-page") {
    auto dir = make_temp_dir("multi");
    auto path = dir / "multi.tif";

    constexpr std::uint32_t W = 8, H = 8;
    utils2::TiffImageInfo info;
    info.width = W;
    info.height = H;
    info.bits_per_sample = 8;
    info.samples_per_pixel = 1;
    info.photometric = 1;

    {
        utils2::TiffWriter writer(path);
        for (int page = 0; page < 3; ++page) {
            std::vector<std::uint8_t> pixels(W * H, static_cast<std::uint8_t>(page * 50));
            writer.write(info, pixels);
        }
        writer.close();
    }

    utils2::TiffReader reader(path);
    REQUIRE_EQ(reader.num_pages(), std::size_t(3));

    for (std::size_t page = 0; page < 3; ++page) {
        auto data = reader.read(page);
        REQUIRE_EQ(data.size(), std::size_t(W * H));
        REQUIRE_EQ(data[0], static_cast<std::uint8_t>(page * 50));
    }

    fs::remove_all(dir);
}

TEST_CASE("TIFF multi-page seek and read individual pages") {
    auto dir = make_temp_dir("seek");
    auto path = dir / "seek.tif";

    constexpr std::uint32_t W = 4, H = 4;
    utils2::TiffImageInfo info;
    info.width = W; info.height = H;
    info.bits_per_sample = 8; info.samples_per_pixel = 1;
    info.photometric = 1;

    {
        utils2::TiffWriter writer(path);
        for (int page = 0; page < 5; ++page) {
            std::vector<std::uint8_t> pixels(W * H, static_cast<std::uint8_t>(page * 10 + 1));
            writer.write(info, pixels);
        }
        writer.close();
    }

    utils2::TiffReader reader(path);
    REQUIRE_EQ(reader.num_pages(), std::size_t(5));

    // Read pages in reverse order
    for (int page = 4; page >= 0; --page) {
        auto data = reader.read(static_cast<std::size_t>(page));
        REQUIRE_EQ(data.size(), std::size_t(W * H));
        REQUIRE_EQ(data[0], static_cast<std::uint8_t>(page * 10 + 1));
    }

    fs::remove_all(dir);
}

TEST_CASE("TIFF multi-page with different sizes") {
    auto dir = make_temp_dir("multi_sizes");
    auto path = dir / "multi_sizes.tif";

    {
        utils2::TiffWriter writer(path);

        // Page 0: 4x4
        utils2::TiffImageInfo info0;
        info0.width = 4; info0.height = 4;
        info0.bits_per_sample = 8; info0.samples_per_pixel = 1;
        info0.photometric = 1;
        std::vector<std::uint8_t> p0(16, 11);
        writer.write(info0, p0);

        // Page 1: 8x2
        utils2::TiffImageInfo info1;
        info1.width = 8; info1.height = 2;
        info1.bits_per_sample = 8; info1.samples_per_pixel = 1;
        info1.photometric = 1;
        std::vector<std::uint8_t> p1(16, 22);
        writer.write(info1, p1);

        writer.close();
    }

    utils2::TiffReader reader(path);
    REQUIRE_EQ(reader.num_pages(), std::size_t(2));

    auto ri0 = reader.info(0);
    REQUIRE_EQ(ri0.width, std::uint32_t(4));
    REQUIRE_EQ(ri0.height, std::uint32_t(4));
    auto d0 = reader.read(0);
    REQUIRE_EQ(d0[0], std::uint8_t(11));

    auto ri1 = reader.info(1);
    REQUIRE_EQ(ri1.width, std::uint32_t(8));
    REQUIRE_EQ(ri1.height, std::uint32_t(2));
    auto d1 = reader.read(1);
    REQUIRE_EQ(d1[0], std::uint8_t(22));

    fs::remove_all(dir);
}

// ============================================================================
// Convenience function tests
// ============================================================================

TEST_CASE("TIFF pixel value verification") {
    auto dir = make_temp_dir("pixval");
    auto path = dir / "pixval.tif";

    constexpr std::uint32_t W = 3, H = 2;
    utils2::TiffImageInfo info;
    info.width = W;
    info.height = H;
    info.bits_per_sample = 8;
    info.samples_per_pixel = 1;
    info.photometric = 1;

    // Specific pattern: row 0 = {10, 20, 30}, row 1 = {40, 50, 60}
    std::vector<std::uint8_t> pixels = {10, 20, 30, 40, 50, 60};

    utils2::write_tiff(path, info, pixels);

    auto [ri, data] = utils2::read_tiff(path);
    REQUIRE_EQ(ri.width, W);
    REQUIRE_EQ(ri.height, H);
    REQUIRE_EQ(data.size(), std::size_t(6));
    REQUIRE_EQ(data[0], std::uint8_t(10));
    REQUIRE_EQ(data[2], std::uint8_t(30));
    REQUIRE_EQ(data[5], std::uint8_t(60));

    fs::remove_all(dir);
}

// ============================================================================
// Edge cases
// ============================================================================

TEST_CASE("TIFF 1x1 image round trip") {
    auto dir = make_temp_dir("tiny");
    auto path = dir / "tiny.tif";

    utils2::TiffImageInfo info;
    info.width = 1; info.height = 1;
    info.bits_per_sample = 8; info.samples_per_pixel = 1;
    info.photometric = 1;

    std::vector<std::uint8_t> pixels = {42};
    utils2::write_tiff(path, info, pixels);

    auto [ri, data] = utils2::read_tiff(path);
    REQUIRE_EQ(ri.width, std::uint32_t(1));
    REQUIRE_EQ(ri.height, std::uint32_t(1));
    REQUIRE_EQ(data.size(), std::size_t(1));
    REQUIRE_EQ(data[0], std::uint8_t(42));

    fs::remove_all(dir);
}

TEST_CASE("TIFF large image write/read") {
    auto dir = make_temp_dir("large");
    auto path = dir / "large.tif";

    constexpr std::uint32_t W = 512, H = 512;
    utils2::TiffImageInfo info;
    info.width = W; info.height = H;
    info.bits_per_sample = 16; info.samples_per_pixel = 1;
    info.photometric = 1;

    std::vector<std::uint16_t> pixels(W * H);
    for (std::size_t i = 0; i < pixels.size(); ++i)
        pixels[i] = static_cast<std::uint16_t>(i % 65536);

    utils2::write_tiff_u16(path, info, pixels);

    utils2::TiffReader reader(path);
    REQUIRE_EQ(reader.num_pages(), std::size_t(1));
    auto ri = reader.info(0);
    REQUIRE_EQ(ri.width, W);
    REQUIRE_EQ(ri.height, H);

    auto data = reader.read_u16(0);
    REQUIRE_EQ(data.size(), pixels.size());
    REQUIRE_EQ(data[0], pixels[0]);
    REQUIRE_EQ(data[1000], pixels[1000]);
    REQUIRE_EQ(data[data.size()-1], pixels[pixels.size()-1]);

    fs::remove_all(dir);
}

TEST_CASE("TIFF all zeros and all 255") {
    auto dir = make_temp_dir("extremes");

    SECTION("all zeros") {
        auto path = dir / "zeros.tif";
        utils2::TiffImageInfo info;
        info.width = 4; info.height = 4;
        info.bits_per_sample = 8; info.samples_per_pixel = 1;
        info.photometric = 1;
        std::vector<std::uint8_t> pixels(16, 0);
        utils2::write_tiff(path, info, pixels);
        auto [ri, data] = utils2::read_tiff(path);
        for (auto v : data) REQUIRE_EQ(v, std::uint8_t(0));
    }

    SECTION("all max") {
        auto path = dir / "max.tif";
        utils2::TiffImageInfo info;
        info.width = 4; info.height = 4;
        info.bits_per_sample = 8; info.samples_per_pixel = 1;
        info.photometric = 1;
        std::vector<std::uint8_t> pixels(16, 255);
        utils2::write_tiff(path, info, pixels);
        auto [ri, data] = utils2::read_tiff(path);
        for (auto v : data) REQUIRE_EQ(v, std::uint8_t(255));
    }

    fs::remove_all(dir);
}

TEST_CASE("TIFF 1xN and Nx1 images") {
    auto dir = make_temp_dir("skinny");

    SECTION("1 column wide") {
        auto path = dir / "col.tif";
        utils2::TiffImageInfo info;
        info.width = 1; info.height = 100;
        info.bits_per_sample = 8; info.samples_per_pixel = 1;
        info.photometric = 1;
        std::vector<std::uint8_t> pixels(100);
        std::iota(pixels.begin(), pixels.end(), std::uint8_t(0));
        utils2::write_tiff(path, info, pixels);
        auto [ri, data] = utils2::read_tiff(path);
        REQUIRE_EQ(ri.width, std::uint32_t(1));
        REQUIRE_EQ(ri.height, std::uint32_t(100));
        REQUIRE_EQ(data.size(), std::size_t(100));
        REQUIRE_EQ(data[0], std::uint8_t(0));
        REQUIRE_EQ(data[99], std::uint8_t(99));
    }

    SECTION("1 row tall") {
        auto path = dir / "row.tif";
        utils2::TiffImageInfo info;
        info.width = 100; info.height = 1;
        info.bits_per_sample = 8; info.samples_per_pixel = 1;
        info.photometric = 1;
        std::vector<std::uint8_t> pixels(100);
        std::iota(pixels.begin(), pixels.end(), std::uint8_t(0));
        utils2::write_tiff(path, info, pixels);
        auto [ri, data] = utils2::read_tiff(path);
        REQUIRE_EQ(ri.width, std::uint32_t(100));
        REQUIRE_EQ(ri.height, std::uint32_t(1));
        REQUIRE_EQ(data.size(), std::size_t(100));
    }

    fs::remove_all(dir);
}

TEST_CASE("TIFF rows_per_strip == 1 (one row per strip)") {
    auto dir = make_temp_dir("rps1");
    auto path = dir / "rps1.tif";

    constexpr std::uint32_t W = 10, H = 5;
    utils2::TiffImageInfo info;
    info.width = W; info.height = H;
    info.bits_per_sample = 8; info.samples_per_pixel = 1;
    info.photometric = 1;
    info.rows_per_strip = 1; // 5 strips, one row each

    std::vector<std::uint8_t> pixels(W * H);
    for (std::size_t i = 0; i < pixels.size(); ++i)
        pixels[i] = static_cast<std::uint8_t>(i % 256);

    utils2::write_tiff(path, info, pixels);

    utils2::TiffReader reader(path);
    auto data = reader.read(0);
    REQUIRE_EQ(data.size(), pixels.size());

    for (std::size_t i = 0; i < pixels.size(); ++i)
        REQUIRE_EQ(data[i], pixels[i]);

    fs::remove_all(dir);
}

// ============================================================================
// Error handling
// ============================================================================

TEST_CASE("TIFF reader throws on nonexistent file") {
    REQUIRE_THROWS(utils2::TiffReader("/tmp/utils2_no_such_file_ever.tif"));
}

TEST_CASE("TIFF reader throws on invalid file content") {
    auto dir = make_temp_dir("invalid");

    SECTION("file too small") {
        auto path = dir / "small.tif";
        // Write only 4 bytes
        std::ofstream f(path, std::ios::binary);
        f.write("II\x2a\x00", 4);
        f.close();
        REQUIRE_THROWS(utils2::TiffReader(path));
    }

    SECTION("bad byte order marker") {
        auto path = dir / "bad_bom.tif";
        std::ofstream f(path, std::ios::binary);
        f.write("XX\x2a\x00\x00\x00\x00\x00", 8);
        f.close();
        REQUIRE_THROWS(utils2::TiffReader(path));
    }

    SECTION("bad magic number") {
        auto path = dir / "bad_magic.tif";
        std::ofstream f(path, std::ios::binary);
        f.write("II\x00\x00\x00\x00\x00\x00", 8);
        f.close();
        REQUIRE_THROWS(utils2::TiffReader(path));
    }

    fs::remove_all(dir);
}

TEST_CASE("TIFF reader throws on out-of-range page") {
    auto dir = make_temp_dir("oor_page");
    auto path = dir / "single.tif";

    utils2::TiffImageInfo info;
    info.width = 2; info.height = 2;
    info.bits_per_sample = 8; info.samples_per_pixel = 1;
    info.photometric = 1;
    std::vector<std::uint8_t> pixels(4, 0);
    utils2::write_tiff(path, info, pixels);

    utils2::TiffReader reader(path);
    REQUIRE_EQ(reader.num_pages(), std::size_t(1));

    REQUIRE_THROWS(reader.info(1));
    REQUIRE_THROWS(reader.read(1));
    REQUIRE_THROWS(reader.read_u16(5));

    fs::remove_all(dir);
}

TEST_CASE("TIFF reader throws on out-of-range strip") {
    auto dir = make_temp_dir("oor_strip");
    auto path = dir / "strip.tif";

    utils2::TiffImageInfo info;
    info.width = 4; info.height = 4;
    info.bits_per_sample = 8; info.samples_per_pixel = 1;
    info.photometric = 1;
    info.rows_per_strip = 2; // 2 strips

    std::vector<std::uint8_t> pixels(16, 0);
    utils2::write_tiff(path, info, pixels);

    utils2::TiffReader reader(path);
    // Strip 0 and 1 should work, strip 2 should throw
    auto s0 = reader.read_strip(0, 0);
    REQUIRE_EQ(s0.size(), std::size_t(8));
    auto s1 = reader.read_strip(0, 1);
    REQUIRE_EQ(s1.size(), std::size_t(8));
    REQUIRE_THROWS(reader.read_strip(0, 2));

    fs::remove_all(dir);
}

// ============================================================================
// WhiteIsZero photometric
// ============================================================================

TEST_CASE("TIFF WhiteIsZero photometric round trip") {
    auto dir = make_temp_dir("whiteiszero");
    auto path = dir / "wiz.tif";

    constexpr std::uint32_t W = 4, H = 4;
    utils2::TiffImageInfo info;
    info.width = W; info.height = H;
    info.bits_per_sample = 8; info.samples_per_pixel = 1;
    info.photometric = 0; // WhiteIsZero

    std::vector<std::uint8_t> pixels(W * H);
    std::iota(pixels.begin(), pixels.end(), std::uint8_t(0));

    utils2::write_tiff(path, info, pixels);

    utils2::TiffReader reader(path);
    auto ri = reader.info(0);
    REQUIRE_EQ(ri.photometric, std::uint16_t(0));

    auto data = reader.read(0);
    for (std::size_t i = 0; i < pixels.size(); ++i)
        REQUIRE_EQ(data[i], pixels[i]);

    fs::remove_all(dir);
}

// ============================================================================
// TiffWriter destructor auto-close
// ============================================================================

TEST_CASE("TiffWriter auto-closes on destruction") {
    auto dir = make_temp_dir("autoclose");
    auto path = dir / "autoclose.tif";

    {
        utils2::TiffWriter writer(path);
        utils2::TiffImageInfo info;
        info.width = 2; info.height = 2;
        info.bits_per_sample = 8; info.samples_per_pixel = 1;
        info.photometric = 1;
        std::vector<std::uint8_t> pixels(4, 77);
        writer.write(info, pixels);
        // No explicit close -- destructor should handle it
    }

    // Should be readable
    auto [ri, data] = utils2::read_tiff(path);
    REQUIRE_EQ(ri.width, std::uint32_t(2));
    REQUIRE_EQ(data[0], std::uint8_t(77));

    fs::remove_all(dir);
}

// ============================================================================
// Calling close() twice is safe
// ============================================================================

TEST_CASE("TiffWriter double close is safe") {
    auto dir = make_temp_dir("dblclose");
    auto path = dir / "dblclose.tif";

    utils2::TiffWriter writer(path);
    utils2::TiffImageInfo info;
    info.width = 2; info.height = 2;
    info.bits_per_sample = 8; info.samples_per_pixel = 1;
    info.photometric = 1;
    std::vector<std::uint8_t> pixels(4, 99);
    writer.write(info, pixels);

    writer.close();
    writer.close(); // second close should be a no-op

    auto [ri, data] = utils2::read_tiff(path);
    REQUIRE_EQ(data[0], std::uint8_t(99));

    fs::remove_all(dir);
}

// ============================================================================
// Multi-strip with 16-bit data
// ============================================================================

TEST_CASE("TIFF multi-strip 16-bit round trip") {
    auto dir = make_temp_dir("multistrip16");
    auto path = dir / "ms16.tif";

    constexpr std::uint32_t W = 6, H = 9;
    utils2::TiffImageInfo info;
    info.width = W; info.height = H;
    info.bits_per_sample = 16; info.samples_per_pixel = 1;
    info.photometric = 1;
    info.rows_per_strip = 4; // 3 strips: 4,4,1 rows

    std::vector<std::uint16_t> pixels(W * H);
    for (std::size_t i = 0; i < pixels.size(); ++i)
        pixels[i] = static_cast<std::uint16_t>(i * 7);

    utils2::write_tiff_u16(path, info, pixels);

    utils2::TiffReader reader(path);
    auto data = reader.read_u16(0);
    REQUIRE_EQ(data.size(), pixels.size());

    for (std::size_t i = 0; i < pixels.size(); ++i)
        REQUIRE_EQ(data[i], pixels[i]);

    fs::remove_all(dir);
}

// ============================================================================
// PackBits decompression (unit test the internal function)
// ============================================================================

TEST_CASE("PackBits decompression") {
    // Test literal run: [2, 0xAA, 0xBB, 0xCC] = 3 literal bytes
    {
        std::vector<std::uint8_t> src = {2, 0xAA, 0xBB, 0xCC};
        std::vector<std::uint8_t> out;
        utils2::detail::tiff::decompress_packbits(src.data(), src.size(), out);
        REQUIRE_EQ(out.size(), std::size_t(3));
        REQUIRE_EQ(out[0], std::uint8_t(0xAA));
        REQUIRE_EQ(out[1], std::uint8_t(0xBB));
        REQUIRE_EQ(out[2], std::uint8_t(0xCC));
    }

    // Test repeat run: [0xFE, 0x42] = repeat 0x42 three times (1 - (-2) = 3)
    {
        std::vector<std::uint8_t> src = {0xFE, 0x42};
        std::vector<std::uint8_t> out;
        utils2::detail::tiff::decompress_packbits(src.data(), src.size(), out);
        REQUIRE_EQ(out.size(), std::size_t(3));
        for (auto v : out) REQUIRE_EQ(v, std::uint8_t(0x42));
    }

    // Test no-op byte (0x80 = -128)
    {
        std::vector<std::uint8_t> src = {0x80};
        std::vector<std::uint8_t> out;
        utils2::detail::tiff::decompress_packbits(src.data(), src.size(), out);
        REQUIRE_EQ(out.size(), std::size_t(0));
    }

    // Mixed: literal + repeat
    {
        // literal 2 bytes, then repeat 4 times
        std::vector<std::uint8_t> src = {
            1, 0x11, 0x22,          // 2 literal bytes
            0xFC, 0xFF              // repeat 0xFF 5 times (1 - (-4) = 5)
        };
        std::vector<std::uint8_t> out;
        utils2::detail::tiff::decompress_packbits(src.data(), src.size(), out);
        REQUIRE_EQ(out.size(), std::size_t(7));
        REQUIRE_EQ(out[0], std::uint8_t(0x11));
        REQUIRE_EQ(out[1], std::uint8_t(0x22));
        for (std::size_t i = 2; i < 7; ++i)
            REQUIRE_EQ(out[i], std::uint8_t(0xFF));
    }
}

// ============================================================================
// decompress_strip with unsupported compression type
// ============================================================================

TEST_CASE("decompress_strip throws on unsupported compression") {
    std::uint8_t dummy = 0;
    REQUIRE_THROWS(utils2::detail::tiff::decompress_strip(&dummy, 1, 99));
}

TEST_CASE("decompress_strip no compression passthrough") {
    std::vector<std::uint8_t> src = {1, 2, 3, 4, 5};
    auto out = utils2::detail::tiff::decompress_strip(src.data(), src.size(), 1);
    REQUIRE_EQ(out.size(), std::size_t(5));
    for (std::size_t i = 0; i < 5; ++i)
        REQUIRE_EQ(out[i], src[i]);
}

// ============================================================================
// detail::tiff::type_size
// ============================================================================

TEST_CASE("tiff type_size returns correct sizes") {
    REQUIRE_EQ(utils2::detail::tiff::type_size(1), std::size_t(1)); // BYTE
    REQUIRE_EQ(utils2::detail::tiff::type_size(2), std::size_t(1)); // ASCII
    REQUIRE_EQ(utils2::detail::tiff::type_size(3), std::size_t(2)); // SHORT
    REQUIRE_EQ(utils2::detail::tiff::type_size(4), std::size_t(4)); // LONG
    REQUIRE_EQ(utils2::detail::tiff::type_size(5), std::size_t(8)); // RATIONAL
    REQUIRE_EQ(utils2::detail::tiff::type_size(99), std::size_t(1)); // unknown defaults to 1
}

// ============================================================================
// Sample format preserved round-trip
// ============================================================================

TEST_CASE("TIFF sample_format preserved in round trip") {
    auto dir = make_temp_dir("samplefmt");

    SECTION("unsigned int (1)") {
        auto path = dir / "uint.tif";
        utils2::TiffImageInfo info;
        info.width = 2; info.height = 2;
        info.bits_per_sample = 8; info.samples_per_pixel = 1;
        info.photometric = 1; info.sample_format = 1;
        std::vector<std::uint8_t> px(4, 1);
        utils2::write_tiff(path, info, px);
        auto ri = utils2::TiffReader(path).info(0);
        REQUIRE_EQ(ri.sample_format, std::uint16_t(1));
    }

    SECTION("signed int (2)") {
        auto path = dir / "sint.tif";
        utils2::TiffImageInfo info;
        info.width = 2; info.height = 2;
        info.bits_per_sample = 16; info.samples_per_pixel = 1;
        info.photometric = 1; info.sample_format = 2;
        std::vector<std::uint8_t> px(8, 0);
        utils2::write_tiff(path, info, px);
        auto ri = utils2::TiffReader(path).info(0);
        REQUIRE_EQ(ri.sample_format, std::uint16_t(2));
    }

    SECTION("float (3)") {
        auto path = dir / "float.tif";
        utils2::TiffImageInfo info;
        info.width = 2; info.height = 2;
        info.bits_per_sample = 32; info.samples_per_pixel = 1;
        info.photometric = 1; info.sample_format = 3;
        std::vector<std::uint8_t> px(16, 0);
        utils2::write_tiff(path, info, px);
        auto ri = utils2::TiffReader(path).info(0);
        REQUIRE_EQ(ri.sample_format, std::uint16_t(3));
    }

    fs::remove_all(dir);
}

// ============================================================================
// Planar config preserved
// ============================================================================

TEST_CASE("TIFF planar_config preserved in round trip") {
    auto dir = make_temp_dir("planar");
    auto path = dir / "planar.tif";

    utils2::TiffImageInfo info;
    info.width = 4; info.height = 4;
    info.bits_per_sample = 8; info.samples_per_pixel = 3;
    info.photometric = 2; info.planar_config = 1; // chunky

    std::vector<std::uint8_t> px(48, 0);
    utils2::write_tiff(path, info, px);

    auto ri = utils2::TiffReader(path).info(0);
    REQUIRE_EQ(ri.planar_config, std::uint16_t(1));

    fs::remove_all(dir);
}

// ============================================================================
// 32-bit float RGB write/read
// ============================================================================

TEST_CASE("TIFF write and read back float32 RGB") {
    auto dir = make_temp_dir("float32_rgb");
    auto path = dir / "float32_rgb.tif";

    constexpr std::uint32_t W = 4, H = 3;
    utils2::TiffImageInfo info;
    info.width = W;
    info.height = H;
    info.bits_per_sample = 32;
    info.samples_per_pixel = 3;
    info.photometric = 2;
    info.sample_format = 3; // float

    std::vector<float> float_pixels(W * H * 3);
    for (std::size_t i = 0; i < float_pixels.size(); ++i)
        float_pixels[i] = static_cast<float>(i) * 0.01f;

    std::vector<std::uint8_t> bytes(float_pixels.size() * sizeof(float));
    std::memcpy(bytes.data(), float_pixels.data(), bytes.size());

    utils2::write_tiff(path, info, bytes);

    utils2::TiffReader reader(path);
    auto ri = reader.info(0);
    REQUIRE_EQ(ri.bits_per_sample, std::uint16_t(32));
    REQUIRE_EQ(ri.samples_per_pixel, std::uint16_t(3));
    REQUIRE_EQ(ri.sample_format, std::uint16_t(3));
    REQUIRE_EQ(ri.pixel_bytes(), std::size_t(12));

    auto data = reader.read(0);
    REQUIRE_EQ(data.size(), bytes.size());

    std::vector<float> read_floats(W * H * 3);
    std::memcpy(read_floats.data(), data.data(), data.size());

    for (std::size_t i = 0; i < float_pixels.size(); ++i)
        REQUIRE_NEAR(read_floats[i], float_pixels[i], 1e-6);

    fs::remove_all(dir);
}

// ============================================================================
// Multi-strip with RGBA
// ============================================================================

TEST_CASE("TIFF multi-strip RGBA") {
    auto dir = make_temp_dir("multistrip_rgba");
    auto path = dir / "multistrip_rgba.tif";

    constexpr std::uint32_t W = 8, H = 12;
    utils2::TiffImageInfo info;
    info.width = W;
    info.height = H;
    info.bits_per_sample = 8;
    info.samples_per_pixel = 4;
    info.photometric = 2;
    info.rows_per_strip = 3; // 4 strips: 3,3,3,3 rows

    std::vector<std::uint8_t> pixels(W * H * 4);
    for (std::size_t i = 0; i < pixels.size(); i += 4) {
        pixels[i + 0] = static_cast<std::uint8_t>((i / 4) % 256);
        pixels[i + 1] = static_cast<std::uint8_t>((i / 4 + 50) % 256);
        pixels[i + 2] = static_cast<std::uint8_t>((i / 4 + 100) % 256);
        pixels[i + 3] = static_cast<std::uint8_t>(255);
    }

    utils2::write_tiff(path, info, pixels);

    utils2::TiffReader reader(path);
    auto ri = reader.info(0);
    REQUIRE_EQ(ri.samples_per_pixel, std::uint16_t(4));
    REQUIRE_EQ(ri.rows_per_strip, std::uint32_t(3));

    auto data = reader.read(0);
    REQUIRE_EQ(data.size(), pixels.size());

    for (std::size_t i = 0; i < pixels.size(); ++i)
        REQUIRE_EQ(data[i], pixels[i]);

    fs::remove_all(dir);
}

// ============================================================================
// Signed integer sample_format round trip
// ============================================================================

TEST_CASE("TIFF signed int16 sample_format round trip") {
    auto dir = make_temp_dir("sint16");
    auto path = dir / "sint16.tif";

    constexpr std::uint32_t W = 6, H = 4;
    utils2::TiffImageInfo info;
    info.width = W;
    info.height = H;
    info.bits_per_sample = 16;
    info.samples_per_pixel = 1;
    info.photometric = 1;
    info.sample_format = 2; // signed int

    // Create signed int16 data
    std::vector<std::int16_t> signed_pixels(W * H);
    for (std::size_t i = 0; i < signed_pixels.size(); ++i)
        signed_pixels[i] = static_cast<std::int16_t>(static_cast<int>(i) - 12);

    std::vector<std::uint16_t> pixels(W * H);
    std::memcpy(pixels.data(), signed_pixels.data(), signed_pixels.size() * 2);

    utils2::write_tiff_u16(path, info, pixels);

    utils2::TiffReader reader(path);
    auto ri = reader.info(0);
    REQUIRE_EQ(ri.sample_format, std::uint16_t(2));
    REQUIRE_EQ(ri.bits_per_sample, std::uint16_t(16));

    auto data = reader.read_u16(0);
    REQUIRE_EQ(data.size(), pixels.size());

    std::vector<std::int16_t> read_signed(W * H);
    std::memcpy(read_signed.data(), data.data(), data.size() * 2);

    for (std::size_t i = 0; i < signed_pixels.size(); ++i)
        REQUIRE_EQ(read_signed[i], signed_pixels[i]);

    fs::remove_all(dir);
}

// ============================================================================
// Multi-strip 16-bit RGB
// ============================================================================

TEST_CASE("TIFF multi-strip 16-bit RGB") {
    auto dir = make_temp_dir("multistrip_rgb16");
    auto path = dir / "ms_rgb16.tif";

    constexpr std::uint32_t W = 4, H = 9;
    utils2::TiffImageInfo info;
    info.width = W;
    info.height = H;
    info.bits_per_sample = 16;
    info.samples_per_pixel = 3;
    info.photometric = 2;
    info.rows_per_strip = 2; // 5 strips: 2,2,2,2,1

    std::vector<std::uint16_t> pixels(W * H * 3);
    for (std::size_t i = 0; i < pixels.size(); ++i)
        pixels[i] = static_cast<std::uint16_t>(i * 13);

    utils2::write_tiff_u16(path, info, pixels);

    utils2::TiffReader reader(path);
    auto data = reader.read_u16(0);
    REQUIRE_EQ(data.size(), pixels.size());

    for (std::size_t i = 0; i < pixels.size(); ++i)
        REQUIRE_EQ(data[i], pixels[i]);

    fs::remove_all(dir);
}

// ============================================================================
// Multi-page with different bit depths and sample formats
// ============================================================================

TEST_CASE("TIFF multi-page with different formats") {
    auto dir = make_temp_dir("multi_formats");
    auto path = dir / "multi_formats.tif";

    {
        utils2::TiffWriter writer(path);

        // Page 0: 8-bit grayscale
        utils2::TiffImageInfo info0;
        info0.width = 4; info0.height = 4;
        info0.bits_per_sample = 8; info0.samples_per_pixel = 1;
        info0.photometric = 1;
        std::vector<std::uint8_t> p0(16, 100);
        writer.write(info0, p0);

        // Page 1: 16-bit grayscale
        utils2::TiffImageInfo info1;
        info1.width = 3; info1.height = 3;
        info1.bits_per_sample = 16; info1.samples_per_pixel = 1;
        info1.photometric = 1;
        std::vector<std::uint16_t> p1_u16(9, 5000);
        std::vector<std::uint8_t> p1(18);
        std::memcpy(p1.data(), p1_u16.data(), 18);
        writer.write(info1, p1);

        // Page 2: 32-bit float
        utils2::TiffImageInfo info2;
        info2.width = 2; info2.height = 2;
        info2.bits_per_sample = 32; info2.samples_per_pixel = 1;
        info2.photometric = 1; info2.sample_format = 3;
        std::vector<float> p2_f = {1.0f, 2.0f, 3.0f, 4.0f};
        std::vector<std::uint8_t> p2(16);
        std::memcpy(p2.data(), p2_f.data(), 16);
        writer.write(info2, p2);

        writer.close();
    }

    utils2::TiffReader reader(path);
    REQUIRE_EQ(reader.num_pages(), std::size_t(3));

    auto ri0 = reader.info(0);
    REQUIRE_EQ(ri0.bits_per_sample, std::uint16_t(8));
    auto d0 = reader.read(0);
    REQUIRE_EQ(d0[0], std::uint8_t(100));

    auto ri1 = reader.info(1);
    REQUIRE_EQ(ri1.bits_per_sample, std::uint16_t(16));

    auto ri2 = reader.info(2);
    REQUIRE_EQ(ri2.bits_per_sample, std::uint16_t(32));
    REQUIRE_EQ(ri2.sample_format, std::uint16_t(3));
    auto d2 = reader.read(2);
    float f_val;
    std::memcpy(&f_val, d2.data(), 4);
    REQUIRE_NEAR(f_val, 1.0f, 1e-6);

    fs::remove_all(dir);
}

// ============================================================================
// LZW decompression
// ============================================================================

TEST_CASE("LZW decompression: basic encoded data") {
    // Craft minimal LZW data: clear code, then a few literal bytes, then EOI
    // TIFF LZW is MSB-first, starts with 9-bit codes
    // Clear code = 256 (0x100), EOI = 257 (0x101)
    // We'll encode: CLEAR, 'A'(65), 'B'(66), EOI

    // 9-bit codes MSB first:
    // 256 = 100000000
    // 65  = 001000001
    // 66  = 001000010
    // 257 = 100000001
    // Total: 36 bits = 4.5 bytes (pad to 5 bytes)

    // Bits: 100000000 001000001 001000010 100000001
    // Byte boundaries:
    //   10000000 0|00100000 1|00100001 0|10000000 1|0000000(pad)
    //   0x80       0x20       0x21       0x80+0x80   -> wait, let me recalculate

    // Bit stream (MSB first): 1 0 0 0 0 0 0 0 0  0 0 1 0 0 0 0 0 1  0 0 1 0 0 0 0 1 0  1 0 0 0 0 0 0 0 1
    // Group into bytes:
    // byte 0: 1 0 0 0 0 0 0 0 = 0x80
    // byte 1: 0 0 0 1 0 0 0 0 = 0x10
    // byte 2: 0 1 0 0 1 0 0 0 = 0x48
    // byte 3: 0 1 0 1 0 0 0 0 = 0x50
    // byte 4: 0 0 0 1 0 0 0 0 = 0x10 (remaining bits + padding)

    std::vector<std::uint8_t> lzw_data = {0x80, 0x10, 0x48, 0x50, 0x10};
    std::vector<std::uint8_t> out;
    utils2::detail::tiff::decompress_lzw(lzw_data.data(), lzw_data.size(), out);

    REQUIRE_EQ(out.size(), std::size_t(2));
    REQUIRE_EQ(out[0], std::uint8_t(65));
    REQUIRE_EQ(out[1], std::uint8_t(66));
}

TEST_CASE("LZW decompression: empty or just EOI after clear") {
    // CLEAR then EOI
    // 256 = 100000000 (9 bits)
    // 257 = 100000001 (9 bits)
    // Bits: 100000000 100000001
    // byte 0: 10000000 = 0x80
    // byte 1: 01000000 = 0x40
    // byte 2: 01000000 = 0x40 (with padding)
    std::vector<std::uint8_t> lzw_data = {0x80, 0x40, 0x40};
    std::vector<std::uint8_t> out;
    utils2::detail::tiff::decompress_lzw(lzw_data.data(), lzw_data.size(), out);
    REQUIRE_EQ(out.size(), std::size_t(0));
}

TEST_CASE("LZW decompression: no clear code at start returns empty") {
    // First code is not clear code -> should return immediately
    std::vector<std::uint8_t> lzw_data = {0x00, 0x00, 0x00};
    std::vector<std::uint8_t> out;
    utils2::detail::tiff::decompress_lzw(lzw_data.data(), lzw_data.size(), out);
    REQUIRE_EQ(out.size(), std::size_t(0));
}

// ============================================================================
// ByteReader bounds checking
// ============================================================================

TEST_CASE("ByteReader u16 out of bounds throws") {
    std::vector<std::uint8_t> data = {0x42};
    utils2::detail::tiff::ByteReader reader{data.data(), data.size(), false};
    REQUIRE_THROWS(reader.u16(0));
}

TEST_CASE("ByteReader u32 out of bounds throws") {
    std::vector<std::uint8_t> data = {0x42, 0x00};
    utils2::detail::tiff::ByteReader reader{data.data(), data.size(), false};
    REQUIRE_THROWS(reader.u32(0));
}

// ============================================================================
// Multi-strip float32
// ============================================================================

TEST_CASE("TIFF multi-strip float32") {
    auto dir = make_temp_dir("multistrip_f32");
    auto path = dir / "ms_f32.tif";

    constexpr std::uint32_t W = 5, H = 8;
    utils2::TiffImageInfo info;
    info.width = W;
    info.height = H;
    info.bits_per_sample = 32;
    info.samples_per_pixel = 1;
    info.photometric = 1;
    info.sample_format = 3;
    info.rows_per_strip = 3; // 3 strips: 3,3,2 rows

    std::vector<float> float_pixels(W * H);
    for (std::size_t i = 0; i < float_pixels.size(); ++i)
        float_pixels[i] = static_cast<float>(i) * 0.5f - 10.0f;

    std::vector<std::uint8_t> bytes(float_pixels.size() * sizeof(float));
    std::memcpy(bytes.data(), float_pixels.data(), bytes.size());

    utils2::write_tiff(path, info, bytes);

    utils2::TiffReader reader(path);
    auto ri = reader.info(0);
    REQUIRE_EQ(ri.rows_per_strip, std::uint32_t(3));

    auto data = reader.read(0);
    REQUIRE_EQ(data.size(), bytes.size());

    std::vector<float> read_floats(W * H);
    std::memcpy(read_floats.data(), data.data(), data.size());

    for (std::size_t i = 0; i < float_pixels.size(); ++i)
        REQUIRE_NEAR(read_floats[i], float_pixels[i], 1e-6);

    fs::remove_all(dir);
}

// ============================================================================
// 16-bit RGBA round trip
// ============================================================================

TEST_CASE("TIFF 16-bit RGBA round trip") {
    auto dir = make_temp_dir("rgba16");
    auto path = dir / "rgba16.tif";

    constexpr std::uint32_t W = 3, H = 3;
    utils2::TiffImageInfo info;
    info.width = W;
    info.height = H;
    info.bits_per_sample = 16;
    info.samples_per_pixel = 4;
    info.photometric = 2;

    std::vector<std::uint16_t> pixels(W * H * 4);
    for (std::size_t i = 0; i < pixels.size(); i += 4) {
        pixels[i + 0] = 1000;
        pixels[i + 1] = 2000;
        pixels[i + 2] = 3000;
        pixels[i + 3] = 65535;
    }

    utils2::write_tiff_u16(path, info, pixels);

    utils2::TiffReader reader(path);
    auto ri = reader.info(0);
    REQUIRE_EQ(ri.samples_per_pixel, std::uint16_t(4));
    REQUIRE_EQ(ri.bits_per_sample, std::uint16_t(16));

    auto data = reader.read_u16(0);
    REQUIRE_EQ(data.size(), pixels.size());
    REQUIRE_EQ(data[0], std::uint16_t(1000));
    REQUIRE_EQ(data[3], std::uint16_t(65535));

    fs::remove_all(dir);
}

// ============================================================================
// Unsupported compression in decompress_strip
// ============================================================================

TEST_CASE("decompress_strip unsupported compression codes") {
    std::uint8_t dummy = 0;
    // Various unsupported compression types
    REQUIRE_THROWS(utils2::detail::tiff::decompress_strip(&dummy, 1, 2));    // CCITT
    REQUIRE_THROWS(utils2::detail::tiff::decompress_strip(&dummy, 1, 3));    // T4
    REQUIRE_THROWS(utils2::detail::tiff::decompress_strip(&dummy, 1, 4));    // T6
    REQUIRE_THROWS(utils2::detail::tiff::decompress_strip(&dummy, 1, 7));    // JPEG
    REQUIRE_THROWS(utils2::detail::tiff::decompress_strip(&dummy, 1, 8));    // Deflate
    REQUIRE_THROWS(utils2::detail::tiff::decompress_strip(&dummy, 1, 34712)); // JP2000
}

// ============================================================================
// Big-endian byte order marker (MM)
// ============================================================================

TEST_CASE("TIFF big-endian byte order marker is recognized") {
    auto dir = make_temp_dir("bigendian_marker");
    // Create a minimal valid big-endian TIFF file manually
    auto path = dir / "be_invalid.tif";

    // Write a minimal big-endian header with magic 42 but IFD offset pointing to 0
    // (no valid IFDs - should throw because no IFDs found)
    std::vector<std::uint8_t> be_tiff = {
        'M', 'M',          // Big endian
        0x00, 0x2A,        // Magic 42 (big endian)
        0x00, 0x00, 0x00, 0x00  // IFD offset = 0 (no IFDs)
    };
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(be_tiff.data()),
            static_cast<std::streamsize>(be_tiff.size()));
    f.close();

    // IFD offset of 0 means no IFDs, which should throw "no IFDs found"
    REQUIRE_THROWS(utils2::TiffReader(path));

    fs::remove_all(dir);
}

// ============================================================================
// read_u16 page out of range
// ============================================================================

TEST_CASE("TIFF read_u16 page out of range throws") {
    auto dir = make_temp_dir("read_u16_oor");
    auto path = dir / "single_u16.tif";

    utils2::TiffImageInfo info;
    info.width = 2; info.height = 2;
    info.bits_per_sample = 16; info.samples_per_pixel = 1;
    info.photometric = 1;

    std::vector<std::uint16_t> pixels = {100, 200, 300, 400};
    utils2::write_tiff_u16(path, info, pixels);

    utils2::TiffReader reader(path);
    // Page 0 should work
    auto data = reader.read_u16(0);
    REQUIRE_EQ(data.size(), std::size_t(4));
    // Page 1 should throw
    REQUIRE_THROWS(reader.read_u16(1));

    fs::remove_all(dir);
}

// ============================================================================
// read_strip page out of range
// ============================================================================

TEST_CASE("TIFF read_strip page out of range throws") {
    auto dir = make_temp_dir("read_strip_oor");
    auto path = dir / "strip_oor.tif";

    utils2::TiffImageInfo info;
    info.width = 4; info.height = 4;
    info.bits_per_sample = 8; info.samples_per_pixel = 1;
    info.photometric = 1;
    std::vector<std::uint8_t> pixels(16, 0);
    utils2::write_tiff(path, info, pixels);

    utils2::TiffReader reader(path);
    REQUIRE_THROWS(reader.read_strip(1, 0));
}

UTILS2_TEST_MAIN()
