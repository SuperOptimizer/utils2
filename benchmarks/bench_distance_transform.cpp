#include <utils2/bench.hpp>
#include <utils2/distance_transform.hpp>
#include <vector>
#include <cstdint>
#include <random>

BENCH_MAIN()

namespace {

// Generate a binary image with ~50% foreground
std::vector<std::uint8_t> make_binary_2d(std::size_t rows, std::size_t cols, unsigned seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(0, 1);
    std::vector<std::uint8_t> img(rows * cols);
    for (auto& v : img) v = static_cast<std::uint8_t>(dist(rng));
    return img;
}

std::vector<std::uint8_t> make_binary_3d(std::size_t d0, std::size_t d1, std::size_t d2, unsigned seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(0, 1);
    std::vector<std::uint8_t> vol(d0 * d1 * d2);
    for (auto& v : vol) v = static_cast<std::uint8_t>(dist(rng));
    return vol;
}

} // namespace

BENCHMARK("EDT 2D 64x64") {
    static auto img = make_binary_2d(64, 64);
    std::mdspan<const std::uint8_t, std::dextents<std::size_t, 2>> view(img.data(), 64, 64);
    auto result = utils2::edt_2d<std::uint8_t, float>(view);
}

BENCHMARK("EDT 2D 256x256") {
    static auto img = make_binary_2d(256, 256);
    std::mdspan<const std::uint8_t, std::dextents<std::size_t, 2>> view(img.data(), 256, 256);
    auto result = utils2::edt_2d<std::uint8_t, float>(view);
}

BENCHMARK("EDT 2D 512x512") {
    static auto img = make_binary_2d(512, 512);
    std::mdspan<const std::uint8_t, std::dextents<std::size_t, 2>> view(img.data(), 512, 512);
    auto result = utils2::edt_2d<std::uint8_t, float>(view);
}

BENCHMARK("EDT 3D 32x32x32") {
    static auto vol = make_binary_3d(32, 32, 32);
    std::mdspan<const std::uint8_t, std::dextents<std::size_t, 3>> view(vol.data(), 32, 32, 32);
    auto result = utils2::edt_3d<std::uint8_t, float>(view);
}

BENCHMARK("EDT 3D 64x64x64") {
    static auto vol = make_binary_3d(64, 64, 64);
    std::mdspan<const std::uint8_t, std::dextents<std::size_t, 3>> view(vol.data(), 64, 64, 64);
    auto result = utils2::edt_3d<std::uint8_t, float>(view);
}
