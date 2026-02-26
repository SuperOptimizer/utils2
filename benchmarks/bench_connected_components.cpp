#include <utils2/bench.hpp>
#include <utils2/connected_components.hpp>
#include <vector>
#include <cstdint>
#include <random>

BENCH_MAIN()

namespace {

std::vector<std::uint8_t> make_binary_2d(std::size_t rows, std::size_t cols,
                                          double density = 0.6, unsigned seed = 42) {
    std::mt19937 rng(seed);
    std::bernoulli_distribution dist(density);
    std::vector<std::uint8_t> img(rows * cols);
    for (auto& v : img) v = dist(rng) ? 1 : 0;
    return img;
}

std::vector<std::uint8_t> make_binary_3d(std::size_t d0, std::size_t d1, std::size_t d2,
                                          double density = 0.5, unsigned seed = 42) {
    std::mt19937 rng(seed);
    std::bernoulli_distribution dist(density);
    std::vector<std::uint8_t> vol(d0 * d1 * d2);
    for (auto& v : vol) v = dist(rng) ? 1 : 0;
    return vol;
}

} // namespace

BENCHMARK("CC 2D 128x128 4-conn") {
    static auto img = make_binary_2d(128, 128);
    std::mdspan<const std::uint8_t, std::dextents<std::size_t, 2>> view(img.data(), 128, 128);
    auto [labels, n] = utils2::connected_components_2d(view, utils2::GridConnectivity::four);
}

BENCHMARK("CC 2D 256x256 8-conn") {
    static auto img = make_binary_2d(256, 256);
    std::mdspan<const std::uint8_t, std::dextents<std::size_t, 2>> view(img.data(), 256, 256);
    auto [labels, n] = utils2::connected_components_2d(view, utils2::GridConnectivity::eight);
}

BENCHMARK("CC 2D 512x512 8-conn") {
    static auto img = make_binary_2d(512, 512);
    std::mdspan<const std::uint8_t, std::dextents<std::size_t, 2>> view(img.data(), 512, 512);
    auto [labels, n] = utils2::connected_components_2d(view, utils2::GridConnectivity::eight);
}

BENCHMARK("CC 3D 32x32x32 6-conn") {
    static auto vol = make_binary_3d(32, 32, 32);
    std::mdspan<const std::uint8_t, std::dextents<std::size_t, 3>> view(vol.data(), 32, 32, 32);
    auto [labels, n] = utils2::connected_components_3d(view, utils2::GridConnectivity::six);
}

BENCHMARK("CC 3D 64x64x64 26-conn") {
    static auto vol = make_binary_3d(64, 64, 64);
    std::mdspan<const std::uint8_t, std::dextents<std::size_t, 3>> view(vol.data(), 64, 64, 64);
    auto [labels, n] = utils2::connected_components_3d(view, utils2::GridConnectivity::twenty_six);
}
