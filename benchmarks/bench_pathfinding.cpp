#include <utils2/bench.hpp>
#include <utils2/pathfinding.hpp>
#include <vector>
#include <random>

BENCH_MAIN()

namespace {

std::vector<float> make_cost_field_2d(std::size_t rows, std::size_t cols, unsigned seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(1.0f, 10.0f);
    std::vector<float> field(rows * cols);
    for (auto& v : field) v = dist(rng);
    return field;
}

} // namespace

BENCHMARK("Dijkstra 2D 64x64 corner-to-corner") {
    static auto field = make_cost_field_2d(64, 64);
    std::mdspan<const float, std::dextents<std::size_t, 2>> view(field.data(), 64, 64);
    auto result = utils2::dijkstra_2d(view, {0, 0}, {63, 63});
}

BENCHMARK("Dijkstra 2D 128x128 corner-to-corner") {
    static auto field = make_cost_field_2d(128, 128);
    std::mdspan<const float, std::dextents<std::size_t, 2>> view(field.data(), 128, 128);
    auto result = utils2::dijkstra_2d(view, {0, 0}, {127, 127});
}

BENCHMARK("Dijkstra 2D 256x256 corner-to-corner") {
    static auto field = make_cost_field_2d(256, 256);
    std::mdspan<const float, std::dextents<std::size_t, 2>> view(field.data(), 256, 256);
    auto result = utils2::dijkstra_2d(view, {0, 0}, {255, 255});
}

BENCHMARK("Distance field 2D 128x128") {
    static auto field = make_cost_field_2d(128, 128);
    std::mdspan<const float, std::dextents<std::size_t, 2>> view(field.data(), 128, 128);
    auto result = utils2::distance_field_2d(view, {64, 64});
}

BENCHMARK("Distance field 2D 256x256") {
    static auto field = make_cost_field_2d(256, 256);
    std::mdspan<const float, std::dextents<std::size_t, 2>> view(field.data(), 256, 256);
    auto result = utils2::distance_field_2d(view, {128, 128});
}
