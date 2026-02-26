#include <utils2/bench.hpp>
#include <utils2/spatial_index.hpp>
#include <vector>
#include <array>
#include <random>

BENCH_MAIN()

using Point3 = std::array<double, 3>;

namespace {

std::vector<Point3> make_points(std::size_t n, unsigned seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(0.0, 1000.0);
    std::vector<Point3> pts(n);
    for (auto& p : pts) {
        p = {dist(rng), dist(rng), dist(rng)};
    }
    return pts;
}

} // namespace

BENCHMARK("KDTree build 1K points") {
    static auto pts = make_points(1'000);
    utils2::KDTree<Point3, 3> tree;
    tree.build(pts);
}

BENCHMARK("KDTree build 10K points") {
    static auto pts = make_points(10'000);
    utils2::KDTree<Point3, 3> tree;
    tree.build(pts);
}

BENCHMARK("KDTree build 100K points") {
    static auto pts = make_points(100'000);
    utils2::KDTree<Point3, 3> tree;
    tree.build(pts);
}

BENCHMARK("KDTree knn(5) 10K points") {
    static auto pts = make_points(10'000);
    static utils2::KDTree<Point3, 3> tree = [&] {
        utils2::KDTree<Point3, 3> t;
        t.build(pts);
        return t;
    }();
    static std::mt19937 rng(123);
    static std::uniform_real_distribution<double> dist(0.0, 1000.0);
    Point3 query = {dist(rng), dist(rng), dist(rng)};
    auto results = tree.knn(query, 5);
}

BENCHMARK("KDTree radius(50) 10K points") {
    static auto pts = make_points(10'000);
    static utils2::KDTree<Point3, 3> tree = [&] {
        utils2::KDTree<Point3, 3> t;
        t.build(pts);
        return t;
    }();
    static std::mt19937 rng(456);
    static std::uniform_real_distribution<double> dist(0.0, 1000.0);
    Point3 query = {dist(rng), dist(rng), dist(rng)};
    auto results = tree.radius(query, 50.0);
}
