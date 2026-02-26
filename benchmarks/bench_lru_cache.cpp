#include <utils2/bench.hpp>
#include <utils2/lru_cache.hpp>
#include <string>
#include <random>

BENCH_MAIN()

BENCHMARK("LRUCache put 1K entries") {
    utils2::LRUCache<int, int> cache({.max_bytes = 1ULL << 20});
    for (int i = 0; i < 1000; ++i)
        cache.put(i, i * 42);
}

BENCHMARK("LRUCache get (hit) 1K lookups") {
    static utils2::LRUCache<int, int> cache({.max_bytes = 1ULL << 20});
    static bool filled = [] {
        for (int i = 0; i < 1000; ++i) cache.put(i, i * 42);
        return true;
    }();
    (void)filled;
    static std::mt19937 rng(42);
    static std::uniform_int_distribution<int> dist(0, 999);
    for (int i = 0; i < 1000; ++i) {
        auto v = cache.get(dist(rng));
    }
}

BENCHMARK("LRUCache get (miss) 1K lookups") {
    static utils2::LRUCache<int, int> cache({.max_bytes = 1ULL << 20});
    static bool filled = [] {
        for (int i = 0; i < 1000; ++i) cache.put(i, i);
        return true;
    }();
    (void)filled;
    for (int i = 1000; i < 2000; ++i) {
        auto v = cache.get(i);
    }
}

BENCHMARK("LRUCache put with eviction") {
    // Small budget forces constant eviction
    utils2::LRUCache<int, int> cache({.max_bytes = 100 * sizeof(int)});
    for (int i = 0; i < 1000; ++i)
        cache.put(i, i);
}

BENCHMARK("LRUCache mixed put/get 10K ops") {
    utils2::LRUCache<int, int> cache({.max_bytes = 1ULL << 20});
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> key_dist(0, 999);
    for (int i = 0; i < 10'000; ++i) {
        int k = key_dist(rng);
        if (i % 3 == 0)
            cache.put(k, k * 7);
        else
            (void)cache.get(k);
    }
}
