#include <utils2/bench.hpp>
#include <utils2/thread_pool.hpp>
#include <atomic>
#include <cstdint>

BENCH_MAIN()

// Shared pool for all benchmarks.
static utils2::ThreadPool& get_pool() {
    static utils2::ThreadPool pool(4);
    return pool;
}

BENCHMARK("ThreadPool submit+get 1 task") {
    auto& pool = get_pool();
    auto fut = pool.submit([] { return 42; });
    fut.get();
}

BENCHMARK("ThreadPool submit+get 100 tasks") {
    auto& pool = get_pool();
    std::vector<std::future<int>> futs;
    futs.reserve(100);
    for (int i = 0; i < 100; ++i)
        futs.push_back(pool.submit([i] { return i * i; }));
    for (auto& f : futs) f.get();
}

BENCHMARK("ThreadPool enqueue 1000 tasks") {
    auto& pool = get_pool();
    // Use futures instead of wait_idle to avoid potential race with rapid re-entry
    std::vector<std::future<int>> futs;
    futs.reserve(1000);
    for (int i = 0; i < 1000; ++i)
        futs.push_back(pool.submit([i] { return i; }));
    for (auto& f : futs) f.get();
}

BENCHMARK("parallel_for 10K iterations") {
    auto& pool = get_pool();
    std::vector<std::size_t> data(10'000, 0);
    utils2::parallel_for(pool, std::size_t(0), std::size_t(10'000),
        [&data](std::size_t i) {
            data[i] = i * i;
        });
}
