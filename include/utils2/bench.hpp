// SPDX-License-Identifier: MIT
// Copyright (c) 2026 SuperOpt
// utils2/bench.hpp -- Minimal self-contained benchmark framework

#pragma once

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <format>
#include <functional>
#include <print>
#include <string>
#include <string_view>
#include <vector>

namespace utils2::bench {

// ---------------------------------------------------------------------------
// Colors
// ---------------------------------------------------------------------------

namespace color {
    inline constexpr const char* green  = "\033[32m";
    inline constexpr const char* cyan   = "\033[36m";
    inline constexpr const char* yellow = "\033[33m";
    inline constexpr const char* reset  = "\033[0m";
    inline constexpr const char* bold   = "\033[1m";
} // namespace color

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

using Clock = std::chrono::high_resolution_clock;

struct BenchCase {
    std::string_view name;
    std::function<void()> func;
};

struct Context {
    bool use_color = true;
    std::string filter;
    std::size_t min_time_ms = 500;   // target wall time per benchmark
    std::size_t max_iters   = 1'000'000;
};

inline std::vector<BenchCase>& registry() {
    static std::vector<BenchCase> r;
    return r;
}

inline Context& ctx() {
    static Context c;
    return c;
}

inline const char* col(const char* code) {
    return ctx().use_color ? code : "";
}

// ---------------------------------------------------------------------------
// Auto-registration
// ---------------------------------------------------------------------------

struct AutoRegister {
    AutoRegister(std::string_view name, std::function<void()> func) {
        registry().push_back({name, std::move(func)});
    }
};

// ---------------------------------------------------------------------------
// Timing helpers
// ---------------------------------------------------------------------------

struct Stats {
    double min_ns{};
    double max_ns{};
    double avg_ns{};
    double median_ns{};
    std::size_t iterations{};
};

inline Stats compute_stats(std::vector<double>& samples) {
    if (samples.empty()) return {};
    std::ranges::sort(samples);
    double sum = 0.0;
    for (auto s : samples) sum += s;
    return {
        .min_ns    = samples.front(),
        .max_ns    = samples.back(),
        .avg_ns    = sum / static_cast<double>(samples.size()),
        .median_ns = samples[samples.size() / 2],
        .iterations = samples.size(),
    };
}

// Format a time value with appropriate unit
inline std::string format_time(double ns) {
    if (ns < 1'000.0)
        return std::format("{:.1f} ns", ns);
    if (ns < 1'000'000.0)
        return std::format("{:.1f} us", ns / 1'000.0);
    if (ns < 1'000'000'000.0)
        return std::format("{:.1f} ms", ns / 1'000'000.0);
    return std::format("{:.2f} s", ns / 1'000'000'000.0);
}

inline void print_stats(std::string_view name, const Stats& s) {
    std::println("  {} {:<40s} {} | min {} | avg {} | median {} | max {} | {} iters",
        col(color::cyan), name, col(color::reset),
        format_time(s.min_ns),
        format_time(s.avg_ns),
        format_time(s.median_ns),
        format_time(s.max_ns),
        s.iterations);
}

// ---------------------------------------------------------------------------
// Core: run a function, auto-calibrate iterations, collect samples
// ---------------------------------------------------------------------------

// The user's benchmark body is called with a reference to an iteration count.
// For simple use: the BENCHMARK macro wraps the body in a lambda that runs
// the body `n` times internally.

/// Run `func` repeatedly, collecting per-invocation timings.
/// Auto-calibrates: first finds how many iterations fit in ~10ms,
/// then collects samples until min_time_ms is exceeded.
inline Stats run_bench(const std::function<void()>& func) {
    auto& c = ctx();

    // Warmup: 1 call
    func();

    // Calibrate: find N so that N calls take ~10ms
    std::size_t n = 1;
    for (;;) {
        auto t0 = Clock::now();
        for (std::size_t i = 0; i < n; ++i) func();
        auto t1 = Clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (elapsed_ms >= 10.0 || n >= c.max_iters) break;
        n = (elapsed_ms < 0.1) ? n * 100 : n * 2;
        if (n > c.max_iters) n = c.max_iters;
    }

    // Collect samples: run batches of `n`, record per-iteration time
    std::vector<double> samples;
    double total_ms = 0.0;
    while (total_ms < static_cast<double>(c.min_time_ms)) {
        auto t0 = Clock::now();
        for (std::size_t i = 0; i < n; ++i) func();
        auto t1 = Clock::now();
        double batch_ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
        total_ms += batch_ns / 1'000'000.0;
        double per_iter_ns = batch_ns / static_cast<double>(n);
        samples.push_back(per_iter_ns);
    }

    return compute_stats(samples);
}

// ---------------------------------------------------------------------------
// Runner
// ---------------------------------------------------------------------------

inline int run_all(int argc = 0, const char** argv = nullptr) {
    auto& c = ctx();

    for (int i = 1; i < argc; ++i) {
        std::string_view arg(argv[i]);
        if (arg.starts_with("--filter=")) {
            c.filter = std::string(arg.substr(9));
        } else if (arg == "--no-color") {
            c.use_color = false;
        } else if (arg.starts_with("--time=")) {
            c.min_time_ms = std::stoull(std::string(arg.substr(7)));
        } else if (arg == "--list") {
            for (auto& bc : registry())
                std::println("{}", bc.name);
            return 0;
        }
    }

    std::vector<BenchCase*> to_run;
    for (auto& bc : registry()) {
        if (c.filter.empty() ||
            std::string_view(bc.name).find(c.filter) != std::string_view::npos) {
            to_run.push_back(&bc);
        }
    }

    std::println("{}[==========]{} Running {} benchmark{}",
        col(color::bold), col(color::reset),
        to_run.size(), to_run.size() == 1 ? "" : "s");

    auto wall_start = Clock::now();

    for (auto* bc : to_run) {
        auto stats = run_bench(bc->func);
        print_stats(bc->name, stats);
    }

    auto wall_end = Clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(wall_end - wall_start).count();

    std::println("{}[==========]{} Done ({:.1f}s total)",
        col(color::bold), col(color::reset), total_ms / 1000.0);

    return 0;
}

} // namespace utils2::bench

// ===========================================================================
// Macros
// ===========================================================================

#define UTILS2_BENCH_CAT2(a, b) a##b
#define UTILS2_BENCH_CAT(a, b)  UTILS2_BENCH_CAT2(a, b)

#define BENCHMARK(bname)                                                       \
    static void UTILS2_BENCH_CAT(utils2_bench_func_, __LINE__)();              \
    static ::utils2::bench::AutoRegister                                       \
        UTILS2_BENCH_CAT(utils2_bench_reg_, __LINE__)(                         \
            bname, UTILS2_BENCH_CAT(utils2_bench_func_, __LINE__));            \
    static void UTILS2_BENCH_CAT(utils2_bench_func_, __LINE__)()

#define BENCH_MAIN()                                                           \
    int main(int argc, const char** argv) {                                    \
        return ::utils2::bench::run_all(argc, argv);                           \
    }
