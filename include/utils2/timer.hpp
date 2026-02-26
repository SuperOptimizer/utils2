// SPDX-License-Identifier: MIT
// Copyright (c) 2026 SuperOpt
// utils2/timer.hpp — Scoped timer, stopwatch, and accumulating profiler

#pragma once

#include <chrono>
#include <cstddef>
#include <format>
#include <limits>
#include <print>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace utils2 {

namespace detail {

using clock = std::chrono::steady_clock;
using time_point = clock::time_point;

[[nodiscard]] inline double to_ms(time_point from, time_point to) noexcept {
    return std::chrono::duration<double, std::milli>(to - from).count();
}

} // namespace detail

// Simple RAII timer that prints elapsed time on destruction
class ScopedTimer final {
public:
    explicit ScopedTimer(std::string_view label) noexcept
        : label_(label), start_(detail::clock::now()) {}

    ~ScopedTimer() {
        std::println(stderr, "[timer] {}: {}ms", label_, elapsed_ms());
    }

    ScopedTimer(const ScopedTimer&) = delete;
    ScopedTimer& operator=(const ScopedTimer&) = delete;

    [[nodiscard]] double elapsed_ms() const noexcept {
        return detail::to_ms(start_, detail::clock::now());
    }

    [[nodiscard]] double elapsed_us() const noexcept {
        return elapsed_ms() * 1000.0;
    }

    [[nodiscard]] double elapsed_s() const noexcept {
        return elapsed_ms() / 1000.0;
    }

    void mark(std::string_view name) {
        std::println(stderr, "[timer] {}/{}: {}ms", label_, name, elapsed_ms());
    }

private:
    std::string label_;
    detail::time_point start_;
};

// Stopwatch: non-RAII, manual start/stop/lap
class Stopwatch final {
public:
    struct Lap {
        std::string name;
        double elapsed_ms;
    };

    Stopwatch() noexcept { start(); }

    void start() noexcept {
        running_ = true;
        start_ = detail::clock::now();
        accumulated_ms_ = 0.0;
    }

    void stop() noexcept {
        if (running_) {
            accumulated_ms_ += detail::to_ms(start_, detail::clock::now());
            running_ = false;
        }
    }

    void reset() noexcept {
        running_ = false;
        accumulated_ms_ = 0.0;
        laps_.clear();
    }

    void lap(std::string_view name = "") {
        laps_.push_back(Lap{std::string(name), elapsed_ms()});
    }

    [[nodiscard]] double elapsed_ms() const noexcept {
        if (running_)
            return accumulated_ms_ + detail::to_ms(start_, detail::clock::now());
        return accumulated_ms_;
    }

    [[nodiscard]] double elapsed_us() const noexcept {
        return elapsed_ms() * 1000.0;
    }

    [[nodiscard]] double elapsed_s() const noexcept {
        return elapsed_ms() / 1000.0;
    }

    [[nodiscard]] bool running() const noexcept { return running_; }

    [[nodiscard]] std::span<const Lap> laps() const noexcept { return laps_; }

    [[nodiscard]] std::string report() const {
        std::string out = std::format("Stopwatch: {}ms total\n", elapsed_ms());
        for (std::size_t i = 0; i < laps_.size(); ++i) {
            const auto& l = laps_[i];
            double delta = (i == 0) ? l.elapsed_ms : l.elapsed_ms - laps_[i - 1].elapsed_ms;
            if (l.name.empty())
                out += std::format("  lap {}: {}ms (+{}ms)\n", i + 1, l.elapsed_ms, delta);
            else
                out += std::format("  {} : {}ms (+{}ms)\n", l.name, l.elapsed_ms, delta);
        }
        return out;
    }

private:
    detail::time_point start_{};
    double accumulated_ms_{0.0};
    bool running_{false};
    std::vector<Lap> laps_;
};

// Accumulating timer for repeated operations
class TimerAccumulator final {
public:
    void start() noexcept {
        start_ = detail::clock::now();
    }

    void stop() noexcept {
        const double ms = detail::to_ms(start_, detail::clock::now());
        total_ms_ += ms;
        ++count_;
        if (ms < min_ms_) min_ms_ = ms;
        if (ms > max_ms_) max_ms_ = ms;
    }

    [[nodiscard]] double total_ms() const noexcept { return total_ms_; }

    [[nodiscard]] double avg_ms() const noexcept {
        return count_ ? total_ms_ / static_cast<double>(count_) : 0.0;
    }

    [[nodiscard]] std::size_t count() const noexcept { return count_; }

    [[nodiscard]] double min_ms() const noexcept {
        return count_ ? min_ms_ : 0.0;
    }

    [[nodiscard]] double max_ms() const noexcept { return max_ms_; }

    [[nodiscard]] std::string report(std::string_view label = "") const {
        if (label.empty())
            return std::format(
                "TimerAccumulator: {}x, total={}ms, avg={}ms, min={}ms, max={}ms",
                count_, total_ms_, avg_ms(), min_ms(), max_ms());
        return std::format(
            "[{}] {}x, total={}ms, avg={}ms, min={}ms, max={}ms",
            label, count_, total_ms_, avg_ms(), min_ms(), max_ms());
    }

private:
    detail::time_point start_{};
    double total_ms_{0.0};
    std::size_t count_{0};
    double min_ms_{std::numeric_limits<double>::max()};
    double max_ms_{0.0};
};

} // namespace utils2
