#pragma once
#include <span>
#include <vector>
#include <array>
#include <cstddef>
#include <cmath>
#include <concepts>
#include <algorithm>
#include <numeric>
#include <ranges>
#include <functional>
#include <optional>

namespace utils2 {

// ---------------------------------------------------------------------------
// DescriptiveStats
// ---------------------------------------------------------------------------
template<std::floating_point T = double>
struct DescriptiveStats {
    T mean{};
    T median{};
    T std_dev{};
    T variance{};
    T min{};
    T max{};
    T range{};
    std::size_t count{};
};

// ---------------------------------------------------------------------------
// Histogram
// ---------------------------------------------------------------------------
template<std::floating_point T = double>
struct Histogram {
    std::vector<std::size_t> counts;
    T bin_min{};
    T bin_max{};
    T bin_width{};
    std::size_t num_bins{};

    [[nodiscard]] std::size_t bin_for(T value) const noexcept {
        if (bin_width == T{0}) return 0;
        auto idx = static_cast<std::ptrdiff_t>((value - bin_min) / bin_width);
        if (idx < 0) return 0;
        if (static_cast<std::size_t>(idx) >= num_bins) return num_bins - 1;
        return static_cast<std::size_t>(idx);
    }
};

// ---------------------------------------------------------------------------
// RunningStats  (Welford's online algorithm)
// ---------------------------------------------------------------------------
template<std::floating_point T = double>
class RunningStats final {
public:
    constexpr void push(T value) noexcept {
        ++count_;
        T delta = value - mean_;
        mean_ += delta / static_cast<T>(count_);
        T delta2 = value - mean_;
        m2_ += delta * delta2;

        if (count_ == 1) { min_ = max_ = value; }
        else { min_ = std::min(min_, value); max_ = std::max(max_, value); }
    }

    [[nodiscard]] constexpr T mean()     const noexcept { return mean_; }
    [[nodiscard]] constexpr T variance() const noexcept { return count_ > 1 ? m2_ / static_cast<T>(count_ - 1) : T{0}; }
    [[nodiscard]] constexpr T std_dev()  const noexcept { return std::sqrt(variance()); }
    [[nodiscard]] constexpr T min()      const noexcept { return min_; }
    [[nodiscard]] constexpr T max()      const noexcept { return max_; }
    [[nodiscard]] constexpr std::size_t count() const noexcept { return count_; }

    constexpr RunningStats& operator+=(const RunningStats& other) noexcept {
        if (other.count_ == 0) return *this;
        if (count_ == 0) { *this = other; return *this; }

        std::size_t combined = count_ + other.count_;
        T delta = other.mean_ - mean_;
        T new_mean = mean_ + delta * static_cast<T>(other.count_) / static_cast<T>(combined);
        T new_m2 = m2_ + other.m2_ + delta * delta
                    * static_cast<T>(count_) * static_cast<T>(other.count_)
                    / static_cast<T>(combined);

        mean_  = new_mean;
        m2_    = new_m2;
        count_ = combined;
        min_   = std::min(min_, other.min_);
        max_   = std::max(max_, other.max_);
        return *this;
    }

private:
    T mean_{};
    T m2_{};
    T min_{};
    T max_{};
    std::size_t count_{};
};

// ---------------------------------------------------------------------------
// median_inplace  -- modifies input via nth_element
// ---------------------------------------------------------------------------
template<std::ranges::random_access_range R>
[[nodiscard]] auto median_inplace(R&& data) -> std::ranges::range_value_t<R> {
    using T = std::ranges::range_value_t<R>;
    auto n = std::ranges::size(data);
    if (n == 0) return T{};
    if (n == 1) return *std::ranges::begin(data);

    auto mid = std::ranges::begin(data) + static_cast<std::ptrdiff_t>(n / 2);
    std::ranges::nth_element(data, mid);

    if (n % 2 != 0) return *mid;

    // even: average of two middle elements
    auto left = std::ranges::max_element(std::ranges::begin(data), mid);
    return (*left + *mid) / T{2};
}

// ---------------------------------------------------------------------------
// median  -- copies first
// ---------------------------------------------------------------------------
template<std::ranges::input_range R>
[[nodiscard]] auto median(R&& data) -> std::ranges::range_value_t<R> {
    using T = std::ranges::range_value_t<R>;
    std::vector<T> buf(std::ranges::begin(data), std::ranges::end(data));
    return median_inplace(buf);
}

// ---------------------------------------------------------------------------
// mad  -- Median Absolute Deviation (* 1.4826)
// ---------------------------------------------------------------------------
template<std::ranges::input_range R>
[[nodiscard]] auto mad(R&& data) -> std::ranges::range_value_t<R> {
    using T = std::ranges::range_value_t<R>;
    constexpr T consistency = T{1.4826};

    std::vector<T> buf(std::ranges::begin(data), std::ranges::end(data));
    if (buf.empty()) return T{};

    T med = median_inplace(buf);

    std::vector<T> abs_devs(buf.size());
    std::ranges::transform(buf, abs_devs.begin(),
        [med](T v) { return std::abs(v - med); });

    return median_inplace(abs_devs) * consistency;
}

// ---------------------------------------------------------------------------
// detect_outliers_mad
// ---------------------------------------------------------------------------
template<std::ranges::random_access_range R>
[[nodiscard]] std::vector<std::size_t> detect_outliers_mad(
        R&& data, double threshold = 3.5) {
    using T = std::ranges::range_value_t<R>;

    std::vector<T> buf(std::ranges::begin(data), std::ranges::end(data));
    if (buf.empty()) return {};

    T med = median(buf);
    T mad_val = mad(buf);

    std::vector<std::size_t> indices;
    if (mad_val == T{0}) return indices;

    for (std::size_t i = 0; i < buf.size(); ++i) {
        T z = std::abs(buf[i] - med) / mad_val;
        if (static_cast<double>(z) > threshold)
            indices.push_back(i);
    }
    return indices;
}

// ---------------------------------------------------------------------------
// remove_outliers_mad
// ---------------------------------------------------------------------------
template<std::ranges::input_range R>
[[nodiscard]] auto remove_outliers_mad(R&& data, double threshold = 3.5)
        -> std::vector<std::ranges::range_value_t<R>> {
    using T = std::ranges::range_value_t<R>;

    std::vector<T> buf(std::ranges::begin(data), std::ranges::end(data));
    if (buf.empty()) return {};

    T med = median(buf);
    T mad_val = mad(buf);

    if (mad_val == T{0}) return buf;

    std::vector<T> result;
    result.reserve(buf.size());
    for (auto& v : buf) {
        T z = std::abs(v - med) / mad_val;
        if (static_cast<double>(z) <= threshold)
            result.push_back(v);
    }
    return result;
}

// ---------------------------------------------------------------------------
// detect_outliers_zscore
// ---------------------------------------------------------------------------
template<std::ranges::random_access_range R>
[[nodiscard]] std::vector<std::size_t> detect_outliers_zscore(
        R&& data, double threshold = 3.0) {
    using T = std::ranges::range_value_t<R>;

    std::vector<T> buf(std::ranges::begin(data), std::ranges::end(data));
    if (buf.size() < 2) return {};

    T sum = std::accumulate(buf.begin(), buf.end(), T{0});
    T m   = sum / static_cast<T>(buf.size());

    T sq_sum = std::accumulate(buf.begin(), buf.end(), T{0},
        [m](T acc, T v) { return acc + (v - m) * (v - m); });
    T sd = std::sqrt(sq_sum / static_cast<T>(buf.size() - 1));

    std::vector<std::size_t> indices;
    if (sd == T{0}) return indices;

    for (std::size_t i = 0; i < buf.size(); ++i) {
        T z = std::abs(buf[i] - m) / sd;
        if (static_cast<double>(z) > threshold)
            indices.push_back(i);
    }
    return indices;
}

// ---------------------------------------------------------------------------
// percentile  (linear interpolation, copies input)
// ---------------------------------------------------------------------------
template<std::ranges::random_access_range R>
[[nodiscard]] auto percentile(R&& data, double p) -> std::ranges::range_value_t<R> {
    using T = std::ranges::range_value_t<R>;

    std::vector<T> buf(std::ranges::begin(data), std::ranges::end(data));
    if (buf.empty()) return T{};

    std::ranges::sort(buf);

    double idx = p / 100.0 * static_cast<double>(buf.size() - 1);
    auto lo = static_cast<std::size_t>(idx);
    double frac = idx - static_cast<double>(lo);

    if (lo + 1 >= buf.size()) return buf.back();
    return static_cast<T>(buf[lo] + static_cast<T>(frac) * (buf[lo + 1] - buf[lo]));
}

// ---------------------------------------------------------------------------
// percentiles  (multiple at once, single sort)
// ---------------------------------------------------------------------------
template<std::ranges::random_access_range R>
[[nodiscard]] auto percentiles(R&& data, std::span<const double> ps)
        -> std::vector<std::ranges::range_value_t<R>> {
    using T = std::ranges::range_value_t<R>;

    std::vector<T> buf(std::ranges::begin(data), std::ranges::end(data));
    std::vector<T> result;
    if (buf.empty() || ps.empty()) return result;

    std::ranges::sort(buf);
    result.reserve(ps.size());

    for (double p : ps) {
        double idx = p / 100.0 * static_cast<double>(buf.size() - 1);
        auto lo = static_cast<std::size_t>(idx);
        double frac = idx - static_cast<double>(lo);

        if (lo + 1 >= buf.size()) { result.push_back(buf.back()); continue; }
        result.push_back(static_cast<T>(buf[lo] + static_cast<T>(frac) * (buf[lo + 1] - buf[lo])));
    }
    return result;
}

// ---------------------------------------------------------------------------
// histogram  (auto range)
// ---------------------------------------------------------------------------
template<std::ranges::input_range R>
[[nodiscard]] Histogram<std::ranges::range_value_t<R>>
histogram(R&& data, std::size_t num_bins = 256) {
    using T = std::ranges::range_value_t<R>;

    std::vector<T> buf(std::ranges::begin(data), std::ranges::end(data));
    if (buf.empty()) return {};

    auto [lo, hi] = std::ranges::minmax_element(buf);
    return histogram(std::span<const T>{buf}, num_bins, *lo, *hi);
}

// ---------------------------------------------------------------------------
// histogram  (explicit range)
// ---------------------------------------------------------------------------
template<std::ranges::input_range R, std::floating_point T>
[[nodiscard]] Histogram<T>
histogram(R&& data, std::size_t num_bins, T min_val, T max_val) {
    Histogram<T> h;
    h.num_bins  = num_bins;
    h.bin_min   = min_val;
    h.bin_max   = max_val;
    h.bin_width = (num_bins > 0 && max_val > min_val)
                  ? (max_val - min_val) / static_cast<T>(num_bins)
                  : T{1};
    h.counts.resize(num_bins, 0);

    for (auto&& v : data) {
        auto idx = h.bin_for(static_cast<T>(v));
        ++h.counts[idx];
    }
    return h;
}

// ---------------------------------------------------------------------------
// describe
// ---------------------------------------------------------------------------
template<std::ranges::input_range R>
[[nodiscard]] DescriptiveStats<std::ranges::range_value_t<R>>
describe(R&& data) {
    using T = std::ranges::range_value_t<R>;

    std::vector<T> buf(std::ranges::begin(data), std::ranges::end(data));
    DescriptiveStats<T> s{};
    s.count = buf.size();
    if (buf.empty()) return s;

    auto [lo, hi] = std::ranges::minmax_element(buf);
    s.min   = *lo;
    s.max   = *hi;
    s.range = s.max - s.min;

    T sum = std::accumulate(buf.begin(), buf.end(), T{0});
    s.mean = sum / static_cast<T>(s.count);

    s.median = median_inplace(buf);

    if (s.count > 1) {
        T sq = std::accumulate(buf.begin(), buf.end(), T{0},
            [m = s.mean](T acc, T v) { return acc + (v - m) * (v - m); });
        s.variance = sq / static_cast<T>(s.count - 1);
        s.std_dev  = std::sqrt(s.variance);
    }
    return s;
}

} // namespace utils2
