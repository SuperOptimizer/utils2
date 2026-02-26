#include <utils2/test.hpp>
#include <utils2/statistics.hpp>
#include <vector>
#include <cmath>

using namespace utils2;

// ---- describe() basic stats ------------------------------------------------

TEST_CASE("describe basic") {
    std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0};
    auto s = describe(data);

    CHECK_EQ(s.count, 5u);
    CHECK_NEAR(s.mean, 3.0, 1e-12);
    CHECK_NEAR(s.median, 3.0, 1e-12);
    CHECK_NEAR(s.min, 1.0, 1e-12);
    CHECK_NEAR(s.max, 5.0, 1e-12);
    CHECK_NEAR(s.range, 4.0, 1e-12);
    // variance of {1,2,3,4,5} = 2.5 (sample variance)
    CHECK_NEAR(s.variance, 2.5, 1e-12);
    CHECK_NEAR(s.std_dev, std::sqrt(2.5), 1e-10);
}

TEST_CASE("describe single element") {
    std::vector<double> data = {42.0};
    auto s = describe(data);
    CHECK_EQ(s.count, 1u);
    CHECK_NEAR(s.mean, 42.0, 1e-12);
    CHECK_NEAR(s.median, 42.0, 1e-12);
    CHECK_NEAR(s.variance, 0.0, 1e-12);
}

TEST_CASE("describe empty") {
    std::vector<double> empty;
    auto s = describe(empty);
    CHECK_EQ(s.count, 0u);
}

// ---- median ----------------------------------------------------------------

TEST_CASE("median odd count") {
    std::vector<double> data = {5.0, 1.0, 3.0};
    CHECK_NEAR(median(data), 3.0, 1e-12);
}

TEST_CASE("median even count") {
    std::vector<double> data = {1.0, 2.0, 3.0, 4.0};
    CHECK_NEAR(median(data), 2.5, 1e-12);
}

TEST_CASE("median single") {
    std::vector<double> data = {7.0};
    CHECK_NEAR(median(data), 7.0, 1e-12);
}

// ---- mad (Median Absolute Deviation) ---------------------------------------

TEST_CASE("mad basic") {
    std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0};
    double m = mad(data);
    // median = 3, deviations = {2,1,0,1,2}, median(dev) = 1
    // MAD = 1 * 1.4826 = 1.4826
    CHECK_NEAR(m, 1.4826, 1e-4);
}

TEST_CASE("mad constant") {
    std::vector<double> data = {5.0, 5.0, 5.0};
    CHECK_NEAR(mad(data), 0.0, 1e-12);
}

// ---- Outlier detection (MAD) -----------------------------------------------

TEST_CASE("detect_outliers_mad") {
    std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0, 100.0};
    auto outliers = detect_outliers_mad(data, 3.5);
    // 100.0 should be detected as outlier
    CHECK_GE(outliers.size(), 1u);
    // Check that index 5 is in the result
    bool found = false;
    for (auto idx : outliers) {
        if (idx == 5) found = true;
    }
    CHECK(found);
}

TEST_CASE("detect_outliers_mad no outliers") {
    std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0};
    auto outliers = detect_outliers_mad(data, 3.5);
    CHECK(outliers.empty());
}

// ---- Outlier detection (z-score) -------------------------------------------

TEST_CASE("detect_outliers_zscore") {
    std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0, 100.0};
    auto outliers = detect_outliers_zscore(data, 2.0);
    CHECK_GE(outliers.size(), 1u);
    bool found = false;
    for (auto idx : outliers) {
        if (idx == 5) found = true;
    }
    CHECK(found);
}

TEST_CASE("detect_outliers_zscore too few elements") {
    std::vector<double> data = {42.0};
    auto outliers = detect_outliers_zscore(data);
    CHECK(outliers.empty());
}

// ---- remove_outliers_mad ---------------------------------------------------

TEST_CASE("remove_outliers_mad") {
    std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0, 100.0};
    auto clean = remove_outliers_mad(data, 3.5);
    CHECK_LT(clean.size(), data.size());
    // 100.0 should be removed
    for (auto v : clean) {
        CHECK_LT(v, 50.0);
    }
}

// ---- Histogram -------------------------------------------------------------

TEST_CASE("histogram basic") {
    std::vector<double> data = {0.0, 0.5, 1.0, 1.5, 2.0};
    auto h = histogram(data, 4, 0.0, 2.0);

    CHECK_EQ(h.num_bins, 4u);
    CHECK_NEAR(h.bin_width, 0.5, 1e-12);
    // Bin 0: [0, 0.5) -> 0.0
    CHECK_EQ(h.counts[0], 1u);
    // Bin 1: [0.5, 1.0) -> 0.5
    CHECK_EQ(h.counts[1], 1u);
    // Bin 2: [1.0, 1.5) -> 1.0
    CHECK_EQ(h.counts[2], 1u);
    // Bin 3: [1.5, 2.0] -> 1.5, 2.0
    CHECK_EQ(h.counts[3], 2u);
}

TEST_CASE("histogram bin_for") {
    Histogram<double> h;
    h.num_bins = 10;
    h.bin_min = 0.0;
    h.bin_max = 10.0;
    h.bin_width = 1.0;
    h.counts.resize(10, 0);

    CHECK_EQ(h.bin_for(0.0), 0u);
    CHECK_EQ(h.bin_for(5.5), 5u);
    CHECK_EQ(h.bin_for(9.9), 9u);
    CHECK_EQ(h.bin_for(100.0), 9u);  // clamp to last bin
    CHECK_EQ(h.bin_for(-5.0), 0u);   // clamp to first bin
}

// ---- RunningStats (Welford) ------------------------------------------------

TEST_CASE("RunningStats basic") {
    RunningStats<double> rs;
    rs.push(1.0);
    rs.push(2.0);
    rs.push(3.0);
    rs.push(4.0);
    rs.push(5.0);

    CHECK_EQ(rs.count(), 5u);
    CHECK_NEAR(rs.mean(), 3.0, 1e-12);
    CHECK_NEAR(rs.min(), 1.0, 1e-12);
    CHECK_NEAR(rs.max(), 5.0, 1e-12);
    CHECK_NEAR(rs.variance(), 2.5, 1e-12);
    CHECK_NEAR(rs.std_dev(), std::sqrt(2.5), 1e-10);
}

TEST_CASE("RunningStats single value") {
    RunningStats<double> rs;
    rs.push(42.0);

    CHECK_EQ(rs.count(), 1u);
    CHECK_NEAR(rs.mean(), 42.0, 1e-12);
    CHECK_NEAR(rs.variance(), 0.0, 1e-12);
}

// ---- RunningStats merge ----------------------------------------------------

TEST_CASE("RunningStats merge") {
    RunningStats<double> a, b;
    a.push(1.0);
    a.push(2.0);
    a.push(3.0);

    b.push(4.0);
    b.push(5.0);

    a += b;

    CHECK_EQ(a.count(), 5u);
    CHECK_NEAR(a.mean(), 3.0, 1e-12);
    CHECK_NEAR(a.min(), 1.0, 1e-12);
    CHECK_NEAR(a.max(), 5.0, 1e-12);
    CHECK_NEAR(a.variance(), 2.5, 1e-10);
}

TEST_CASE("RunningStats merge empty") {
    RunningStats<double> a, empty;
    a.push(1.0);
    a.push(2.0);

    a += empty;
    CHECK_EQ(a.count(), 2u);
    CHECK_NEAR(a.mean(), 1.5, 1e-12);
}

TEST_CASE("RunningStats merge into empty") {
    RunningStats<double> empty, b;
    b.push(10.0);
    b.push(20.0);

    empty += b;
    CHECK_EQ(empty.count(), 2u);
    CHECK_NEAR(empty.mean(), 15.0, 1e-12);
}

// ---- Percentile ------------------------------------------------------------

TEST_CASE("percentile") {
    std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0};
    CHECK_NEAR(percentile(data, 0.0), 1.0, 1e-12);
    CHECK_NEAR(percentile(data, 100.0), 5.0, 1e-12);
    CHECK_NEAR(percentile(data, 50.0), 3.0, 1e-12);
}

TEST_CASE("percentile interpolation") {
    std::vector<double> data = {10.0, 20.0, 30.0, 40.0};
    // 25th percentile: index = 0.25 * 3 = 0.75 -> lerp(10, 20, 0.75) = 17.5
    CHECK_NEAR(percentile(data, 25.0), 17.5, 1e-10);
}

TEST_CASE("percentiles multiple") {
    std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> ps = {0.0, 25.0, 50.0, 75.0, 100.0};
    auto result = percentiles(data, ps);
    REQUIRE_EQ(result.size(), 5u);
    CHECK_NEAR(result[0], 1.0, 1e-12);
    CHECK_NEAR(result[2], 3.0, 1e-12);
    CHECK_NEAR(result[4], 5.0, 1e-12);
}

TEST_CASE("percentile single element") {
    std::vector<double> data = {42.0};
    CHECK_NEAR(percentile(data, 0.0), 42.0, 1e-12);
    CHECK_NEAR(percentile(data, 50.0), 42.0, 1e-12);
    CHECK_NEAR(percentile(data, 100.0), 42.0, 1e-12);
}

// ---- Bug hunt: numerical stability of variance/stddev ----------------------

TEST_CASE("describe large mean small variance") {
    // Values near 1e8 with small variance -- tests for catastrophic cancellation
    std::vector<double> data;
    double base = 1e8;
    for (int i = 0; i < 1000; ++i)
        data.push_back(base + static_cast<double>(i) * 0.001);

    auto s = describe(data);
    // True mean = 1e8 + 0.4995
    double true_mean = base + 999.0 * 0.001 / 2.0;
    CHECK_NEAR(s.mean, true_mean, 1e-4);

    // Sample variance of arithmetic sequence a, a+d, ..., a+(n-1)*d:
    // sum((x_i - mean)^2) / (n-1) = d^2 * n * (n + 1) / 12
    double d = 0.001;
    double n = 1000.0;
    double true_var = d * d * n * (n + 1.0) / 12.0;
    CHECK_NEAR(s.variance, true_var, true_var * 1e-4);
    CHECK_GT(s.std_dev, 0.0);
}

TEST_CASE("RunningStats large mean small variance") {
    // Same test but using Welford's algorithm (should be numerically stable)
    RunningStats<double> rs;
    double base = 1e8;
    for (int i = 0; i < 1000; ++i)
        rs.push(base + static_cast<double>(i) * 0.001);

    double true_mean = base + 999.0 * 0.001 / 2.0;
    CHECK_NEAR(rs.mean(), true_mean, 1e-4);

    double d = 0.001;
    double n = 1000.0;
    double true_var = d * d * n * (n + 1.0) / 12.0;
    // Welford's is stable but with 1e8 mean, expect some floating-point noise
    CHECK_NEAR(rs.variance(), true_var, true_var * 1e-3);
}

TEST_CASE("describe all identical values") {
    std::vector<double> data(100, 42.0);
    auto s = describe(data);

    CHECK_EQ(s.count, 100u);
    CHECK_NEAR(s.mean, 42.0, 1e-12);
    CHECK_NEAR(s.median, 42.0, 1e-12);
    CHECK_NEAR(s.variance, 0.0, 1e-12);
    CHECK_NEAR(s.std_dev, 0.0, 1e-12);
    CHECK_NEAR(s.min, 42.0, 1e-12);
    CHECK_NEAR(s.max, 42.0, 1e-12);
    CHECK_NEAR(s.range, 0.0, 1e-12);
}

TEST_CASE("RunningStats all identical values") {
    RunningStats<double> rs;
    for (int i = 0; i < 100; ++i)
        rs.push(42.0);

    CHECK_NEAR(rs.mean(), 42.0, 1e-12);
    CHECK_NEAR(rs.variance(), 0.0, 1e-12);
    CHECK_NEAR(rs.std_dev(), 0.0, 1e-12);
}

TEST_CASE("RunningStats vs describe agreement") {
    // Compare RunningStats (Welford) with describe (two-pass) on same data
    std::vector<double> data = {1.0, 3.0, 5.0, 7.0, 9.0, 2.0, 4.0, 6.0, 8.0, 10.0};

    RunningStats<double> rs;
    for (auto v : data) rs.push(v);

    auto s = describe(data);

    CHECK_NEAR(rs.mean(), s.mean, 1e-12);
    CHECK_NEAR(rs.variance(), s.variance, 1e-10);
    CHECK_NEAR(rs.std_dev(), s.std_dev, 1e-10);
    CHECK_NEAR(rs.min(), s.min, 1e-12);
    CHECK_NEAR(rs.max(), s.max, 1e-12);
}

TEST_CASE("describe two elements") {
    std::vector<double> data = {1.0, 3.0};
    auto s = describe(data);
    CHECK_NEAR(s.mean, 2.0, 1e-12);
    CHECK_NEAR(s.median, 2.0, 1e-12);
    // Sample variance of {1,3} = (1-2)^2 + (3-2)^2 = 2, divided by (2-1) = 2
    CHECK_NEAR(s.variance, 2.0, 1e-12);
    CHECK_NEAR(s.std_dev, std::sqrt(2.0), 1e-10);
}

UTILS2_TEST_MAIN()
