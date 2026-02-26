#include <utils2/test.hpp>
#include <utils2/timer.hpp>
#include <thread>

using namespace utils2;

// ============================================================================
// ScopedTimer tests
// ============================================================================

TEST_CASE("ScopedTimer: elapsed tracking") {
    ScopedTimer t("test_scoped");
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    REQUIRE_GT(t.elapsed_ms(), 0.0);
    REQUIRE_GT(t.elapsed_us(), 0.0);
    REQUIRE_GT(t.elapsed_s(), 0.0);
}

TEST_CASE("ScopedTimer: elapsed_us is 1000x elapsed_ms") {
    ScopedTimer t("test_units");
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    double ms = t.elapsed_ms();
    double us = t.elapsed_us();
    // us should be approximately 1000x ms (with some tolerance for timing)
    REQUIRE_NEAR(us / ms, 1000.0, 1.0);
}

TEST_CASE("ScopedTimer: elapsed_s is elapsed_ms / 1000") {
    ScopedTimer t("test_s");
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    double ms = t.elapsed_ms();
    double s = t.elapsed_s();
    REQUIRE_NEAR(s * 1000.0, ms, 0.01);
}

// ============================================================================
// Stopwatch tests
// ============================================================================

TEST_CASE("Stopwatch: starts running on construction") {
    Stopwatch sw;
    REQUIRE(sw.running());
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    REQUIRE_GT(sw.elapsed_ms(), 0.0);
}

TEST_CASE("Stopwatch: stop freezes elapsed time") {
    Stopwatch sw;
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    sw.stop();
    REQUIRE(!sw.running());
    double t1 = sw.elapsed_ms();
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    double t2 = sw.elapsed_ms();
    REQUIRE_NEAR(t1, t2, 0.01);
}

TEST_CASE("Stopwatch: reset clears state") {
    Stopwatch sw;
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    sw.lap("first");
    sw.reset();
    REQUIRE(!sw.running());
    REQUIRE_NEAR(sw.elapsed_ms(), 0.0, 0.01);
    REQUIRE_EQ(sw.laps().size(), 0u);
}

TEST_CASE("Stopwatch: lap records timestamps") {
    Stopwatch sw;
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    sw.lap("one");
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    sw.lap("two");

    auto laps = sw.laps();
    REQUIRE_EQ(laps.size(), 2u);
    REQUIRE_EQ(laps[0].name, std::string("one"));
    REQUIRE_EQ(laps[1].name, std::string("two"));
    REQUIRE_GT(laps[0].elapsed_ms, 0.0);
    REQUIRE_GT(laps[1].elapsed_ms, laps[0].elapsed_ms);
}

TEST_CASE("Stopwatch: accumulation across start/stop cycles") {
    Stopwatch sw;
    sw.stop();
    sw.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    sw.stop();
    double first = sw.elapsed_ms();
    REQUIRE_GT(first, 0.0);

    // start() resets accumulated, so it does NOT add
    sw.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    sw.stop();
    double second = sw.elapsed_ms();
    // second measures only the second interval (start resets accumulated_ms_)
    REQUIRE_GT(second, 0.0);
}

TEST_CASE("Stopwatch: report format") {
    Stopwatch sw;
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    sw.lap("alpha");
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    sw.lap("beta");

    auto r = sw.report();
    REQUIRE(r.find("Stopwatch:") != std::string::npos);
    REQUIRE(r.find("total") != std::string::npos);
    REQUIRE(r.find("alpha") != std::string::npos);
    REQUIRE(r.find("beta") != std::string::npos);
}

TEST_CASE("Stopwatch: unnamed laps use index") {
    Stopwatch sw;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    sw.lap();
    auto r = sw.report();
    REQUIRE(r.find("lap 1") != std::string::npos);
}

// ============================================================================
// TimerAccumulator tests
// ============================================================================

TEST_CASE("TimerAccumulator: count/total tracking") {
    TimerAccumulator acc;
    REQUIRE_EQ(acc.count(), 0u);
    REQUIRE_NEAR(acc.total_ms(), 0.0, 0.01);

    for (int i = 0; i < 3; ++i) {
        acc.start();
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        acc.stop();
    }

    REQUIRE_EQ(acc.count(), 3u);
    REQUIRE_GT(acc.total_ms(), 0.0);
}

TEST_CASE("TimerAccumulator: min/max/avg") {
    TimerAccumulator acc;

    // Before any measurements, min/max should return 0
    REQUIRE_NEAR(acc.min_ms(), 0.0, 0.01);
    REQUIRE_NEAR(acc.max_ms(), 0.0, 0.01);
    REQUIRE_NEAR(acc.avg_ms(), 0.0, 0.01);

    acc.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    acc.stop();

    REQUIRE_GT(acc.min_ms(), 0.0);
    REQUIRE_GT(acc.max_ms(), 0.0);
    REQUIRE_LE(acc.min_ms(), acc.max_ms());
    REQUIRE_NEAR(acc.avg_ms(), acc.total_ms(), 0.01);

    acc.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    acc.stop();

    REQUIRE_LE(acc.min_ms(), acc.avg_ms());
    REQUIRE_GE(acc.max_ms(), acc.avg_ms());
}

TEST_CASE("TimerAccumulator: report format (no label)") {
    TimerAccumulator acc;
    acc.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    acc.stop();

    auto r = acc.report();
    REQUIRE(r.find("TimerAccumulator:") != std::string::npos);
    REQUIRE(r.find("1x") != std::string::npos);
    REQUIRE(r.find("total=") != std::string::npos);
    REQUIRE(r.find("avg=") != std::string::npos);
    REQUIRE(r.find("min=") != std::string::npos);
    REQUIRE(r.find("max=") != std::string::npos);
}

TEST_CASE("TimerAccumulator: report format (with label)") {
    TimerAccumulator acc;
    acc.start();
    acc.stop();

    auto r = acc.report("myop");
    REQUIRE(r.find("[myop]") != std::string::npos);
    REQUIRE(r.find("TimerAccumulator:") == std::string::npos);
}

UTILS2_TEST_MAIN()
