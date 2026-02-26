#include <utils2/test.hpp>
#include <utils2/log.hpp>
#include <string>
#include <vector>
#include <utility>

using namespace utils2;

// Helper: RAII cleanup that restores logger state after each test
struct LoggerGuard {
    LogLevel prev_level;
    LoggerGuard() : prev_level(Logger::level()) {}
    ~LoggerGuard() {
        Logger::set_level(prev_level);
        Logger::set_sink(nullptr);
        Logger::set_file(nullptr);
    }
};

// ============================================================================
// Compile-time tests
// ============================================================================

static_assert(log_level_name(LogLevel::trace) == "TRACE");
static_assert(log_level_name(LogLevel::debug) == "DEBUG");
static_assert(log_level_name(LogLevel::info)  == "INFO ");
static_assert(log_level_name(LogLevel::warn)  == "WARN ");
static_assert(log_level_name(LogLevel::error) == "ERROR");
static_assert(log_level_name(LogLevel::fatal) == "FATAL");

static_assert(static_cast<std::uint8_t>(LogLevel::trace) < static_cast<std::uint8_t>(LogLevel::debug));
static_assert(static_cast<std::uint8_t>(LogLevel::debug) < static_cast<std::uint8_t>(LogLevel::info));
static_assert(static_cast<std::uint8_t>(LogLevel::info)  < static_cast<std::uint8_t>(LogLevel::warn));
static_assert(static_cast<std::uint8_t>(LogLevel::warn)  < static_cast<std::uint8_t>(LogLevel::error));
static_assert(static_cast<std::uint8_t>(LogLevel::error) < static_cast<std::uint8_t>(LogLevel::fatal));
static_assert(static_cast<std::uint8_t>(LogLevel::fatal) < static_cast<std::uint8_t>(LogLevel::off));

// ============================================================================
// Runtime tests
// ============================================================================

TEST_CASE("Logger: set and get level") {
    LoggerGuard guard;

    Logger::set_level(LogLevel::warn);
    REQUIRE_EQ(static_cast<int>(Logger::level()), static_cast<int>(LogLevel::warn));

    Logger::set_level(LogLevel::trace);
    REQUIRE_EQ(static_cast<int>(Logger::level()), static_cast<int>(LogLevel::trace));

    Logger::set_level(LogLevel::off);
    REQUIRE_EQ(static_cast<int>(Logger::level()), static_cast<int>(LogLevel::off));
}

TEST_CASE("Logger: custom sink captures messages") {
    LoggerGuard guard;

    std::vector<std::pair<LogLevel, std::string>> captured;
    Logger::set_sink([&](LogLevel lvl, std::string_view msg) {
        captured.emplace_back(lvl, std::string(msg));
    });

    Logger::set_level(LogLevel::trace);
    Logger::info("hello {}", "world");

    REQUIRE_EQ(captured.size(), std::size_t{1});
    REQUIRE_EQ(static_cast<int>(captured[0].first), static_cast<int>(LogLevel::info));
    // The message should contain the formatted text
    REQUIRE(captured[0].second.find("hello world") != std::string::npos);
}

TEST_CASE("Logger: level filtering - messages below threshold are dropped") {
    LoggerGuard guard;

    std::vector<std::pair<LogLevel, std::string>> captured;
    Logger::set_sink([&](LogLevel lvl, std::string_view msg) {
        captured.emplace_back(lvl, std::string(msg));
    });

    Logger::set_level(LogLevel::warn);

    Logger::trace("should not appear");
    Logger::debug("should not appear");
    Logger::info("should not appear");
    Logger::warn("this should appear");
    Logger::error("this should appear too");

    REQUIRE_EQ(captured.size(), std::size_t{2});
    REQUIRE_EQ(static_cast<int>(captured[0].first), static_cast<int>(LogLevel::warn));
    REQUIRE_EQ(static_cast<int>(captured[1].first), static_cast<int>(LogLevel::error));
}

TEST_CASE("Logger: off level drops everything") {
    LoggerGuard guard;

    std::vector<std::string> captured;
    Logger::set_sink([&](LogLevel, std::string_view msg) {
        captured.emplace_back(msg);
    });

    Logger::set_level(LogLevel::off);

    Logger::trace("nope");
    Logger::debug("nope");
    Logger::info("nope");
    Logger::warn("nope");
    Logger::error("nope");
    Logger::fatal("nope");

    REQUIRE_EQ(captured.size(), std::size_t{0});
}

TEST_CASE("Logger: all convenience methods at trace level") {
    LoggerGuard guard;

    std::vector<LogLevel> levels;
    Logger::set_sink([&](LogLevel lvl, std::string_view) {
        levels.push_back(lvl);
    });

    Logger::set_level(LogLevel::trace);

    Logger::trace("t");
    Logger::debug("d");
    Logger::info("i");
    Logger::warn("w");
    Logger::error("e");
    Logger::fatal("f");

    REQUIRE_EQ(levels.size(), std::size_t{6});
    REQUIRE_EQ(static_cast<int>(levels[0]), static_cast<int>(LogLevel::trace));
    REQUIRE_EQ(static_cast<int>(levels[1]), static_cast<int>(LogLevel::debug));
    REQUIRE_EQ(static_cast<int>(levels[2]), static_cast<int>(LogLevel::info));
    REQUIRE_EQ(static_cast<int>(levels[3]), static_cast<int>(LogLevel::warn));
    REQUIRE_EQ(static_cast<int>(levels[4]), static_cast<int>(LogLevel::error));
    REQUIRE_EQ(static_cast<int>(levels[5]), static_cast<int>(LogLevel::fatal));
}

TEST_CASE("ScopedLogLevel: RAII level restoration") {
    LoggerGuard guard;

    Logger::set_level(LogLevel::info);
    REQUIRE_EQ(static_cast<int>(Logger::level()), static_cast<int>(LogLevel::info));

    {
        ScopedLogLevel scoped(LogLevel::trace);
        REQUIRE_EQ(static_cast<int>(Logger::level()), static_cast<int>(LogLevel::trace));
    }

    REQUIRE_EQ(static_cast<int>(Logger::level()), static_cast<int>(LogLevel::info));
}

TEST_CASE("ScopedLogLevel: nested scopes") {
    LoggerGuard guard;

    Logger::set_level(LogLevel::error);

    {
        ScopedLogLevel outer(LogLevel::warn);
        REQUIRE_EQ(static_cast<int>(Logger::level()), static_cast<int>(LogLevel::warn));

        {
            ScopedLogLevel inner(LogLevel::trace);
            REQUIRE_EQ(static_cast<int>(Logger::level()), static_cast<int>(LogLevel::trace));
        }

        REQUIRE_EQ(static_cast<int>(Logger::level()), static_cast<int>(LogLevel::warn));
    }

    REQUIRE_EQ(static_cast<int>(Logger::level()), static_cast<int>(LogLevel::error));
}

TEST_CASE("Logger: format string correctness") {
    LoggerGuard guard;

    std::string captured_msg;
    Logger::set_sink([&](LogLevel, std::string_view msg) {
        captured_msg = std::string(msg);
    });

    Logger::set_level(LogLevel::trace);

    SECTION("integer formatting") {
        Logger::info("value={}", 42);
        REQUIRE(captured_msg.find("value=42") != std::string::npos);
    }

    SECTION("float formatting") {
        Logger::info("pi={:.2f}", 3.14159);
        REQUIRE(captured_msg.find("pi=3.14") != std::string::npos);
    }

    SECTION("string formatting") {
        Logger::info("name={}", "test");
        REQUIRE(captured_msg.find("name=test") != std::string::npos);
    }

    SECTION("multiple arguments") {
        Logger::info("{} + {} = {}", 1, 2, 3);
        REQUIRE(captured_msg.find("1 + 2 = 3") != std::string::npos);
    }
}

TEST_CASE("Logger: message contains level tag") {
    LoggerGuard guard;

    std::string captured_msg;
    Logger::set_sink([&](LogLevel, std::string_view msg) {
        captured_msg = std::string(msg);
    });

    Logger::set_level(LogLevel::trace);

    Logger::warn("test message");
    REQUIRE(captured_msg.find("WARN") != std::string::npos);

    Logger::error("test message");
    REQUIRE(captured_msg.find("ERROR") != std::string::npos);
}

TEST_CASE("Logger: message contains timestamp bracket") {
    LoggerGuard guard;

    std::string captured_msg;
    Logger::set_sink([&](LogLevel, std::string_view msg) {
        captured_msg = std::string(msg);
    });

    Logger::set_level(LogLevel::trace);
    Logger::info("timing check");

    // Should have two bracket sections: [LEVEL] [HH:MM:SS...]
    auto first = captured_msg.find('[');
    REQUIRE(first != std::string::npos);
    auto second = captured_msg.find('[', first + 1);
    REQUIRE(second != std::string::npos);
}

TEST_CASE("Logger: removing sink stops capture") {
    LoggerGuard guard;

    int count = 0;
    Logger::set_sink([&](LogLevel, std::string_view) { count++; });
    Logger::set_level(LogLevel::trace);

    Logger::info("one");
    REQUIRE_EQ(count, 1);

    Logger::set_sink(nullptr);
    Logger::info("two");
    REQUIRE_EQ(count, 1); // should not have incremented
}

TEST_CASE("Logger: file output writes to file") {
    LoggerGuard guard;

    // Create a temporary file for logging
    auto* tmpf = std::tmpfile();
    REQUIRE(tmpf != nullptr);

    Logger::set_file(tmpf);
    Logger::set_level(LogLevel::trace);

    Logger::info("file output test {}", 42);

    // Read back the file
    std::fflush(tmpf);
    std::rewind(tmpf);

    char buf[512] = {};
    auto n = std::fread(buf, 1, sizeof(buf) - 1, tmpf);
    std::fclose(tmpf);

    std::string_view content(buf, n);
    REQUIRE(content.find("file output test 42") != std::string_view::npos);
    REQUIRE(content.find("INFO") != std::string_view::npos);
}

TEST_CASE("Logger: file output with multiple levels") {
    LoggerGuard guard;

    auto* tmpf = std::tmpfile();
    REQUIRE(tmpf != nullptr);

    Logger::set_file(tmpf);
    Logger::set_level(LogLevel::warn);

    Logger::debug("should not appear");
    Logger::warn("warning msg");
    Logger::error("error msg");

    std::fflush(tmpf);
    std::rewind(tmpf);

    char buf[1024] = {};
    auto n = std::fread(buf, 1, sizeof(buf) - 1, tmpf);
    std::fclose(tmpf);

    std::string_view content(buf, n);
    REQUIRE(content.find("should not appear") == std::string_view::npos);
    REQUIRE(content.find("warning msg") != std::string_view::npos);
    REQUIRE(content.find("error msg") != std::string_view::npos);
}

TEST_CASE("Logger: log_level_name default case") {
    // Cast an invalid underlying value to LogLevel to exercise the default branch
    auto invalid_level = static_cast<LogLevel>(200);
    auto name = log_level_name(invalid_level);
    REQUIRE_EQ(name, std::string_view("?????"));
}

TEST_CASE("Logger: trace and debug convenience methods produce output") {
    LoggerGuard guard;

    std::vector<std::pair<LogLevel, std::string>> captured;
    Logger::set_sink([&](LogLevel lvl, std::string_view msg) {
        captured.emplace_back(lvl, std::string(msg));
    });

    Logger::set_level(LogLevel::trace);

    Logger::trace("trace message {}", 1);
    Logger::debug("debug message {}", 2);

    REQUIRE_EQ(captured.size(), std::size_t{2});
    REQUIRE(captured[0].second.find("trace message 1") != std::string::npos);
    REQUIRE(captured[0].second.find("TRACE") != std::string::npos);
    REQUIRE(captured[1].second.find("debug message 2") != std::string::npos);
    REQUIRE(captured[1].second.find("DEBUG") != std::string::npos);
}

TEST_CASE("Logger: fatal convenience method") {
    LoggerGuard guard;

    std::string captured_msg;
    Logger::set_sink([&](LogLevel, std::string_view msg) {
        captured_msg = std::string(msg);
    });

    Logger::set_level(LogLevel::trace);
    Logger::fatal("fatal: {} happened", "crash");
    REQUIRE(captured_msg.find("fatal: crash happened") != std::string::npos);
    REQUIRE(captured_msg.find("FATAL") != std::string::npos);
}

TEST_CASE("Logger: file and sink both receive output") {
    LoggerGuard guard;

    auto* tmpf = std::tmpfile();
    REQUIRE(tmpf != nullptr);

    int sink_count = 0;
    Logger::set_file(tmpf);
    Logger::set_sink([&](LogLevel, std::string_view) { sink_count++; });
    Logger::set_level(LogLevel::trace);

    Logger::info("both outputs");

    REQUIRE_EQ(sink_count, 1);

    std::fflush(tmpf);
    std::rewind(tmpf);
    char buf[256] = {};
    auto n = std::fread(buf, 1, sizeof(buf) - 1, tmpf);
    std::fclose(tmpf);

    std::string_view content(buf, n);
    REQUIRE(content.find("both outputs") != std::string_view::npos);
}

UTILS2_TEST_MAIN()
