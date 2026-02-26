#pragma once
#include <format>
#include <print>
#include <string_view>
#include <string>
#include <mutex>
#include <cstdio>
#include <cstdint>
#include <source_location>
#include <chrono>
#include <utility>
#include <functional>

namespace utils2 {

// --- Log severity levels ---
enum class LogLevel : std::uint8_t {
    trace = 0,
    debug = 1,
    info = 2,
    warn = 3,
    error = 4,
    fatal = 5,
    off = 6
};

// --- Level name for output ---
[[nodiscard]] constexpr std::string_view log_level_name(LogLevel lvl) noexcept
{
    switch (lvl) {
        case LogLevel::trace: return "TRACE";
        case LogLevel::debug: return "DEBUG";
        case LogLevel::info:  return "INFO ";
        case LogLevel::warn:  return "WARN ";
        case LogLevel::error: return "ERROR";
        case LogLevel::fatal: return "FATAL";
        default:              return "?????";
    }
}

// --- Global logger (all-static, no instances) ---
class Logger final {
public:
    Logger() = delete;

    static void set_level(LogLevel level) noexcept
    {
        auto lock = std::lock_guard{mutex_()};
        level_() = level;
    }

    [[nodiscard]] static LogLevel level() noexcept
    {
        auto lock = std::lock_guard{mutex_()};
        return level_();
    }

    // Set an additional output file (nullptr to disable file output).
    static void set_file(std::FILE* f) noexcept
    {
        auto lock = std::lock_guard{mutex_()};
        file_() = f;
    }

    // Set a custom sink callback. Called after stderr/file output.
    using Sink = std::function<void(LogLevel, std::string_view)>;
    static void set_sink(Sink sink)
    {
        auto lock = std::lock_guard{mutex_()};
        sink_() = std::move(sink);
    }

    // Core log function. Early-returns before formatting if level is filtered.
    template <typename... Args>
    static void log(LogLevel lvl, std::format_string<Args...> fmt, Args&&... args)
    {
        // Fast check: avoid formatting entirely when filtered out.
        if (static_cast<std::uint8_t>(lvl) < static_cast<std::uint8_t>(level_())) {
            return;
        }

        // Format the user message.
        auto msg = std::format(fmt, std::forward<Args>(args)...);

        // Build timestamp.
        auto now = std::chrono::system_clock::now();
        auto time = std::chrono::floor<std::chrono::milliseconds>(now);
        auto dp = std::chrono::floor<std::chrono::days>(time);
        auto tod = std::chrono::hh_mm_ss{time - dp};

        auto line = std::format("[{}] [{:%H:%M:%S}] {}\n",
                                log_level_name(lvl), tod, msg);

        auto lock = std::lock_guard{mutex_()};

        // Write to stderr.
        std::print(stderr, "{}", line);

        // Write to file if set.
        if (file_()) {
            std::print(file_(), "{}", line);
            std::fflush(file_());
        }

        // Invoke custom sink if set.
        if (sink_()) {
            sink_()(lvl, line);
        }
    }

    // --- Convenience methods ---
    template <typename... Args>
    static void trace(std::format_string<Args...> fmt, Args&&... args)
    {
        log(LogLevel::trace, fmt, std::forward<Args>(args)...);
    }

    template <typename... Args>
    static void debug(std::format_string<Args...> fmt, Args&&... args)
    {
        log(LogLevel::debug, fmt, std::forward<Args>(args)...);
    }

    template <typename... Args>
    static void info(std::format_string<Args...> fmt, Args&&... args)
    {
        log(LogLevel::info, fmt, std::forward<Args>(args)...);
    }

    template <typename... Args>
    static void warn(std::format_string<Args...> fmt, Args&&... args)
    {
        log(LogLevel::warn, fmt, std::forward<Args>(args)...);
    }

    template <typename... Args>
    static void error(std::format_string<Args...> fmt, Args&&... args)
    {
        log(LogLevel::error, fmt, std::forward<Args>(args)...);
    }

    template <typename... Args>
    static void fatal(std::format_string<Args...> fmt, Args&&... args)
    {
        log(LogLevel::fatal, fmt, std::forward<Args>(args)...);
    }

private:
    // Use function-local statics to avoid static-init-order fiasco.
    static std::mutex& mutex_()
    {
        static std::mutex m;
        return m;
    }

    static LogLevel& level_()
    {
        static LogLevel lvl = LogLevel::info;
        return lvl;
    }

    static std::FILE*& file_()
    {
        static std::FILE* f = nullptr;
        return f;
    }

    static Sink& sink_()
    {
        static Sink s;
        return s;
    }
};

// --- RAII scoped log level override ---
class ScopedLogLevel final {
    LogLevel prev_;

public:
    explicit ScopedLogLevel(LogLevel level) noexcept
        : prev_{Logger::level()}
    {
        Logger::set_level(level);
    }

    ~ScopedLogLevel() noexcept { Logger::set_level(prev_); }

    ScopedLogLevel(const ScopedLogLevel&) = delete;
    ScopedLogLevel& operator=(const ScopedLogLevel&) = delete;
    ScopedLogLevel(ScopedLogLevel&&) = delete;
    ScopedLogLevel& operator=(ScopedLogLevel&&) = delete;
};

} // namespace utils2
