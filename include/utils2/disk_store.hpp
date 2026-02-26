#pragma once
#include <filesystem>
#include <fstream>
#include <mutex>
#include <atomic>
#include <vector>
#include <array>
#include <span>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <concepts>
#include <functional>
#include <optional>
#include <algorithm>
#include <chrono>

namespace utils2 {

// ---------------------------------------------------------------------------
// DiskStoreConfig
// ---------------------------------------------------------------------------
struct DiskStoreConfig {
    std::filesystem::path root;
    std::size_t max_bytes = 50ULL << 30; // 50 GB default
    bool persistent = true;              // if false, cleanup on destruction
    std::size_t lock_pool_size = 32;
};

// ---------------------------------------------------------------------------
// DiskStore -- persistent file-based cache with atomic writes, LRU eviction
//              by modification time, and per-key lock serialization.
// ---------------------------------------------------------------------------
class DiskStore final {
public:
    using KeyToPath = std::function<std::filesystem::path(std::string_view key)>;

    explicit DiskStore(DiskStoreConfig config)
        : config_{std::move(config)}
        , total_bytes_{0}
        , locks_(config_.lock_pool_size)
    {
        std::filesystem::create_directories(config_.root);
        rescan_total();
    }

    ~DiskStore()
    {
        if (!config_.persistent) {
            std::error_code ec;
            std::filesystem::remove_all(config_.root, ec);
        }
    }

    DiskStore(const DiskStore&)            = delete;
    DiskStore& operator=(const DiskStore&) = delete;
    DiskStore(DiskStore&&)                 = delete;
    DiskStore& operator=(DiskStore&&)      = delete;

    // -- key mapper ----------------------------------------------------------

    void set_key_mapper(KeyToPath mapper) { key_to_path_ = std::move(mapper); }

    // -- core operations -----------------------------------------------------

    /// Store data atomically (temp file + rename).
    void put(std::string_view key, std::span<const std::byte> data)
    {
        auto path = resolve(key);
        auto tmp_path = path;
        tmp_path += ".tmp";

        auto& mtx = lock_for(key);
        std::lock_guard lock{mtx};

        std::filesystem::create_directories(path.parent_path());

        // If the file already exists, subtract its old size.
        std::error_code ec;
        auto old_size = std::filesystem::file_size(path, ec);
        if (!ec) {
            total_bytes_.fetch_sub(old_size, std::memory_order_relaxed);
        }

        // Write to temp file.
        {
            std::ofstream ofs(tmp_path, std::ios::binary | std::ios::trunc);
            ofs.write(reinterpret_cast<const char*>(data.data()),
                      static_cast<std::streamsize>(data.size()));
            ofs.close();
        }

        // Atomic rename.
        std::filesystem::rename(tmp_path, path);

        // Update tracked size.
        total_bytes_.fetch_add(data.size(), std::memory_order_relaxed);
    }

    /// Store data atomically (uint8_t overload).
    void put(std::string_view key, std::span<const std::uint8_t> data)
    {
        put(key, std::span<const std::byte>{
            reinterpret_cast<const std::byte*>(data.data()), data.size()});
    }

    /// Read data. Returns nullopt if the file does not exist.
    [[nodiscard]] std::optional<std::vector<std::byte>>
    get(std::string_view key) const
    {
        auto path = resolve(key);

        auto& mtx = lock_for(key);
        std::lock_guard lock{mtx};

        std::error_code ec;
        auto size = std::filesystem::file_size(path, ec);
        if (ec) return std::nullopt;

        std::ifstream ifs(path, std::ios::binary);
        if (!ifs) return std::nullopt;

        std::vector<std::byte> buf(size);
        ifs.read(reinterpret_cast<char*>(buf.data()),
                 static_cast<std::streamsize>(size));
        if (!ifs) return std::nullopt;

        return buf;
    }

    /// Check whether the key exists on disk.
    [[nodiscard]] bool contains(std::string_view key) const
    {
        auto path = resolve(key);
        std::error_code ec;
        return std::filesystem::exists(path, ec);
    }

    /// Remove a specific key. Returns true if the file existed.
    bool remove(std::string_view key)
    {
        auto path = resolve(key);

        auto& mtx = lock_for(key);
        std::lock_guard lock{mtx};

        std::error_code ec;
        auto size = std::filesystem::file_size(path, ec);
        if (ec) return false;

        if (!std::filesystem::remove(path, ec) || ec) return false;

        total_bytes_.fetch_sub(size, std::memory_order_relaxed);
        return true;
    }

    // -- size management -----------------------------------------------------

    /// Current tracked total bytes on disk.
    [[nodiscard]] std::size_t total_bytes() const noexcept
    {
        return total_bytes_.load(std::memory_order_relaxed);
    }

    /// Configured maximum bytes before eviction.
    [[nodiscard]] std::size_t max_bytes() const noexcept
    {
        return config_.max_bytes;
    }

    /// Evict oldest files (by mtime) until total bytes <= target.
    void evict_to_size(std::size_t target_bytes)
    {
        if (total_bytes_.load(std::memory_order_relaxed) <= target_bytes) return;

        struct FileInfo {
            std::filesystem::path                       p;
            std::size_t                                 bytes;
            std::filesystem::file_time_type             mtime;
        };

        std::vector<FileInfo> files;
        std::error_code ec;
        for (auto& entry :
             std::filesystem::recursive_directory_iterator(config_.root, ec))
        {
            if (!entry.is_regular_file()) continue;
            // Skip temp files that may be in-flight.
            if (entry.path().extension() == ".tmp") continue;
            files.push_back({
                entry.path(),
                entry.file_size(),
                entry.last_write_time()
            });
        }

        // Sort oldest first.
        std::sort(files.begin(), files.end(),
                  [](const FileInfo& a, const FileInfo& b) {
                      return a.mtime < b.mtime;
                  });

        auto current = total_bytes_.load(std::memory_order_relaxed);
        for (const auto& fi : files) {
            if (current <= target_bytes) break;
            std::error_code rm_ec;
            if (std::filesystem::remove(fi.p, rm_ec) && !rm_ec) {
                current -= fi.bytes;
            }
        }
        total_bytes_.store(current, std::memory_order_relaxed);
    }

    /// Trigger eviction if total bytes exceed the configured maximum.
    void evict_if_needed()
    {
        if (total_bytes_.load(std::memory_order_relaxed) > config_.max_bytes) {
            evict_to_size(config_.max_bytes);
        }
    }

    // -- bulk operations -----------------------------------------------------

    /// Scan the root directory and recompute total_bytes_ from scratch.
    void rescan_total()
    {
        std::size_t total = 0;
        std::error_code ec;
        for (auto& entry :
             std::filesystem::recursive_directory_iterator(config_.root, ec))
        {
            if (entry.is_regular_file() && entry.path().extension() != ".tmp") {
                total += entry.file_size();
            }
        }
        total_bytes_.store(total, std::memory_order_relaxed);
    }

    /// Remove all cached files under the root directory.
    void clear()
    {
        std::error_code ec;
        for (auto& entry :
             std::filesystem::directory_iterator(config_.root, ec))
        {
            std::filesystem::remove_all(entry.path(), ec);
        }
        total_bytes_.store(0, std::memory_order_relaxed);
    }

    // -- stats ---------------------------------------------------------------

    /// Count the number of regular files (excluding .tmp) under root.
    [[nodiscard]] std::size_t file_count() const
    {
        std::size_t count = 0;
        std::error_code ec;
        for (auto& entry :
             std::filesystem::recursive_directory_iterator(config_.root, ec))
        {
            if (entry.is_regular_file() && entry.path().extension() != ".tmp") {
                ++count;
            }
        }
        return count;
    }

private:
    // -- default key-to-path: hash-based directory sharding ------------------
    [[nodiscard]] std::filesystem::path
    resolve(std::string_view key) const
    {
        if (key_to_path_) return key_to_path_(key);
        return default_key_path(key);
    }

    [[nodiscard]] std::filesystem::path
    default_key_path(std::string_view key) const
    {
        auto h = hash_key(key);
        // Use first 2 hex chars of the hash as a subdirectory shard.
        char shard[3];
        static constexpr char hex[] = "0123456789abcdef";
        shard[0] = hex[(h >> 4) & 0xf];
        shard[1] = hex[h & 0xf];
        shard[2] = '\0';

        // Full hex hash as the filename.
        char name[17];
        for (int i = 15; i >= 0; --i) {
            name[i] = hex[h & 0xf];
            h >>= 4;
        }
        name[16] = '\0';

        return config_.root / std::string_view{shard, 2}
                            / std::string_view{name, 16};
    }

    // -- FNV-1a hash of key --------------------------------------------------
    [[nodiscard]] static std::uint64_t
    hash_key(std::string_view key) noexcept
    {
        constexpr std::uint64_t basis = 14695981039346656037ULL;
        constexpr std::uint64_t prime = 1099511628211ULL;
        auto h = basis;
        for (auto c : key) {
            h ^= static_cast<std::uint64_t>(static_cast<unsigned char>(c));
            h *= prime;
        }
        return h;
    }

    // -- per-key lock pool ---------------------------------------------------
    [[nodiscard]] std::mutex& lock_for(std::string_view key) const
    {
        auto idx = hash_key(key) % locks_.size();
        return locks_[idx];
    }

    // -- data members --------------------------------------------------------
    DiskStoreConfig                    config_;
    mutable std::atomic<std::size_t>   total_bytes_;
    mutable std::vector<std::mutex>    locks_;
    KeyToPath                          key_to_path_;
};

} // namespace utils2
