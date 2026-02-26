#pragma once
#include <unordered_map>
#include <shared_mutex>
#include <mutex>
#include <atomic>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <concepts>
#include <functional>
#include <optional>
#include <utility>
#include <algorithm>
#include <span>
#include <string>
#include <unordered_set>

namespace utils2 {

// ---------------------------------------------------------------------------
// TieredCacheConfig -- sizing and policy knobs
// ---------------------------------------------------------------------------
struct TieredCacheConfig {
    std::size_t hot_max_bytes       = 10ULL << 30;   // 10 GB
    std::size_t warm_max_bytes      = 2ULL << 30;    // 2 GB
    double      evict_ratio         = 15.0 / 16.0;   // hysteresis target
    bool        enable_negative_cache = true;
    std::size_t max_negative_entries  = 500'000;
};

// ---------------------------------------------------------------------------
// TieredCache -- 4-tier cache: HOT -> WARM -> COLD -> ICE
// ---------------------------------------------------------------------------
template<typename K, typename V,
         typename Hash     = std::hash<K>,
         typename KeyEqual = std::equal_to<K>>
class TieredCache final {
public:
    // -- types --------------------------------------------------------------
    using CompressedV = std::vector<std::byte>;

    struct Callbacks {
        /// Size of a value in bytes (required).
        std::function<std::size_t(const V&)> value_size;

        /// Compress value for warm tier. nullptr = warm tier disabled.
        std::function<CompressedV(const V&)> compress = nullptr;

        /// Decompress from warm tier.
        std::function<V(const CompressedV&)> decompress = nullptr;

        /// Fetch from cold tier (disk). nullopt = not found.
        std::function<std::optional<CompressedV>(const K&)> cold_fetch = nullptr;

        /// Store to cold tier.
        std::function<void(const K&, const CompressedV&)> cold_store = nullptr;

        /// Fetch from ice tier (remote). nullopt = not found.
        std::function<std::optional<CompressedV>(const K&)> ice_fetch = nullptr;
    };

    using CoarsenFn    = std::function<std::optional<K>(const K&)>;
    using ReadyCallback = std::function<void(const K&)>;

    // -- construction -------------------------------------------------------
    explicit TieredCache(TieredCacheConfig config, Callbacks callbacks)
        : config_{std::move(config)}
        , cb_{std::move(callbacks)}
        , hot_gen_{0}
        , hot_bytes_{0}
        , warm_gen_{0}
        , warm_bytes_{0}
        , arrived_{false}
        , next_cb_id_{1}
    {
        stats_.hot_hits.store(0, std::memory_order_relaxed);
        stats_.warm_hits.store(0, std::memory_order_relaxed);
        stats_.cold_hits.store(0, std::memory_order_relaxed);
        stats_.ice_hits.store(0, std::memory_order_relaxed);
        stats_.misses.store(0, std::memory_order_relaxed);
        stats_.hot_evictions.store(0, std::memory_order_relaxed);
        stats_.warm_evictions.store(0, std::memory_order_relaxed);
    }

    // -- non-copyable, non-movable ------------------------------------------
    TieredCache(const TieredCache&)            = delete;
    TieredCache& operator=(const TieredCache&) = delete;
    TieredCache(TieredCache&&)                 = delete;
    TieredCache& operator=(TieredCache&&)      = delete;

    // -- hot tier read (non-blocking) ---------------------------------------

    /// Fast lookup in hot tier only. Returns nullopt on miss.
    [[nodiscard]] std::optional<V> get(const K& key) const
    {
        std::shared_lock lock{hot_mutex_};
        auto it = hot_.find(key);
        if (it == hot_.end()) {
            stats_.misses.fetch_add(1, std::memory_order_relaxed);
            return std::nullopt;
        }
        it->second.generation.store(hot_gen_.fetch_add(1, std::memory_order_relaxed), std::memory_order_relaxed);
        stats_.hot_hits.fetch_add(1, std::memory_order_relaxed);
        return it->second.value;
    }

    // -- blocking get (traverses all tiers) ---------------------------------

    /// Tries hot -> warm -> cold -> ice. Promotes to hot on hit.
    [[nodiscard]] std::optional<V> get_blocking(const K& key)
    {
        // 1) Hot tier.
        {
            std::shared_lock lock{hot_mutex_};
            auto it = hot_.find(key);
            if (it != hot_.end()) {
                it->second.generation.store(hot_gen_.fetch_add(1, std::memory_order_relaxed), std::memory_order_relaxed);
                stats_.hot_hits.fetch_add(1, std::memory_order_relaxed);
                return it->second.value;
            }
        }

        // 2) Negative cache short-circuit.
        if (config_.enable_negative_cache && is_negative(key)) {
            stats_.misses.fetch_add(1, std::memory_order_relaxed);
            return std::nullopt;
        }

        // 3) Warm tier.
        if (cb_.compress && cb_.decompress) {
            std::optional<CompressedV> compressed;
            {
                std::shared_lock lock{warm_mutex_};
                auto it = warm_.find(key);
                if (it != warm_.end()) {
                    it->second.generation.store(warm_gen_.fetch_add(1, std::memory_order_relaxed), std::memory_order_relaxed);
                    compressed = it->second.data;
                }
            }
            if (compressed) {
                V val = cb_.decompress(*compressed);
                stats_.warm_hits.fetch_add(1, std::memory_order_relaxed);
                promote_to_hot(key, std::move(val));
                return get_from_hot(key);
            }
        }

        // 4) Cold tier (disk).
        if (cb_.cold_fetch && cb_.decompress) {
            auto compressed = cb_.cold_fetch(key);
            if (compressed) {
                V val = cb_.decompress(*compressed);
                stats_.cold_hits.fetch_add(1, std::memory_order_relaxed);
                promote_to_hot(key, std::move(val));
                return get_from_hot(key);
            }
        }

        // 5) Ice tier (remote).
        if (cb_.ice_fetch && cb_.decompress) {
            auto compressed = cb_.ice_fetch(key);
            if (compressed) {
                V val = cb_.decompress(*compressed);
                stats_.ice_hits.fetch_add(1, std::memory_order_relaxed);
                // Also store to cold tier for future access.
                if (cb_.cold_store) {
                    cb_.cold_store(key, *compressed);
                }
                promote_to_hot(key, std::move(val));
                return get_from_hot(key);
            }
        }

        stats_.misses.fetch_add(1, std::memory_order_relaxed);
        return std::nullopt;
    }

    // -- best available (coarsening search) ---------------------------------

    /// Walk coarser keys until a hit is found in the hot tier.
    [[nodiscard]] std::optional<std::pair<V, K>> get_best_available(
        const K& key, CoarsenFn coarsen) const
    {
        // Try exact key first.
        if (auto v = get(key)) {
            return std::pair{std::move(*v), key};
        }

        // Walk coarser keys.
        auto cur = coarsen(key);
        while (cur) {
            if (auto v = get(*cur)) {
                return std::pair{std::move(*v), *cur};
            }
            cur = coarsen(*cur);
        }
        return std::nullopt;
    }

    // -- write (hot tier) ---------------------------------------------------

    /// Insert or update in the hot tier. May trigger eviction.
    void put(const K& key, V value)
    {
        put_impl(key, std::move(value), /*pinned=*/false);
    }

    /// Insert or update a pinned entry (never evicted from hot tier).
    void put_pinned(const K& key, V value)
    {
        put_impl(key, std::move(value), /*pinned=*/true);
    }

    // -- batch operations ---------------------------------------------------

    /// Return keys from [begin, end) absent from the hot tier.
    template<typename Iter>
    [[nodiscard]] std::vector<K> missing_keys(Iter begin, Iter end) const
    {
        std::vector<K> result;
        std::shared_lock lock{hot_mutex_};
        for (auto it = begin; it != end; ++it) {
            if (!hot_.contains(*it)) {
                result.push_back(*it);
            }
        }
        return result;
    }

    // -- negative cache -----------------------------------------------------

    /// Mark a key as known-missing (avoids repeated cold/ice lookups).
    void mark_negative(const K& key)
    {
        if (!config_.enable_negative_cache) return;

        std::unique_lock lock{neg_mutex_};
        if (negative_.size() >= config_.max_negative_entries) {
            // Simple eviction: clear the whole set when full.
            negative_.clear();
        }
        negative_.insert(key);
    }

    /// Check whether a key is in the negative cache.
    [[nodiscard]] bool is_negative(const K& key) const
    {
        if (!config_.enable_negative_cache) return false;
        std::shared_lock lock{neg_mutex_};
        return negative_.contains(key);
    }

    // -- prefetch -----------------------------------------------------------

    /// Submit keys for background loading. Fetches via get_blocking and
    /// fires ready callbacks for each key that resolves.
    template<typename Iter>
    void prefetch(Iter keys_begin, Iter keys_end)
    {
        for (auto it = keys_begin; it != keys_end; ++it) {
            const K& key = *it;

            // Skip keys already in hot tier.
            {
                std::shared_lock lock{hot_mutex_};
                if (hot_.contains(key)) continue;
            }

            auto result = get_blocking(key);
            if (result) {
                arrived_.store(true, std::memory_order_release);
                fire_ready_callbacks(key);
            }
        }
    }

    // -- ready callbacks ----------------------------------------------------

    /// Register a callback invoked when a key is promoted to hot tier via
    /// prefetch. Returns an ID for later removal.
    std::size_t on_ready(ReadyCallback cb)
    {
        std::unique_lock lock{cb_mutex_};
        auto id = next_cb_id_++;
        ready_cbs_.emplace_back(id, std::move(cb));
        return id;
    }

    /// Remove a previously-registered callback.
    void remove_callback(std::size_t id)
    {
        std::unique_lock lock{cb_mutex_};
        std::erase_if(ready_cbs_, [id](const auto& p) { return p.first == id; });
    }

    // -- debounce flag ------------------------------------------------------

    /// Atomically check and clear the "data arrived" flag.
    [[nodiscard]] bool check_and_clear_arrived() noexcept
    {
        return arrived_.exchange(false, std::memory_order_acq_rel);
    }

    // -- stats --------------------------------------------------------------

    struct Stats {
        std::uint64_t hot_hits, warm_hits, cold_hits, ice_hits, misses;
        std::uint64_t hot_evictions, warm_evictions;
        std::size_t   hot_bytes, warm_bytes;
        std::size_t   hot_count, warm_count;
        std::size_t   negative_count;
    };

    [[nodiscard]] Stats stats() const
    {
        Stats s{};
        s.hot_hits        = stats_.hot_hits.load(std::memory_order_relaxed);
        s.warm_hits       = stats_.warm_hits.load(std::memory_order_relaxed);
        s.cold_hits       = stats_.cold_hits.load(std::memory_order_relaxed);
        s.ice_hits        = stats_.ice_hits.load(std::memory_order_relaxed);
        s.misses          = stats_.misses.load(std::memory_order_relaxed);
        s.hot_evictions   = stats_.hot_evictions.load(std::memory_order_relaxed);
        s.warm_evictions  = stats_.warm_evictions.load(std::memory_order_relaxed);
        s.hot_bytes       = hot_bytes_.load(std::memory_order_relaxed);
        s.warm_bytes      = warm_bytes_.load(std::memory_order_relaxed);
        {
            std::shared_lock lock{hot_mutex_};
            s.hot_count = hot_.size();
        }
        {
            std::shared_lock lock{warm_mutex_};
            s.warm_count = warm_.size();
        }
        {
            std::shared_lock lock{neg_mutex_};
            s.negative_count = negative_.size();
        }
        return s;
    }

    // -- control ------------------------------------------------------------

    /// Clear all tiers and reset stats.
    void clear()
    {
        {
            std::unique_lock lock{hot_mutex_};
            hot_.clear();
            hot_bytes_.store(0, std::memory_order_relaxed);
        }
        {
            std::unique_lock lock{warm_mutex_};
            warm_.clear();
            warm_bytes_.store(0, std::memory_order_relaxed);
        }
        clear_negative_cache();
    }

    /// Clear only the negative cache.
    void clear_negative_cache()
    {
        std::unique_lock lock{neg_mutex_};
        negative_.clear();
    }

private:
    // -- internal entry types -----------------------------------------------
    struct HotEntry {
        V                                  value;
        std::size_t                        bytes;
        mutable std::atomic<std::uint64_t> generation;
        bool                               pinned;

        HotEntry(V v, std::size_t b, std::uint64_t g, bool p)
            : value(std::move(v)), bytes(b), generation(g), pinned(p) {}
        HotEntry(HotEntry&& o) noexcept(std::is_nothrow_move_constructible_v<V>)
            : value(std::move(o.value)), bytes(o.bytes)
            , generation(o.generation.load(std::memory_order_relaxed)), pinned(o.pinned) {}
        HotEntry& operator=(HotEntry&& o) noexcept(std::is_nothrow_move_assignable_v<V>) {
            value = std::move(o.value); bytes = o.bytes;
            generation.store(o.generation.load(std::memory_order_relaxed), std::memory_order_relaxed);
            pinned = o.pinned; return *this;
        }
    };

    struct WarmEntry {
        CompressedV                        data;
        std::size_t                        bytes;
        mutable std::atomic<std::uint64_t> generation;

        WarmEntry(CompressedV d, std::size_t b, std::uint64_t g)
            : data(std::move(d)), bytes(b), generation(g) {}
        WarmEntry(WarmEntry&& o) noexcept(std::is_nothrow_move_constructible_v<CompressedV>)
            : data(std::move(o.data)), bytes(o.bytes)
            , generation(o.generation.load(std::memory_order_relaxed)) {}
        WarmEntry& operator=(WarmEntry&& o) noexcept(std::is_nothrow_move_assignable_v<CompressedV>) {
            data = std::move(o.data); bytes = o.bytes;
            generation.store(o.generation.load(std::memory_order_relaxed), std::memory_order_relaxed);
            return *this;
        }
    };

    // -- hot tier helpers ---------------------------------------------------

    void put_impl(const K& key, V value, bool pinned)
    {
        const auto val_bytes = cb_.value_size(value);

        {
            std::unique_lock lock{hot_mutex_};
            if (auto it = hot_.find(key); it != hot_.end()) {
                hot_bytes_.fetch_sub(it->second.bytes, std::memory_order_relaxed);
                it->second.value      = std::move(value);
                it->second.bytes      = val_bytes;
                it->second.generation.store(hot_gen_.fetch_add(1, std::memory_order_relaxed), std::memory_order_relaxed);
                it->second.pinned     = pinned;
                hot_bytes_.fetch_add(val_bytes, std::memory_order_relaxed);
                return;
            }

            auto gen = hot_gen_.fetch_add(1, std::memory_order_relaxed);
            hot_.emplace(key, HotEntry{std::move(value), val_bytes, gen, pinned});
            hot_bytes_.fetch_add(val_bytes, std::memory_order_relaxed);
        }

        if (hot_bytes_.load(std::memory_order_relaxed) > config_.hot_max_bytes) {
            evict_hot();
        }
    }

    void promote_to_hot(const K& key, V value)
    {
        put_impl(key, std::move(value), /*pinned=*/false);
    }

    /// Read back from hot tier after promotion (avoids keeping a second copy).
    [[nodiscard]] std::optional<V> get_from_hot(const K& key) const
    {
        std::shared_lock lock{hot_mutex_};
        auto it = hot_.find(key);
        if (it == hot_.end()) return std::nullopt;
        it->second.generation.store(hot_gen_.fetch_add(1, std::memory_order_relaxed), std::memory_order_relaxed);
        return it->second.value;
    }

    // -- hot eviction -------------------------------------------------------

    void evict_hot()
    {
        const auto target = static_cast<std::size_t>(
            static_cast<double>(config_.hot_max_bytes) * config_.evict_ratio);

        struct Candidate {
            K             key;
            std::size_t   bytes;
            std::uint64_t generation;
        };
        std::vector<Candidate> candidates;

        // Phase 1: snapshot under shared lock.
        {
            std::shared_lock lock{hot_mutex_};
            candidates.reserve(hot_.size());
            for (const auto& [k, entry] : hot_) {
                if (!entry.pinned) {
                    candidates.push_back({k, entry.bytes, entry.generation.load(std::memory_order_relaxed)});
                }
            }
        }

        std::sort(candidates.begin(), candidates.end(),
                  [](const Candidate& a, const Candidate& b) {
                      return a.generation < b.generation;
                  });

        // Phase 2: evict under unique lock, demoting to warm tier.
        std::unique_lock lock{hot_mutex_};
        for (const auto& cand : candidates) {
            if (hot_bytes_.load(std::memory_order_relaxed) <= target) break;

            auto it = hot_.find(cand.key);
            if (it == hot_.end() || it->second.pinned) continue;
            if (it->second.generation.load(std::memory_order_relaxed) != cand.generation) continue;

            // Demote to warm tier if compression is available.
            if (cb_.compress && cb_.decompress) {
                auto compressed = cb_.compress(it->second.value);
                demote_to_warm(cand.key, std::move(compressed));
            }

            hot_bytes_.fetch_sub(it->second.bytes, std::memory_order_relaxed);
            hot_.erase(it);
            stats_.hot_evictions.fetch_add(1, std::memory_order_relaxed);
        }
    }

    // -- warm tier helpers --------------------------------------------------

    void demote_to_warm(const K& key, CompressedV data)
    {
        const auto data_bytes = data.size();

        {
            std::unique_lock lock{warm_mutex_};
            if (auto it = warm_.find(key); it != warm_.end()) {
                warm_bytes_.fetch_sub(it->second.bytes, std::memory_order_relaxed);
                it->second.data       = std::move(data);
                it->second.bytes      = data_bytes;
                it->second.generation.store(warm_gen_.fetch_add(1, std::memory_order_relaxed), std::memory_order_relaxed);
                warm_bytes_.fetch_add(data_bytes, std::memory_order_relaxed);
            } else {
                auto gen = warm_gen_.fetch_add(1, std::memory_order_relaxed);
                warm_.emplace(key, WarmEntry{std::move(data), data_bytes, gen});
                warm_bytes_.fetch_add(data_bytes, std::memory_order_relaxed);
            }
        }

        if (warm_bytes_.load(std::memory_order_relaxed) > config_.warm_max_bytes) {
            evict_warm();
        }
    }

    void evict_warm()
    {
        const auto target = static_cast<std::size_t>(
            static_cast<double>(config_.warm_max_bytes) * config_.evict_ratio);

        struct Candidate {
            K             key;
            std::size_t   bytes;
            std::uint64_t generation;
        };
        std::vector<Candidate> candidates;

        {
            std::shared_lock lock{warm_mutex_};
            candidates.reserve(warm_.size());
            for (const auto& [k, entry] : warm_) {
                candidates.push_back({k, entry.bytes, entry.generation.load(std::memory_order_relaxed)});
            }
        }

        std::sort(candidates.begin(), candidates.end(),
                  [](const Candidate& a, const Candidate& b) {
                      return a.generation < b.generation;
                  });

        std::unique_lock lock{warm_mutex_};
        for (const auto& cand : candidates) {
            if (warm_bytes_.load(std::memory_order_relaxed) <= target) break;

            auto it = warm_.find(cand.key);
            if (it == warm_.end()) continue;
            if (it->second.generation.load(std::memory_order_relaxed) != cand.generation) continue;

            // Optionally persist to cold tier before evicting.
            if (cb_.cold_store) {
                cb_.cold_store(cand.key, it->second.data);
            }

            warm_bytes_.fetch_sub(it->second.bytes, std::memory_order_relaxed);
            warm_.erase(it);
            stats_.warm_evictions.fetch_add(1, std::memory_order_relaxed);
        }
    }

    // -- callback helpers ---------------------------------------------------

    void fire_ready_callbacks(const K& key) const
    {
        std::shared_lock lock{cb_mutex_};
        for (const auto& [id, cb] : ready_cbs_) {
            cb(key);
        }
    }

    // -- data members -------------------------------------------------------
    TieredCacheConfig config_;
    Callbacks         cb_;

    // Hot tier.
    using HotMap = std::unordered_map<K, HotEntry, Hash, KeyEqual>;
    mutable HotMap                      hot_;
    mutable std::shared_mutex           hot_mutex_;
    mutable std::atomic<std::uint64_t>  hot_gen_;
    mutable std::atomic<std::size_t>    hot_bytes_;

    // Warm tier.
    using WarmMap = std::unordered_map<K, WarmEntry, Hash, KeyEqual>;
    mutable WarmMap                     warm_;
    mutable std::shared_mutex           warm_mutex_;
    mutable std::atomic<std::uint64_t>  warm_gen_;
    mutable std::atomic<std::size_t>    warm_bytes_;

    // Negative cache.
    using NegSet = std::unordered_set<K, Hash, KeyEqual>;
    mutable NegSet                      negative_;
    mutable std::shared_mutex           neg_mutex_;

    // Arrived flag for debounced notification.
    mutable std::atomic<bool>           arrived_;

    // Ready callbacks.
    std::vector<std::pair<std::size_t, ReadyCallback>> ready_cbs_;
    mutable std::shared_mutex                          cb_mutex_;
    std::size_t                                        next_cb_id_;

    // Atomic stats.
    struct AtomicStats {
        mutable std::atomic<std::uint64_t> hot_hits{0};
        mutable std::atomic<std::uint64_t> warm_hits{0};
        mutable std::atomic<std::uint64_t> cold_hits{0};
        mutable std::atomic<std::uint64_t> ice_hits{0};
        mutable std::atomic<std::uint64_t> misses{0};
        mutable std::atomic<std::uint64_t> hot_evictions{0};
        mutable std::atomic<std::uint64_t> warm_evictions{0};
    } stats_;
};

} // namespace utils2
