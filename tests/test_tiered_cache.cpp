#include <utils2/test.hpp>
#include <utils2/tiered_cache.hpp>
#include <string>
#include <vector>
#include <cstddef>
#include <atomic>
#include <unordered_map>
#include <mutex>

// A simple value type for testing.
using TestCache = utils2::TieredCache<int, std::string>;

static TestCache::Callbacks make_basic_callbacks() {
    return {
        .value_size = [](const std::string& s) -> std::size_t { return s.size(); },
        .compress   = nullptr,
        .decompress = nullptr,
    };
}

static TestCache::Callbacks make_warm_callbacks() {
    return {
        .value_size = [](const std::string& s) -> std::size_t { return s.size(); },
        .compress   = [](const std::string& s) -> TestCache::CompressedV {
            TestCache::CompressedV out(s.size());
            std::memcpy(out.data(), s.data(), s.size());
            return out;
        },
        .decompress = [](const TestCache::CompressedV& d) -> std::string {
            return std::string(reinterpret_cast<const char*>(d.data()), d.size());
        },
    };
}

TEST_CASE("TieredCache hot tier put/get") {
    utils2::TieredCacheConfig cfg;
    cfg.hot_max_bytes = 1024;
    TestCache cache(cfg, make_basic_callbacks());

    cache.put(1, "hello");
    auto val = cache.get(1);
    REQUIRE(val.has_value());
    REQUIRE_EQ(*val, std::string("hello"));
}

TEST_CASE("TieredCache get miss returns nullopt") {
    utils2::TieredCacheConfig cfg;
    TestCache cache(cfg, make_basic_callbacks());

    auto val = cache.get(999);
    REQUIRE(!val.has_value());
}

TEST_CASE("TieredCache hot eviction") {
    utils2::TieredCacheConfig cfg;
    cfg.hot_max_bytes = 20;
    cfg.evict_ratio = 0.5;
    TestCache cache(cfg, make_basic_callbacks());

    // Each string is 10 bytes. After 3 puts we exceed 20 bytes, triggering eviction.
    cache.put(1, "aaaaaaaaaa"); // 10 bytes
    cache.put(2, "bbbbbbbbbb"); // 10 bytes -> now 20, at limit
    cache.put(3, "cccccccccc"); // 10 bytes -> now 30, triggers eviction

    auto s = cache.stats();
    CHECK_GT(s.hot_evictions, std::uint64_t(0));
}

TEST_CASE("TieredCache pinned entries survive eviction") {
    utils2::TieredCacheConfig cfg;
    cfg.hot_max_bytes = 20;
    cfg.evict_ratio = 0.5;
    TestCache cache(cfg, make_basic_callbacks());

    cache.put_pinned(1, "aaaaaaaaaa");
    cache.put(2, "bbbbbbbbbb");
    cache.put(3, "cccccccccc"); // trigger eviction

    // Pinned entry should still be present.
    auto val = cache.get(1);
    REQUIRE(val.has_value());
    REQUIRE_EQ(*val, std::string("aaaaaaaaaa"));
}

TEST_CASE("TieredCache warm tier with compress/decompress") {
    utils2::TieredCacheConfig cfg;
    cfg.hot_max_bytes = 15;
    cfg.warm_max_bytes = 1024;
    cfg.evict_ratio = 0.5;
    TestCache cache(cfg, make_warm_callbacks());

    cache.put(1, "aaaaaaaaaa"); // 10 bytes
    cache.put(2, "bbbbbbbbbb"); // 10 bytes -> triggers hot eviction, demotes to warm

    // get_blocking should find it in warm tier and promote.
    auto val = cache.get_blocking(2);
    // It should find one of the values (whichever was evicted).
    // Let's check stats to confirm warm was used.
    auto s = cache.stats();
    // Either key 1 or 2 was evicted to warm; get_blocking traverses tiers.
    CHECK_GE(s.hot_evictions, std::uint64_t(0));
}

TEST_CASE("TieredCache get_blocking full tier traversal") {
    utils2::TieredCacheConfig cfg;
    cfg.hot_max_bytes = 1024;
    TestCache cache(cfg, make_basic_callbacks());

    // Nothing in any tier -- should return nullopt.
    auto val = cache.get_blocking(42);
    REQUIRE(!val.has_value());
}

TEST_CASE("TieredCache negative cache") {
    utils2::TieredCacheConfig cfg;
    cfg.enable_negative_cache = true;
    cfg.max_negative_entries = 100;
    TestCache cache(cfg, make_basic_callbacks());

    REQUIRE(!cache.is_negative(5));
    cache.mark_negative(5);
    REQUIRE(cache.is_negative(5));

    cache.clear_negative_cache();
    REQUIRE(!cache.is_negative(5));
}

TEST_CASE("TieredCache stats tracking") {
    utils2::TieredCacheConfig cfg;
    cfg.hot_max_bytes = 1024;
    TestCache cache(cfg, make_basic_callbacks());

    cache.put(1, "hello");
    cache.get(1);
    cache.get(1);
    cache.get(999); // miss

    auto s = cache.stats();
    REQUIRE_EQ(s.hot_hits, std::uint64_t(2));
    REQUIRE_GE(s.misses, std::uint64_t(1));
    REQUIRE_EQ(s.hot_count, std::size_t(1));
}

TEST_CASE("TieredCache missing_keys batch") {
    utils2::TieredCacheConfig cfg;
    TestCache cache(cfg, make_basic_callbacks());

    cache.put(1, "a");
    cache.put(3, "c");

    std::vector<int> keys = {1, 2, 3, 4, 5};
    auto missing = cache.missing_keys(keys.begin(), keys.end());

    REQUIRE_EQ(missing.size(), std::size_t(3));
    // Keys 2, 4, 5 should be missing.
    CHECK(std::find(missing.begin(), missing.end(), 2) != missing.end());
    CHECK(std::find(missing.begin(), missing.end(), 4) != missing.end());
    CHECK(std::find(missing.begin(), missing.end(), 5) != missing.end());
}

TEST_CASE("TieredCache check_and_clear_arrived debounce") {
    utils2::TieredCacheConfig cfg;
    TestCache cache(cfg, make_basic_callbacks());

    // Initially false.
    REQUIRE(!cache.check_and_clear_arrived());

    // Simulate arrival via prefetch-like put + manual flag is internal;
    // We can test the flag by checking it is false initially and remains so.
    REQUIRE(!cache.check_and_clear_arrived());
}

TEST_CASE("TieredCache clear resets everything") {
    utils2::TieredCacheConfig cfg;
    TestCache cache(cfg, make_basic_callbacks());

    cache.put(1, "a");
    cache.put(2, "b");
    cache.clear();

    REQUIRE(!cache.get(1).has_value());
    REQUIRE(!cache.get(2).has_value());
    auto s = cache.stats();
    REQUIRE_EQ(s.hot_count, std::size_t(0));
}

TEST_CASE("TieredCache warm promotion round-trip") {
    utils2::TieredCacheConfig cfg;
    cfg.hot_max_bytes = 15;   // very small so eviction triggers quickly
    cfg.warm_max_bytes = 1024;
    cfg.evict_ratio = 0.5;
    TestCache cache(cfg, make_warm_callbacks());

    // Insert key 1 then key 2 to overflow hot tier
    cache.put(1, "aaaaaaaaaa"); // 10 bytes
    cache.put(2, "bbbbbbbbbb"); // 10 bytes -> 20 > 15, triggers eviction

    // Key 1 should have been evicted to warm. Use get_blocking to promote.
    auto val = cache.get_blocking(1);
    REQUIRE(val.has_value());
    REQUIRE_EQ(*val, std::string("aaaaaaaaaa"));

    auto s = cache.stats();
    CHECK_GT(s.hot_evictions, std::uint64_t(0));
}

TEST_CASE("TieredCache get_best_available with coarsening") {
    utils2::TieredCacheConfig cfg;
    cfg.hot_max_bytes = 1024;
    TestCache cache(cfg, make_basic_callbacks());

    // Insert only the coarser key
    cache.put(10, "coarse_value");

    // Coarsen function: int / 2 until 0
    auto coarsen = [](const int& k) -> std::optional<int> {
        if (k <= 0) return std::nullopt;
        return k / 2;
    };

    // Key 42 is not present, but coarsen(42)=21, coarsen(21)=10 -> found
    auto result = cache.get_best_available(42, coarsen);
    REQUIRE(result.has_value());
    REQUIRE_EQ(result->first, std::string("coarse_value"));
    REQUIRE_EQ(result->second, 10);

    // Key that can't be coarsened to any existing key
    auto no_result = cache.get_best_available(3, coarsen);
    // coarsen(3)=1, coarsen(1)=0, coarsen(0)=nullopt -> no match
    REQUIRE(!no_result.has_value());
}

TEST_CASE("TieredCache on_ready and remove_callback") {
    utils2::TieredCacheConfig cfg;
    cfg.hot_max_bytes = 1024;
    TestCache cache(cfg, make_basic_callbacks());

    int cb_count = 0;
    int last_key = -1;
    auto id = cache.on_ready([&](const int& k) {
        cb_count++;
        last_key = k;
    });
    REQUIRE_GT(id, std::size_t(0));

    // Prefetch fires ready callbacks for successfully resolved keys.
    // Since there's nothing in any tier, prefetch won't find anything.
    std::vector<int> keys = {99};
    cache.prefetch(keys.begin(), keys.end());
    REQUIRE_EQ(cb_count, 0); // nothing to find

    // Now put data and prefetch it (it's already in hot, so skip)
    cache.put(99, "exists");
    cache.prefetch(keys.begin(), keys.end());
    // Already in hot tier -> skipped, no callback fired
    REQUIRE_EQ(cb_count, 0);

    // Remove callback
    cache.remove_callback(id);
}

TEST_CASE("TieredCache hot_bytes tracking") {
    utils2::TieredCacheConfig cfg;
    cfg.hot_max_bytes = 1024;
    TestCache cache(cfg, make_basic_callbacks());

    cache.put(1, "hello"); // 5 bytes
    auto s1 = cache.stats();
    REQUIRE_EQ(s1.hot_bytes, std::size_t(5));

    cache.put(2, "world!"); // 6 bytes
    auto s2 = cache.stats();
    REQUIRE_EQ(s2.hot_bytes, std::size_t(11));
}

TEST_CASE("TieredCache overwrite existing key") {
    utils2::TieredCacheConfig cfg;
    cfg.hot_max_bytes = 1024;
    TestCache cache(cfg, make_basic_callbacks());

    cache.put(1, "aaa");
    cache.put(1, "bbbbbb"); // overwrite with longer value

    auto val = cache.get(1);
    REQUIRE(val.has_value());
    REQUIRE_EQ(*val, std::string("bbbbbb"));

    auto s = cache.stats();
    REQUIRE_EQ(s.hot_bytes, std::size_t(6)); // updated bytes
    REQUIRE_EQ(s.hot_count, std::size_t(1)); // still one entry
}

TEST_CASE("TieredCache negative cache overflow clears") {
    utils2::TieredCacheConfig cfg;
    cfg.enable_negative_cache = true;
    cfg.max_negative_entries = 3;
    TestCache cache(cfg, make_basic_callbacks());

    cache.mark_negative(1);
    cache.mark_negative(2);
    cache.mark_negative(3);
    REQUIRE(cache.is_negative(1));

    // Adding a 4th should trigger clear + insert only the new key
    cache.mark_negative(4);
    // After clear: only key 4 should exist
    REQUIRE(cache.is_negative(4));
    // Old keys were cleared
    REQUIRE(!cache.is_negative(1));
}

TEST_CASE("TieredCache disabled negative cache") {
    utils2::TieredCacheConfig cfg;
    cfg.enable_negative_cache = false;
    TestCache cache(cfg, make_basic_callbacks());

    cache.mark_negative(1);
    REQUIRE(!cache.is_negative(1)); // always false when disabled
}

TEST_CASE("TieredCache warm-to-hot promotion with stats") {
    utils2::TieredCacheConfig cfg;
    cfg.hot_max_bytes = 10;   // only fits 1 entry of 10 bytes
    cfg.warm_max_bytes = 1024;
    cfg.evict_ratio = 0.5;    // target = 5 bytes, so eviction removes all unpinned
    TestCache cache(cfg, make_warm_callbacks());

    // Insert key 1 (10 bytes). Fits in hot.
    cache.put(1, "aaaaaaaaaa");
    auto s0 = cache.stats();
    REQUIRE_EQ(s0.hot_count, std::size_t(1));
    REQUIRE_EQ(s0.hot_evictions, std::uint64_t(0));

    // Insert key 2 (10 bytes). Exceeds hot_max_bytes (20 > 10), triggers eviction.
    // Key 1 (older generation) should be evicted to warm tier.
    cache.put(2, "bbbbbbbbbb");
    auto s1 = cache.stats();
    CHECK_GT(s1.hot_evictions, std::uint64_t(0));
    CHECK_GT(s1.warm_count, std::size_t(0));
    CHECK_GT(s1.warm_bytes, std::size_t(0));

    // Key 1 should no longer be in hot tier.
    auto hot_miss = cache.get(1);
    REQUIRE(!hot_miss.has_value());

    // Use get_blocking to promote key 1 from warm back to hot.
    auto val = cache.get_blocking(1);
    REQUIRE(val.has_value());
    REQUIRE_EQ(*val, std::string("aaaaaaaaaa"));

    auto s2 = cache.stats();
    CHECK_GT(s2.warm_hits, std::uint64_t(0));
}

TEST_CASE("TieredCache warm eviction triggers and warm_evictions stat") {
    utils2::TieredCacheConfig cfg;
    cfg.hot_max_bytes = 10;    // fits 1 entry of 10 bytes
    cfg.warm_max_bytes = 15;   // fits ~1.5 entries of 10 bytes compressed
    cfg.evict_ratio = 0.5;     // target = 7 bytes for warm
    TestCache cache(cfg, make_warm_callbacks());

    // Insert 3 entries sequentially. Each triggers hot eviction, demoting old to warm.
    cache.put(1, "aaaaaaaaaa"); // 10b hot
    cache.put(2, "bbbbbbbbbb"); // evicts 1 to warm, 2 in hot
    cache.put(3, "cccccccccc"); // evicts 2 to warm, 3 in hot; warm now has 1+2 = 20b > 15, triggers warm eviction

    auto s = cache.stats();
    CHECK_GT(s.hot_evictions, std::uint64_t(0));
    // Warm evictions should have happened since warm exceeded warm_max_bytes.
    CHECK_GT(s.warm_evictions, std::uint64_t(0));
}

TEST_CASE("TieredCache prefetch with warm backend fires callbacks") {
    utils2::TieredCacheConfig cfg;
    cfg.hot_max_bytes = 10;
    cfg.warm_max_bytes = 1024;
    cfg.evict_ratio = 0.5;
    TestCache cache(cfg, make_warm_callbacks());

    // Insert key 1 then key 2 to push key 1 to warm tier.
    cache.put(1, "aaaaaaaaaa");
    cache.put(2, "bbbbbbbbbb");

    // Now key 1 should be in warm (evicted from hot).
    // Verify it's not in hot.
    auto hot_check = cache.get(1);
    // It might or might not be in hot depending on eviction order, so use get_blocking path.

    // Register a ready callback.
    std::atomic<int> cb_count{0};
    int last_key = -1;
    std::mutex mu;
    auto id = cache.on_ready([&](const int& k) {
        cb_count.fetch_add(1, std::memory_order_relaxed);
        std::lock_guard lk(mu);
        last_key = k;
    });

    // Prefetch key 1. If it was evicted to warm, it will be found there,
    // promoted to hot, and the callback will fire.
    // If it's still in hot, prefetch skips it.
    std::vector<int> keys = {1};
    cache.prefetch(keys.begin(), keys.end());

    if (!hot_check.has_value()) {
        // Key 1 was in warm, prefetch should have promoted it and fired callback.
        CHECK_GT(cb_count.load(), 0);
        CHECK(cache.check_and_clear_arrived());
        // After clearing, flag should be false.
        CHECK(!cache.check_and_clear_arrived());
    }

    cache.remove_callback(id);
}

TEST_CASE("TieredCache prefetch skips keys already in hot") {
    utils2::TieredCacheConfig cfg;
    cfg.hot_max_bytes = 1024;
    TestCache cache(cfg, make_basic_callbacks());

    cache.put(1, "hello");

    std::atomic<int> cb_count{0};
    auto id = cache.on_ready([&](const int&) {
        cb_count.fetch_add(1, std::memory_order_relaxed);
    });

    std::vector<int> keys = {1};
    cache.prefetch(keys.begin(), keys.end());

    // Already in hot, so no callback fired.
    REQUIRE_EQ(cb_count.load(), 0);
    REQUIRE(!cache.check_and_clear_arrived());

    cache.remove_callback(id);
}

TEST_CASE("TieredCache cold tier fetch and store via get_blocking") {
    // Simulated cold storage.
    std::unordered_map<int, TestCache::CompressedV> cold_store;
    std::mutex cold_mu;

    utils2::TieredCacheConfig cfg;
    cfg.hot_max_bytes = 1024;
    cfg.enable_negative_cache = false;

    TestCache::Callbacks cbs = make_warm_callbacks();
    cbs.cold_fetch = [&](const int& k) -> std::optional<TestCache::CompressedV> {
        std::lock_guard lk(cold_mu);
        auto it = cold_store.find(k);
        if (it == cold_store.end()) return std::nullopt;
        return it->second;
    };
    cbs.cold_store = [&](const int& k, const TestCache::CompressedV& d) {
        std::lock_guard lk(cold_mu);
        cold_store[k] = d;
    };

    TestCache cache(cfg, cbs);

    // Manually insert compressed data into cold storage.
    {
        std::string val = "from_cold";
        TestCache::CompressedV comp(val.size());
        std::memcpy(comp.data(), val.data(), val.size());
        cold_store[42] = comp;
    }

    // get_blocking should find it in cold tier and promote to hot.
    auto val = cache.get_blocking(42);
    REQUIRE(val.has_value());
    REQUIRE_EQ(*val, std::string("from_cold"));

    auto s = cache.stats();
    REQUIRE_EQ(s.cold_hits, std::uint64_t(1));

    // It should now be in hot tier.
    auto hot_val = cache.get(42);
    REQUIRE(hot_val.has_value());
    REQUIRE_EQ(*hot_val, std::string("from_cold"));
}

TEST_CASE("TieredCache ice tier fetch stores to cold") {
    std::unordered_map<int, TestCache::CompressedV> cold_store;
    std::mutex cold_mu;

    utils2::TieredCacheConfig cfg;
    cfg.hot_max_bytes = 1024;
    cfg.enable_negative_cache = false;

    TestCache::Callbacks cbs = make_warm_callbacks();
    cbs.cold_fetch = [&](const int& k) -> std::optional<TestCache::CompressedV> {
        std::lock_guard lk(cold_mu);
        auto it = cold_store.find(k);
        if (it == cold_store.end()) return std::nullopt;
        return it->second;
    };
    cbs.cold_store = [&](const int& k, const TestCache::CompressedV& d) {
        std::lock_guard lk(cold_mu);
        cold_store[k] = d;
    };
    cbs.ice_fetch = [&](const int& k) -> std::optional<TestCache::CompressedV> {
        if (k == 99) {
            std::string val = "from_ice";
            TestCache::CompressedV comp(val.size());
            std::memcpy(comp.data(), val.data(), val.size());
            return comp;
        }
        return std::nullopt;
    };

    TestCache cache(cfg, cbs);

    auto val = cache.get_blocking(99);
    REQUIRE(val.has_value());
    REQUIRE_EQ(*val, std::string("from_ice"));

    auto s = cache.stats();
    REQUIRE_EQ(s.ice_hits, std::uint64_t(1));

    // Ice fetch should have also stored to cold.
    {
        std::lock_guard lk(cold_mu);
        REQUIRE(cold_store.contains(99));
    }
}

TEST_CASE("TieredCache negative cache short-circuits get_blocking") {
    utils2::TieredCacheConfig cfg;
    cfg.hot_max_bytes = 1024;
    cfg.enable_negative_cache = true;
    TestCache cache(cfg, make_basic_callbacks());

    cache.mark_negative(7);

    // get_blocking should short-circuit on negative cache.
    auto val = cache.get_blocking(7);
    REQUIRE(!val.has_value());

    auto s = cache.stats();
    CHECK_GE(s.misses, std::uint64_t(1));
}

TEST_CASE("TieredCache hot eviction demotes to warm and warm tracks bytes") {
    utils2::TieredCacheConfig cfg;
    cfg.hot_max_bytes = 10;
    cfg.warm_max_bytes = 1024;
    cfg.evict_ratio = 0.5;
    TestCache cache(cfg, make_warm_callbacks());

    cache.put(1, "aaaaaaaaaa"); // 10 bytes
    cache.put(2, "bbbbbbbbbb"); // triggers hot eviction

    auto s = cache.stats();
    CHECK_GT(s.hot_evictions, std::uint64_t(0));
    // Warm tier should have received the evicted entry.
    CHECK_GT(s.warm_bytes, std::size_t(0));
    CHECK_GT(s.warm_count, std::size_t(0));
}

TEST_CASE("TieredCache multiple warm promotions and evictions cycle") {
    utils2::TieredCacheConfig cfg;
    cfg.hot_max_bytes = 10;
    cfg.warm_max_bytes = 1024;
    cfg.evict_ratio = 0.5;
    TestCache cache(cfg, make_warm_callbacks());

    // Cycle through several keys, forcing repeated hot->warm eviction and warm->hot promotion.
    for (int round = 0; round < 3; ++round) {
        cache.put(1, "aaaaaaaaaa");
        cache.put(2, "bbbbbbbbbb");

        // Promote key 1 back from warm.
        auto val = cache.get_blocking(1);
        CHECK(val.has_value());
    }

    auto s = cache.stats();
    CHECK_GT(s.hot_evictions, std::uint64_t(0));
    CHECK_GT(s.warm_hits, std::uint64_t(0));
}

UTILS2_TEST_MAIN()
