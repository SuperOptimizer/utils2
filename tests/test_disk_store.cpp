#include <utils2/test.hpp>
#include <utils2/disk_store.hpp>
#include <filesystem>
#include <cstring>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static fs::path make_temp_dir(const char* suffix) {
    auto p = fs::temp_directory_path() / ("utils2_test_disk_store_" + std::string(suffix));
    fs::remove_all(p);
    return p;
}

TEST_CASE("DiskStore put/get round trip") {
    auto dir = make_temp_dir("roundtrip");
    utils2::DiskStore store(utils2::DiskStoreConfig{.root = dir, .persistent = false});

    std::string data = "hello disk store";
    auto bytes = std::span<const std::byte>(
        reinterpret_cast<const std::byte*>(data.data()), data.size());

    store.put("key1", bytes);

    auto result = store.get("key1");
    REQUIRE(result.has_value());
    REQUIRE_EQ(result->size(), data.size());

    std::string got(reinterpret_cast<const char*>(result->data()), result->size());
    REQUIRE_EQ(got, data);
}

TEST_CASE("DiskStore contains/remove") {
    auto dir = make_temp_dir("contains");
    utils2::DiskStore store(utils2::DiskStoreConfig{.root = dir, .persistent = false});

    std::string data = "test";
    store.put("mykey", std::span<const std::byte>(
        reinterpret_cast<const std::byte*>(data.data()), data.size()));

    REQUIRE(store.contains("mykey"));
    REQUIRE(!store.contains("nokey"));

    bool removed = store.remove("mykey");
    REQUIRE(removed);
    REQUIRE(!store.contains("mykey"));

    // Remove again should return false.
    REQUIRE(!store.remove("mykey"));
}

TEST_CASE("DiskStore get nonexistent returns nullopt") {
    auto dir = make_temp_dir("noexist");
    utils2::DiskStore store(utils2::DiskStoreConfig{.root = dir, .persistent = false});

    auto result = store.get("does_not_exist");
    REQUIRE(!result.has_value());
}

TEST_CASE("DiskStore atomic write safety") {
    // Verify that no .tmp files remain after put.
    auto dir = make_temp_dir("atomic");
    utils2::DiskStore store(utils2::DiskStoreConfig{.root = dir, .persistent = false});

    std::string data = "atomic test data";
    store.put("atomic_key", std::span<const std::byte>(
        reinterpret_cast<const std::byte*>(data.data()), data.size()));

    // Scan for .tmp files.
    bool found_tmp = false;
    for (auto& entry : fs::recursive_directory_iterator(dir)) {
        if (entry.path().extension() == ".tmp") found_tmp = true;
    }
    REQUIRE(!found_tmp);
}

TEST_CASE("DiskStore evict_to_size") {
    auto dir = make_temp_dir("evict");
    utils2::DiskStore store(utils2::DiskStoreConfig{
        .root = dir, .max_bytes = 10000, .persistent = false});

    // Write several keys.
    for (int i = 0; i < 10; ++i) {
        std::string data(100, static_cast<char>('a' + i));
        store.put("key" + std::to_string(i), std::span<const std::byte>(
            reinterpret_cast<const std::byte*>(data.data()), data.size()));
    }

    REQUIRE_GE(store.total_bytes(), std::size_t(1000));

    // Evict down to 500 bytes.
    store.evict_to_size(500);
    REQUIRE_LE(store.total_bytes(), std::size_t(500));
}

TEST_CASE("DiskStore rescan_total") {
    auto dir = make_temp_dir("rescan");
    utils2::DiskStore store(utils2::DiskStoreConfig{.root = dir, .persistent = false});

    std::string data(256, 'x');
    store.put("k1", std::span<const std::byte>(
        reinterpret_cast<const std::byte*>(data.data()), data.size()));
    store.put("k2", std::span<const std::byte>(
        reinterpret_cast<const std::byte*>(data.data()), data.size()));

    auto before = store.total_bytes();
    store.rescan_total();
    auto after = store.total_bytes();

    REQUIRE_EQ(before, after);
    REQUIRE_EQ(after, std::size_t(512));
}

TEST_CASE("DiskStore multiple keys") {
    auto dir = make_temp_dir("multi");
    utils2::DiskStore store(utils2::DiskStoreConfig{.root = dir, .persistent = false});

    for (int i = 0; i < 20; ++i) {
        std::string data = "value_" + std::to_string(i);
        store.put("key_" + std::to_string(i), std::span<const std::byte>(
            reinterpret_cast<const std::byte*>(data.data()), data.size()));
    }

    REQUIRE_EQ(store.file_count(), std::size_t(20));

    for (int i = 0; i < 20; ++i) {
        auto result = store.get("key_" + std::to_string(i));
        REQUIRE(result.has_value());
        std::string got(reinterpret_cast<const char*>(result->data()), result->size());
        REQUIRE_EQ(got, "value_" + std::to_string(i));
    }
}

TEST_CASE("DiskStore non-persistent cleanup") {
    auto dir = make_temp_dir("cleanup");
    {
        utils2::DiskStore store(utils2::DiskStoreConfig{
            .root = dir, .persistent = false});
        std::string data = "temp";
        store.put("k", std::span<const std::byte>(
            reinterpret_cast<const std::byte*>(data.data()), data.size()));
    }
    // After destruction, directory should be removed.
    REQUIRE(!fs::exists(dir));
}

TEST_CASE("DiskStore overwrite existing key") {
    auto dir = make_temp_dir("overwrite");
    utils2::DiskStore store(utils2::DiskStoreConfig{.root = dir, .persistent = false});

    std::string data1 = "original";
    store.put("key", std::span<const std::byte>(
        reinterpret_cast<const std::byte*>(data1.data()), data1.size()));
    REQUIRE_EQ(store.total_bytes(), data1.size());

    std::string data2 = "replaced_with_longer";
    store.put("key", std::span<const std::byte>(
        reinterpret_cast<const std::byte*>(data2.data()), data2.size()));
    REQUIRE_EQ(store.total_bytes(), data2.size());

    auto result = store.get("key");
    REQUIRE(result.has_value());
    std::string got(reinterpret_cast<const char*>(result->data()), result->size());
    REQUIRE_EQ(got, data2);
}

TEST_CASE("DiskStore clear") {
    auto dir = make_temp_dir("clear");
    utils2::DiskStore store(utils2::DiskStoreConfig{.root = dir, .persistent = false});

    for (int i = 0; i < 5; ++i) {
        std::string data = "val" + std::to_string(i);
        store.put("key" + std::to_string(i), std::span<const std::byte>(
            reinterpret_cast<const std::byte*>(data.data()), data.size()));
    }
    REQUIRE_EQ(store.file_count(), std::size_t(5));

    store.clear();
    REQUIRE_EQ(store.total_bytes(), std::size_t(0));
    REQUIRE_EQ(store.file_count(), std::size_t(0));
}

TEST_CASE("DiskStore custom key_mapper") {
    auto dir = make_temp_dir("mapper");
    utils2::DiskStore store(utils2::DiskStoreConfig{.root = dir, .persistent = false});

    // Custom mapper: key directly becomes the filename
    store.set_key_mapper([&dir](std::string_view key) {
        return dir / std::string(key);
    });

    std::string data = "mapped";
    store.put("myfile.bin", std::span<const std::byte>(
        reinterpret_cast<const std::byte*>(data.data()), data.size()));

    REQUIRE(fs::exists(dir / "myfile.bin"));

    auto result = store.get("myfile.bin");
    REQUIRE(result.has_value());
    std::string got(reinterpret_cast<const char*>(result->data()), result->size());
    REQUIRE_EQ(got, data);
}

TEST_CASE("DiskStore persistent mode") {
    auto dir = make_temp_dir("persistent");
    {
        utils2::DiskStore store(utils2::DiskStoreConfig{
            .root = dir, .persistent = true});
        std::string data = "persist";
        store.put("k", std::span<const std::byte>(
            reinterpret_cast<const std::byte*>(data.data()), data.size()));
    }
    // After destruction, directory should still exist.
    REQUIRE(fs::exists(dir));

    // Re-open and read back.
    {
        utils2::DiskStore store(utils2::DiskStoreConfig{
            .root = dir, .persistent = true});
        auto result = store.get("k");
        REQUIRE(result.has_value());
        std::string got(reinterpret_cast<const char*>(result->data()), result->size());
        REQUIRE_EQ(got, std::string("persist"));
    }

    fs::remove_all(dir);
}

TEST_CASE("DiskStore evict_if_needed") {
    auto dir = make_temp_dir("evict_auto");
    utils2::DiskStore store(utils2::DiskStoreConfig{
        .root = dir, .max_bytes = 500, .persistent = false});

    // Write 10 keys of 100 bytes each = 1000 bytes > max_bytes(500)
    for (int i = 0; i < 10; ++i) {
        std::string data(100, static_cast<char>('a' + i));
        store.put("key" + std::to_string(i), std::span<const std::byte>(
            reinterpret_cast<const std::byte*>(data.data()), data.size()));
    }

    store.evict_if_needed();
    REQUIRE_LE(store.total_bytes(), std::size_t(500));
}

TEST_CASE("DiskStore max_bytes getter") {
    auto dir = make_temp_dir("maxbytes");
    utils2::DiskStore store(utils2::DiskStoreConfig{
        .root = dir, .max_bytes = 12345, .persistent = false});
    REQUIRE_EQ(store.max_bytes(), std::size_t(12345));
}

TEST_CASE("DiskStore uint8_t put overload") {
    auto dir = make_temp_dir("u8put");
    utils2::DiskStore store(utils2::DiskStoreConfig{.root = dir, .persistent = false});

    std::vector<std::uint8_t> data = {0x01, 0x02, 0xFF, 0x00};
    store.put("binkey", std::span<const std::uint8_t>(data));

    auto result = store.get("binkey");
    REQUIRE(result.has_value());
    REQUIRE_EQ(result->size(), std::size_t(4));
    REQUIRE_EQ(static_cast<std::uint8_t>((*result)[0]), std::uint8_t(0x01));
    REQUIRE_EQ(static_cast<std::uint8_t>((*result)[3]), std::uint8_t(0x00));
}

UTILS2_TEST_MAIN()
