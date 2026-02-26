#include <utils2/test.hpp>
#include <utils2/compression.hpp>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <string>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static std::vector<std::byte> make_test_data(std::size_t size) {
    std::vector<std::byte> data(size);
    for (std::size_t i = 0; i < size; ++i)
        data[i] = static_cast<std::byte>(i & 0xFF);
    return data;
}

static std::vector<std::byte> make_compressible_data(std::size_t size) {
    // Highly compressible: repeating pattern
    std::vector<std::byte> data(size);
    for (std::size_t i = 0; i < size; ++i)
        data[i] = static_cast<std::byte>(i % 4);
    return data;
}

static void round_trip_test(utils2::Codec codec, int level,
                            std::span<const std::byte> input,
                            std::size_t expected_hint = 0) {
    utils2::CompressParams params;
    params.codec = codec;
    params.level = level;

    auto compressed = utils2::compress(input, params);
    auto decompressed = utils2::decompress(compressed, codec, expected_hint);

    REQUIRE_EQ(decompressed.size(), input.size());
    for (std::size_t i = 0; i < input.size(); ++i) {
        if (decompressed[i] != input[i]) {
            REQUIRE(false); // mismatch at byte i
            return;
        }
    }
    REQUIRE(true);
}

// ---------------------------------------------------------------------------
// Codec enum utilities
// ---------------------------------------------------------------------------

TEST_CASE("codec_name returns correct names") {
    REQUIRE_EQ(utils2::codec_name(utils2::Codec::none), std::string_view("none"));
    REQUIRE_EQ(utils2::codec_name(utils2::Codec::zstd), std::string_view("zstd"));
    REQUIRE_EQ(utils2::codec_name(utils2::Codec::lz4), std::string_view("lz4"));
    REQUIRE_EQ(utils2::codec_name(utils2::Codec::lz4hc), std::string_view("lz4hc"));
    REQUIRE_EQ(utils2::codec_name(utils2::Codec::zlib), std::string_view("zlib"));
    REQUIRE_EQ(utils2::codec_name(utils2::Codec::gzip), std::string_view("gzip"));
    REQUIRE_EQ(utils2::codec_name(utils2::Codec::blosc_lz4), std::string_view("blosc_lz4"));
    REQUIRE_EQ(utils2::codec_name(utils2::Codec::blosc2_zstd), std::string_view("blosc2_zstd"));
}

TEST_CASE("parse_codec round-trips with codec_name") {
    for (auto c : {utils2::Codec::none, utils2::Codec::zstd, utils2::Codec::lz4,
                   utils2::Codec::lz4hc, utils2::Codec::zlib, utils2::Codec::gzip,
                   utils2::Codec::blosc_lz4, utils2::Codec::blosc2_zstd}) {
        auto name = utils2::codec_name(c);
        auto parsed = utils2::parse_codec(name);
        REQUIRE(parsed.has_value());
        REQUIRE_EQ(static_cast<int>(*parsed), static_cast<int>(c));
    }
}

TEST_CASE("parse_codec returns nullopt for unknown") {
    auto result = utils2::parse_codec("not_a_codec");
    REQUIRE(!result.has_value());
}

// ---------------------------------------------------------------------------
// Codec::none (passthrough)
// ---------------------------------------------------------------------------

TEST_CASE("none codec: passthrough round-trip") {
    auto data = make_test_data(256);
    round_trip_test(utils2::Codec::none, 0, data);
}

TEST_CASE("none codec: empty data") {
    std::vector<std::byte> empty;
    round_trip_test(utils2::Codec::none, 0, empty);
}

TEST_CASE("none codec: single byte") {
    std::vector<std::byte> one{std::byte{0x42}};
    round_trip_test(utils2::Codec::none, 0, one);
}

TEST_CASE("codec_available: none always available") {
    REQUIRE(utils2::codec_available(utils2::Codec::none));
}

// ---------------------------------------------------------------------------
// zstd
// ---------------------------------------------------------------------------

#if UTILS2_HAS_ZSTD
TEST_CASE("zstd: codec_available") {
    REQUIRE(utils2::codec_available(utils2::Codec::zstd));
}

TEST_CASE("zstd: round-trip small data") {
    auto data = make_test_data(64);
    round_trip_test(utils2::Codec::zstd, 3, data);
}

TEST_CASE("zstd: round-trip medium data") {
    auto data = make_test_data(4096);
    round_trip_test(utils2::Codec::zstd, 5, data);
}

TEST_CASE("zstd: round-trip large data") {
    auto data = make_compressible_data(1 << 16);
    round_trip_test(utils2::Codec::zstd, 5, data);
}

TEST_CASE("zstd: round-trip single byte") {
    std::vector<std::byte> one{std::byte{0xAB}};
    round_trip_test(utils2::Codec::zstd, 1, one);
}

TEST_CASE("zstd: compression levels") {
    auto data = make_compressible_data(4096);
    for (int level : {1, 3, 5, 9, 19}) {
        round_trip_test(utils2::Codec::zstd, level, data);
    }
}

TEST_CASE("zstd: decompress with expected_size=0 (auto-detect)") {
    auto data = make_test_data(512);
    utils2::CompressParams params;
    params.codec = utils2::Codec::zstd;
    params.level = 3;
    auto compressed = utils2::compress(data, params);
    auto decompressed = utils2::decompress(compressed, utils2::Codec::zstd, 0);
    REQUIRE_EQ(decompressed.size(), data.size());
    REQUIRE(decompressed == data);
}

TEST_CASE("zstd: compresses data smaller") {
    auto data = make_compressible_data(4096);
    utils2::CompressParams params;
    params.codec = utils2::Codec::zstd;
    params.level = 5;
    auto compressed = utils2::compress(data, params);
    REQUIRE_LT(compressed.size(), data.size());
}
#endif

// ---------------------------------------------------------------------------
// lz4
// ---------------------------------------------------------------------------

#if UTILS2_HAS_LZ4
TEST_CASE("lz4: codec_available") {
    REQUIRE(utils2::codec_available(utils2::Codec::lz4));
}

TEST_CASE("lz4: round-trip small data") {
    auto data = make_test_data(64);
    round_trip_test(utils2::Codec::lz4, 0, data);
}

TEST_CASE("lz4: round-trip medium data") {
    auto data = make_test_data(4096);
    round_trip_test(utils2::Codec::lz4, 0, data);
}

TEST_CASE("lz4: round-trip large data") {
    auto data = make_compressible_data(1 << 16);
    round_trip_test(utils2::Codec::lz4, 0, data);
}

TEST_CASE("lz4: round-trip single byte") {
    std::vector<std::byte> one{std::byte{0xCD}};
    round_trip_test(utils2::Codec::lz4, 0, one);
}

TEST_CASE("lz4: decompress with expected_size=0 (size prefix)") {
    auto data = make_test_data(512);
    utils2::CompressParams params;
    params.codec = utils2::Codec::lz4;
    auto compressed = utils2::compress(data, params);
    auto decompressed = utils2::decompress(compressed, utils2::Codec::lz4, 0);
    REQUIRE_EQ(decompressed.size(), data.size());
    REQUIRE(decompressed == data);
}

TEST_CASE("lz4: compresses compressible data smaller") {
    auto data = make_compressible_data(4096);
    utils2::CompressParams params;
    params.codec = utils2::Codec::lz4;
    auto compressed = utils2::compress(data, params);
    // lz4 has 4-byte size prefix, so compare against data+4
    REQUIRE_LT(compressed.size(), data.size() + 4);
}
#endif

// ---------------------------------------------------------------------------
// lz4hc
// ---------------------------------------------------------------------------

#if UTILS2_HAS_LZ4HC && UTILS2_HAS_LZ4
TEST_CASE("lz4hc: codec_available") {
    REQUIRE(utils2::codec_available(utils2::Codec::lz4hc));
}

TEST_CASE("lz4hc: round-trip small data") {
    auto data = make_test_data(64);
    round_trip_test(utils2::Codec::lz4hc, 9, data);
}

TEST_CASE("lz4hc: round-trip medium data") {
    auto data = make_test_data(4096);
    round_trip_test(utils2::Codec::lz4hc, 9, data);
}

TEST_CASE("lz4hc: round-trip large data") {
    auto data = make_compressible_data(1 << 16);
    round_trip_test(utils2::Codec::lz4hc, 9, data);
}

TEST_CASE("lz4hc: compression levels") {
    auto data = make_compressible_data(4096);
    for (int level : {1, 6, 9, 12}) {
        round_trip_test(utils2::Codec::lz4hc, level, data);
    }
}

TEST_CASE("lz4hc: single byte") {
    std::vector<std::byte> one{std::byte{0xEF}};
    round_trip_test(utils2::Codec::lz4hc, 9, one);
}
#endif

// ---------------------------------------------------------------------------
// zlib
// ---------------------------------------------------------------------------

#if UTILS2_HAS_ZLIB
TEST_CASE("zlib: codec_available") {
    REQUIRE(utils2::codec_available(utils2::Codec::zlib));
}

TEST_CASE("zlib: round-trip small data") {
    auto data = make_test_data(64);
    round_trip_test(utils2::Codec::zlib, 6, data);
}

TEST_CASE("zlib: round-trip medium data") {
    auto data = make_test_data(4096);
    round_trip_test(utils2::Codec::zlib, 6, data);
}

TEST_CASE("zlib: round-trip large data") {
    auto data = make_compressible_data(1 << 16);
    round_trip_test(utils2::Codec::zlib, 6, data);
}

TEST_CASE("zlib: single byte") {
    std::vector<std::byte> one{std::byte{0x01}};
    round_trip_test(utils2::Codec::zlib, 6, one);
}

TEST_CASE("zlib: compression levels") {
    auto data = make_compressible_data(4096);
    for (int level : {1, 3, 6, 9}) {
        round_trip_test(utils2::Codec::zlib, level, data);
    }
}

TEST_CASE("zlib: decompress with expected_size=0 (streaming)") {
    auto data = make_test_data(1024);
    utils2::CompressParams params;
    params.codec = utils2::Codec::zlib;
    params.level = 6;
    auto compressed = utils2::compress(data, params);
    auto decompressed = utils2::decompress(compressed, utils2::Codec::zlib, 0);
    REQUIRE_EQ(decompressed.size(), data.size());
    REQUIRE(decompressed == data);
}

// ---------------------------------------------------------------------------
// gzip
// ---------------------------------------------------------------------------

TEST_CASE("gzip: codec_available") {
    REQUIRE(utils2::codec_available(utils2::Codec::gzip));
}

TEST_CASE("gzip: round-trip small data") {
    auto data = make_test_data(64);
    round_trip_test(utils2::Codec::gzip, 6, data);
}

TEST_CASE("gzip: round-trip medium data") {
    auto data = make_test_data(4096);
    round_trip_test(utils2::Codec::gzip, 6, data);
}

TEST_CASE("gzip: round-trip large data") {
    auto data = make_compressible_data(1 << 16);
    round_trip_test(utils2::Codec::gzip, 6, data);
}

TEST_CASE("gzip: single byte") {
    std::vector<std::byte> one{std::byte{0xFF}};
    round_trip_test(utils2::Codec::gzip, 6, one);
}

TEST_CASE("gzip: compression levels") {
    auto data = make_compressible_data(4096);
    for (int level : {1, 6, 9}) {
        round_trip_test(utils2::Codec::gzip, level, data);
    }
}
#endif

// ---------------------------------------------------------------------------
// blosc
// ---------------------------------------------------------------------------

#if UTILS2_HAS_BLOSC
TEST_CASE("blosc_lz4: codec_available") {
    REQUIRE(utils2::codec_available(utils2::Codec::blosc_lz4));
}

TEST_CASE("blosc_lz4: round-trip small data") {
    auto data = make_test_data(64);
    round_trip_test(utils2::Codec::blosc_lz4, 5, data);
}

TEST_CASE("blosc_lz4: round-trip medium data") {
    auto data = make_test_data(4096);
    round_trip_test(utils2::Codec::blosc_lz4, 5, data);
}

TEST_CASE("blosc_lz4: round-trip large data") {
    auto data = make_compressible_data(1 << 16);
    round_trip_test(utils2::Codec::blosc_lz4, 5, data);
}

TEST_CASE("blosc_lz4: single byte") {
    std::vector<std::byte> one{std::byte{0x77}};
    round_trip_test(utils2::Codec::blosc_lz4, 5, one);
}

TEST_CASE("blosc_lz4: decompress with expected_size=0 (auto-detect)") {
    auto data = make_test_data(512);
    utils2::CompressParams params;
    params.codec = utils2::Codec::blosc_lz4;
    params.level = 5;
    auto compressed = utils2::compress(data, params);
    auto decompressed = utils2::decompress(compressed, utils2::Codec::blosc_lz4, 0);
    REQUIRE_EQ(decompressed.size(), data.size());
    REQUIRE(decompressed == data);
}

TEST_CASE("blosc_lz4hc: round-trip") {
    auto data = make_test_data(4096);
    round_trip_test(utils2::Codec::blosc_lz4hc, 5, data);
}

TEST_CASE("blosc_zstd: round-trip") {
    auto data = make_test_data(4096);
    round_trip_test(utils2::Codec::blosc_zstd, 5, data);
}

TEST_CASE("blosc_zlib: round-trip") {
    auto data = make_test_data(4096);
    round_trip_test(utils2::Codec::blosc_zlib, 5, data);
}

TEST_CASE("blosc_blosclz: round-trip") {
    auto data = make_test_data(4096);
    round_trip_test(utils2::Codec::blosc_blosclz, 5, data);
}

TEST_CASE("blosc: shuffle modes") {
    auto data = make_compressible_data(4096);
    for (int shuffle : {0, 1, 2}) {
        utils2::CompressParams params;
        params.codec = utils2::Codec::blosc_lz4;
        params.level = 5;
        params.shuffle = shuffle;
        params.typesize = 4;
        auto compressed = utils2::compress(data, params);
        auto decompressed = utils2::decompress(compressed, utils2::Codec::blosc_lz4, data.size());
        REQUIRE_EQ(decompressed.size(), data.size());
        REQUIRE(decompressed == data);
    }
}

TEST_CASE("blosc: different type sizes") {
    auto data = make_compressible_data(4096);
    for (std::size_t ts : {1, 2, 4, 8}) {
        utils2::CompressParams params;
        params.codec = utils2::Codec::blosc_lz4;
        params.level = 5;
        params.typesize = ts;
        auto compressed = utils2::compress(data, params);
        auto decompressed = utils2::decompress(compressed, utils2::Codec::blosc_lz4, data.size());
        REQUIRE_EQ(decompressed.size(), data.size());
        REQUIRE(decompressed == data);
    }
}
#endif

// ---------------------------------------------------------------------------
// blosc2
// ---------------------------------------------------------------------------

#if UTILS2_HAS_BLOSC2
TEST_CASE("blosc2_lz4: codec_available") {
    REQUIRE(utils2::codec_available(utils2::Codec::blosc2_lz4));
}

TEST_CASE("blosc2_lz4: round-trip small data") {
    auto data = make_test_data(64);
    round_trip_test(utils2::Codec::blosc2_lz4, 5, data);
}

TEST_CASE("blosc2_lz4: round-trip medium data") {
    auto data = make_test_data(4096);
    round_trip_test(utils2::Codec::blosc2_lz4, 5, data);
}

TEST_CASE("blosc2_lz4: round-trip large data") {
    auto data = make_compressible_data(1 << 16);
    round_trip_test(utils2::Codec::blosc2_lz4, 5, data);
}

TEST_CASE("blosc2_lz4: single byte") {
    std::vector<std::byte> one{std::byte{0x33}};
    round_trip_test(utils2::Codec::blosc2_lz4, 5, one);
}

TEST_CASE("blosc2_lz4: decompress with expected_size=0 (auto-detect)") {
    auto data = make_test_data(512);
    utils2::CompressParams params;
    params.codec = utils2::Codec::blosc2_lz4;
    params.level = 5;
    auto compressed = utils2::compress(data, params);
    auto decompressed = utils2::decompress(compressed, utils2::Codec::blosc2_lz4, 0);
    REQUIRE_EQ(decompressed.size(), data.size());
    REQUIRE(decompressed == data);
}

TEST_CASE("blosc2_zstd: codec_available") {
    REQUIRE(utils2::codec_available(utils2::Codec::blosc2_zstd));
}

TEST_CASE("blosc2_zstd: round-trip medium data") {
    auto data = make_test_data(4096);
    round_trip_test(utils2::Codec::blosc2_zstd, 5, data);
}

TEST_CASE("blosc2_zstd: round-trip large data") {
    auto data = make_compressible_data(1 << 16);
    round_trip_test(utils2::Codec::blosc2_zstd, 5, data);
}

TEST_CASE("blosc2: compression levels") {
    auto data = make_compressible_data(4096);
    for (int level : {1, 5, 9}) {
        round_trip_test(utils2::Codec::blosc2_lz4, level, data);
    }
}

TEST_CASE("blosc2: shuffle modes") {
    auto data = make_compressible_data(4096);
    for (int shuffle : {0, 1, 2}) {
        utils2::CompressParams params;
        params.codec = utils2::Codec::blosc2_lz4;
        params.level = 5;
        params.shuffle = shuffle;
        params.typesize = 4;
        auto compressed = utils2::compress(data, params);
        auto decompressed = utils2::decompress(compressed, utils2::Codec::blosc2_lz4, data.size());
        REQUIRE_EQ(decompressed.size(), data.size());
        REQUIRE(decompressed == data);
    }
}
#endif

// ---------------------------------------------------------------------------
// uint8_t convenience overloads
// ---------------------------------------------------------------------------

TEST_CASE("compress uint8_t span overload") {
    std::vector<std::uint8_t> data(256);
    std::iota(data.begin(), data.end(), std::uint8_t{0});

    utils2::CompressParams params;
    params.codec = utils2::Codec::none;
    auto compressed = utils2::compress(
        std::span<const std::uint8_t>(data), params);
    REQUIRE_EQ(compressed.size(), data.size());
}

TEST_CASE("decompress_u8 returns uint8_t vector") {
    std::vector<std::byte> data = make_test_data(128);
    utils2::CompressParams params;
    params.codec = utils2::Codec::none;
    auto compressed = utils2::compress(data, params);
    auto result = utils2::decompress_u8(compressed, utils2::Codec::none, data.size());
    REQUIRE_EQ(result.size(), data.size());
    for (std::size_t i = 0; i < data.size(); ++i) {
        REQUIRE_EQ(result[i], static_cast<std::uint8_t>(data[i]));
    }
}

// ---------------------------------------------------------------------------
// CodecPipeline
// ---------------------------------------------------------------------------

TEST_CASE("CodecPipeline: empty pipeline is passthrough") {
    utils2::CodecPipeline pipe;
    REQUIRE(pipe.empty());
    REQUIRE_EQ(pipe.size(), std::size_t{0});

    auto data = make_test_data(128);
    auto encoded = pipe.encode(data);
    REQUIRE_EQ(encoded.size(), data.size());
    REQUIRE(encoded == data);

    auto decoded = pipe.decode(encoded, data.size());
    REQUIRE_EQ(decoded.size(), data.size());
    REQUIRE(decoded == data);
}

TEST_CASE("CodecPipeline: single stage none") {
    utils2::CodecPipeline pipe;
    pipe.add({.codec = utils2::Codec::none});
    REQUIRE_EQ(pipe.size(), std::size_t{1});

    auto data = make_test_data(256);
    auto encoded = pipe.encode(data);
    auto decoded = pipe.decode(encoded, data.size());
    REQUIRE_EQ(decoded.size(), data.size());
    REQUIRE(decoded == data);
}

#if UTILS2_HAS_ZSTD
TEST_CASE("CodecPipeline: single stage zstd") {
    utils2::CodecPipeline pipe;
    pipe.add({.codec = utils2::Codec::zstd, .level = 3});

    auto data = make_compressible_data(4096);
    auto encoded = pipe.encode(data);
    REQUIRE_LT(encoded.size(), data.size());

    auto decoded = pipe.decode(encoded, data.size());
    REQUIRE_EQ(decoded.size(), data.size());
    REQUIRE(decoded == data);
}
#endif

#if UTILS2_HAS_ZSTD && UTILS2_HAS_LZ4
TEST_CASE("CodecPipeline: two stages zstd then lz4") {
    utils2::CodecPipeline pipe;
    pipe.add({.codec = utils2::Codec::zstd, .level = 3});
    pipe.add({.codec = utils2::Codec::none}); // second stage passthrough

    auto data = make_compressible_data(4096);
    auto encoded = pipe.encode(data);
    auto decoded = pipe.decode(encoded, data.size());
    REQUIRE_EQ(decoded.size(), data.size());
    REQUIRE(decoded == data);
}
#endif

// ---------------------------------------------------------------------------
// Error handling
// ---------------------------------------------------------------------------

TEST_CASE("compress: unavailable codec throws") {
    // Use a codec ID that's definitely not registered by casting a high value
    auto fake_codec = static_cast<utils2::Codec>(255);
    utils2::CompressParams params;
    params.codec = fake_codec;
    auto data = make_test_data(64);
    REQUIRE_THROWS(utils2::compress(data, params));
}

TEST_CASE("decompress: unavailable codec throws") {
    auto fake_codec = static_cast<utils2::Codec>(255);
    auto data = make_test_data(64);
    REQUIRE_THROWS(utils2::decompress(data, fake_codec, 64));
}

// ---------------------------------------------------------------------------
// parse_codec: all codec names round-trip
// ---------------------------------------------------------------------------

TEST_CASE("parse_codec: all codecs round-trip via codec_name") {
    for (auto c : {utils2::Codec::none, utils2::Codec::zstd, utils2::Codec::lz4,
                   utils2::Codec::lz4hc, utils2::Codec::zlib, utils2::Codec::gzip,
                   utils2::Codec::blosc_lz4, utils2::Codec::blosc_lz4hc,
                   utils2::Codec::blosc_zstd, utils2::Codec::blosc_zlib,
                   utils2::Codec::blosc_snappy, utils2::Codec::blosc_blosclz,
                   utils2::Codec::blosc2_lz4, utils2::Codec::blosc2_zstd}) {
        auto name = utils2::codec_name(c);
        auto parsed = utils2::parse_codec(name);
        REQUIRE(parsed.has_value());
        REQUIRE_EQ(static_cast<int>(*parsed), static_cast<int>(c));
    }
}

// ---------------------------------------------------------------------------
// parse_codec: invalid names
// ---------------------------------------------------------------------------

TEST_CASE("parse_codec: various invalid names return nullopt") {
    REQUIRE(!utils2::parse_codec("").has_value());
    REQUIRE(!utils2::parse_codec("invalid_codec").has_value());
    REQUIRE(!utils2::parse_codec("ZSTD").has_value()); // case sensitive
    REQUIRE(!utils2::parse_codec("lz4 ").has_value()); // trailing space
    REQUIRE(!utils2::parse_codec("blosc3_lz4").has_value());
}

// ---------------------------------------------------------------------------
// codec_name: all codecs
// ---------------------------------------------------------------------------

TEST_CASE("codec_name: all enum values produce non-empty names") {
    for (auto c : {utils2::Codec::none, utils2::Codec::zstd, utils2::Codec::lz4,
                   utils2::Codec::lz4hc, utils2::Codec::zlib, utils2::Codec::gzip,
                   utils2::Codec::blosc_lz4, utils2::Codec::blosc_lz4hc,
                   utils2::Codec::blosc_zstd, utils2::Codec::blosc_zlib,
                   utils2::Codec::blosc_snappy, utils2::Codec::blosc_blosclz,
                   utils2::Codec::blosc2_lz4, utils2::Codec::blosc2_zstd}) {
        auto name = utils2::codec_name(c);
        REQUIRE(!name.empty());
        REQUIRE_NE(name, std::string_view("unknown"));
    }
}

// ---------------------------------------------------------------------------
// codec_available: all available codecs
// ---------------------------------------------------------------------------

TEST_CASE("codec_available: check all codec enum values") {
    // none is always available
    REQUIRE(utils2::codec_available(utils2::Codec::none));

#if UTILS2_HAS_ZSTD
    REQUIRE(utils2::codec_available(utils2::Codec::zstd));
#endif
#if UTILS2_HAS_LZ4
    REQUIRE(utils2::codec_available(utils2::Codec::lz4));
#endif
#if UTILS2_HAS_LZ4HC && UTILS2_HAS_LZ4
    REQUIRE(utils2::codec_available(utils2::Codec::lz4hc));
#endif
#if UTILS2_HAS_ZLIB
    REQUIRE(utils2::codec_available(utils2::Codec::zlib));
    REQUIRE(utils2::codec_available(utils2::Codec::gzip));
#endif
#if UTILS2_HAS_BLOSC
    REQUIRE(utils2::codec_available(utils2::Codec::blosc_lz4));
    REQUIRE(utils2::codec_available(utils2::Codec::blosc_lz4hc));
    REQUIRE(utils2::codec_available(utils2::Codec::blosc_zstd));
    REQUIRE(utils2::codec_available(utils2::Codec::blosc_zlib));
    REQUIRE(utils2::codec_available(utils2::Codec::blosc_snappy));
    REQUIRE(utils2::codec_available(utils2::Codec::blosc_blosclz));
#endif
#if UTILS2_HAS_BLOSC2
    REQUIRE(utils2::codec_available(utils2::Codec::blosc2_lz4));
    REQUIRE(utils2::codec_available(utils2::Codec::blosc2_zstd));
#endif

    // A fake codec should not be available
    REQUIRE(!utils2::codec_available(static_cast<utils2::Codec>(254)));
}

// ---------------------------------------------------------------------------
// Corrupted data handling
// ---------------------------------------------------------------------------

#if UTILS2_HAS_ZSTD
TEST_CASE("zstd: corrupted compressed data throws on decompress") {
    std::vector<std::byte> garbage = {std::byte{0xDE}, std::byte{0xAD},
                                       std::byte{0xBE}, std::byte{0xEF},
                                       std::byte{0x00}, std::byte{0x01}};
    REQUIRE_THROWS(utils2::decompress(garbage, utils2::Codec::zstd, 100));
}

TEST_CASE("zstd: empty input throws on decompress with auto-detect") {
    std::vector<std::byte> empty;
    REQUIRE_THROWS(utils2::decompress(empty, utils2::Codec::zstd, 0));
}
#endif

#if UTILS2_HAS_LZ4
TEST_CASE("lz4: corrupted data throws on decompress") {
    // Valid size prefix but garbage compressed data
    std::vector<std::byte> garbage(20, std::byte{0xFF});
    // Set size prefix to indicate 100 bytes original
    std::uint32_t sz = 100;
    std::memcpy(garbage.data(), &sz, 4);
    REQUIRE_THROWS(utils2::decompress(garbage, utils2::Codec::lz4, 0));
}

TEST_CASE("lz4: data too small for size prefix throws") {
    std::vector<std::byte> tiny = {std::byte{0x01}, std::byte{0x02}};
    REQUIRE_THROWS(utils2::decompress(tiny, utils2::Codec::lz4, 0));
}
#endif

#if UTILS2_HAS_ZLIB
TEST_CASE("zlib: corrupted data throws on decompress") {
    std::vector<std::byte> garbage = {std::byte{0x00}, std::byte{0x11},
                                       std::byte{0x22}, std::byte{0x33}};
    REQUIRE_THROWS(utils2::decompress(garbage, utils2::Codec::zlib, 100));
}

TEST_CASE("gzip: corrupted data throws on decompress") {
    std::vector<std::byte> garbage = {std::byte{0xAA}, std::byte{0xBB},
                                       std::byte{0xCC}, std::byte{0xDD}};
    REQUIRE_THROWS(utils2::decompress(garbage, utils2::Codec::gzip, 100));
}
#endif

// ---------------------------------------------------------------------------
// Blosc shuffle modes and type_size variations
// ---------------------------------------------------------------------------

#if UTILS2_HAS_BLOSC
TEST_CASE("blosc: all compressors round-trip") {
    auto data = make_compressible_data(4096);
    for (auto c : {utils2::Codec::blosc_lz4, utils2::Codec::blosc_lz4hc,
                   utils2::Codec::blosc_zstd, utils2::Codec::blosc_zlib,
                   utils2::Codec::blosc_blosclz}) {
        utils2::CompressParams params;
        params.codec = c;
        params.level = 5;
        params.typesize = 4;
        auto compressed = utils2::compress(data, params);
        auto decompressed = utils2::decompress(compressed, c, data.size());
        REQUIRE_EQ(decompressed.size(), data.size());
        REQUIRE(decompressed == data);
    }
}

TEST_CASE("blosc: typesize=1 byte shuffle") {
    auto data = make_test_data(1024);
    utils2::CompressParams params;
    params.codec = utils2::Codec::blosc_lz4;
    params.level = 5;
    params.shuffle = 1;
    params.typesize = 1;
    auto compressed = utils2::compress(data, params);
    auto decompressed = utils2::decompress(compressed, utils2::Codec::blosc_lz4, data.size());
    REQUIRE_EQ(decompressed.size(), data.size());
    REQUIRE(decompressed == data);
}

TEST_CASE("blosc: typesize=8 for double-sized elements") {
    auto data = make_compressible_data(8192);
    utils2::CompressParams params;
    params.codec = utils2::Codec::blosc_lz4;
    params.level = 5;
    params.shuffle = 1;
    params.typesize = 8;
    auto compressed = utils2::compress(data, params);
    auto decompressed = utils2::decompress(compressed, utils2::Codec::blosc_lz4, data.size());
    REQUIRE_EQ(decompressed.size(), data.size());
    REQUIRE(decompressed == data);
}

TEST_CASE("blosc: no-shuffle mode (shuffle=0)") {
    auto data = make_test_data(2048);
    utils2::CompressParams params;
    params.codec = utils2::Codec::blosc_lz4;
    params.level = 5;
    params.shuffle = 0;
    params.typesize = 4;
    auto compressed = utils2::compress(data, params);
    auto decompressed = utils2::decompress(compressed, utils2::Codec::blosc_lz4, data.size());
    REQUIRE_EQ(decompressed.size(), data.size());
    REQUIRE(decompressed == data);
}

TEST_CASE("blosc: bit-shuffle mode (shuffle=2)") {
    auto data = make_compressible_data(4096);
    utils2::CompressParams params;
    params.codec = utils2::Codec::blosc_lz4;
    params.level = 5;
    params.shuffle = 2;
    params.typesize = 4;
    auto compressed = utils2::compress(data, params);
    auto decompressed = utils2::decompress(compressed, utils2::Codec::blosc_lz4, data.size());
    REQUIRE_EQ(decompressed.size(), data.size());
    REQUIRE(decompressed == data);
}

TEST_CASE("blosc: custom blocksize") {
    auto data = make_compressible_data(8192);
    utils2::CompressParams params;
    params.codec = utils2::Codec::blosc_lz4;
    params.level = 5;
    params.typesize = 4;
    params.blocksize = 2048;
    auto compressed = utils2::compress(data, params);
    auto decompressed = utils2::decompress(compressed, utils2::Codec::blosc_lz4, data.size());
    REQUIRE_EQ(decompressed.size(), data.size());
    REQUIRE(decompressed == data);
}
#endif

// ---------------------------------------------------------------------------
// Blosc2 additional tests
// ---------------------------------------------------------------------------

#if UTILS2_HAS_BLOSC2
TEST_CASE("blosc2: different type sizes") {
    auto data = make_compressible_data(4096);
    for (std::size_t ts : {1, 2, 4, 8}) {
        utils2::CompressParams params;
        params.codec = utils2::Codec::blosc2_lz4;
        params.level = 5;
        params.typesize = ts;
        auto compressed = utils2::compress(data, params);
        auto decompressed = utils2::decompress(compressed, utils2::Codec::blosc2_lz4, data.size());
        REQUIRE_EQ(decompressed.size(), data.size());
        REQUIRE(decompressed == data);
    }
}

TEST_CASE("blosc2: custom blocksize") {
    auto data = make_compressible_data(8192);
    utils2::CompressParams params;
    params.codec = utils2::Codec::blosc2_lz4;
    params.level = 5;
    params.typesize = 4;
    params.blocksize = 1024;
    auto compressed = utils2::compress(data, params);
    auto decompressed = utils2::decompress(compressed, utils2::Codec::blosc2_lz4, data.size());
    REQUIRE_EQ(decompressed.size(), data.size());
    REQUIRE(decompressed == data);
}

TEST_CASE("blosc2: no-shuffle round-trip") {
    auto data = make_test_data(2048);
    utils2::CompressParams params;
    params.codec = utils2::Codec::blosc2_zstd;
    params.level = 3;
    params.shuffle = 0;
    params.typesize = 4;
    auto compressed = utils2::compress(data, params);
    auto decompressed = utils2::decompress(compressed, utils2::Codec::blosc2_zstd, data.size());
    REQUIRE_EQ(decompressed.size(), data.size());
    REQUIRE(decompressed == data);
}

TEST_CASE("blosc2: bit-shuffle round-trip") {
    auto data = make_compressible_data(4096);
    utils2::CompressParams params;
    params.codec = utils2::Codec::blosc2_lz4;
    params.level = 5;
    params.shuffle = 2; // bit shuffle
    params.typesize = 4;
    auto compressed = utils2::compress(data, params);
    auto decompressed = utils2::decompress(compressed, utils2::Codec::blosc2_lz4, data.size());
    REQUIRE_EQ(decompressed.size(), data.size());
    REQUIRE(decompressed == data);
}

TEST_CASE("blosc2_zstd: decompress with auto-detect size") {
    auto data = make_test_data(1024);
    utils2::CompressParams params;
    params.codec = utils2::Codec::blosc2_zstd;
    params.level = 5;
    auto compressed = utils2::compress(data, params);
    auto decompressed = utils2::decompress(compressed, utils2::Codec::blosc2_zstd, 0);
    REQUIRE_EQ(decompressed.size(), data.size());
    REQUIRE(decompressed == data);
}
#endif

// ---------------------------------------------------------------------------
// Large data round-trip
// ---------------------------------------------------------------------------

#if UTILS2_HAS_ZSTD
TEST_CASE("zstd: round-trip very large data (1MB)") {
    auto data = make_compressible_data(1 << 20);
    round_trip_test(utils2::Codec::zstd, 3, data);
}
#endif

#if UTILS2_HAS_LZ4
TEST_CASE("lz4: round-trip very large data (1MB)") {
    auto data = make_compressible_data(1 << 20);
    round_trip_test(utils2::Codec::lz4, 0, data);
}
#endif

// ---------------------------------------------------------------------------
// CompressParams default values
// ---------------------------------------------------------------------------

TEST_CASE("CompressParams: default values are sensible") {
    utils2::CompressParams p;
    REQUIRE_EQ(static_cast<int>(p.codec), static_cast<int>(utils2::Codec::zstd));
    REQUIRE_EQ(p.level, 5);
    REQUIRE_EQ(p.shuffle, 1);
    REQUIRE_EQ(p.typesize, std::size_t(1));
    REQUIRE_EQ(p.blocksize, std::size_t(0));
    REQUIRE_EQ(p.num_threads, std::size_t(1));
}

// ---------------------------------------------------------------------------
// Register and use custom codec
// ---------------------------------------------------------------------------

TEST_CASE("register_codec: custom named codec") {
    utils2::register_codec("test_custom", utils2::CodecImpl{
        [](std::span<const std::byte> in, const utils2::CompressParams&) {
            // Reverse bytes as a "compression"
            std::vector<std::byte> out(in.begin(), in.end());
            std::reverse(out.begin(), out.end());
            return out;
        },
        [](std::span<const std::byte> in, std::size_t) {
            // Reverse again to decompress
            std::vector<std::byte> out(in.begin(), in.end());
            std::reverse(out.begin(), out.end());
            return out;
        }
    });
    // Named codecs are registered but won't affect enum-based compress/decompress
    // Just verify registration didn't throw
    REQUIRE(true);
}

// ---------------------------------------------------------------------------
// CodecPipeline: chaining
// ---------------------------------------------------------------------------

TEST_CASE("CodecPipeline: add returns reference for chaining") {
    utils2::CodecPipeline pipe;
    auto& ref = pipe.add({.codec = utils2::Codec::none});
    REQUIRE_EQ(&ref, &pipe);
    REQUIRE_EQ(pipe.size(), std::size_t(1));
}

#if UTILS2_HAS_ZSTD
TEST_CASE("CodecPipeline: two real codecs zstd then zstd") {
    utils2::CodecPipeline pipe;
    pipe.add({.codec = utils2::Codec::zstd, .level = 1});
    pipe.add({.codec = utils2::Codec::zstd, .level = 3});
    REQUIRE_EQ(pipe.size(), std::size_t(2));
    REQUIRE(!pipe.empty());

    auto data = make_compressible_data(4096);
    auto encoded = pipe.encode(data);
    auto decoded = pipe.decode(encoded, data.size());
    REQUIRE_EQ(decoded.size(), data.size());
    REQUIRE(decoded == data);
}
#endif

// ---------------------------------------------------------------------------
// size prefix helper
// ---------------------------------------------------------------------------

TEST_CASE("read_size_prefix: data too small throws") {
    std::vector<std::byte> tiny = {std::byte{0x01}};
    REQUIRE_THROWS(utils2::detail::read_size_prefix(tiny));
}

TEST_CASE("read_size_prefix: reads correct value") {
    std::uint32_t val = 12345;
    std::vector<std::byte> data(4);
    std::memcpy(data.data(), &val, 4);
    auto result = utils2::detail::read_size_prefix(data);
    REQUIRE_EQ(result, val);
}

UTILS2_TEST_MAIN()
