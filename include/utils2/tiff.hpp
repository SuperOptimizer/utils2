#pragma once
#include <vector>
#include <array>
#include <span>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <filesystem>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <algorithm>
#include <bit>
#include <concepts>
#include <cstring>
#include <utility>

namespace utils2 {

// ---------------------------------------------------------------------------
// TiffImageInfo
// ---------------------------------------------------------------------------

struct TiffImageInfo {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint16_t bits_per_sample = 8;
    std::uint16_t samples_per_pixel = 1;  // 1=grayscale, 3=RGB, 4=RGBA
    std::uint16_t compression = 1;         // 1=none, 5=LZW, 32773=PackBits
    std::uint16_t photometric = 1;         // 0=WhiteIsZero, 1=BlackIsZero, 2=RGB
    std::uint32_t rows_per_strip = 0;
    std::uint16_t sample_format = 1;       // 1=uint, 2=int, 3=float
    std::uint16_t planar_config = 1;       // 1=chunky (RGBRGB), 2=planar

    [[nodiscard]] std::size_t pixel_bytes() const noexcept {
        return static_cast<std::size_t>(bits_per_sample / 8) * samples_per_pixel;
    }
    [[nodiscard]] std::size_t row_bytes() const noexcept {
        return static_cast<std::size_t>(width) * pixel_bytes();
    }
    [[nodiscard]] std::size_t image_bytes() const noexcept {
        return static_cast<std::size_t>(height) * row_bytes();
    }
};

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

namespace detail::tiff {

// TIFF tag constants
inline constexpr std::uint16_t tag_image_width        = 256;
inline constexpr std::uint16_t tag_image_length       = 257;
inline constexpr std::uint16_t tag_bits_per_sample    = 258;
inline constexpr std::uint16_t tag_compression        = 259;
inline constexpr std::uint16_t tag_photometric        = 262;
inline constexpr std::uint16_t tag_strip_offsets      = 273;
inline constexpr std::uint16_t tag_samples_per_pixel  = 277;
inline constexpr std::uint16_t tag_rows_per_strip     = 278;
inline constexpr std::uint16_t tag_strip_byte_counts  = 279;
inline constexpr std::uint16_t tag_planar_config      = 284;
inline constexpr std::uint16_t tag_sample_format      = 339;

// TIFF type IDs and sizes
inline constexpr std::uint16_t type_byte   = 1;  // 1 byte
inline constexpr std::uint16_t type_short  = 3;  // 2 bytes
inline constexpr std::uint16_t type_long   = 4;  // 4 bytes

[[nodiscard]] inline std::size_t type_size(std::uint16_t t) noexcept {
    switch (t) {
        case 1:  return 1;  // BYTE
        case 2:  return 1;  // ASCII
        case 3:  return 2;  // SHORT
        case 4:  return 4;  // LONG
        case 5:  return 8;  // RATIONAL
        default: return 1;
    }
}

// Endian-aware read helpers operating on raw bytes
struct ByteReader {
    const std::uint8_t* data;
    std::size_t size;
    bool big_endian;

    [[nodiscard]] std::uint16_t u16(std::size_t off) const {
        if (off + 2 > size) throw std::runtime_error("TIFF: read out of bounds");
        std::uint16_t v;
        std::memcpy(&v, data + off, 2);
        if (big_endian && std::endian::native == std::endian::little)
            v = std::byteswap(v);
        else if (!big_endian && std::endian::native == std::endian::big)
            v = std::byteswap(v);
        return v;
    }

    [[nodiscard]] std::uint32_t u32(std::size_t off) const {
        if (off + 4 > size) throw std::runtime_error("TIFF: read out of bounds");
        std::uint32_t v;
        std::memcpy(&v, data + off, 4);
        if (big_endian && std::endian::native == std::endian::little)
            v = std::byteswap(v);
        else if (!big_endian && std::endian::native == std::endian::big)
            v = std::byteswap(v);
        return v;
    }

    // Read a tag value which may be inlined or at an offset
    [[nodiscard]] std::uint32_t tag_value(std::size_t entry_off) const {
        auto type  = u16(entry_off + 2);
        auto count = u32(entry_off + 4);
        std::size_t total = count * type_size(type);
        if (total <= 4) {
            // Value is inlined in the offset field
            if (type == type_short) return u16(entry_off + 8);
            if (type == type_byte)  return data[entry_off + 8];
            return u32(entry_off + 8);
        }
        // Value is at an offset
        auto off = u32(entry_off + 8);
        if (type == type_short) return u16(off);
        return u32(off);
    }

    // Read an array of LONG or SHORT values
    [[nodiscard]] std::vector<std::uint32_t> tag_array(std::size_t entry_off) const {
        auto type  = u16(entry_off + 2);
        auto count = u32(entry_off + 4);
        std::size_t total = count * type_size(type);

        std::size_t val_off = (total <= 4) ? entry_off + 8 : u32(entry_off + 8);

        std::vector<std::uint32_t> out(count);
        for (std::uint32_t i = 0; i < count; ++i) {
            if (type == type_short)
                out[i] = u16(val_off + i * 2);
            else
                out[i] = u32(val_off + i * 4);
        }
        return out;
    }
};

// IFD parsed from file bytes
struct IfdData {
    TiffImageInfo info;
    std::vector<std::uint32_t> strip_offsets;
    std::vector<std::uint32_t> strip_byte_counts;
};

[[nodiscard]] inline IfdData parse_ifd(const ByteReader& r, std::size_t ifd_off) {
    IfdData ifd{};
    auto num_entries = r.u16(ifd_off);
    std::size_t entry_base = ifd_off + 2;

    for (std::uint16_t i = 0; i < num_entries; ++i) {
        std::size_t e = entry_base + static_cast<std::size_t>(i) * 12;
        auto tag = r.u16(e);

        switch (tag) {
            case tag_image_width:       ifd.info.width            = r.tag_value(e); break;
            case tag_image_length:      ifd.info.height           = r.tag_value(e); break;
            case tag_bits_per_sample:   ifd.info.bits_per_sample  = static_cast<std::uint16_t>(r.tag_value(e)); break;
            case tag_compression:       ifd.info.compression      = static_cast<std::uint16_t>(r.tag_value(e)); break;
            case tag_photometric:       ifd.info.photometric      = static_cast<std::uint16_t>(r.tag_value(e)); break;
            case tag_samples_per_pixel: ifd.info.samples_per_pixel = static_cast<std::uint16_t>(r.tag_value(e)); break;
            case tag_rows_per_strip:    ifd.info.rows_per_strip   = r.tag_value(e); break;
            case tag_planar_config:     ifd.info.planar_config    = static_cast<std::uint16_t>(r.tag_value(e)); break;
            case tag_sample_format:     ifd.info.sample_format    = static_cast<std::uint16_t>(r.tag_value(e)); break;
            case tag_strip_offsets:     ifd.strip_offsets         = r.tag_array(e); break;
            case tag_strip_byte_counts: ifd.strip_byte_counts     = r.tag_array(e); break;
            default: break;
        }
    }

    if (ifd.info.rows_per_strip == 0)
        ifd.info.rows_per_strip = ifd.info.height;

    return ifd;
}

// ---------------------------------------------------------------------------
// LZW decompression (TIFF variant: MSB-first bit packing)
// ---------------------------------------------------------------------------

inline void decompress_lzw(const std::uint8_t* src, std::size_t src_len,
                           std::vector<std::uint8_t>& out) {
    constexpr int clear_code = 256;
    constexpr int eoi_code   = 257;
    constexpr int first_code = 258;
    constexpr int max_code   = 4093;

    // Dictionary: each entry is (prefix_index, append_byte)
    // We reconstruct strings by chasing prefix chains.
    struct Entry {
        int prefix;           // -1 for single-byte entries
        std::uint8_t value;
        std::uint16_t length; // total decoded length
    };

    std::vector<Entry> table;
    table.reserve(4096);

    auto reset_table = [&] {
        table.clear();
        for (int i = 0; i < 256; ++i)
            table.push_back({-1, static_cast<std::uint8_t>(i), 1});
        table.push_back({-1, 0, 0}); // clear code placeholder
        table.push_back({-1, 0, 0}); // eoi code placeholder
    };

    // Decode a table entry into a byte sequence appended to out
    auto decode_string = [&](int code) {
        auto len = table[static_cast<std::size_t>(code)].length;
        auto pos = out.size();
        out.resize(pos + len);
        auto idx = code;
        for (int i = len - 1; i >= 0; --i) {
            out[pos + static_cast<std::size_t>(i)] = table[static_cast<std::size_t>(idx)].value;
            idx = table[static_cast<std::size_t>(idx)].prefix;
        }
    };

    auto first_char = [&](int code) -> std::uint8_t {
        while (table[static_cast<std::size_t>(code)].prefix != -1)
            code = table[static_cast<std::size_t>(code)].prefix;
        return table[static_cast<std::size_t>(code)].value;
    };

    // Bit reader: MSB-first (TIFF LZW convention)
    std::size_t bit_pos = 0;
    auto read_bits = [&](int n) -> int {
        int result = 0;
        for (int i = 0; i < n; ++i) {
            std::size_t byte_idx = (bit_pos + static_cast<std::size_t>(i)) / 8;
            int bit_idx = 7 - static_cast<int>((bit_pos + static_cast<std::size_t>(i)) % 8);
            if (byte_idx < src_len)
                result = (result << 1) | ((src[byte_idx] >> bit_idx) & 1);
            else
                result <<= 1;
        }
        bit_pos += static_cast<std::size_t>(n);
        return result;
    };

    reset_table();
    int code_size = 9;
    int next_code = first_code;

    int code = read_bits(code_size);
    if (code != clear_code) return; // first code must be clear

    reset_table();
    code_size = 9;
    next_code = first_code;

    int old_code = read_bits(code_size);
    if (old_code == eoi_code) return;
    decode_string(old_code);

    while (true) {
        code = read_bits(code_size);
        if (code == eoi_code) break;

        if (code == clear_code) {
            reset_table();
            code_size = 9;
            next_code = first_code;
            old_code = read_bits(code_size);
            if (old_code == eoi_code) break;
            decode_string(old_code);
            continue;
        }

        if (code < static_cast<int>(table.size())) {
            decode_string(code);
            if (next_code <= max_code) {
                auto fc = first_char(code);
                auto plen = table[static_cast<std::size_t>(old_code)].length;
                table.push_back({old_code, fc, static_cast<std::uint16_t>(plen + 1)});
                ++next_code;
            }
        } else {
            // code == next_code (the special KwKwK case)
            auto fc = first_char(old_code);
            auto plen = table[static_cast<std::size_t>(old_code)].length;
            table.push_back({old_code, fc, static_cast<std::uint16_t>(plen + 1)});
            ++next_code;
            decode_string(next_code - 1);
        }

        if (next_code > (1 << code_size) - 1 && code_size < 12)
            ++code_size;

        old_code = code;
    }
}

// ---------------------------------------------------------------------------
// PackBits decompression
// ---------------------------------------------------------------------------

inline void decompress_packbits(const std::uint8_t* src, std::size_t src_len,
                                std::vector<std::uint8_t>& out) {
    std::size_t pos = 0;
    while (pos < src_len) {
        auto n = static_cast<std::int8_t>(src[pos++]);
        if (n >= 0) {
            // Copy next n+1 bytes literally
            std::size_t count = static_cast<std::size_t>(n) + 1;
            if (pos + count > src_len) break;
            out.insert(out.end(), src + pos, src + pos + count);
            pos += count;
        } else if (n == -128) {
            // No-op
        } else {
            // Repeat next byte (1 - n) times
            if (pos >= src_len) break;
            std::size_t count = static_cast<std::size_t>(1 - n);
            auto val = src[pos++];
            out.insert(out.end(), count, val);
        }
    }
}

// ---------------------------------------------------------------------------
// Decompress a strip based on compression type
// ---------------------------------------------------------------------------

inline std::vector<std::uint8_t> decompress_strip(
    const std::uint8_t* src, std::size_t src_len, std::uint16_t compression)
{
    if (compression == 1) {
        // No compression
        return {src, src + src_len};
    }

    std::vector<std::uint8_t> out;
    if (compression == 5) {
        decompress_lzw(src, src_len, out);
    } else if (compression == 32773) {
        decompress_packbits(src, src_len, out);
    } else {
        throw std::runtime_error("TIFF: unsupported compression type " +
                                 std::to_string(compression));
    }
    return out;
}

// ---------------------------------------------------------------------------
// Endian-aware write helpers
// ---------------------------------------------------------------------------

struct ByteWriter {
    std::vector<std::uint8_t> buf;
    bool big_endian = false;

    void u8(std::uint8_t v) { buf.push_back(v); }

    void u16(std::uint16_t v) {
        if (big_endian && std::endian::native == std::endian::little)
            v = std::byteswap(v);
        else if (!big_endian && std::endian::native == std::endian::big)
            v = std::byteswap(v);
        auto off = buf.size();
        buf.resize(off + 2);
        std::memcpy(buf.data() + off, &v, 2);
    }

    void u32(std::uint32_t v) {
        if (big_endian && std::endian::native == std::endian::little)
            v = std::byteswap(v);
        else if (!big_endian && std::endian::native == std::endian::big)
            v = std::byteswap(v);
        auto off = buf.size();
        buf.resize(off + 4);
        std::memcpy(buf.data() + off, &v, 4);
    }

    void raw(const std::uint8_t* data, std::size_t len) {
        buf.insert(buf.end(), data, data + len);
    }

    void pad_to_word() {
        if (buf.size() % 2 != 0) buf.push_back(0);
    }

    [[nodiscard]] std::uint32_t pos() const noexcept {
        return static_cast<std::uint32_t>(buf.size());
    }

    // Patch a u32 at a given offset
    void patch_u32(std::size_t off, std::uint32_t v) {
        if (big_endian && std::endian::native == std::endian::little)
            v = std::byteswap(v);
        else if (!big_endian && std::endian::native == std::endian::big)
            v = std::byteswap(v);
        std::memcpy(buf.data() + off, &v, 4);
    }
};

// Write an IFD entry: tag, type, count, value/offset (always 12 bytes)
inline void write_ifd_entry(ByteWriter& w, std::uint16_t tag, std::uint16_t type,
                            std::uint32_t count, std::uint32_t value) {
    w.u16(tag);
    w.u16(type);
    w.u32(count);
    // If value fits in 4 bytes, store inline; otherwise it is an offset
    if (type == type_short && count == 1) {
        // SHORT value left-justified in 4 bytes
        w.u16(value);
        w.u16(0);
    } else if (type == type_short && count == 2) {
        w.u16(value & 0xFFFF);
        w.u16((value >> 16) & 0xFFFF);
    } else {
        w.u32(value);
    }
}

} // namespace detail::tiff

// ---------------------------------------------------------------------------
// TiffReader
// ---------------------------------------------------------------------------

class TiffReader final {
public:
    explicit TiffReader(const std::filesystem::path& path) {
        std::ifstream fs(path, std::ios::binary | std::ios::ate);
        if (!fs) throw std::runtime_error("TIFF: cannot open " + path.string());

        auto fsize = fs.tellg();
        fs.seekg(0);
        data_.resize(static_cast<std::size_t>(fsize));
        fs.read(reinterpret_cast<char*>(data_.data()), fsize);

        if (data_.size() < 8)
            throw std::runtime_error("TIFF: file too small");

        // Determine byte order
        if (data_[0] == 'I' && data_[1] == 'I')
            big_endian_ = false;
        else if (data_[0] == 'M' && data_[1] == 'M')
            big_endian_ = true;
        else
            throw std::runtime_error("TIFF: invalid byte order marker");

        detail::tiff::ByteReader reader{data_.data(), data_.size(), big_endian_};

        auto magic = reader.u16(2);
        if (magic != 42)
            throw std::runtime_error("TIFF: invalid magic number");

        // Walk IFD chain
        auto ifd_offset = reader.u32(4);
        while (ifd_offset != 0 && ifd_offset < data_.size()) {
            ifds_.push_back(detail::tiff::parse_ifd(reader, ifd_offset));
            auto num_entries = reader.u16(ifd_offset);
            auto next_off_pos = ifd_offset + 2 +
                                static_cast<std::size_t>(num_entries) * 12;
            ifd_offset = reader.u32(next_off_pos);
        }

        if (ifds_.empty())
            throw std::runtime_error("TIFF: no IFDs found");
    }

    [[nodiscard]] std::size_t num_pages() const noexcept { return ifds_.size(); }

    [[nodiscard]] TiffImageInfo info(std::size_t page = 0) const {
        if (page >= ifds_.size())
            throw std::out_of_range("TIFF: page index out of range");
        return ifds_[page].info;
    }

    [[nodiscard]] std::vector<std::uint8_t> read(std::size_t page = 0) const {
        if (page >= ifds_.size())
            throw std::out_of_range("TIFF: page index out of range");

        const auto& ifd = ifds_[page];
        std::vector<std::uint8_t> out;
        out.reserve(ifd.info.image_bytes());

        for (std::size_t i = 0; i < ifd.strip_offsets.size(); ++i) {
            auto strip = read_strip_impl(ifd, i);
            out.insert(out.end(), strip.begin(), strip.end());
        }

        // Handle big-endian 16-bit data: swap bytes to native order
        if (ifd.info.bits_per_sample == 16 && big_endian_ &&
            std::endian::native == std::endian::little) {
            for (std::size_t i = 0; i + 1 < out.size(); i += 2)
                std::swap(out[i], out[i + 1]);
        } else if (ifd.info.bits_per_sample == 16 && !big_endian_ &&
                   std::endian::native == std::endian::big) {
            for (std::size_t i = 0; i + 1 < out.size(); i += 2)
                std::swap(out[i], out[i + 1]);
        }

        return out;
    }

    [[nodiscard]] std::vector<std::uint16_t> read_u16(std::size_t page = 0) const {
        auto bytes = read(page);
        std::vector<std::uint16_t> out(bytes.size() / 2);
        std::memcpy(out.data(), bytes.data(), out.size() * 2);
        return out;
    }

    [[nodiscard]] std::vector<std::uint8_t> read_strip(
        std::size_t page, std::size_t strip_index) const
    {
        if (page >= ifds_.size())
            throw std::out_of_range("TIFF: page index out of range");
        return read_strip_impl(ifds_[page], strip_index);
    }

private:
    [[nodiscard]] std::vector<std::uint8_t> read_strip_impl(
        const detail::tiff::IfdData& ifd, std::size_t strip_index) const
    {
        if (strip_index >= ifd.strip_offsets.size())
            throw std::out_of_range("TIFF: strip index out of range");

        auto offset = ifd.strip_offsets[strip_index];
        auto length = ifd.strip_byte_counts[strip_index];

        if (static_cast<std::size_t>(offset) + length > data_.size())
            throw std::runtime_error("TIFF: strip data out of bounds");

        return detail::tiff::decompress_strip(
            data_.data() + offset, length, ifd.info.compression);
    }

    std::vector<std::uint8_t> data_;
    bool big_endian_ = false;
    std::vector<detail::tiff::IfdData> ifds_;
};

// ---------------------------------------------------------------------------
// TiffWriter
// ---------------------------------------------------------------------------

class TiffWriter final {
public:
    explicit TiffWriter(const std::filesystem::path& path)
        : path_(path) {
        // Write native endian (little on most platforms)
        w_.big_endian = (std::endian::native == std::endian::big);

        // Header: byte order, magic 42, offset to first IFD (patched later)
        if (w_.big_endian) {
            w_.u8('M'); w_.u8('M');
        } else {
            w_.u8('I'); w_.u8('I');
        }
        w_.u16(42);
        first_ifd_offset_pos_ = w_.pos();
        w_.u32(0); // placeholder for first IFD offset
    }

    ~TiffWriter() {
        try { close(); } catch (...) {}
    }

    void write(const TiffImageInfo& info, std::span<const std::uint8_t> data) {
        write_page(info, data.data(), data.size());
    }

    void write_u16(const TiffImageInfo& info, std::span<const std::uint16_t> data) {
        // Convert to bytes, handling endianness
        std::vector<std::uint8_t> bytes(data.size() * 2);
        if (w_.big_endian && std::endian::native == std::endian::little) {
            for (std::size_t i = 0; i < data.size(); ++i) {
                auto v = std::byteswap(data[i]);
                std::memcpy(bytes.data() + i * 2, &v, 2);
            }
        } else if (!w_.big_endian && std::endian::native == std::endian::big) {
            for (std::size_t i = 0; i < data.size(); ++i) {
                auto v = std::byteswap(data[i]);
                std::memcpy(bytes.data() + i * 2, &v, 2);
            }
        } else {
            std::memcpy(bytes.data(), data.data(), bytes.size());
        }

        write_page(info, bytes.data(), bytes.size());
    }

    void close() {
        if (closed_) return;
        closed_ = true;

        // Patch first IFD offset
        if (!ifd_offsets_.empty())
            w_.patch_u32(first_ifd_offset_pos_, ifd_offsets_.front());

        // Write file
        std::ofstream fs(path_, std::ios::binary | std::ios::trunc);
        if (!fs) throw std::runtime_error("TIFF: cannot write " + path_.string());
        fs.write(reinterpret_cast<const char*>(w_.buf.data()),
                 static_cast<std::streamsize>(w_.buf.size()));
    }

private:
    void write_page(const TiffImageInfo& info,
                    const std::uint8_t* pixel_data, std::size_t pixel_len) {
        auto rps = info.rows_per_strip > 0 ? info.rows_per_strip : info.height;
        auto rb = info.row_bytes();
        std::uint32_t num_strips =
            (info.height + rps - 1) / rps;

        // Write strip data and record offsets / byte counts
        std::vector<std::uint32_t> strip_offsets(num_strips);
        std::vector<std::uint32_t> strip_byte_counts(num_strips);

        for (std::uint32_t s = 0; s < num_strips; ++s) {
            auto first_row = s * rps;
            auto rows = std::min(rps, info.height - first_row);
            auto nbytes = static_cast<std::uint32_t>(rows * rb);
            auto src_off = static_cast<std::size_t>(first_row) * rb;

            w_.pad_to_word();
            strip_offsets[s] = w_.pos();
            strip_byte_counts[s] = nbytes;

            if (src_off + nbytes <= pixel_len)
                w_.raw(pixel_data + src_off, nbytes);
            else
                w_.raw(pixel_data + src_off, pixel_len - src_off);
        }

        w_.pad_to_word();

        // If multiple strips, write offset/count arrays and record their positions
        std::uint32_t offsets_pos = 0;
        std::uint32_t counts_pos = 0;
        bool arrays_external = (num_strips > 1);

        if (arrays_external) {
            offsets_pos = w_.pos();
            for (auto o : strip_offsets) w_.u32(o);
            counts_pos = w_.pos();
            for (auto c : strip_byte_counts) w_.u32(c);
            w_.pad_to_word();
        }

        // If BitsPerSample > 1 sample, write the array externally
        std::uint32_t bps_offset = 0;
        bool bps_external = (info.samples_per_pixel > 1);
        if (bps_external) {
            w_.pad_to_word();
            bps_offset = w_.pos();
            for (std::uint16_t i = 0; i < info.samples_per_pixel; ++i)
                w_.u16(info.bits_per_sample);
        }

        // Patch previous IFD's next-offset to point here
        if (!prev_ifd_next_offset_pos_.empty()) {
            auto patch_pos = prev_ifd_next_offset_pos_.back();
            w_.patch_u32(patch_pos, w_.pos());
        }

        // Write IFD
        ifd_offsets_.push_back(w_.pos());

        // Count tags: width, height, bps, compression, photometric,
        //             strip_offsets, spp, rows_per_strip, strip_byte_counts,
        //             planar_config, sample_format = 11
        std::uint16_t num_tags = 11;
        w_.u16(num_tags);

        // Tags must be in ascending order
        using detail::tiff::write_ifd_entry;
        using namespace detail::tiff;

        write_ifd_entry(w_, tag_image_width, type_long, 1, info.width);
        write_ifd_entry(w_, tag_image_length, type_long, 1, info.height);

        if (bps_external)
            write_ifd_entry(w_, tag_bits_per_sample, type_short,
                            info.samples_per_pixel, bps_offset);
        else
            write_ifd_entry(w_, tag_bits_per_sample, type_short, 1,
                            info.bits_per_sample);

        write_ifd_entry(w_, tag_compression, type_short, 1, 1); // uncompressed
        write_ifd_entry(w_, tag_photometric, type_short, 1, info.photometric);

        if (arrays_external)
            write_ifd_entry(w_, tag_strip_offsets, type_long,
                            num_strips, offsets_pos);
        else
            write_ifd_entry(w_, tag_strip_offsets, type_long, 1,
                            strip_offsets[0]);

        write_ifd_entry(w_, tag_samples_per_pixel, type_short, 1,
                        info.samples_per_pixel);
        write_ifd_entry(w_, tag_rows_per_strip, type_long, 1, rps);

        if (arrays_external)
            write_ifd_entry(w_, tag_strip_byte_counts, type_long,
                            num_strips, counts_pos);
        else
            write_ifd_entry(w_, tag_strip_byte_counts, type_long, 1,
                            strip_byte_counts[0]);

        write_ifd_entry(w_, tag_planar_config, type_short, 1,
                        info.planar_config);
        write_ifd_entry(w_, tag_sample_format, type_short, 1,
                        info.sample_format);

        // Next IFD offset (0 for now; patched if another page is added)
        prev_ifd_next_offset_pos_.push_back(w_.pos());
        w_.u32(0);
    }

    std::filesystem::path path_;
    detail::tiff::ByteWriter w_;
    bool closed_ = false;
    std::size_t first_ifd_offset_pos_ = 0;
    std::vector<std::uint32_t> ifd_offsets_;
    std::vector<std::size_t> prev_ifd_next_offset_pos_;
};

// ---------------------------------------------------------------------------
// Convenience functions
// ---------------------------------------------------------------------------

[[nodiscard]] inline std::pair<TiffImageInfo, std::vector<std::uint8_t>>
read_tiff(const std::filesystem::path& path) {
    TiffReader reader(path);
    auto inf = reader.info(0);
    auto data = reader.read(0);
    return {inf, std::move(data)};
}

inline void write_tiff(const std::filesystem::path& path,
                       const TiffImageInfo& info,
                       std::span<const std::uint8_t> data) {
    TiffWriter writer(path);
    writer.write(info, data);
    writer.close();
}

inline void write_tiff_u16(const std::filesystem::path& path,
                           const TiffImageInfo& info,
                           std::span<const std::uint16_t> data) {
    TiffWriter writer(path);
    writer.write_u16(info, data);
    writer.close();
}

} // namespace utils2
