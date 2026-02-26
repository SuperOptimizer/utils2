#pragma once
#include "mdspan.hpp"
#include <vector>
#include <array>
#include <span>
#include <memory>
#include <cstddef>
#include <concepts>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <type_traits>

namespace utils2 {

// ---------------------------------------------------------------------------
// Stride helpers (row-major / C-order: last index varies fastest)
// ---------------------------------------------------------------------------

template <std::size_t Dims>
[[nodiscard]] constexpr auto compute_strides(
    const std::array<std::size_t, Dims>& shape) noexcept
    -> std::array<std::size_t, Dims>
{
    std::array<std::size_t, Dims> strides{};
    if constexpr (Dims > 0) {
        strides[Dims - 1] = 1;
        for (std::size_t i = Dims - 1; i > 0; --i) {
            strides[i - 1] = strides[i] * shape[i];
        }
    }
    return strides;
}

template <std::size_t Dims>
[[nodiscard]] constexpr auto linear_index(
    const std::array<std::size_t, Dims>& strides,
    const std::array<std::size_t, Dims>& indices) noexcept -> std::size_t
{
    std::size_t idx = 0;
    for (std::size_t i = 0; i < Dims; ++i) {
        idx += strides[i] * indices[i];
    }
    return idx;
}

// ---------------------------------------------------------------------------
// Internal: recursive N-dimensional index iteration
// ---------------------------------------------------------------------------

namespace detail {

template <std::size_t Rank, typename F, std::size_t... I>
void for_each_idx(
    const std::array<std::size_t, Rank>& extents,
    std::array<std::size_t, Rank>& idx,
    std::size_t dim,
    F&& func,
    std::index_sequence<I...>)
{
    if (dim == Rank) {
        func(idx[I]...);
        return;
    }
    for (std::size_t i = 0; i < extents[dim]; ++i) {
        idx[dim] = i;
        for_each_idx(extents, idx, dim + 1, func,
                     std::make_index_sequence<Rank>{});
    }
}

// Collect extents from an mdspan into an array.
template <typename Span, std::size_t... I>
auto extents_array(Span s, std::index_sequence<I...>)
    -> std::array<std::size_t, Span::rank()>
{
    return {s.extent(I)...};
}

} // namespace detail

// ---------------------------------------------------------------------------
// NDArray<T, Dims> -- owning N-dimensional array backed by std::vector
// ---------------------------------------------------------------------------

template <typename T, std::size_t Dims>
class NDArray {
public:
    using value_type = T;
    static constexpr std::size_t dimensions = Dims;

    // Construct from shape (value-initialized).
    explicit NDArray(std::array<std::size_t, Dims> shape)
        : shape_{shape}, storage_(total_size(shape)) {}

    // Construct from shape + fill value.
    NDArray(std::array<std::size_t, Dims> shape, T init)
        : shape_{shape}, storage_(total_size(shape), init) {}

    // -- mdspan view --------------------------------------------------------

    [[nodiscard]] auto view() noexcept { return make_mdspan(storage_.data()); }
    [[nodiscard]] auto view() const noexcept { return make_mdspan(storage_.data()); }

    // -- element access (variadic indices) ----------------------------------

    template <typename... Indices>
        requires (sizeof...(Indices) == Dims)
              && (std::convertible_to<Indices, std::size_t> && ...)
    [[nodiscard]] T& operator()(Indices... idx) noexcept
    {
        const auto i = linear_index(
            compute_strides(shape_),
            std::array<std::size_t, Dims>{static_cast<std::size_t>(idx)...});
        return storage_[i];
    }

    template <typename... Indices>
        requires (sizeof...(Indices) == Dims)
              && (std::convertible_to<Indices, std::size_t> && ...)
    [[nodiscard]] const T& operator()(Indices... idx) const noexcept
    {
        const auto i = linear_index(
            compute_strides(shape_),
            std::array<std::size_t, Dims>{static_cast<std::size_t>(idx)...});
        return storage_[i];
    }

    // -- shape info ---------------------------------------------------------

    [[nodiscard]] const auto& shape() const noexcept { return shape_; }
    [[nodiscard]] std::size_t size() const noexcept { return storage_.size(); }
    [[nodiscard]] std::size_t extent(std::size_t dim) const noexcept { return shape_[dim]; }

    // -- raw data -----------------------------------------------------------

    [[nodiscard]] T* data() noexcept { return storage_.data(); }
    [[nodiscard]] const T* data() const noexcept { return storage_.data(); }
    [[nodiscard]] std::span<T> flat() noexcept { return {storage_.data(), storage_.size()}; }
    [[nodiscard]] std::span<const T> flat() const noexcept { return {storage_.data(), storage_.size()}; }

private:
    std::array<std::size_t, Dims> shape_;
    std::vector<T> storage_;

    [[nodiscard]] static constexpr std::size_t total_size(
        const std::array<std::size_t, Dims>& shape) noexcept
    {
        std::size_t n = 1;
        for (auto s : shape) n *= s;
        return n;
    }

    template <typename U>
    [[nodiscard]] auto make_mdspan(U* ptr) const noexcept
    {
        return [&]<std::size_t... I>(std::index_sequence<I...>) {
            return std::mdspan<U, std::dextents<std::size_t, Dims>>(
                ptr, shape_[I]...);
        }(std::make_index_sequence<Dims>{});
    }
};

// Deduction guides
template <typename T, std::size_t N>
NDArray(std::array<std::size_t, N>) -> NDArray<T, N>;

template <typename T, std::size_t N>
NDArray(std::array<std::size_t, N>, T) -> NDArray<T, N>;

// ---------------------------------------------------------------------------
// Slice descriptor for subview extraction
// ---------------------------------------------------------------------------

struct Slice {
    std::size_t start;
    std::size_t stop; // exclusive

    [[nodiscard]] constexpr std::size_t size() const noexcept
    {
        return stop - start;
    }
};

// ---------------------------------------------------------------------------
// subview -- extract a sub-region from an mdspan (returns owning NDArray)
// ---------------------------------------------------------------------------

template <typename T, typename Extents, typename Layout, typename Accessor,
          typename... Slices>
    requires (sizeof...(Slices) == Extents::rank())
          && (std::same_as<Slices, Slice> && ...)
[[nodiscard]] auto subview(
    std::mdspan<T, Extents, Layout, Accessor> src, Slices... slices)
{
    constexpr std::size_t Dims = Extents::rank();
    using value_type = std::remove_const_t<T>;

    const std::array<Slice, Dims> sl{slices...};
    std::array<std::size_t, Dims> out_shape;
    for (std::size_t i = 0; i < Dims; ++i) {
        out_shape[i] = sl[i].size();
    }

    NDArray<value_type, Dims> dst(out_shape);

    std::array<std::size_t, Dims> didx{};
    auto copy_recursive = [&](auto& self, std::size_t dim) -> void {
        if (dim == Dims) {
            auto src_idx = didx;
            for (std::size_t d = 0; d < Dims; ++d) {
                src_idx[d] += sl[d].start;
            }
            [&]<std::size_t... I>(std::index_sequence<I...>) {
                dst(didx[I]...) =
                    static_cast<value_type>(src[src_idx[I]...]);
            }(std::make_index_sequence<Dims>{});
            return;
        }
        for (std::size_t i = 0; i < out_shape[dim]; ++i) {
            didx[dim] = i;
            self(self, dim + 1);
        }
    };
    copy_recursive(copy_recursive, 0);

    return dst;
}

// ---------------------------------------------------------------------------
// TypeErasedBuffer -- type-erased contiguous storage (up to 3D)
// ---------------------------------------------------------------------------

class TypeErasedBuffer {
public:
    TypeErasedBuffer() = default;

    TypeErasedBuffer(std::size_t elem_size, std::array<std::size_t, 3> shape)
        : element_size_{elem_size}
        , shape_{shape}
        , bytes_(elem_size * shape[0] * shape[1] * shape[2])
    {
    }

    template <typename T>
    [[nodiscard]] T* as() noexcept
    {
        return reinterpret_cast<T*>(bytes_.data());
    }

    template <typename T>
    [[nodiscard]] const T* as() const noexcept
    {
        return reinterpret_cast<const T*>(bytes_.data());
    }

    [[nodiscard]] const auto& shape() const noexcept { return shape_; }
    [[nodiscard]] std::size_t element_size() const noexcept { return element_size_; }
    [[nodiscard]] std::span<std::byte> raw() noexcept { return {bytes_.data(), bytes_.size()}; }
    [[nodiscard]] std::span<const std::byte> raw() const noexcept { return {bytes_.data(), bytes_.size()}; }
    [[nodiscard]] std::size_t size_bytes() const noexcept { return bytes_.size(); }

    [[nodiscard]] std::size_t total_elements() const noexcept
    {
        return shape_[0] * shape_[1] * shape_[2];
    }

    template <typename T>
    [[nodiscard]] auto mdspan_view() noexcept
    {
        return std::mdspan<T, std::dextents<std::size_t, 3>>(
            as<T>(), shape_[0], shape_[1], shape_[2]);
    }

    template <typename T>
    [[nodiscard]] auto mdspan_view() const noexcept
    {
        return std::mdspan<const T, std::dextents<std::size_t, 3>>(
            as<T>(), shape_[0], shape_[1], shape_[2]);
    }

private:
    std::vector<std::byte> bytes_;
    std::size_t element_size_{0};
    std::array<std::size_t, 3> shape_{0, 0, 0};
};

// ---------------------------------------------------------------------------
// Bulk operations on mdspan views
// ---------------------------------------------------------------------------

template <typename T, typename Extents>
void fill(std::mdspan<T, Extents> dst, T value)
{
    constexpr std::size_t Rank = Extents::rank();
    const auto exts = detail::extents_array(dst, std::make_index_sequence<Rank>{});
    std::array<std::size_t, Rank> idx{};

    detail::for_each_idx<Rank>(
        exts, idx, 0,
        [&](auto... is) { dst[static_cast<std::size_t>(is)...] = value; },
        std::make_index_sequence<Rank>{});
}

template <typename T, typename U, typename Extents>
void copy(std::mdspan<const T, Extents> src, std::mdspan<U, Extents> dst)
{
    constexpr std::size_t Rank = Extents::rank();
    const auto exts = detail::extents_array(src, std::make_index_sequence<Rank>{});
    std::array<std::size_t, Rank> idx{};

    detail::for_each_idx<Rank>(
        exts, idx, 0,
        [&](auto... is) {
            dst[static_cast<std::size_t>(is)...] =
                static_cast<U>(src[static_cast<std::size_t>(is)...]);
        },
        std::make_index_sequence<Rank>{});
}

template <typename T, typename Extents, typename F>
void transform(std::mdspan<T, Extents> arr, F&& func)
{
    constexpr std::size_t Rank = Extents::rank();
    const auto exts = detail::extents_array(arr, std::make_index_sequence<Rank>{});
    std::array<std::size_t, Rank> idx{};

    detail::for_each_idx<Rank>(
        exts, idx, 0,
        [&](auto... is) {
            auto& elem = arr[static_cast<std::size_t>(is)...];
            elem = func(elem);
        },
        std::make_index_sequence<Rank>{});
}

} // namespace utils2
