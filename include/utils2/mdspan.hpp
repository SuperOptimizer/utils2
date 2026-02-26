#pragma once

// ---------------------------------------------------------------------------
// Minimal <mdspan> polyfill for GCC 15 (which does not yet ship <mdspan>).
// When the standard header becomes available, we simply include it instead.
// ---------------------------------------------------------------------------

#if __has_include(<mdspan>)
#include <mdspan>
#else

#include <array>
#include <cstddef>
#include <span>        // for std::dynamic_extent
#include <type_traits>

namespace std {

// std::dynamic_extent is already defined in <span>. No need to redefine.

// ---------------------------------------------------------------------------
// extents<IndexType, Extents...>
// ---------------------------------------------------------------------------

namespace detail_mdspan {

template <class IndexType, std::size_t Rank>
struct dextents_storage {
    std::array<IndexType, Rank> exts_{};

    constexpr dextents_storage() noexcept = default;

    template <typename... Sizes>
        requires (sizeof...(Sizes) == Rank)
              && (std::is_convertible_v<Sizes, IndexType> && ...)
    constexpr dextents_storage(Sizes... sizes) noexcept
        : exts_{static_cast<IndexType>(sizes)...} {}

    constexpr dextents_storage(const std::array<IndexType, Rank>& a) noexcept
        : exts_{a} {}
};

template <class IndexType>
struct dextents_storage<IndexType, 0> {
    constexpr dextents_storage() noexcept = default;
};

} // namespace detail_mdspan

template <class IndexType, std::size_t... StaticExtents>
class extents : private detail_mdspan::dextents_storage<IndexType, ((StaticExtents == dynamic_extent) + ... + 0)> {
    static constexpr std::size_t rank_v = sizeof...(StaticExtents);
    static constexpr std::size_t rank_dynamic_v = ((StaticExtents == dynamic_extent) + ... + 0);

    using base = detail_mdspan::dextents_storage<IndexType, rank_dynamic_v>;

    static constexpr std::array<std::size_t, rank_v> static_extents_{StaticExtents...};

public:
    using index_type = IndexType;
    using size_type = std::make_unsigned_t<IndexType>;
    using rank_type = std::size_t;

    static constexpr rank_type rank() noexcept { return rank_v; }
    static constexpr rank_type rank_dynamic() noexcept { return rank_dynamic_v; }

    constexpr extents() noexcept = default;

    // Constructor from individual dynamic sizes.
    template <typename... Sizes>
        requires (sizeof...(Sizes) == rank_dynamic_v)
              && (rank_dynamic_v > 0)
              && (std::is_convertible_v<Sizes, IndexType> && ...)
    constexpr extents(Sizes... sizes) noexcept : base(sizes...) {}

    // Constructor from array of dynamic sizes.
    constexpr extents(const std::array<IndexType, rank_dynamic_v>& a) noexcept
        requires (rank_dynamic_v > 0)
        : base(a) {}

    [[nodiscard]] constexpr IndexType extent(rank_type i) const noexcept {
        if constexpr (rank_dynamic_v == rank_v) {
            return this->exts_[i];
        } else if constexpr (rank_dynamic_v == 0) {
            return static_cast<IndexType>(static_extents_[i]);
        } else {
            // Mixed: not fully supported in this polyfill.
            return this->exts_[i];
        }
    }

    static constexpr std::size_t static_extent(rank_type i) noexcept {
        return static_extents_[i];
    }
};

// ---------------------------------------------------------------------------
// dextents<IndexType, Rank> -- all-dynamic extents
// ---------------------------------------------------------------------------

namespace detail_mdspan {

template <class IndexType, std::size_t Rank, class = std::make_index_sequence<Rank>>
struct make_dextents;

template <class IndexType, std::size_t Rank, std::size_t... Is>
struct make_dextents<IndexType, Rank, std::index_sequence<Is...>> {
    template <std::size_t> static constexpr std::size_t dyn = dynamic_extent;
    using type = extents<IndexType, dyn<Is>...>;
};

} // namespace detail_mdspan

template <class IndexType, std::size_t Rank>
using dextents = typename detail_mdspan::make_dextents<IndexType, Rank>::type;

// ---------------------------------------------------------------------------
// layout_right -- row-major layout mapping
// ---------------------------------------------------------------------------

struct layout_right {
    template <class Extents>
    class mapping {
    public:
        using extents_type = Extents;
        using index_type = typename Extents::index_type;
        using size_type = typename Extents::size_type;
        using rank_type = typename Extents::rank_type;
        using layout_type = layout_right;

        constexpr mapping() noexcept = default;
        constexpr mapping(const Extents& e) noexcept : extents_(e) {}

        [[nodiscard]] constexpr const Extents& extents() const noexcept { return extents_; }

        [[nodiscard]] constexpr index_type required_span_size() const noexcept {
            index_type s = 1;
            for (rank_type i = 0; i < Extents::rank(); ++i) {
                s *= extents_.extent(i);
            }
            return s;
        }

        // Row-major: offset = ((i0 * e1 + i1) * e2 + i2) * e3 + i3 ...
        template <typename... Indices>
            requires (sizeof...(Indices) == Extents::rank())
        [[nodiscard]] constexpr index_type operator()(Indices... indices) const noexcept {
            const index_type idx[] = {static_cast<index_type>(indices)...};
            index_type offset = idx[0];
            for (rank_type i = 1; i < Extents::rank(); ++i) {
                offset = offset * extents_.extent(i) + idx[i];
            }
            return offset;
        }

        [[nodiscard]] constexpr index_type stride(rank_type i) const noexcept {
            index_type s = 1;
            for (rank_type j = i + 1; j < Extents::rank(); ++j) {
                s *= extents_.extent(j);
            }
            return s;
        }

    private:
        Extents extents_{};
    };
};

// ---------------------------------------------------------------------------
// default_accessor<T>
// ---------------------------------------------------------------------------

template <class ElementType>
struct default_accessor {
    using offset_policy = default_accessor;
    using element_type = ElementType;
    using reference = ElementType&;
    using data_handle_type = ElementType*;

    constexpr default_accessor() noexcept = default;

    [[nodiscard]] constexpr reference access(data_handle_type p, std::size_t i) const noexcept {
        return p[i];
    }

    [[nodiscard]] constexpr data_handle_type offset(data_handle_type p, std::size_t i) const noexcept {
        return p + i;
    }
};

// ---------------------------------------------------------------------------
// mdspan<ElementType, Extents, LayoutPolicy, AccessorPolicy>
// ---------------------------------------------------------------------------

template <
    class ElementType,
    class Extents,
    class LayoutPolicy = layout_right,
    class AccessorPolicy = default_accessor<ElementType>
>
class mdspan {
public:
    using extents_type = Extents;
    using layout_type = LayoutPolicy;
    using accessor_type = AccessorPolicy;
    using mapping_type = typename LayoutPolicy::template mapping<Extents>;
    using element_type = ElementType;
    using value_type = std::remove_cv_t<ElementType>;
    using index_type = typename Extents::index_type;
    using size_type = typename Extents::size_type;
    using rank_type = typename Extents::rank_type;
    using data_handle_type = typename AccessorPolicy::data_handle_type;
    using reference = typename AccessorPolicy::reference;

    static constexpr rank_type rank() noexcept { return Extents::rank(); }
    static constexpr rank_type rank_dynamic() noexcept { return Extents::rank_dynamic(); }

    constexpr mdspan() noexcept = default;

    // Construct from pointer + individual extents.
    template <typename... SizeTypes>
        requires (sizeof...(SizeTypes) == Extents::rank())
              && (std::is_convertible_v<SizeTypes, index_type> && ...)
    constexpr mdspan(data_handle_type ptr, SizeTypes... exts) noexcept
        : ptr_(ptr), map_(Extents(static_cast<index_type>(exts)...)) {}

    // Construct from pointer + extents object.
    constexpr mdspan(data_handle_type ptr, const Extents& e) noexcept
        : ptr_(ptr), map_(e) {}

    // Construct from pointer + mapping.
    constexpr mdspan(data_handle_type ptr, const mapping_type& m) noexcept
        : ptr_(ptr), map_(m) {}

    // -- element access -------------------------------------------------------

    // C++23 multidimensional subscript operator.
    template <typename... Indices>
        requires (sizeof...(Indices) == Extents::rank())
              && (std::is_convertible_v<Indices, index_type> && ...)
    [[nodiscard]] constexpr reference operator[](Indices... indices) const noexcept {
        return acc_.access(ptr_, static_cast<std::size_t>(map_(static_cast<index_type>(indices)...)));
    }

    // -- observers ------------------------------------------------------------

    [[nodiscard]] constexpr index_type extent(rank_type i) const noexcept {
        return map_.extents().extent(i);
    }

    [[nodiscard]] constexpr const Extents& extents() const noexcept {
        return map_.extents();
    }

    [[nodiscard]] constexpr data_handle_type data_handle() const noexcept {
        return ptr_;
    }

    [[nodiscard]] constexpr const mapping_type& mapping() const noexcept {
        return map_;
    }

    [[nodiscard]] constexpr const accessor_type& accessor() const noexcept {
        return acc_;
    }

    [[nodiscard]] constexpr size_type size() const noexcept {
        size_type s = 1;
        for (rank_type i = 0; i < Extents::rank(); ++i) {
            s *= static_cast<size_type>(map_.extents().extent(i));
        }
        return s;
    }

    [[nodiscard]] constexpr bool empty() const noexcept { return size() == 0; }

private:
    data_handle_type ptr_{};
    mapping_type map_{};
    [[no_unique_address]] accessor_type acc_{};
};

// ---------------------------------------------------------------------------
// CTAD deduction guides for mdspan
// ---------------------------------------------------------------------------

// mdspan(T*, SizeTypes...) -> mdspan<T, dextents<size_t, sizeof...(SizeTypes)>>
template <class ElementType, typename... SizeTypes>
    requires (std::is_convertible_v<SizeTypes, std::size_t> && ...)
mdspan(ElementType*, SizeTypes...)
    -> mdspan<ElementType, dextents<std::size_t, sizeof...(SizeTypes)>>;

} // namespace std

#endif // __has_include(<mdspan>)
