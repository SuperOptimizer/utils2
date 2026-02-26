#pragma once
#include <vector>
#include <cstddef>
#include <cstdint>
#include <concepts>
#include <numeric>
#include <unordered_map>
#include <type_traits>

namespace utils2 {

/// Integer-label disjoint set (union-find) with path compression and union by
/// rank. Extracted from the cc3d library in volume-cartographer.
template <std::integral T = std::uint32_t>
class DisjointSet final {
    std::vector<T> parent_;
    std::vector<std::uint8_t> rank_;
    std::vector<std::size_t> size_;
    std::size_t num_sets_;

public:
    /// Create with @p n elements labelled 0..n-1.
    explicit constexpr DisjointSet(std::size_t n)
        : parent_(n), rank_(n, 0), size_(n, 1), num_sets_{n}
    {
        std::iota(parent_.begin(), parent_.end(), T{0});
    }

    /// Find the root of @p x with iterative path compression.
    [[nodiscard]] T find(T x) noexcept
    {
        T root = x;
        while (parent_[root] != root) {
            root = parent_[root];
        }
        // Path compression: point every node on the path directly to root.
        while (parent_[x] != root) {
            T next = parent_[x];
            parent_[x] = root;
            x = next;
        }
        return root;
    }

    /// Union the sets containing @p x and @p y. Returns the root of the
    /// merged set.
    T unite(T x, T y) noexcept
    {
        T rx = find(x);
        T ry = find(y);
        if (rx == ry) {
            return rx;
        }
        // Union by rank: attach the shorter tree under the taller one.
        if (rank_[rx] < rank_[ry]) {
            std::swap(rx, ry);
        }
        parent_[ry] = rx;
        size_[rx] += size_[ry];
        if (rank_[rx] == rank_[ry]) {
            ++rank_[rx];
        }
        --num_sets_;
        return rx;
    }

    /// Check whether @p x and @p y belong to the same set.
    [[nodiscard]] bool connected(T x, T y) noexcept
    {
        return find(x) == find(y);
    }

    /// Return the number of disjoint sets.
    [[nodiscard]] std::size_t num_sets() const noexcept { return num_sets_; }

    /// Return the total number of elements.
    [[nodiscard]] std::size_t size() const noexcept { return parent_.size(); }

    /// Grow the capacity to at least @p new_size elements, preserving all
    /// existing elements and their union relationships. New elements are
    /// initialized as singletons.
    void grow(std::size_t new_size)
    {
        const auto old_size = parent_.size();
        if (new_size <= old_size) return;
        parent_.resize(new_size);
        rank_.resize(new_size, 0);
        size_.resize(new_size, 1);
        for (std::size_t i = old_size; i < new_size; ++i) {
            parent_[i] = static_cast<T>(i);
        }
        num_sets_ += (new_size - old_size);
    }

    /// Return the size of the set containing @p x.
    [[nodiscard]] std::size_t set_size(T x) noexcept { return size_[find(x)]; }

    /// Flatten all paths so every element points directly to its root.
    void flatten()
    {
        for (std::size_t i = 0; i < parent_.size(); ++i) {
            parent_[i] = find(static_cast<T>(i));
        }
    }

    /// Return a vector of all current root elements.
    [[nodiscard]] std::vector<T> roots() const
    {
        std::vector<T> result;
        result.reserve(num_sets_);
        for (std::size_t i = 0; i < parent_.size(); ++i) {
            if (parent_[i] == static_cast<T>(i)) {
                result.push_back(static_cast<T>(i));
            }
        }
        return result;
    }

    /// Relabel every element so that set IDs are consecutive (0, 1, 2, ...).
    /// Returns a vector where result[i] is the new label for element i.
    [[nodiscard]] std::vector<T> relabel() const
    {
        std::vector<T> labels(parent_.size());
        T next_label{0};
        std::unordered_map<T, T> root_map;
        root_map.reserve(num_sets_);
        for (std::size_t i = 0; i < parent_.size(); ++i) {
            // Walk to root without mutating (const method).
            T root = static_cast<T>(i);
            while (parent_[root] != root) {
                root = parent_[root];
            }
            auto [it, inserted] = root_map.try_emplace(root, next_label);
            if (inserted) {
                ++next_label;
            }
            labels[i] = it->second;
        }
        return labels;
    }
};

/// Dynamic disjoint set that works with any hashable key type.
template <
    typename K,
    typename Hash = std::hash<K>,
    typename KeyEqual = std::equal_to<K>>
class DynamicDisjointSet final {
    struct Node {
        K parent;
        std::uint8_t rank{0};
    };

    std::unordered_map<K, Node, Hash, KeyEqual> nodes_;
    std::size_t num_sets_{0};

public:
    /// Add an element. No-op if the element already exists.
    void add(const K& key)
    {
        auto [it, inserted] = nodes_.try_emplace(key, Node{key, 0});
        if (inserted) {
            ++num_sets_;
        }
    }

    /// Find the root of @p key with iterative path compression.
    [[nodiscard]] const K& find(const K& key)
    {
        K root = key;
        while (nodes_.at(root).parent != root) {
            root = nodes_.at(root).parent;
        }
        // Path compression.
        K cur = key;
        while (nodes_.at(cur).parent != root) {
            K next = nodes_.at(cur).parent;
            nodes_.at(cur).parent = root;
            cur = next;
        }
        return nodes_.at(root).parent;
    }

    /// Union the sets containing @p a and @p b.
    void unite(const K& a, const K& b)
    {
        K ra = find(a);
        K rb = find(b);
        if (KeyEqual{}(ra, rb)) {
            return;
        }
        auto& na = nodes_.at(ra);
        auto& nb = nodes_.at(rb);
        if (na.rank < nb.rank) {
            na.parent = rb;
        } else if (na.rank > nb.rank) {
            nb.parent = ra;
        } else {
            nb.parent = ra;
            ++na.rank;
        }
        --num_sets_;
    }

    /// Check whether @p a and @p b belong to the same set.
    [[nodiscard]] bool connected(const K& a, const K& b)
    {
        return KeyEqual{}(find(a), find(b));
    }

    /// Return the number of disjoint sets.
    [[nodiscard]] std::size_t num_sets() const noexcept { return num_sets_; }

    /// Return the total number of elements.
    [[nodiscard]] std::size_t size() const noexcept { return nodes_.size(); }
};

}  // namespace utils2
