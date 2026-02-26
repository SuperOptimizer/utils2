#pragma once
#include <vector>
#include <array>
#include <algorithm>
#include <cstddef>
#include <cmath>
#include <concepts>
#include <optional>
#include <functional>
#include <numeric>
#include <limits>
#include <span>
#include <queue>
#include <ranges>
#include <set>

namespace utils2 {

// ---------------------------------------------------------------------------
// Point concept
// ---------------------------------------------------------------------------
template <typename P, std::size_t Dims>
concept PointLike = requires(const P& p) {
    { p[0] } -> std::convertible_to<double>;
};

// ---------------------------------------------------------------------------
// AABB -- axis-aligned bounding box
// ---------------------------------------------------------------------------
template <std::size_t Dims>
struct AABB {
    std::array<double, Dims> lo{};
    std::array<double, Dims> hi{};

    [[nodiscard]] constexpr bool contains(
        const std::array<double, Dims>& p) const noexcept
    {
        for (std::size_t i = 0; i < Dims; ++i)
            if (p[i] < lo[i] || p[i] > hi[i]) return false;
        return true;
    }

    [[nodiscard]] constexpr bool intersects(const AABB& other) const noexcept
    {
        for (std::size_t i = 0; i < Dims; ++i)
            if (lo[i] > other.hi[i] || hi[i] < other.lo[i]) return false;
        return true;
    }

    [[nodiscard]] constexpr AABB merge(const AABB& other) const noexcept
    {
        AABB r;
        for (std::size_t i = 0; i < Dims; ++i) {
            r.lo[i] = std::min(lo[i], other.lo[i]);
            r.hi[i] = std::max(hi[i], other.hi[i]);
        }
        return r;
    }

    [[nodiscard]] constexpr double distance_sq(
        const std::array<double, Dims>& p) const noexcept
    {
        double d = 0.0;
        for (std::size_t i = 0; i < Dims; ++i) {
            if (p[i] < lo[i]) { double g = lo[i] - p[i]; d += g * g; }
            else if (p[i] > hi[i]) { double g = p[i] - hi[i]; d += g * g; }
        }
        return d;
    }

    [[nodiscard]] constexpr std::array<double, Dims> center() const noexcept
    {
        std::array<double, Dims> c{};
        for (std::size_t i = 0; i < Dims; ++i)
            c[i] = (lo[i] + hi[i]) * 0.5;
        return c;
    }
};

// ---------------------------------------------------------------------------
// KDTree
// ---------------------------------------------------------------------------
template <typename T, std::size_t Dims,
          typename PointExtractor = std::identity>
class KDTree final {
public:
    using point_type = std::array<double, Dims>;

    struct QueryResult {
        const T* item;
        double distance_sq;
    };

    // -- construction -------------------------------------------------------

    void build(std::vector<T> items)
    {
        items_ = std::move(items);
        nodes_.clear();
        erased_.clear();
        next_node_ = 0;
        root_ = 0;
        size_ = items_.size();
        insertions_since_build_ = 0;
        if (items_.empty()) return;
        nodes_.resize(items_.size());
        std::vector<std::size_t> idx(items_.size());
        std::iota(idx.begin(), idx.end(), 0);
        root_ = build_rec(idx.data(), idx.size(), 0);
    }

    void insert(T item)
    {
        items_.push_back(std::move(item));
        ++size_;
        ++insertions_since_build_;
        // Rebuild when insertions exceed half the original tree
        if (nodes_.empty() ||
            insertions_since_build_ > items_.size() / 2) {
            rebuild_all();
        }
    }

    // -- queries ------------------------------------------------------------

    [[nodiscard]] std::vector<QueryResult> knn(
        const point_type& query, std::size_t k,
        double max_distance =
            std::numeric_limits<double>::infinity()) const
    {
        double max_dist_sq = max_distance * max_distance;
        // max-heap on distance_sq
        using Entry = std::pair<double, const T*>;
        auto cmp = [](const Entry& a, const Entry& b) {
            return a.first < b.first;
        };
        std::priority_queue<Entry, std::vector<Entry>, decltype(cmp)> heap(cmp);

        knn_rec(query, k, max_dist_sq, root_, heap);

        // also scan un-indexed tail
        for (std::size_t i = nodes_.size(); i < items_.size(); ++i) {
            double d = dist_sq(query, point_of(items_[i]));
            if (d <= max_dist_sq) {
                heap.push({d, &items_[i]});
                if (heap.size() > k) heap.pop();
                if (heap.size() == k) max_dist_sq = heap.top().first;
            }
        }

        std::vector<QueryResult> res;
        res.reserve(heap.size());
        while (!heap.empty()) {
            auto [d, ptr] = heap.top();
            heap.pop();
            res.push_back({ptr, d});
        }
        std::ranges::sort(res, {}, &QueryResult::distance_sq);
        return res;
    }

    [[nodiscard]] std::optional<QueryResult> nearest(
        const point_type& query,
        double max_distance =
            std::numeric_limits<double>::infinity()) const
    {
        auto r = knn(query, 1, max_distance);
        if (r.empty()) return std::nullopt;
        return r.front();
    }

    [[nodiscard]] std::vector<QueryResult> radius(
        const point_type& query, double r) const
    {
        double r_sq = r * r;
        std::vector<QueryResult> res;
        radius_rec(query, r_sq, root_, res);
        // scan un-indexed tail
        for (std::size_t i = nodes_.size(); i < items_.size(); ++i) {
            double d = dist_sq(query, point_of(items_[i]));
            if (d <= r_sq) res.push_back({&items_[i], d});
        }
        std::ranges::sort(res, {}, &QueryResult::distance_sq);
        return res;
    }

    [[nodiscard]] std::vector<const T*> range(
        const point_type& lo, const point_type& hi) const
    {
        std::vector<const T*> res;
        range_rec(lo, hi, root_, res);
        for (std::size_t i = nodes_.size(); i < items_.size(); ++i) {
            auto pt = point_of(items_[i]);
            bool inside = true;
            for (std::size_t d = 0; d < Dims; ++d)
                if (pt[d] < lo[d] || pt[d] > hi[d]) { inside = false; break; }
            if (inside) res.push_back(&items_[i]);
        }
        return res;
    }

    // -- mutation ------------------------------------------------------------

    bool remove(const T& item)
    {
        for (std::size_t i = 0; i < items_.size(); ++i) {
            if (erased_.count(i)) continue;
            if constexpr (requires { items_[i] == item; }) {
                if (!(items_[i] == item)) continue;
            } else {
                if (!(point_of(items_[i]) == point_of(item))) continue;
            }
            erased_.insert(i);
            --size_;
            if (erased_.size() > items_.size() / 4 && !items_.empty())
                rebuild_all();
            return true;
        }
        return false;
    }

    // -- capacity ------------------------------------------------------------

    [[nodiscard]] std::size_t size() const noexcept { return size_; }
    [[nodiscard]] bool empty() const noexcept { return size_ == 0; }

    void clear()
    {
        items_.clear();
        nodes_.clear();
        erased_.clear();
        size_ = 0;
        root_ = 0;
        insertions_since_build_ = 0;
    }

private:
    // Internal flat-array KD-tree node
    struct Node {
        std::size_t item_idx{};  // index into items_
        std::size_t left{0};     // 0 = none
        std::size_t right{0};
        std::size_t split_dim{};
    };

    std::vector<T> items_;
    std::vector<Node> nodes_;       // 1-indexed (0 = null)
    std::set<std::size_t> erased_;  // lazy-deleted indices
    std::size_t size_{0};
    std::size_t insertions_since_build_{0};
    std::size_t root_{0};           // 1-based index of root node (0 = empty)

    // -- helpers ------------------------------------------------------------

    [[nodiscard]] point_type point_of(const T& v) const
    {
        if constexpr (std::is_same_v<PointExtractor, std::identity>) {
            return to_point(v);
        } else {
            return to_point(PointExtractor{}(v));
        }
    }

    template <typename U>
    [[nodiscard]] static point_type to_point(const U& v)
    {
        point_type p;
        for (std::size_t i = 0; i < Dims; ++i)
            p[i] = static_cast<double>(v[i]);
        return p;
    }

    [[nodiscard]] static point_type to_point(const point_type& v)
    {
        return v;
    }

    [[nodiscard]] static double dist_sq(const point_type& a,
                                         const point_type& b) noexcept
    {
        double s = 0.0;
        for (std::size_t i = 0; i < Dims; ++i) {
            double d = a[i] - b[i];
            s += d * d;
        }
        return s;
    }

    [[nodiscard]] bool is_erased(std::size_t item_idx) const
    {
        return erased_.count(item_idx);
    }

    // -- build recursion (returns 1-based node index) -----------------------

    std::size_t next_node_{0};

    std::size_t build_rec(std::size_t* idx, std::size_t n, std::size_t depth)
    {
        if (n == 0) return 0;
        std::size_t dim = depth % Dims;
        std::size_t mid = n / 2;

        std::nth_element(idx, idx + mid, idx + n,
            [&](std::size_t a, std::size_t b) {
                return point_of(items_[a])[dim] < point_of(items_[b])[dim];
            });

        std::size_t ni = ++next_node_;  // 1-based
        if (ni > nodes_.size()) nodes_.resize(ni);
        nodes_[ni - 1].item_idx = idx[mid];
        nodes_[ni - 1].split_dim = dim;
        nodes_[ni - 1].left  = build_rec(idx, mid, depth + 1);
        nodes_[ni - 1].right = build_rec(idx + mid + 1, n - mid - 1, depth + 1);
        return ni;
    }

    void rebuild_all()
    {
        // compact: remove erased items
        if (!erased_.empty()) {
            std::vector<T> keep;
            keep.reserve(size_);
            for (std::size_t i = 0; i < items_.size(); ++i)
                if (!erased_.count(i))
                    keep.push_back(std::move(items_[i]));
            items_ = std::move(keep);
            erased_.clear();
        }
        nodes_.clear();
        next_node_ = 0;
        root_ = 0;
        size_ = items_.size();
        insertions_since_build_ = 0;
        if (items_.empty()) return;
        nodes_.resize(items_.size());
        std::vector<std::size_t> idx(items_.size());
        std::iota(idx.begin(), idx.end(), 0);
        root_ = build_rec(idx.data(), idx.size(), 0);
    }

    // -- knn recursion ------------------------------------------------------

    template <typename Heap>
    void knn_rec(const point_type& q, std::size_t k,
                 double& max_dist_sq, std::size_t ni, Heap& heap) const
    {
        if (ni == 0) return;
        const auto& node = nodes_[ni - 1];
        auto pt = point_of(items_[node.item_idx]);

        if (!is_erased(node.item_idx)) {
            double d = dist_sq(q, pt);
            if (d <= max_dist_sq) {
                heap.push({d, &items_[node.item_idx]});
                if (heap.size() > k) heap.pop();
                if (heap.size() == k) max_dist_sq = heap.top().first;
            }
        }

        std::size_t dim = node.split_dim;
        double diff = q[dim] - pt[dim];
        std::size_t near = diff <= 0.0 ? node.left : node.right;
        std::size_t far  = diff <= 0.0 ? node.right : node.left;

        knn_rec(q, k, max_dist_sq, near, heap);
        if (diff * diff <= max_dist_sq)
            knn_rec(q, k, max_dist_sq, far, heap);
    }

    // -- radius recursion ---------------------------------------------------

    void radius_rec(const point_type& q, double r_sq,
                    std::size_t ni,
                    std::vector<QueryResult>& out) const
    {
        if (ni == 0) return;
        const auto& node = nodes_[ni - 1];
        auto pt = point_of(items_[node.item_idx]);

        if (!is_erased(node.item_idx)) {
            double d = dist_sq(q, pt);
            if (d <= r_sq) out.push_back({&items_[node.item_idx], d});
        }

        std::size_t dim = node.split_dim;
        double diff = q[dim] - pt[dim];
        std::size_t near = diff <= 0.0 ? node.left : node.right;
        std::size_t far  = diff <= 0.0 ? node.right : node.left;

        radius_rec(q, r_sq, near, out);
        if (diff * diff <= r_sq)
            radius_rec(q, r_sq, far, out);
    }

    // -- range recursion ----------------------------------------------------

    void range_rec(const point_type& lo, const point_type& hi,
                   std::size_t ni, std::vector<const T*>& out) const
    {
        if (ni == 0) return;
        const auto& node = nodes_[ni - 1];
        auto pt = point_of(items_[node.item_idx]);

        if (!is_erased(node.item_idx)) {
            bool inside = true;
            for (std::size_t d = 0; d < Dims; ++d)
                if (pt[d] < lo[d] || pt[d] > hi[d]) { inside = false; break; }
            if (inside) out.push_back(&items_[node.item_idx]);
        }

        std::size_t dim = node.split_dim;
        if (lo[dim] <= pt[dim]) range_rec(lo, hi, node.left, out);
        if (hi[dim] >= pt[dim]) range_rec(lo, hi, node.right, out);
    }
};

// ---------------------------------------------------------------------------
// BVH -- bounding volume hierarchy
// ---------------------------------------------------------------------------
template <typename T, std::size_t Dims, typename BoundsExtractor>
class BVH final {
public:
    using aabb_type = AABB<Dims>;

    void build(std::vector<T> items)
    {
        items_ = std::move(items);
        nodes_.clear();
        if (items_.empty()) return;
        std::vector<std::size_t> idx(items_.size());
        std::iota(idx.begin(), idx.end(), 0);
        build_rec(idx.data(), idx.size());
    }

    [[nodiscard]] std::vector<const T*> query(const aabb_type& box) const
    {
        std::vector<const T*> res;
        if (!nodes_.empty())
            query_rec(box, 0, res);
        return res;
    }

    [[nodiscard]] std::vector<const T*> query_point(
        const std::array<double, Dims>& point) const
    {
        std::vector<const T*> res;
        if (!nodes_.empty())
            query_point_rec(point, 0, res);
        return res;
    }

    [[nodiscard]] std::size_t size() const noexcept { return items_.size(); }

private:
    struct Node {
        aabb_type bounds{};
        std::size_t left{0};     // index into nodes_
        std::size_t right{0};
        std::size_t item_idx{};  // leaf: index into items_
        bool is_leaf{false};
    };

    std::vector<T> items_;
    std::vector<Node> nodes_;

    [[nodiscard]] aabb_type bounds_of(const T& item) const
    {
        return BoundsExtractor{}(item);
    }

    // Build returns index into nodes_
    std::size_t build_rec(std::size_t* idx, std::size_t n)
    {
        if (n == 1) {
            Node node;
            node.bounds = bounds_of(items_[idx[0]]);
            node.is_leaf = true;
            node.item_idx = idx[0];
            nodes_.push_back(node);
            return nodes_.size() - 1;
        }

        // Compute combined bounds
        aabb_type total = bounds_of(items_[idx[0]]);
        for (std::size_t i = 1; i < n; ++i)
            total = total.merge(bounds_of(items_[idx[i]]));

        // Pick axis with largest extent
        std::size_t best_dim = 0;
        double best_extent = 0.0;
        for (std::size_t d = 0; d < Dims; ++d) {
            double ext = total.hi[d] - total.lo[d];
            if (ext > best_extent) { best_extent = ext; best_dim = d; }
        }

        // Partition at midpoint of that axis
        std::size_t mid = n / 2;
        std::nth_element(idx, idx + mid, idx + n,
            [&](std::size_t a, std::size_t b) {
                return bounds_of(items_[a]).center()[best_dim] <
                       bounds_of(items_[b]).center()[best_dim];
            });

        // Reserve a slot for this internal node
        nodes_.push_back({});
        std::size_t self = nodes_.size() - 1;

        std::size_t li = build_rec(idx, mid);
        std::size_t ri = build_rec(idx + mid, n - mid);

        nodes_[self].bounds = total;
        nodes_[self].left = li;
        nodes_[self].right = ri;
        nodes_[self].is_leaf = false;
        return self;
    }

    void query_rec(const aabb_type& box, std::size_t ni,
                   std::vector<const T*>& out) const
    {
        const auto& node = nodes_[ni];
        if (!node.bounds.intersects(box)) return;
        if (node.is_leaf) {
            out.push_back(&items_[node.item_idx]);
            return;
        }
        query_rec(box, node.left, out);
        query_rec(box, node.right, out);
    }

    void query_point_rec(const std::array<double, Dims>& point,
                         std::size_t ni,
                         std::vector<const T*>& out) const
    {
        const auto& node = nodes_[ni];
        if (!node.bounds.contains(point)) return;
        if (node.is_leaf) {
            out.push_back(&items_[node.item_idx]);
            return;
        }
        query_point_rec(point, node.left, out);
        query_point_rec(point, node.right, out);
    }
};

}  // namespace utils2
