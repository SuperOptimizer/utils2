#include <utils2/test.hpp>
#include <utils2/connectivity.hpp>
#include <cmath>
#include <numeric>
#include <vector>
#include <set>
#include <utility>

using namespace utils2;

// ============================================================================
// 2D offset tables
// ============================================================================

TEST_CASE("offsets_2d(four) returns 4 offsets") {
    auto [ptr, n] = offsets_2d(GridConnectivity::four);
    REQUIRE_EQ(n, 4u);
}

TEST_CASE("offsets_2d(eight) returns 8 offsets") {
    auto [ptr, n] = offsets_2d(GridConnectivity::eight);
    REQUIRE_EQ(n, 8u);
}

TEST_CASE("offsets_2d(four): all distances are 1.0") {
    auto [ptr, n] = offsets_2d(GridConnectivity::four);
    for (std::size_t i = 0; i < n; ++i) {
        REQUIRE_NEAR(ptr[i].dist, 1.0, 1e-12);
    }
}

TEST_CASE("offsets_2d(eight): face neighbors have dist 1.0, diagonals have sqrt(2)") {
    auto [ptr, n] = offsets_2d(GridConnectivity::eight);
    constexpr double sqrt2 = 1.4142135623730951;
    int face_count = 0;
    int diag_count = 0;
    for (std::size_t i = 0; i < n; ++i) {
        bool is_diag = (ptr[i].dy != 0 && ptr[i].dx != 0);
        if (is_diag) {
            REQUIRE_NEAR(ptr[i].dist, sqrt2, 1e-12);
            ++diag_count;
        } else {
            REQUIRE_NEAR(ptr[i].dist, 1.0, 1e-12);
            ++face_count;
        }
    }
    REQUIRE_EQ(face_count, 4);
    REQUIRE_EQ(diag_count, 4);
}

TEST_CASE("offsets_2d(four): offsets are unique") {
    auto [ptr, n] = offsets_2d(GridConnectivity::four);
    std::set<std::pair<int,int>> seen;
    for (std::size_t i = 0; i < n; ++i) {
        auto inserted = seen.insert({ptr[i].dy, ptr[i].dx}).second;
        REQUIRE(inserted);
    }
}

TEST_CASE("offsets_2d(eight): offsets are unique") {
    auto [ptr, n] = offsets_2d(GridConnectivity::eight);
    std::set<std::pair<int,int>> seen;
    for (std::size_t i = 0; i < n; ++i) {
        auto inserted = seen.insert({ptr[i].dy, ptr[i].dx}).second;
        REQUIRE(inserted);
    }
}

TEST_CASE("offsets_2d(four): no diagonal offsets") {
    auto [ptr, n] = offsets_2d(GridConnectivity::four);
    for (std::size_t i = 0; i < n; ++i) {
        // Each offset has exactly one nonzero component
        bool one_nonzero = (ptr[i].dy != 0) != (ptr[i].dx != 0);
        REQUIRE(one_nonzero);
    }
}

TEST_CASE("offsets_2d(four): includes all four cardinal directions") {
    auto [ptr, n] = offsets_2d(GridConnectivity::four);
    std::set<std::pair<int,int>> s;
    for (std::size_t i = 0; i < n; ++i)
        s.insert({ptr[i].dy, ptr[i].dx});

    REQUIRE(s.contains({-1, 0}));  // up
    REQUIRE(s.contains({ 1, 0}));  // down
    REQUIRE(s.contains({ 0,-1}));  // left
    REQUIRE(s.contains({ 0, 1}));  // right
}

TEST_CASE("offsets_2d(eight): includes all 8 directions") {
    auto [ptr, n] = offsets_2d(GridConnectivity::eight);
    std::set<std::pair<int,int>> s;
    for (std::size_t i = 0; i < n; ++i)
        s.insert({ptr[i].dy, ptr[i].dx});

    REQUIRE_EQ(s.size(), std::size_t(8));
    REQUIRE(s.contains({-1,-1}));
    REQUIRE(s.contains({-1, 0}));
    REQUIRE(s.contains({-1, 1}));
    REQUIRE(s.contains({ 0,-1}));
    REQUIRE(s.contains({ 0, 1}));
    REQUIRE(s.contains({ 1,-1}));
    REQUIRE(s.contains({ 1, 0}));
    REQUIRE(s.contains({ 1, 1}));
}

TEST_CASE("offsets_2d(four): distances match Euclidean") {
    auto [ptr, n] = offsets_2d(GridConnectivity::four);
    for (std::size_t i = 0; i < n; ++i) {
        double expected = std::sqrt(ptr[i].dy * ptr[i].dy + ptr[i].dx * ptr[i].dx);
        REQUIRE_NEAR(ptr[i].dist, expected, 1e-12);
    }
}

TEST_CASE("offsets_2d(eight): distances match Euclidean") {
    auto [ptr, n] = offsets_2d(GridConnectivity::eight);
    for (std::size_t i = 0; i < n; ++i) {
        double expected = std::sqrt(ptr[i].dy * ptr[i].dy + ptr[i].dx * ptr[i].dx);
        REQUIRE_NEAR(ptr[i].dist, expected, 1e-12);
    }
}

// ============================================================================
// 3D offset tables
// ============================================================================

TEST_CASE("offsets_3d(six) returns 6 offsets") {
    auto [ptr, n] = offsets_3d(GridConnectivity::six);
    REQUIRE_EQ(n, 6u);
}

TEST_CASE("offsets_3d(eighteen) returns 18 offsets") {
    auto [ptr, n] = offsets_3d(GridConnectivity::eighteen);
    REQUIRE_EQ(n, 18u);
}

TEST_CASE("offsets_3d(twenty_six) returns 26 offsets") {
    auto [ptr, n] = offsets_3d(GridConnectivity::twenty_six);
    REQUIRE_EQ(n, 26u);
}

TEST_CASE("offsets_3d(six): all distances are 1.0") {
    auto [ptr, n] = offsets_3d(GridConnectivity::six);
    for (std::size_t i = 0; i < n; ++i) {
        REQUIRE_NEAR(ptr[i].dist, 1.0, 1e-12);
    }
}

TEST_CASE("offsets_3d(six): each offset has exactly one nonzero component") {
    auto [ptr, n] = offsets_3d(GridConnectivity::six);
    for (std::size_t i = 0; i < n; ++i) {
        int nonzero = (ptr[i].dz != 0) + (ptr[i].dy != 0) + (ptr[i].dx != 0);
        REQUIRE_EQ(nonzero, 1);
    }
}

TEST_CASE("offsets_3d(six): includes all 6 face directions") {
    auto [ptr, n] = offsets_3d(GridConnectivity::six);
    using T = std::tuple<int,int,int>;
    std::set<T> s;
    for (std::size_t i = 0; i < n; ++i)
        s.insert({ptr[i].dz, ptr[i].dy, ptr[i].dx});

    REQUIRE(s.contains({-1, 0, 0}));
    REQUIRE(s.contains({ 1, 0, 0}));
    REQUIRE(s.contains({ 0,-1, 0}));
    REQUIRE(s.contains({ 0, 1, 0}));
    REQUIRE(s.contains({ 0, 0,-1}));
    REQUIRE(s.contains({ 0, 0, 1}));
}

TEST_CASE("offsets_3d(eighteen): 6 face at 1.0, 12 edge at sqrt(2)") {
    auto [ptr, n] = offsets_3d(GridConnectivity::eighteen);
    constexpr double sqrt2 = 1.4142135623730951;
    int face_count = 0;
    int edge_count = 0;
    for (std::size_t i = 0; i < n; ++i) {
        int nonzero = (ptr[i].dz != 0) + (ptr[i].dy != 0) + (ptr[i].dx != 0);
        if (nonzero == 1) {
            REQUIRE_NEAR(ptr[i].dist, 1.0, 1e-12);
            ++face_count;
        } else if (nonzero == 2) {
            REQUIRE_NEAR(ptr[i].dist, sqrt2, 1e-12);
            ++edge_count;
        }
    }
    REQUIRE_EQ(face_count, 6);
    REQUIRE_EQ(edge_count, 12);
}

TEST_CASE("offsets_3d(eighteen): no corner offsets") {
    auto [ptr, n] = offsets_3d(GridConnectivity::eighteen);
    for (std::size_t i = 0; i < n; ++i) {
        int nonzero = (ptr[i].dz != 0) + (ptr[i].dy != 0) + (ptr[i].dx != 0);
        REQUIRE_LE(nonzero, 2);
    }
}

TEST_CASE("offsets_3d(eighteen): offsets are unique") {
    auto [ptr, n] = offsets_3d(GridConnectivity::eighteen);
    using T = std::tuple<int,int,int>;
    std::set<T> s;
    for (std::size_t i = 0; i < n; ++i) {
        auto inserted = s.insert({ptr[i].dz, ptr[i].dy, ptr[i].dx}).second;
        REQUIRE(inserted);
    }
}

TEST_CASE("offsets_3d(twenty_six): 6 face, 12 edge, 8 corner with correct distances") {
    auto [ptr, n] = offsets_3d(GridConnectivity::twenty_six);
    constexpr double sqrt2 = 1.4142135623730951;
    constexpr double sqrt3 = 1.7320508075688772;
    int face_count = 0;
    int edge_count = 0;
    int corner_count = 0;
    for (std::size_t i = 0; i < n; ++i) {
        int nonzero = (ptr[i].dz != 0) + (ptr[i].dy != 0) + (ptr[i].dx != 0);
        if (nonzero == 1) {
            REQUIRE_NEAR(ptr[i].dist, 1.0, 1e-12);
            ++face_count;
        } else if (nonzero == 2) {
            REQUIRE_NEAR(ptr[i].dist, sqrt2, 1e-12);
            ++edge_count;
        } else if (nonzero == 3) {
            REQUIRE_NEAR(ptr[i].dist, sqrt3, 1e-12);
            ++corner_count;
        }
    }
    REQUIRE_EQ(face_count, 6);
    REQUIRE_EQ(edge_count, 12);
    REQUIRE_EQ(corner_count, 8);
}

TEST_CASE("offsets_3d(twenty_six): offsets are unique") {
    auto [ptr, n] = offsets_3d(GridConnectivity::twenty_six);
    using T = std::tuple<int,int,int>;
    std::set<T> s;
    for (std::size_t i = 0; i < n; ++i) {
        auto inserted = s.insert({ptr[i].dz, ptr[i].dy, ptr[i].dx}).second;
        REQUIRE(inserted);
    }
    REQUIRE_EQ(s.size(), std::size_t(26));
}

TEST_CASE("offsets_3d(twenty_six): distances match Euclidean") {
    auto [ptr, n] = offsets_3d(GridConnectivity::twenty_six);
    for (std::size_t i = 0; i < n; ++i) {
        double expected = std::sqrt(
            ptr[i].dz * ptr[i].dz +
            ptr[i].dy * ptr[i].dy +
            ptr[i].dx * ptr[i].dx);
        REQUIRE_NEAR(ptr[i].dist, expected, 1e-12);
    }
}

TEST_CASE("offsets_3d(twenty_six): is a superset of eighteen which is superset of six") {
    using T = std::tuple<int,int,int>;

    auto [p6, n6] = offsets_3d(GridConnectivity::six);
    auto [p18, n18] = offsets_3d(GridConnectivity::eighteen);
    auto [p26, n26] = offsets_3d(GridConnectivity::twenty_six);

    std::set<T> s6, s18, s26;
    for (std::size_t i = 0; i < n6; ++i)
        s6.insert({p6[i].dz, p6[i].dy, p6[i].dx});
    for (std::size_t i = 0; i < n18; ++i)
        s18.insert({p18[i].dz, p18[i].dy, p18[i].dx});
    for (std::size_t i = 0; i < n26; ++i)
        s26.insert({p26[i].dz, p26[i].dy, p26[i].dx});

    // six is a subset of eighteen
    for (const auto& o : s6)
        REQUIRE(s18.contains(o));

    // eighteen is a subset of twenty_six
    for (const auto& o : s18)
        REQUIRE(s26.contains(o));
}

// ============================================================================
// Predecessor tables - 2D
// ============================================================================

TEST_CASE("predecessors_2d(four) returns 2") {
    auto [ptr, n] = predecessors_2d(GridConnectivity::four);
    REQUIRE_EQ(n, 2u);
}

TEST_CASE("predecessors_2d(eight) returns 4") {
    auto [ptr, n] = predecessors_2d(GridConnectivity::eight);
    REQUIRE_EQ(n, 4u);
}

TEST_CASE("predecessors_2d: all offsets precede in raster order") {
    auto [ptr, n] = predecessors_2d(GridConnectivity::eight);
    for (std::size_t i = 0; i < n; ++i) {
        // In raster order, predecessors have dy < 0, or (dy == 0 and dx < 0)
        bool precedes = (ptr[i].dy < 0) || (ptr[i].dy == 0 && ptr[i].dx < 0);
        REQUIRE(precedes);
    }
}

TEST_CASE("predecessors_2d(four): all offsets precede in raster order") {
    auto [ptr, n] = predecessors_2d(GridConnectivity::four);
    for (std::size_t i = 0; i < n; ++i) {
        bool precedes = (ptr[i].dy < 0) || (ptr[i].dy == 0 && ptr[i].dx < 0);
        REQUIRE(precedes);
    }
}

TEST_CASE("predecessors_2d(four): offsets are unique") {
    auto [ptr, n] = predecessors_2d(GridConnectivity::four);
    std::set<std::pair<int,int>> seen;
    for (std::size_t i = 0; i < n; ++i) {
        auto inserted = seen.insert({ptr[i].dy, ptr[i].dx}).second;
        REQUIRE(inserted);
    }
}

TEST_CASE("predecessors_2d(eight): offsets are unique") {
    auto [ptr, n] = predecessors_2d(GridConnectivity::eight);
    std::set<std::pair<int,int>> seen;
    for (std::size_t i = 0; i < n; ++i) {
        auto inserted = seen.insert({ptr[i].dy, ptr[i].dx}).second;
        REQUIRE(inserted);
    }
}

TEST_CASE("predecessors_2d(four): is a subset of offsets_2d(four)") {
    auto [pred, np] = predecessors_2d(GridConnectivity::four);
    auto [full, nf] = offsets_2d(GridConnectivity::four);

    std::set<std::pair<int,int>> full_set;
    for (std::size_t i = 0; i < nf; ++i)
        full_set.insert({full[i].dy, full[i].dx});

    for (std::size_t i = 0; i < np; ++i)
        REQUIRE(full_set.contains({pred[i].dy, pred[i].dx}));
}

TEST_CASE("predecessors_2d(eight): is a subset of offsets_2d(eight)") {
    auto [pred, np] = predecessors_2d(GridConnectivity::eight);
    auto [full, nf] = offsets_2d(GridConnectivity::eight);

    std::set<std::pair<int,int>> full_set;
    for (std::size_t i = 0; i < nf; ++i)
        full_set.insert({full[i].dy, full[i].dx});

    for (std::size_t i = 0; i < np; ++i)
        REQUIRE(full_set.contains({pred[i].dy, pred[i].dx}));
}

// ============================================================================
// Predecessor tables - 3D
// ============================================================================

TEST_CASE("predecessors_3d(six) returns 3") {
    auto [ptr, n] = predecessors_3d(GridConnectivity::six);
    REQUIRE_EQ(n, 3u);
}

TEST_CASE("predecessors_3d(eighteen) returns 9") {
    auto [ptr, n] = predecessors_3d(GridConnectivity::eighteen);
    REQUIRE_EQ(n, 9u);
}

TEST_CASE("predecessors_3d(twenty_six) returns 13") {
    auto [ptr, n] = predecessors_3d(GridConnectivity::twenty_six);
    REQUIRE_EQ(n, 13u);
}

TEST_CASE("predecessors_3d: all offsets precede in raster order") {
    auto [ptr, n] = predecessors_3d(GridConnectivity::twenty_six);
    for (std::size_t i = 0; i < n; ++i) {
        bool precedes = (ptr[i].dz < 0) ||
                        (ptr[i].dz == 0 && ptr[i].dy < 0) ||
                        (ptr[i].dz == 0 && ptr[i].dy == 0 && ptr[i].dx < 0);
        REQUIRE(precedes);
    }
}

TEST_CASE("predecessors_3d(six): all precede in raster order") {
    auto [ptr, n] = predecessors_3d(GridConnectivity::six);
    for (std::size_t i = 0; i < n; ++i) {
        bool precedes = (ptr[i].dz < 0) ||
                        (ptr[i].dz == 0 && ptr[i].dy < 0) ||
                        (ptr[i].dz == 0 && ptr[i].dy == 0 && ptr[i].dx < 0);
        REQUIRE(precedes);
    }
}

TEST_CASE("predecessors_3d(eighteen): all precede in raster order") {
    auto [ptr, n] = predecessors_3d(GridConnectivity::eighteen);
    for (std::size_t i = 0; i < n; ++i) {
        bool precedes = (ptr[i].dz < 0) ||
                        (ptr[i].dz == 0 && ptr[i].dy < 0) ||
                        (ptr[i].dz == 0 && ptr[i].dy == 0 && ptr[i].dx < 0);
        REQUIRE(precedes);
    }
}

TEST_CASE("predecessors_3d(six): offsets are unique") {
    auto [ptr, n] = predecessors_3d(GridConnectivity::six);
    using T = std::tuple<int,int,int>;
    std::set<T> seen;
    for (std::size_t i = 0; i < n; ++i) {
        auto inserted = seen.insert({ptr[i].dz, ptr[i].dy, ptr[i].dx}).second;
        REQUIRE(inserted);
    }
}

TEST_CASE("predecessors_3d(eighteen): offsets are unique") {
    auto [ptr, n] = predecessors_3d(GridConnectivity::eighteen);
    using T = std::tuple<int,int,int>;
    std::set<T> seen;
    for (std::size_t i = 0; i < n; ++i) {
        auto inserted = seen.insert({ptr[i].dz, ptr[i].dy, ptr[i].dx}).second;
        REQUIRE(inserted);
    }
}

TEST_CASE("predecessors_3d(twenty_six): offsets are unique") {
    auto [ptr, n] = predecessors_3d(GridConnectivity::twenty_six);
    using T = std::tuple<int,int,int>;
    std::set<T> seen;
    for (std::size_t i = 0; i < n; ++i) {
        auto inserted = seen.insert({ptr[i].dz, ptr[i].dy, ptr[i].dx}).second;
        REQUIRE(inserted);
    }
    REQUIRE_EQ(seen.size(), std::size_t(13));
}

TEST_CASE("predecessors_3d(six): is a subset of offsets_3d(six)") {
    auto [pred, np] = predecessors_3d(GridConnectivity::six);
    auto [full, nf] = offsets_3d(GridConnectivity::six);

    using T = std::tuple<int,int,int>;
    std::set<T> full_set;
    for (std::size_t i = 0; i < nf; ++i)
        full_set.insert({full[i].dz, full[i].dy, full[i].dx});

    for (std::size_t i = 0; i < np; ++i)
        REQUIRE(full_set.contains({pred[i].dz, pred[i].dy, pred[i].dx}));
}

TEST_CASE("predecessors_3d(twenty_six): is a subset of offsets_3d(twenty_six)") {
    auto [pred, np] = predecessors_3d(GridConnectivity::twenty_six);
    auto [full, nf] = offsets_3d(GridConnectivity::twenty_six);

    using T = std::tuple<int,int,int>;
    std::set<T> full_set;
    for (std::size_t i = 0; i < nf; ++i)
        full_set.insert({full[i].dz, full[i].dy, full[i].dx});

    for (std::size_t i = 0; i < np; ++i)
        REQUIRE(full_set.contains({pred[i].dz, pred[i].dy, pred[i].dx}));
}

// ============================================================================
// Structuring element offset counts
// ============================================================================

TEST_CASE("StructuringElement: cross has 5 offsets") {
    REQUIRE_EQ(detail::cross_offsets.size(), 5u);
}

TEST_CASE("StructuringElement: square has 9 offsets") {
    REQUIRE_EQ(detail::square_offsets.size(), 9u);
}

TEST_CASE("StructuringElement: disk_3 same as square (9 offsets)") {
    REQUIRE_EQ(detail::disk3_offsets.size(), 9u);
}

TEST_CASE("StructuringElement: disk_5 has 21 offsets") {
    REQUIRE_EQ(detail::disk5_offsets.size(), 21u);
}

TEST_CASE("StructuringElement: cross includes center") {
    bool has_center = false;
    for (const auto& [dy, dx] : detail::cross_offsets) {
        if (dy == 0 && dx == 0) { has_center = true; break; }
    }
    REQUIRE(has_center);
}

TEST_CASE("StructuringElement: square includes center") {
    bool has_center = false;
    for (const auto& [dy, dx] : detail::square_offsets) {
        if (dy == 0 && dx == 0) { has_center = true; break; }
    }
    REQUIRE(has_center);
}

TEST_CASE("StructuringElement: disk_5 includes center") {
    bool has_center = false;
    for (const auto& [dy, dx] : detail::disk5_offsets) {
        if (dy == 0 && dx == 0) { has_center = true; break; }
    }
    REQUIRE(has_center);
}

TEST_CASE("StructuringElement: cross offsets are within radius 1") {
    for (const auto& [dy, dx] : detail::cross_offsets) {
        REQUIRE_LE(std::abs(dy), 1);
        REQUIRE_LE(std::abs(dx), 1);
    }
}

TEST_CASE("StructuringElement: square offsets are within radius 1") {
    for (const auto& [dy, dx] : detail::square_offsets) {
        REQUIRE_LE(std::abs(dy), 1);
        REQUIRE_LE(std::abs(dx), 1);
    }
}

TEST_CASE("StructuringElement: disk_5 offsets are within radius 2") {
    for (const auto& [dy, dx] : detail::disk5_offsets) {
        REQUIRE_LE(std::abs(dy), 2);
        REQUIRE_LE(std::abs(dx), 2);
    }
}

TEST_CASE("StructuringElement: cross has only cardinal + center offsets") {
    for (const auto& [dy, dx] : detail::cross_offsets) {
        // Must be center or have exactly one nonzero component
        bool is_cardinal_or_center = (dy == 0 && dx == 0) ||
                                     (dy == 0) != (dx == 0);
        REQUIRE(is_cardinal_or_center);
    }
}

TEST_CASE("StructuringElement: square includes all 9 offsets in 3x3") {
    std::set<std::pair<int,int>> s;
    for (const auto& [dy, dx] : detail::square_offsets)
        s.insert({dy, dx});

    REQUIRE_EQ(s.size(), std::size_t(9));
    for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx)
            REQUIRE(s.contains({dy, dx}));
}

TEST_CASE("StructuringElement: disk_3 offsets match square offsets") {
    // disk_3 is defined as an alias for square
    for (std::size_t i = 0; i < detail::disk3_offsets.size(); ++i) {
        REQUIRE_EQ(detail::disk3_offsets[i][0], detail::square_offsets[i][0]);
        REQUIRE_EQ(detail::disk3_offsets[i][1], detail::square_offsets[i][1]);
    }
}

TEST_CASE("StructuringElement: disk_5 offsets are unique") {
    std::set<std::pair<int,int>> s;
    for (const auto& [dy, dx] : detail::disk5_offsets)
        s.insert({dy, dx});
    REQUIRE_EQ(s.size(), std::size_t(21));
}

TEST_CASE("StructuringElement: disk_5 is symmetric") {
    std::set<std::pair<int,int>> s;
    for (const auto& [dy, dx] : detail::disk5_offsets)
        s.insert({dy, dx});

    // For every offset (dy, dx), the opposite (-dy, -dx) should also exist
    for (const auto& [dy, dx] : detail::disk5_offsets) {
        REQUIRE(s.contains({-dy, -dx}));
    }
}

TEST_CASE("StructuringElement: disk_5 does not include extreme corners") {
    // The corners (+-2, +-2) should not be in the disk approximation
    std::set<std::pair<int,int>> s;
    for (const auto& [dy, dx] : detail::disk5_offsets)
        s.insert({dy, dx});

    REQUIRE(!s.contains({ 2,  2}));
    REQUIRE(!s.contains({ 2, -2}));
    REQUIRE(!s.contains({-2,  2}));
    REQUIRE(!s.contains({-2, -2}));
}

// ============================================================================
// dispatch_se
// ============================================================================

TEST_CASE("StructuringElement: dispatch_se selects correct table") {
    detail::dispatch_se(StructuringElement::cross, [](const auto& offsets) {
        REQUIRE_EQ(offsets.size(), 5u);
    });
    detail::dispatch_se(StructuringElement::square, [](const auto& offsets) {
        REQUIRE_EQ(offsets.size(), 9u);
    });
    detail::dispatch_se(StructuringElement::disk_3, [](const auto& offsets) {
        REQUIRE_EQ(offsets.size(), 9u);
    });
    detail::dispatch_se(StructuringElement::disk_5, [](const auto& offsets) {
        REQUIRE_EQ(offsets.size(), 21u);
    });
}

TEST_CASE("dispatch_se: cross body can accumulate offsets") {
    int count = 0;
    detail::dispatch_se(StructuringElement::cross, [&](const auto& offsets) {
        for (const auto& [dy, dx] : offsets)
            ++count;
    });
    REQUIRE_EQ(count, 5);
}

TEST_CASE("dispatch_se: disk_5 body can accumulate offsets") {
    int count = 0;
    detail::dispatch_se(StructuringElement::disk_5, [&](const auto& offsets) {
        for (const auto& [dy, dx] : offsets)
            ++count;
    });
    REQUIRE_EQ(count, 21);
}

TEST_CASE("dispatch_se: disk_3 returns same as square") {
    auto disk3_size = detail::dispatch_se(StructuringElement::disk_3, [](const auto& offsets) {
        return offsets.size();
    });
    auto square_size = detail::dispatch_se(StructuringElement::square, [](const auto& offsets) {
        return offsets.size();
    });
    REQUIRE_EQ(disk3_size, square_size);
}

// ============================================================================
// for_each_neighbor with mdspan
// ============================================================================

TEST_CASE("for_each_neighbor: center pixel 4-connectivity") {
    // 3x3 image, all ones except center is 5
    std::vector<int> data = {1,1,1, 1,5,1, 1,1,1};
    std::mdspan<const int, std::dextents<std::size_t, 2>> img(data.data(), 3, 3);

    std::vector<int> neighbors;
    detail::for_each_neighbor(img, 1, 1, detail::cross_offsets,
        [&](int v) { neighbors.push_back(v); });

    // cross has 5 offsets (including center)
    REQUIRE_EQ(neighbors.size(), std::size_t(5));

    // Count the center value (5) and surrounding values (1)
    int count_5 = 0, count_1 = 0;
    for (auto v : neighbors) {
        if (v == 5) ++count_5;
        else if (v == 1) ++count_1;
    }
    REQUIRE_EQ(count_5, 1); // center
    REQUIRE_EQ(count_1, 4); // 4 cardinal neighbors
}

TEST_CASE("for_each_neighbor: center pixel 8-connectivity (square)") {
    std::vector<int> data = {1,2,3, 4,5,6, 7,8,9};
    std::mdspan<const int, std::dextents<std::size_t, 2>> img(data.data(), 3, 3);

    std::vector<int> neighbors;
    detail::for_each_neighbor(img, 1, 1, detail::square_offsets,
        [&](int v) { neighbors.push_back(v); });

    REQUIRE_EQ(neighbors.size(), std::size_t(9));

    // All values 1-9 should appear
    std::set<int> s(neighbors.begin(), neighbors.end());
    for (int i = 1; i <= 9; ++i)
        REQUIRE(s.contains(i));
}

TEST_CASE("for_each_neighbor: corner pixel clips correctly") {
    // 3x3 image
    std::vector<int> data = {1,2,3, 4,5,6, 7,8,9};
    std::mdspan<const int, std::dextents<std::size_t, 2>> img(data.data(), 3, 3);

    // Top-left corner (0,0) with square SE -- only 4 neighbors in bounds
    std::vector<int> neighbors;
    detail::for_each_neighbor(img, 0, 0, detail::square_offsets,
        [&](int v) { neighbors.push_back(v); });

    // Of the 9 offsets, only (0,0), (0,1), (1,0), (1,1) are in bounds
    REQUIRE_EQ(neighbors.size(), std::size_t(4));
    std::set<int> s(neighbors.begin(), neighbors.end());
    REQUIRE(s.contains(1)); // (0,0)
    REQUIRE(s.contains(2)); // (0,1)
    REQUIRE(s.contains(4)); // (1,0)
    REQUIRE(s.contains(5)); // (1,1)
}

TEST_CASE("for_each_neighbor: bottom-right corner clips correctly") {
    std::vector<int> data = {1,2,3, 4,5,6, 7,8,9};
    std::mdspan<const int, std::dextents<std::size_t, 2>> img(data.data(), 3, 3);

    std::vector<int> neighbors;
    detail::for_each_neighbor(img, 2, 2, detail::square_offsets,
        [&](int v) { neighbors.push_back(v); });

    // Only (1,1), (1,2), (2,1), (2,2) are in bounds
    REQUIRE_EQ(neighbors.size(), std::size_t(4));
    std::set<int> s(neighbors.begin(), neighbors.end());
    REQUIRE(s.contains(5));
    REQUIRE(s.contains(6));
    REQUIRE(s.contains(8));
    REQUIRE(s.contains(9));
}

TEST_CASE("for_each_neighbor: edge pixel with cross SE") {
    // 4x4 image
    std::vector<int> data(16);
    std::iota(data.begin(), data.end(), 0);
    std::mdspan<const int, std::dextents<std::size_t, 2>> img(data.data(), 4, 4);

    // Pixel at (0, 2) -- top edge
    std::vector<int> neighbors;
    detail::for_each_neighbor(img, 0, 2, detail::cross_offsets,
        [&](int v) { neighbors.push_back(v); });

    // Cross at (0,2): center(0,2)=2, right(0,3)=3, left(0,1)=1, down(1,2)=6
    // up(-1,2) is out of bounds
    REQUIRE_EQ(neighbors.size(), std::size_t(4));
}

TEST_CASE("for_each_neighbor: 1x1 image") {
    std::vector<int> data = {42};
    std::mdspan<const int, std::dextents<std::size_t, 2>> img(data.data(), 1, 1);

    std::vector<int> neighbors;
    detail::for_each_neighbor(img, 0, 0, detail::square_offsets,
        [&](int v) { neighbors.push_back(v); });

    // Only center is in bounds
    REQUIRE_EQ(neighbors.size(), std::size_t(1));
    REQUIRE_EQ(neighbors[0], 42);
}

TEST_CASE("for_each_neighbor: disk_5 on large image center") {
    // 5x5 image
    std::vector<int> data(25);
    std::iota(data.begin(), data.end(), 0);
    std::mdspan<const int, std::dextents<std::size_t, 2>> img(data.data(), 5, 5);

    // Center pixel (2,2) -- all 21 disk_5 offsets should be in bounds
    std::vector<int> neighbors;
    detail::for_each_neighbor(img, 2, 2, detail::disk5_offsets,
        [&](int v) { neighbors.push_back(v); });

    REQUIRE_EQ(neighbors.size(), std::size_t(21));
}

TEST_CASE("for_each_neighbor: disk_5 on corner clips many offsets") {
    std::vector<int> data(25, 1);
    std::mdspan<const int, std::dextents<std::size_t, 2>> img(data.data(), 5, 5);

    // Corner (0,0): only offsets with dy>=0 and dx>=0 are valid
    std::vector<int> neighbors;
    detail::for_each_neighbor(img, 0, 0, detail::disk5_offsets,
        [&](int v) { neighbors.push_back(v); });

    // Count how many disk_5 offsets have dy>=0 and dx>=0
    int expected = 0;
    for (const auto& [dy, dx] : detail::disk5_offsets) {
        if (dy >= 0 && dx >= 0 &&
            static_cast<std::size_t>(dy) < 5 &&
            static_cast<std::size_t>(dx) < 5)
            ++expected;
    }
    REQUIRE_EQ(neighbors.size(), static_cast<std::size_t>(expected));
}

// ============================================================================
// GridConnectivity enum values
// ============================================================================

TEST_CASE("GridConnectivity enum has correct underlying values") {
    REQUIRE_EQ(static_cast<std::uint8_t>(GridConnectivity::four), std::uint8_t(4));
    REQUIRE_EQ(static_cast<std::uint8_t>(GridConnectivity::eight), std::uint8_t(8));
    REQUIRE_EQ(static_cast<std::uint8_t>(GridConnectivity::six), std::uint8_t(6));
    REQUIRE_EQ(static_cast<std::uint8_t>(GridConnectivity::eighteen), std::uint8_t(18));
    REQUIRE_EQ(static_cast<std::uint8_t>(GridConnectivity::twenty_six), std::uint8_t(26));
}

// ============================================================================
// StructuringElement enum values
// ============================================================================

TEST_CASE("StructuringElement enum values are distinct") {
    auto a = static_cast<std::uint8_t>(StructuringElement::cross);
    auto b = static_cast<std::uint8_t>(StructuringElement::square);
    auto c = static_cast<std::uint8_t>(StructuringElement::disk_3);
    auto d = static_cast<std::uint8_t>(StructuringElement::disk_5);

    REQUIRE_NE(a, b);
    REQUIRE_NE(a, d);
    REQUIRE_NE(b, d);
    // disk_3 and square may or may not be the same enum value, but they are different enum entries
    (void)c; // disk_3 is a separate enum entry
}

// ============================================================================
// Offset struct sizes
// ============================================================================

TEST_CASE("Offset2D has expected fields") {
    Offset2D o{-1, 2, 3.0};
    REQUIRE_EQ(o.dy, -1);
    REQUIRE_EQ(o.dx, 2);
    REQUIRE_NEAR(o.dist, 3.0, 1e-12);
}

TEST_CASE("Offset3D has expected fields") {
    Offset3D o{1, -2, 3, 4.0};
    REQUIRE_EQ(o.dz, 1);
    REQUIRE_EQ(o.dy, -2);
    REQUIRE_EQ(o.dx, 3);
    REQUIRE_NEAR(o.dist, 4.0, 1e-12);
}

TEST_CASE("PredOffset2D has expected fields") {
    PredOffset2D o{-1, 0};
    REQUIRE_EQ(o.dy, -1);
    REQUIRE_EQ(o.dx, 0);
}

TEST_CASE("PredOffset3D has expected fields") {
    PredOffset3D o{-1, 0, 1};
    REQUIRE_EQ(o.dz, -1);
    REQUIRE_EQ(o.dy, 0);
    REQUIRE_EQ(o.dx, 1);
}

// ============================================================================
// for_each_neighbor with various integral types
// ============================================================================

TEST_CASE("for_each_neighbor: works with uint8_t data") {
    std::vector<std::uint8_t> data = {10, 20, 30, 40, 50, 60, 70, 80, 90};
    std::mdspan<const std::uint8_t, std::dextents<std::size_t, 2>> img(data.data(), 3, 3);

    std::vector<std::uint8_t> neighbors;
    detail::for_each_neighbor(img, 1, 1, detail::cross_offsets,
        [&](std::uint8_t v) { neighbors.push_back(v); });

    REQUIRE_EQ(neighbors.size(), std::size_t(5));
    // center=50, up=20, down=80, left=40, right=60
    std::set<std::uint8_t> s(neighbors.begin(), neighbors.end());
    REQUIRE(s.contains(50)); // center
    REQUIRE(s.contains(20)); // up
    REQUIRE(s.contains(80)); // down
    REQUIRE(s.contains(40)); // left
    REQUIRE(s.contains(60)); // right
}

TEST_CASE("for_each_neighbor: works with int16_t data") {
    std::vector<std::int16_t> data = {-5, -4, -3, -2, -1, 0, 1, 2, 3};
    std::mdspan<const std::int16_t, std::dextents<std::size_t, 2>> img(data.data(), 3, 3);

    std::vector<std::int16_t> neighbors;
    detail::for_each_neighbor(img, 1, 1, detail::square_offsets,
        [&](std::int16_t v) { neighbors.push_back(v); });

    REQUIRE_EQ(neighbors.size(), std::size_t(9));
}

TEST_CASE("for_each_neighbor: works with uint32_t data") {
    std::vector<std::uint32_t> data = {100, 200, 300, 400, 500, 600, 700, 800, 900};
    std::mdspan<const std::uint32_t, std::dextents<std::size_t, 2>> img(data.data(), 3, 3);

    std::uint32_t sum = 0;
    detail::for_each_neighbor(img, 1, 1, detail::cross_offsets,
        [&](std::uint32_t v) { sum += v; });

    // center(500) + up(200) + down(800) + left(400) + right(600) = 2500
    REQUIRE_EQ(sum, 2500u);
}

TEST_CASE("for_each_neighbor: works with int64_t data and disk_5") {
    std::vector<std::int64_t> data(25, 1);
    std::mdspan<const std::int64_t, std::dextents<std::size_t, 2>> img(data.data(), 5, 5);

    std::int64_t count = 0;
    detail::for_each_neighbor(img, 2, 2, detail::disk5_offsets,
        [&](std::int64_t v) { count += v; });

    // Center (2,2) on 5x5 -- all 21 disk_5 offsets should be in bounds
    REQUIRE_EQ(count, 21);
}

// ============================================================================
// for_each_neighbor: edge and boundary iterations
// ============================================================================

TEST_CASE("for_each_neighbor: top-right corner with cross") {
    std::vector<int> data = {1,2,3, 4,5,6, 7,8,9};
    std::mdspan<const int, std::dextents<std::size_t, 2>> img(data.data(), 3, 3);

    std::vector<int> neighbors;
    detail::for_each_neighbor(img, 0, 2, detail::cross_offsets,
        [&](int v) { neighbors.push_back(v); });

    // Center(0,2)=3, left(0,1)=2, down(1,2)=6 -- up and right out of bounds
    REQUIRE_EQ(neighbors.size(), std::size_t(3));
    std::set<int> s(neighbors.begin(), neighbors.end());
    REQUIRE(s.contains(3)); // center
    REQUIRE(s.contains(2)); // left
    REQUIRE(s.contains(6)); // down
}

TEST_CASE("for_each_neighbor: bottom-left corner with cross") {
    std::vector<int> data = {1,2,3, 4,5,6, 7,8,9};
    std::mdspan<const int, std::dextents<std::size_t, 2>> img(data.data(), 3, 3);

    std::vector<int> neighbors;
    detail::for_each_neighbor(img, 2, 0, detail::cross_offsets,
        [&](int v) { neighbors.push_back(v); });

    // Center(2,0)=7, up(1,0)=4, right(2,1)=8
    REQUIRE_EQ(neighbors.size(), std::size_t(3));
    std::set<int> s(neighbors.begin(), neighbors.end());
    REQUIRE(s.contains(7));
    REQUIRE(s.contains(4));
    REQUIRE(s.contains(8));
}

// ============================================================================
// for_each_neighbor: disk_5 on edges
// ============================================================================

TEST_CASE("for_each_neighbor: disk_5 on top edge center") {
    std::vector<int> data(25);
    std::iota(data.begin(), data.end(), 0);
    std::mdspan<const int, std::dextents<std::size_t, 2>> img(data.data(), 5, 5);

    // (0,2) is top edge center
    std::vector<int> neighbors;
    detail::for_each_neighbor(img, 0, 2, detail::disk5_offsets,
        [&](int v) { neighbors.push_back(v); });

    // Count expected: offsets with dy+0>=0 and dx+2 in [0,4]
    int expected = 0;
    for (const auto& [dy, dx] : detail::disk5_offsets) {
        int nr = 0 + dy;
        int nc = 2 + dx;
        if (nr >= 0 && nr < 5 && nc >= 0 && nc < 5)
            ++expected;
    }
    REQUIRE_EQ(neighbors.size(), static_cast<std::size_t>(expected));
}

// ============================================================================
// for_each_neighbor: accumulate sum
// ============================================================================

TEST_CASE("for_each_neighbor: sum of neighbors") {
    std::vector<int> data = {1, 1, 1, 1, 1, 1, 1, 1, 1};
    std::mdspan<const int, std::dextents<std::size_t, 2>> img(data.data(), 3, 3);

    int sum = 0;
    detail::for_each_neighbor(img, 1, 1, detail::square_offsets,
        [&](int v) { sum += v; });

    REQUIRE_EQ(sum, 9); // all 9 cells are 1
}

// ============================================================================
// dispatch_se: return value from body
// ============================================================================

TEST_CASE("dispatch_se: body returns value") {
    auto size = detail::dispatch_se(StructuringElement::cross, [](const auto& offsets) {
        return offsets.size();
    });
    REQUIRE_EQ(size, std::size_t(5));

    auto size2 = detail::dispatch_se(StructuringElement::disk_5, [](const auto& offsets) {
        return offsets.size();
    });
    REQUIRE_EQ(size2, std::size_t(21));
}

// ============================================================================
// for_each_neighbor: rectangular image (non-square)
// ============================================================================

TEST_CASE("for_each_neighbor: non-square image") {
    // 2 rows x 5 cols
    std::vector<int> data = {1,2,3,4,5, 6,7,8,9,10};
    std::mdspan<const int, std::dextents<std::size_t, 2>> img(data.data(), 2, 5);

    // Center of bottom row: (1,2)=8
    std::vector<int> neighbors;
    detail::for_each_neighbor(img, 1, 2, detail::cross_offsets,
        [&](int v) { neighbors.push_back(v); });

    // center(1,2)=8, up(0,2)=3, left(1,1)=7, right(1,3)=9
    // down(2,2) is out of bounds
    REQUIRE_EQ(neighbors.size(), std::size_t(4));
    std::set<int> s(neighbors.begin(), neighbors.end());
    REQUIRE(s.contains(8));
    REQUIRE(s.contains(3));
    REQUIRE(s.contains(7));
    REQUIRE(s.contains(9));
}

UTILS2_TEST_MAIN()
