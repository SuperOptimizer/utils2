#include <utils2/test.hpp>
#include <utils2/spatial_index.hpp>
#include <vector>
#include <cmath>
#include <array>

UTILS2_TEST_MAIN()

using Point2 = std::array<double, 2>;
using Point3 = std::array<double, 3>;

// ---------------------------------------------------------------------------
// AABB tests
// ---------------------------------------------------------------------------

TEST_CASE("AABB: contains point") {
    utils2::AABB<2> box{.lo = {0.0, 0.0}, .hi = {10.0, 10.0}};
    REQUIRE(box.contains({5.0, 5.0}));
    REQUIRE(box.contains({0.0, 0.0}));   // on boundary
    REQUIRE(box.contains({10.0, 10.0})); // on boundary
    REQUIRE(!box.contains({-1.0, 5.0}));
    REQUIRE(!box.contains({5.0, 11.0}));
}

TEST_CASE("AABB: intersects") {
    utils2::AABB<2> a{.lo = {0.0, 0.0}, .hi = {10.0, 10.0}};
    utils2::AABB<2> b{.lo = {5.0, 5.0}, .hi = {15.0, 15.0}};
    utils2::AABB<2> c{.lo = {20.0, 20.0}, .hi = {30.0, 30.0}};

    REQUIRE(a.intersects(b));
    REQUIRE(b.intersects(a));
    REQUIRE(!a.intersects(c));
}

TEST_CASE("AABB: merge") {
    utils2::AABB<2> a{.lo = {0.0, 0.0}, .hi = {5.0, 5.0}};
    utils2::AABB<2> b{.lo = {3.0, 3.0}, .hi = {10.0, 10.0}};
    auto m = a.merge(b);

    REQUIRE_NEAR(m.lo[0], 0.0, 1e-9);
    REQUIRE_NEAR(m.lo[1], 0.0, 1e-9);
    REQUIRE_NEAR(m.hi[0], 10.0, 1e-9);
    REQUIRE_NEAR(m.hi[1], 10.0, 1e-9);
}

TEST_CASE("AABB: distance_sq to point") {
    utils2::AABB<2> box{.lo = {0.0, 0.0}, .hi = {10.0, 10.0}};

    // Point inside -- distance is 0
    REQUIRE_NEAR(box.distance_sq({5.0, 5.0}), 0.0, 1e-9);

    // Point outside along x-axis
    REQUIRE_NEAR(box.distance_sq({15.0, 5.0}), 25.0, 1e-9);

    // Point at corner diagonal
    REQUIRE_NEAR(box.distance_sq({12.0, 13.0}), 4.0 + 9.0, 1e-9);
}

TEST_CASE("AABB: center") {
    utils2::AABB<3> box{.lo = {0.0, 2.0, 4.0}, .hi = {10.0, 8.0, 6.0}};
    auto c = box.center();
    REQUIRE_NEAR(c[0], 5.0, 1e-9);
    REQUIRE_NEAR(c[1], 5.0, 1e-9);
    REQUIRE_NEAR(c[2], 5.0, 1e-9);
}

// ---------------------------------------------------------------------------
// KDTree tests
// ---------------------------------------------------------------------------

TEST_CASE("KDTree: empty tree") {
    utils2::KDTree<Point2, 2> tree;
    tree.build({});
    REQUIRE_EQ(tree.size(), 0u);
    REQUIRE(tree.empty());

    auto r = tree.knn({0.0, 0.0}, 1);
    REQUIRE(r.empty());
}

TEST_CASE("KDTree: single point") {
    utils2::KDTree<Point2, 2> tree;
    tree.build({{1.0, 2.0}});
    REQUIRE_EQ(tree.size(), 1u);

    auto r = tree.nearest({0.0, 0.0});
    REQUIRE(r.has_value());
    REQUIRE_NEAR((*r->item)[0], 1.0, 1e-9);
    REQUIRE_NEAR((*r->item)[1], 2.0, 1e-9);
}

TEST_CASE("KDTree: knn query") {
    utils2::KDTree<Point2, 2> tree;
    std::vector<Point2> pts = {
        {0.0, 0.0}, {1.0, 0.0}, {2.0, 0.0}, {10.0, 10.0}, {5.0, 5.0}
    };
    tree.build(pts);

    auto r = tree.knn({0.0, 0.0}, 3);
    REQUIRE_EQ(r.size(), 3u);

    // Closest first: (0,0), (1,0), (2,0)
    REQUIRE_NEAR(r[0].distance_sq, 0.0, 1e-9);
    REQUIRE_NEAR(r[1].distance_sq, 1.0, 1e-9);
    REQUIRE_NEAR(r[2].distance_sq, 4.0, 1e-9);
}

TEST_CASE("KDTree: radius query") {
    utils2::KDTree<Point2, 2> tree;
    std::vector<Point2> pts = {
        {0.0, 0.0}, {1.0, 0.0}, {2.0, 0.0}, {10.0, 10.0}
    };
    tree.build(pts);

    auto r = tree.radius({0.0, 0.0}, 1.5);
    REQUIRE_EQ(r.size(), 2u);  // (0,0) and (1,0)
}

TEST_CASE("KDTree: range (AABB) query") {
    utils2::KDTree<Point2, 2> tree;
    std::vector<Point2> pts = {
        {1.0, 1.0}, {3.0, 3.0}, {5.0, 5.0}, {7.0, 7.0}
    };
    tree.build(pts);

    auto r = tree.range({0.0, 0.0}, {4.0, 4.0});
    REQUIRE_EQ(r.size(), 2u);  // (1,1) and (3,3)
}

TEST_CASE("KDTree: insert and remove") {
    utils2::KDTree<Point2, 2> tree;
    tree.build({{0.0, 0.0}, {1.0, 1.0}});
    REQUIRE_EQ(tree.size(), 2u);

    tree.insert({2.0, 2.0});
    REQUIRE_EQ(tree.size(), 3u);

    bool removed = tree.remove({1.0, 1.0});
    REQUIRE(removed);
    REQUIRE_EQ(tree.size(), 2u);

    // Removed item should not appear in queries
    auto r = tree.knn({1.0, 1.0}, 5);
    for (auto& qr : r) {
        double dx = (*qr.item)[0] - 1.0;
        double dy = (*qr.item)[1] - 1.0;
        CHECK(std::abs(dx) > 1e-9 || std::abs(dy) > 1e-9);
    }
}

TEST_CASE("KDTree: knn with max_distance") {
    utils2::KDTree<Point2, 2> tree;
    std::vector<Point2> pts = {{0.0, 0.0}, {100.0, 100.0}};
    tree.build(pts);

    auto r = tree.knn({0.0, 0.0}, 10, 5.0);
    REQUIRE_EQ(r.size(), 1u);  // only (0,0) within distance 5
}

// ---------------------------------------------------------------------------
// BVH tests
// ---------------------------------------------------------------------------

struct BoxItem {
    utils2::AABB<2> bounds;
    int id;
};

struct BoxExtractor {
    utils2::AABB<2> operator()(const BoxItem& item) const {
        return item.bounds;
    }
};

TEST_CASE("BVH: build and query") {
    utils2::BVH<BoxItem, 2, BoxExtractor> bvh;
    std::vector<BoxItem> items = {
        {{{0.0, 0.0}, {2.0, 2.0}}, 1},
        {{{5.0, 5.0}, {7.0, 7.0}}, 2},
        {{{10.0, 10.0}, {12.0, 12.0}}, 3},
    };
    bvh.build(items);
    REQUIRE_EQ(bvh.size(), 3u);

    // Query overlapping first box
    utils2::AABB<2> qbox{.lo = {1.0, 1.0}, .hi = {3.0, 3.0}};
    auto r = bvh.query(qbox);
    REQUIRE_EQ(r.size(), 1u);
    REQUIRE_EQ(r[0]->id, 1);
}

TEST_CASE("BVH: query_point") {
    utils2::BVH<BoxItem, 2, BoxExtractor> bvh;
    std::vector<BoxItem> items = {
        {{{0.0, 0.0}, {5.0, 5.0}}, 1},
        {{{3.0, 3.0}, {8.0, 8.0}}, 2},
    };
    bvh.build(items);

    auto r = bvh.query_point({4.0, 4.0});
    REQUIRE_EQ(r.size(), 2u);  // both boxes contain the point
}

// ---- Bug hunt: KDTree with duplicate points --------------------------------

TEST_CASE("KDTree: duplicate points knn") {
    utils2::KDTree<Point2, 2> tree;
    std::vector<Point2> pts = {
        {1.0, 1.0}, {1.0, 1.0}, {1.0, 1.0}, {5.0, 5.0}
    };
    tree.build(pts);
    REQUIRE_EQ(tree.size(), 4u);

    // k=2 nearest to origin should return 2 of the duplicates at (1,1)
    auto r = tree.knn({0.0, 0.0}, 2);
    REQUIRE_EQ(r.size(), 2u);
    CHECK_NEAR(r[0].distance_sq, 2.0, 1e-9);
    CHECK_NEAR(r[1].distance_sq, 2.0, 1e-9);

    // k=4 should return all
    auto r4 = tree.knn({0.0, 0.0}, 4);
    REQUIRE_EQ(r4.size(), 4u);
}

TEST_CASE("KDTree: collinear points (degenerate structure)") {
    // All points on the x-axis
    utils2::KDTree<Point2, 2> tree;
    std::vector<Point2> pts;
    for (int i = 0; i < 20; ++i)
        pts.push_back({static_cast<double>(i), 0.0});
    tree.build(pts);

    // Query point above the line
    auto r = tree.nearest({10.0, 5.0});
    REQUIRE(r.has_value());
    CHECK_NEAR((*r->item)[0], 10.0, 1e-9);
    CHECK_NEAR((*r->item)[1], 0.0, 1e-9);
    CHECK_NEAR(r->distance_sq, 25.0, 1e-9);

    // knn with 3
    auto r3 = tree.knn({10.0, 0.0}, 3);
    REQUIRE_EQ(r3.size(), 3u);
    // Should return 10, 9, 11 (or 10, 11, 9)
    CHECK_NEAR(r3[0].distance_sq, 0.0, 1e-9);
    CHECK_NEAR(r3[1].distance_sq, 1.0, 1e-9);
    CHECK_NEAR(r3[2].distance_sq, 1.0, 1e-9);
}

TEST_CASE("KDTree: nearest brute-force verification") {
    // Build a tree with random-ish points and verify nearest matches brute force
    utils2::KDTree<Point2, 2> tree;
    std::vector<Point2> pts;
    // Deterministic "random" points using a simple formula
    for (int i = 0; i < 50; ++i) {
        double x = static_cast<double>((i * 7 + 3) % 41);
        double y = static_cast<double>((i * 13 + 5) % 37);
        pts.push_back({x, y});
    }
    tree.build(pts);

    // Test 10 query points
    for (int q = 0; q < 10; ++q) {
        double qx = static_cast<double>((q * 11 + 1) % 43);
        double qy = static_cast<double>((q * 17 + 2) % 39);
        Point2 query{qx, qy};

        auto result = tree.nearest(query);
        REQUIRE(result.has_value());

        // Brute-force find nearest
        double best_dist = 1e18;
        std::size_t best_idx = 0;
        for (std::size_t i = 0; i < pts.size(); ++i) {
            double dx = pts[i][0] - qx;
            double dy = pts[i][1] - qy;
            double d = dx * dx + dy * dy;
            if (d < best_dist) {
                best_dist = d;
                best_idx = i;
            }
        }

        CHECK_NEAR(result->distance_sq, best_dist, 1e-9);
    }
}

TEST_CASE("KDTree: remove then query") {
    utils2::KDTree<Point2, 2> tree;
    std::vector<Point2> pts = {
        {0.0, 0.0}, {1.0, 0.0}, {2.0, 0.0}, {3.0, 0.0}, {4.0, 0.0}
    };
    tree.build(pts);

    // Remove the closest point to query
    bool removed = tree.remove({1.0, 0.0});
    REQUIRE(removed);

    auto r = tree.nearest({1.0, 0.0});
    REQUIRE(r.has_value());
    // Should return either (0,0) or (2,0), both at distance 1
    CHECK_NEAR(r->distance_sq, 1.0, 1e-9);
    // It should NOT return (1,0)
    double returned_x = (*r->item)[0];
    CHECK(std::abs(returned_x - 1.0) > 0.5);

    // Remove more and verify
    tree.remove({0.0, 0.0});
    tree.remove({2.0, 0.0});
    REQUIRE_EQ(tree.size(), 2u);

    auto r2 = tree.nearest({0.0, 0.0});
    REQUIRE(r2.has_value());
    // Should be (3,0) or (4,0)
    CHECK_GE((*r2->item)[0], 3.0 - 1e-9);
}

TEST_CASE("KDTree: 3D nearest") {
    utils2::KDTree<Point3, 3> tree;
    std::vector<Point3> pts = {
        {0.0, 0.0, 0.0}, {10.0, 0.0, 0.0}, {0.0, 10.0, 0.0}, {0.0, 0.0, 10.0}
    };
    tree.build(pts);

    auto r = tree.nearest({1.0, 1.0, 1.0});
    REQUIRE(r.has_value());
    CHECK_NEAR(r->distance_sq, 3.0, 1e-9);
    CHECK_NEAR((*r->item)[0], 0.0, 1e-9);
}

TEST_CASE("BVH: empty tree") {
    utils2::BVH<BoxItem, 2, BoxExtractor> bvh;
    bvh.build({});
    REQUIRE_EQ(bvh.size(), 0u);

    auto r = bvh.query(utils2::AABB<2>{.lo = {0.0, 0.0}, .hi = {10.0, 10.0}});
    REQUIRE(r.empty());
}
