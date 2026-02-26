#include <utils2/test.hpp>
#include <utils2/disjoint_set.hpp>
#include <string>
#include <algorithm>
#include <set>

UTILS2_TEST_MAIN()

// ---------------------------------------------------------------------------
// DisjointSet (integer-label)
// ---------------------------------------------------------------------------

TEST_CASE("DisjointSet: initial state") {
    utils2::DisjointSet ds(5);
    REQUIRE_EQ(ds.size(), 5u);
    REQUIRE_EQ(ds.num_sets(), 5u);

    // Each element is its own root
    for (uint32_t i = 0; i < 5; ++i) {
        REQUIRE_EQ(ds.find(i), i);
    }
}

TEST_CASE("DisjointSet: basic find and unite") {
    utils2::DisjointSet ds(6);
    ds.unite(0, 1);
    ds.unite(2, 3);
    ds.unite(4, 5);

    REQUIRE(ds.connected(0, 1));
    REQUIRE(ds.connected(2, 3));
    REQUIRE(!ds.connected(0, 2));
    REQUIRE_EQ(ds.num_sets(), 3u);

    ds.unite(0, 2);
    REQUIRE(ds.connected(0, 3));
    REQUIRE(ds.connected(1, 2));
    REQUIRE_EQ(ds.num_sets(), 2u);
}

TEST_CASE("DisjointSet: path compression") {
    utils2::DisjointSet ds(8);
    // Build a chain: 0->1->2->3
    ds.unite(0, 1);
    ds.unite(1, 2);
    ds.unite(2, 3);

    // find(0) should walk to root and compress
    auto root = ds.find(0);
    // After path compression, find(0) should return root directly
    REQUIRE_EQ(ds.find(0), root);
    // All should share the same root
    REQUIRE_EQ(ds.find(1), root);
    REQUIRE_EQ(ds.find(2), root);
    REQUIRE_EQ(ds.find(3), root);
}

TEST_CASE("DisjointSet: connected check") {
    utils2::DisjointSet ds(4);
    REQUIRE(!ds.connected(0, 3));
    ds.unite(0, 1);
    ds.unite(1, 3);
    REQUIRE(ds.connected(0, 3));
    REQUIRE(!ds.connected(0, 2));
}

TEST_CASE("DisjointSet: num_sets tracking") {
    utils2::DisjointSet ds(10);
    REQUIRE_EQ(ds.num_sets(), 10u);

    ds.unite(0, 1);
    REQUIRE_EQ(ds.num_sets(), 9u);

    // Uniting already-connected elements should not change count
    ds.unite(0, 1);
    REQUIRE_EQ(ds.num_sets(), 9u);

    for (uint32_t i = 0; i < 9; ++i) {
        ds.unite(i, i + 1);
    }
    REQUIRE_EQ(ds.num_sets(), 1u);
}

TEST_CASE("DisjointSet: set_size") {
    utils2::DisjointSet ds(5);
    REQUIRE_EQ(ds.set_size(0), 1u);

    ds.unite(0, 1);
    REQUIRE_EQ(ds.set_size(0), 2u);
    REQUIRE_EQ(ds.set_size(1), 2u);

    ds.unite(2, 3);
    ds.unite(0, 2);
    REQUIRE_EQ(ds.set_size(0), 4u);
    REQUIRE_EQ(ds.set_size(3), 4u);
    REQUIRE_EQ(ds.set_size(4), 1u);
}

TEST_CASE("DisjointSet: flatten") {
    utils2::DisjointSet ds(6);
    ds.unite(0, 1);
    ds.unite(1, 2);
    ds.unite(3, 4);
    ds.unite(4, 5);

    ds.flatten();

    // After flatten, find should be a direct lookup (parent == root)
    auto r1 = ds.find(0);
    REQUIRE_EQ(ds.find(1), r1);
    REQUIRE_EQ(ds.find(2), r1);

    auto r2 = ds.find(3);
    REQUIRE_EQ(ds.find(4), r2);
    REQUIRE_EQ(ds.find(5), r2);
}

TEST_CASE("DisjointSet: roots") {
    utils2::DisjointSet ds(5);
    ds.unite(0, 1);
    ds.unite(2, 3);

    auto r = ds.roots();
    REQUIRE_EQ(r.size(), 3u);  // 3 sets remain
}

TEST_CASE("DisjointSet: relabel") {
    utils2::DisjointSet ds(6);
    ds.unite(0, 1);
    ds.unite(2, 3);
    ds.unite(4, 5);

    auto labels = ds.relabel();
    REQUIRE_EQ(labels.size(), 6u);

    // Elements in the same set should have the same label
    REQUIRE_EQ(labels[0], labels[1]);
    REQUIRE_EQ(labels[2], labels[3]);
    REQUIRE_EQ(labels[4], labels[5]);

    // Elements in different sets should have different labels
    std::set<uint32_t> unique_labels(labels.begin(), labels.end());
    REQUIRE_EQ(unique_labels.size(), 3u);

    // Labels should be consecutive starting from 0
    REQUIRE(unique_labels.contains(0u));
    REQUIRE(unique_labels.contains(1u));
    REQUIRE(unique_labels.contains(2u));
}

// ---------------------------------------------------------------------------
// DynamicDisjointSet (string keys)
// ---------------------------------------------------------------------------

TEST_CASE("DynamicDisjointSet: add and find") {
    utils2::DynamicDisjointSet<std::string> ds;
    ds.add("a");
    ds.add("b");
    ds.add("c");

    REQUIRE_EQ(ds.size(), 3u);
    REQUIRE_EQ(ds.num_sets(), 3u);

    // Each element is its own root
    REQUIRE_EQ(ds.find("a"), std::string("a"));
    REQUIRE_EQ(ds.find("b"), std::string("b"));
}

TEST_CASE("DynamicDisjointSet: unite and connected") {
    utils2::DynamicDisjointSet<std::string> ds;
    ds.add("x");
    ds.add("y");
    ds.add("z");

    ds.unite("x", "y");
    REQUIRE(ds.connected("x", "y"));
    REQUIRE(!ds.connected("x", "z"));
    REQUIRE_EQ(ds.num_sets(), 2u);

    ds.unite("y", "z");
    REQUIRE(ds.connected("x", "z"));
    REQUIRE_EQ(ds.num_sets(), 1u);
}

TEST_CASE("DynamicDisjointSet: add is idempotent") {
    utils2::DynamicDisjointSet<std::string> ds;
    ds.add("a");
    ds.add("a");
    ds.add("a");
    REQUIRE_EQ(ds.size(), 1u);
    REQUIRE_EQ(ds.num_sets(), 1u);
}

// ---- Bug hunt: large element count and flatten/relabel ---------------------

TEST_CASE("DisjointSet: large N flatten correctness") {
    const std::size_t N = 10000;
    utils2::DisjointSet ds(N);

    // Unite all even elements into one set, all odd into another
    for (uint32_t i = 2; i < N; i += 2) ds.unite(0, i);
    for (uint32_t i = 3; i < N; i += 2) ds.unite(1, i);

    REQUIRE_EQ(ds.num_sets(), 2u);

    ds.flatten();

    // After flatten, every element's find should be O(1) and correct
    auto root_even = ds.find(0);
    auto root_odd = ds.find(1);
    REQUIRE_NE(root_even, root_odd);

    for (uint32_t i = 0; i < N; i += 2)
        CHECK_EQ(ds.find(i), root_even);
    for (uint32_t i = 1; i < N; i += 2)
        CHECK_EQ(ds.find(i), root_odd);

    // Verify roots() returns exactly 2
    auto r = ds.roots();
    REQUIRE_EQ(r.size(), 2u);

    // After flatten, roots should be actual roots (parent[root] == root)
    for (auto root : r) {
        CHECK_EQ(ds.find(root), root);
    }
}

TEST_CASE("DisjointSet: relabel produces consecutive labels") {
    utils2::DisjointSet ds(100);

    // Create 5 groups: elements 0-19, 20-39, 40-59, 60-79, 80-99
    for (uint32_t base = 0; base < 100; base += 20) {
        for (uint32_t i = base + 1; i < base + 20; ++i)
            ds.unite(base, i);
    }

    REQUIRE_EQ(ds.num_sets(), 5u);

    auto labels = ds.relabel();
    REQUIRE_EQ(labels.size(), 100u);

    // Labels should be 0..4
    std::set<uint32_t> unique_labels(labels.begin(), labels.end());
    REQUIRE_EQ(unique_labels.size(), 5u);
    REQUIRE(unique_labels.contains(0u));
    REQUIRE(unique_labels.contains(1u));
    REQUIRE(unique_labels.contains(2u));
    REQUIRE(unique_labels.contains(3u));
    REQUIRE(unique_labels.contains(4u));

    // Elements in same group should have same label
    for (uint32_t base = 0; base < 100; base += 20) {
        for (uint32_t i = base + 1; i < base + 20; ++i) {
            CHECK_EQ(labels[i], labels[base]);
        }
    }
}

TEST_CASE("DisjointSet: set_size after many merges") {
    utils2::DisjointSet ds(1000);
    // Merge all into one set by chain
    for (uint32_t i = 0; i < 999; ++i)
        ds.unite(i, i + 1);

    REQUIRE_EQ(ds.num_sets(), 1u);
    REQUIRE_EQ(ds.set_size(0), 1000u);
    REQUIRE_EQ(ds.set_size(999), 1000u);
    REQUIRE_EQ(ds.set_size(500), 1000u);
}

TEST_CASE("DisjointSet: unite returns correct root") {
    utils2::DisjointSet ds(10);
    // Unite and verify the returned root is indeed a root
    for (uint32_t i = 0; i < 9; ++i) {
        auto root = ds.unite(i, i + 1);
        CHECK_EQ(ds.find(root), root);
        CHECK(ds.connected(i, i + 1));
    }
}

TEST_CASE("DynamicDisjointSet: path compression with strings") {
    utils2::DynamicDisjointSet<std::string> ds;
    ds.add("a");
    ds.add("b");
    ds.add("c");
    ds.add("d");

    ds.unite("a", "b");
    ds.unite("b", "c");
    ds.unite("c", "d");

    // find should compress paths
    auto root = ds.find("a");
    REQUIRE_EQ(ds.find("d"), root);
    REQUIRE_EQ(ds.num_sets(), 1u);
}
