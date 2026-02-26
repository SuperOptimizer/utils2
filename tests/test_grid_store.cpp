#include <utils2/test.hpp>
#include <utils2/grid_store.hpp>
#include <string>
#include <vector>
#include <algorithm>

UTILS2_TEST_MAIN()

// ---------------------------------------------------------------------------
// 2D GridStore
// ---------------------------------------------------------------------------

TEST_CASE("GridStore2D: insert and at_point") {
    utils2::GridStore<int, 2> grid(1.0);
    grid.insert({0.5, 0.5}, 10);
    grid.insert({0.7, 0.7}, 20);

    auto items = grid.at_point({0.5, 0.5});
    REQUIRE_EQ(items.size(), 2u);  // same cell
    REQUIRE_EQ(grid.size(), 2u);
}

TEST_CASE("GridStore2D: at_point empty cell") {
    utils2::GridStore<int, 2> grid(1.0);
    auto items = grid.at_point({5.0, 5.0});
    REQUIRE_EQ(items.size(), 0u);
}

TEST_CASE("GridStore2D: box query") {
    utils2::GridStore<int, 2> grid(1.0);
    grid.insert({0.5, 0.5}, 1);
    grid.insert({1.5, 1.5}, 2);
    grid.insert({5.5, 5.5}, 3);

    auto results = grid.box_query({0.0, 0.0}, {2.0, 2.0});
    // Should find items in cells overlapping [0,2] x [0,2]
    REQUIRE_EQ(results.size(), 2u);
}

TEST_CASE("GridStore2D: radius query") {
    utils2::GridStore<int, 2> grid(1.0);
    grid.insert({0.5, 0.5}, 1);
    grid.insert({1.5, 1.5}, 2);
    grid.insert({10.5, 10.5}, 3);

    // Radius query is coarse (cell-level), so items in cells overlapping
    // the bounding box of the sphere are returned
    auto results = grid.radius_query({0.5, 0.5}, 2.0);
    REQUIRE_GE(results.size(), 1u);

    // The distant item should NOT be included
    bool found_distant = false;
    for (auto* ptr : results) {
        if (*ptr == 3) found_distant = true;
    }
    REQUIRE(!found_distant);
}

TEST_CASE("GridStore2D: neighbors query") {
    utils2::GridStore<int, 2> grid(1.0);
    // Place items in a 3x3 pattern of cells
    grid.insert({0.5, 0.5}, 1);    // cell (0,0)
    grid.insert({1.5, 0.5}, 2);    // cell (1,0)
    grid.insert({-0.5, -0.5}, 3);  // cell (-1,-1)
    grid.insert({10.5, 10.5}, 4);  // far away

    auto results = grid.neighbors({0.5, 0.5});
    // Should find items in the 3x3 neighborhood around cell (0,0)
    REQUIRE_GE(results.size(), 2u);  // at least items 1, 2, and 3

    // Far-away item should not appear
    bool found_far = false;
    for (auto* ptr : results) {
        if (*ptr == 4) found_far = true;
    }
    REQUIRE(!found_far);
}

TEST_CASE("GridStore2D: remove") {
    utils2::GridStore<int, 2> grid(1.0);
    grid.insert({0.5, 0.5}, 10);
    grid.insert({0.5, 0.5}, 20);
    REQUIRE_EQ(grid.size(), 2u);

    bool removed = grid.remove({0.5, 0.5}, 10);
    REQUIRE(removed);
    REQUIRE_EQ(grid.size(), 1u);

    // Removing non-existent item returns false
    REQUIRE(!grid.remove({0.5, 0.5}, 10));

    // Remove last item from cell
    REQUIRE(grid.remove({0.5, 0.5}, 20));
    REQUIRE_EQ(grid.size(), 0u);
    REQUIRE_EQ(grid.cell_count(), 0u);
}

TEST_CASE("GridStore2D: clear") {
    utils2::GridStore<int, 2> grid(1.0);
    grid.insert({0.5, 0.5}, 1);
    grid.insert({1.5, 1.5}, 2);
    grid.clear();
    REQUIRE(grid.empty());
    REQUIRE_EQ(grid.size(), 0u);
    REQUIRE_EQ(grid.cell_count(), 0u);
}

TEST_CASE("GridStore2D: for_each") {
    utils2::GridStore<int, 2> grid(1.0);
    grid.insert({0.5, 0.5}, 10);
    grid.insert({1.5, 1.5}, 20);
    grid.insert({2.5, 2.5}, 30);

    int sum = 0;
    grid.for_each([&](const int& v) { sum += v; });
    REQUIRE_EQ(sum, 60);
}

// ---------------------------------------------------------------------------
// 3D GridStore
// ---------------------------------------------------------------------------

TEST_CASE("GridStore3D: insert and at_point") {
    utils2::GridStore<std::string, 3> grid(2.0);
    grid.insert({1.0, 1.0, 1.0}, "a");
    grid.insert({1.5, 1.5, 1.5}, "b");

    auto items = grid.at_point({1.0, 1.0, 1.0});
    REQUIRE_EQ(items.size(), 2u);  // same cell [0,2)^3
}

TEST_CASE("GridStore3D: box query") {
    utils2::GridStore<int, 3> grid(1.0);
    grid.insert({0.5, 0.5, 0.5}, 1);
    grid.insert({1.5, 1.5, 1.5}, 2);
    grid.insert({10.5, 10.5, 10.5}, 3);

    auto results = grid.box_query({0.0, 0.0, 0.0}, {2.0, 2.0, 2.0});
    REQUIRE_EQ(results.size(), 2u);
}

TEST_CASE("GridStore3D: neighbors query") {
    utils2::GridStore<int, 3> grid(1.0);
    grid.insert({0.5, 0.5, 0.5}, 1);
    grid.insert({1.5, 0.5, 0.5}, 2);
    grid.insert({50.0, 50.0, 50.0}, 99);

    auto results = grid.neighbors({0.5, 0.5, 0.5});
    // 3D neighbors: 3^3 = 27 cells
    REQUIRE_GE(results.size(), 1u);

    // Far-away item should not appear
    bool found_far = false;
    for (auto* ptr : results) {
        if (*ptr == 99) found_far = true;
    }
    REQUIRE(!found_far);
}

TEST_CASE("GridStore3D: remove") {
    utils2::GridStore<int, 3> grid(1.0);
    grid.insert({0.5, 0.5, 0.5}, 42);
    REQUIRE_EQ(grid.size(), 1u);

    REQUIRE(grid.remove({0.5, 0.5, 0.5}, 42));
    REQUIRE_EQ(grid.size(), 0u);
    REQUIRE(!grid.remove({0.5, 0.5, 0.5}, 42));
}

TEST_CASE("GridStore: non-uniform cell sizes") {
    utils2::GridStore<int, 2> grid(std::array<double, 2>{2.0, 5.0});
    grid.insert({1.0, 1.0}, 1);  // cell (0, 0)
    grid.insert({3.0, 1.0}, 2);  // cell (1, 0) -- different x cell
    grid.insert({1.0, 6.0}, 3);  // cell (0, 1) -- different y cell

    auto items = grid.at_point({0.5, 0.5});
    REQUIRE_EQ(items.size(), 1u);  // only item 1

    auto items2 = grid.at_point({3.5, 0.5});
    REQUIRE_EQ(items2.size(), 1u);  // only item 2
}
