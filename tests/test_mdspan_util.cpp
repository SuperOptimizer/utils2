#include <utils2/test.hpp>
#include <utils2/mdspan_util.hpp>
#include <numeric>

using namespace utils2;

// ============================================================================
// Compile-time tests
// ============================================================================

// Stride computation
static_assert(compute_strides<3>({2, 3, 4}) == std::array<std::size_t, 3>{12, 4, 1});
static_assert(compute_strides<2>({5, 10}) == std::array<std::size_t, 2>{10, 1});
static_assert(compute_strides<1>({7}) == std::array<std::size_t, 1>{1});

// Linear index computation
static_assert(linear_index<3>({12, 4, 1}, {0, 0, 0}) == 0);
static_assert(linear_index<3>({12, 4, 1}, {1, 0, 0}) == 12);
static_assert(linear_index<3>({12, 4, 1}, {0, 1, 0}) == 4);
static_assert(linear_index<3>({12, 4, 1}, {0, 0, 1}) == 1);
static_assert(linear_index<2>({10, 1}, {3, 7}) == 37);

// Slice size
static_assert(Slice{2, 5}.size() == 3);
static_assert(Slice{0, 0}.size() == 0);

// ============================================================================
// Runtime tests
// ============================================================================

TEST_CASE("NDArray: 1D construction and access") {
    NDArray<int, 1> arr(std::array<std::size_t, 1>{5});
    REQUIRE_EQ(arr.size(), std::size_t{5});
    REQUIRE_EQ(arr.extent(0), std::size_t{5});

    for (std::size_t i = 0; i < 5; ++i)
        arr(i) = static_cast<int>(i * 10);

    REQUIRE_EQ(arr(0), 0);
    REQUIRE_EQ(arr(3), 30);
    REQUIRE_EQ(arr(4), 40);
}

TEST_CASE("NDArray: 2D construction and access") {
    NDArray<double, 2> arr(std::array<std::size_t, 2>{3, 4});
    REQUIRE_EQ(arr.size(), std::size_t{12});
    REQUIRE_EQ(arr.extent(0), std::size_t{3});
    REQUIRE_EQ(arr.extent(1), std::size_t{4});

    for (std::size_t r = 0; r < 3; ++r)
        for (std::size_t c = 0; c < 4; ++c)
            arr(r, c) = static_cast<double>(r * 10 + c);

    REQUIRE_NEAR(arr(0, 0), 0.0, 1e-9);
    REQUIRE_NEAR(arr(1, 2), 12.0, 1e-9);
    REQUIRE_NEAR(arr(2, 3), 23.0, 1e-9);
}

TEST_CASE("NDArray: 3D construction and access") {
    NDArray<int, 3> arr(std::array<std::size_t, 3>{2, 3, 4});
    REQUIRE_EQ(arr.size(), std::size_t{24});

    int val = 0;
    for (std::size_t i = 0; i < 2; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            for (std::size_t k = 0; k < 4; ++k)
                arr(i, j, k) = val++;

    REQUIRE_EQ(arr(0, 0, 0), 0);
    REQUIRE_EQ(arr(0, 0, 3), 3);
    REQUIRE_EQ(arr(1, 2, 3), 23);
}

TEST_CASE("NDArray: fill value constructor") {
    NDArray<int, 2> arr(std::array<std::size_t, 2>{3, 3}, 42);
    REQUIRE_EQ(arr.size(), std::size_t{9});
    for (std::size_t r = 0; r < 3; ++r)
        for (std::size_t c = 0; c < 3; ++c)
            REQUIRE_EQ(arr(r, c), 42);
}

TEST_CASE("NDArray: shape and flat access") {
    NDArray<int, 2> arr(std::array<std::size_t, 2>{2, 3}, 7);
    auto shape = arr.shape();
    REQUIRE_EQ(shape[0], std::size_t{2});
    REQUIRE_EQ(shape[1], std::size_t{3});

    auto flat = arr.flat();
    REQUIRE_EQ(flat.size(), std::size_t{6});
    for (auto v : flat)
        REQUIRE_EQ(v, 7);
}

TEST_CASE("NDArray: data pointer") {
    NDArray<float, 1> arr(std::array<std::size_t, 1>{4}, 1.5f);
    float* p = arr.data();
    REQUIRE(p != nullptr);
    REQUIRE_NEAR(p[0], 1.5f, 1e-6);
    REQUIRE_NEAR(p[3], 1.5f, 1e-6);
}

TEST_CASE("NDArray: mdspan view") {
    NDArray<int, 2> arr(std::array<std::size_t, 2>{3, 4}, 0);
    auto v = arr.view();

    // Write through the view
    for (std::size_t r = 0; r < 3; ++r)
        for (std::size_t c = 0; c < 4; ++c)
            v[r, c] = static_cast<int>(r * 4 + c);

    // Read back through NDArray
    REQUIRE_EQ(arr(0, 0), 0);
    REQUIRE_EQ(arr(2, 3), 11);
}

TEST_CASE("subview: 2D extraction") {
    NDArray<int, 2> arr(std::array<std::size_t, 2>{4, 5}, 0);
    // Fill with row*10 + col
    for (std::size_t r = 0; r < 4; ++r)
        for (std::size_t c = 0; c < 5; ++c)
            arr(r, c) = static_cast<int>(r * 10 + c);

    auto v = arr.view();
    // Extract rows [1,3), cols [2,5) => 2x3 sub-array
    auto sub = subview(
        std::mdspan<const int, std::dextents<std::size_t, 2>>(arr.data(), 4, 5),
        Slice{1, 3}, Slice{2, 5});

    REQUIRE_EQ(sub.extent(0), std::size_t{2});
    REQUIRE_EQ(sub.extent(1), std::size_t{3});
    REQUIRE_EQ(sub(0, 0), 12); // arr(1,2)
    REQUIRE_EQ(sub(0, 2), 14); // arr(1,4)
    REQUIRE_EQ(sub(1, 0), 22); // arr(2,2)
    REQUIRE_EQ(sub(1, 2), 24); // arr(2,4)
}

TEST_CASE("TypeErasedBuffer: construction and typed access") {
    TypeErasedBuffer buf(sizeof(float), {4, 3, 2});
    REQUIRE_EQ(buf.element_size(), sizeof(float));
    REQUIRE_EQ(buf.total_elements(), std::size_t{24});
    REQUIRE_EQ(buf.size_bytes(), sizeof(float) * 24);

    auto shape = buf.shape();
    REQUIRE_EQ(shape[0], std::size_t{4});
    REQUIRE_EQ(shape[1], std::size_t{3});
    REQUIRE_EQ(shape[2], std::size_t{2});

    // Write via typed pointer
    float* p = buf.as<float>();
    for (std::size_t i = 0; i < 24; ++i)
        p[i] = static_cast<float>(i);

    // Read back
    const float* cp = buf.as<float>();
    REQUIRE_NEAR(cp[0], 0.f, 1e-6);
    REQUIRE_NEAR(cp[23], 23.f, 1e-6);
}

TEST_CASE("TypeErasedBuffer: mdspan view") {
    TypeErasedBuffer buf(sizeof(int), {2, 3, 4});
    auto view = buf.mdspan_view<int>();

    int val = 0;
    for (std::size_t i = 0; i < 2; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            for (std::size_t k = 0; k < 4; ++k)
                view[i, j, k] = val++;

    REQUIRE_EQ((view[0, 0, 0]), 0);
    REQUIRE_EQ((view[1, 2, 3]), 23);
}

TEST_CASE("TypeErasedBuffer: raw byte access") {
    TypeErasedBuffer buf(sizeof(int), {2, 2, 1});
    auto raw = buf.raw();
    REQUIRE_EQ(raw.size(), sizeof(int) * 4);
    // Initially zero
    for (auto b : raw)
        REQUIRE_EQ(static_cast<int>(b), 0);
}

TEST_CASE("TypeErasedBuffer: default construction") {
    TypeErasedBuffer buf;
    REQUIRE_EQ(buf.element_size(), std::size_t{0});
    REQUIRE_EQ(buf.total_elements(), std::size_t{0});
    REQUIRE_EQ(buf.size_bytes(), std::size_t{0});
}

TEST_CASE("fill: mdspan bulk fill") {
    NDArray<int, 2> arr(std::array<std::size_t, 2>{3, 4}, 0);
    auto v = arr.view();
    utils2::fill(v, 99);

    for (std::size_t r = 0; r < 3; ++r)
        for (std::size_t c = 0; c < 4; ++c)
            REQUIRE_EQ(arr(r, c), 99);
}

TEST_CASE("copy: mdspan bulk copy") {
    NDArray<int, 2> src(std::array<std::size_t, 2>{2, 3}, 0);
    NDArray<int, 2> dst(std::array<std::size_t, 2>{2, 3}, 0);

    for (std::size_t r = 0; r < 2; ++r)
        for (std::size_t c = 0; c < 3; ++c)
            src(r, c) = static_cast<int>(r * 3 + c);

    auto src_view = std::mdspan<const int, std::dextents<std::size_t, 2>>(
        src.data(), 2, 3);
    auto dst_view = dst.view();
    utils2::copy(src_view, dst_view);

    for (std::size_t r = 0; r < 2; ++r)
        for (std::size_t c = 0; c < 3; ++c)
            REQUIRE_EQ(dst(r, c), src(r, c));
}

TEST_CASE("transform: mdspan bulk transform") {
    NDArray<int, 2> arr(std::array<std::size_t, 2>{2, 3}, 5);
    auto v = arr.view();
    utils2::transform(v, [](int x) { return x * 2 + 1; });

    for (std::size_t r = 0; r < 2; ++r)
        for (std::size_t c = 0; c < 3; ++c)
            REQUIRE_EQ(arr(r, c), 11);
}

TEST_CASE("compute_strides: edge cases") {
    // Single dimension
    auto s1 = compute_strides<1>({100});
    REQUIRE_EQ(s1[0], std::size_t{1});

    // All size-1 dimensions
    auto s3 = compute_strides<3>({1, 1, 1});
    REQUIRE_EQ(s3[0], std::size_t{1});
    REQUIRE_EQ(s3[1], std::size_t{1});
    REQUIRE_EQ(s3[2], std::size_t{1});
}

UTILS2_TEST_MAIN()
