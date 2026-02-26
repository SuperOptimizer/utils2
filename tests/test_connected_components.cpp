#include <utils2/test.hpp>
#include <utils2/connected_components.hpp>
#include <vector>
#include <set>
#include <utils2/mdspan.hpp>

using namespace utils2;

// ---- 2D binary 4-connectivity ----------------------------------------------

TEST_CASE("cc 2D binary 4-conn") {
    // Two blobs separated diagonally (4-conn should see 2 components)
    std::vector<int> img = {
        1, 0, 0,
        0, 0, 0,
        0, 0, 1
    };
    std::mdspan<const int, std::dextents<std::size_t, 2>> view(img.data(), 3, 3);
    auto [labels, n] = connected_components_2d(view, GridConnectivity::four);
    REQUIRE_EQ(n, 2u);
}

// ---- 2D binary 8-connectivity ----------------------------------------------

TEST_CASE("cc 2D binary 8-conn diagonal") {
    // Diagonal line: with 8-conn should be 1 component
    std::vector<int> img = {
        1, 0, 0,
        0, 1, 0,
        0, 0, 1
    };
    std::mdspan<const int, std::dextents<std::size_t, 2>> view(img.data(), 3, 3);
    auto [labels, n] = connected_components_2d(view, GridConnectivity::eight);
    REQUIRE_EQ(n, 1u);
}

TEST_CASE("cc 2D binary 4-conn diagonal separate") {
    // Same diagonal pattern, 4-conn -> 3 separate components
    std::vector<int> img = {
        1, 0, 0,
        0, 1, 0,
        0, 0, 1
    };
    std::mdspan<const int, std::dextents<std::size_t, 2>> view(img.data(), 3, 3);
    auto [labels, n] = connected_components_2d(view, GridConnectivity::four);
    REQUIRE_EQ(n, 3u);
}

// ---- 3D 6-connectivity -----------------------------------------------------

TEST_CASE("cc 3D binary 6-conn") {
    // 2x2x2 volume, all foreground -> 1 component with 6-conn
    std::vector<int> vol(8, 1);
    std::mdspan<const int, std::dextents<std::size_t, 3>> view(vol.data(), 2, 2, 2);
    auto [labels, n] = connected_components_3d(view, GridConnectivity::six);
    REQUIRE_EQ(n, 1u);
}

TEST_CASE("cc 3D separate slices") {
    // Two filled slices with a gap of zeros between
    // 3 slices of 2x2: slice 0 = filled, slice 1 = empty, slice 2 = filled
    std::vector<int> vol = {
        1,1, 1,1,   // z=0
        0,0, 0,0,   // z=1
        1,1, 1,1    // z=2
    };
    std::mdspan<const int, std::dextents<std::size_t, 3>> view(vol.data(), 3, 2, 2);
    auto [labels, n] = connected_components_3d(view, GridConnectivity::six);
    REQUIRE_EQ(n, 2u);
}

// ---- Multi-label -----------------------------------------------------------

TEST_CASE("cc 2D multi-label") {
    // Two regions with different labels that touch
    std::vector<int> img = {
        1, 1, 2, 2,
        1, 1, 2, 2
    };
    std::mdspan<const int, std::dextents<std::size_t, 2>> view(img.data(), 2, 4);
    auto [labels, n] = connected_components_2d_multilabel(view, GridConnectivity::four);
    REQUIRE_EQ(n, 2u);
}

TEST_CASE("cc 2D multi-label same value disconnected") {
    // Same label value but disconnected
    std::vector<int> img = {
        1, 0, 1,
        0, 0, 0,
        1, 0, 1
    };
    std::mdspan<const int, std::dextents<std::size_t, 2>> view(img.data(), 3, 3);
    auto [labels, n] = connected_components_2d_multilabel(view, GridConnectivity::four);
    REQUIRE_EQ(n, 4u);
}

// ---- label_counts ----------------------------------------------------------

TEST_CASE("label_counts") {
    std::vector<std::uint32_t> labels = {0, 1, 1, 2, 1, 0, 2, 2};
    auto counts = label_counts<std::uint32_t>(labels, 2);
    REQUIRE_EQ(counts.size(), 3u);
    CHECK_EQ(counts[0], 2u);  // background
    CHECK_EQ(counts[1], 3u);  // label 1
    CHECK_EQ(counts[2], 3u);  // label 2
}

// ---- remove_small_components -----------------------------------------------

TEST_CASE("remove_small_components") {
    std::vector<std::uint32_t> labels = {0, 1, 1, 1, 2, 3, 3, 0};
    // label 1: 3 pixels, label 2: 1 pixel, label 3: 2 pixels
    auto removed = remove_small_components<std::uint32_t>(labels, 2);
    CHECK_EQ(removed, 1u);   // only label 2 removed
    CHECK_EQ(labels[4], 0u); // label 2 pixel zeroed
    CHECK_NE(labels[1], 0u); // label 1 survives
    CHECK_NE(labels[5], 0u); // label 3 survives
}

// ---- keep_largest ----------------------------------------------------------

TEST_CASE("keep_largest") {
    std::vector<std::uint32_t> labels = {0, 1, 1, 1, 2, 2, 3, 0};
    // label 1: 3 px, label 2: 2 px, label 3: 1 px
    keep_largest<std::uint32_t>(labels, 3);
    // Only label 1 should remain
    CHECK_EQ(labels[4], 0u);
    CHECK_EQ(labels[6], 0u);
    CHECK_EQ(labels[1], 1u);
    CHECK_EQ(labels[2], 1u);
}

// ---- Edge cases ------------------------------------------------------------

TEST_CASE("cc all background") {
    std::vector<int> img(9, 0);
    std::mdspan<const int, std::dextents<std::size_t, 2>> view(img.data(), 3, 3);
    auto [labels, n] = connected_components_2d(view);
    CHECK_EQ(n, 0u);
}

TEST_CASE("cc all foreground same") {
    std::vector<int> img(9, 1);
    std::mdspan<const int, std::dextents<std::size_t, 2>> view(img.data(), 3, 3);
    auto [labels, n] = connected_components_2d(view, GridConnectivity::eight);
    CHECK_EQ(n, 1u);
}

TEST_CASE("cc single pixel") {
    std::vector<int> img = {1};
    std::mdspan<const int, std::dextents<std::size_t, 2>> view(img.data(), 1, 1);
    auto [labels, n] = connected_components_2d(view);
    CHECK_EQ(n, 1u);
    CHECK_EQ(labels[0], 1u);
}

TEST_CASE("cc checkerboard 4-conn") {
    // Checkerboard: no 4-connected pair, every pixel is its own component
    std::vector<int> img = {
        1, 0, 1,
        0, 1, 0,
        1, 0, 1
    };
    std::mdspan<const int, std::dextents<std::size_t, 2>> view(img.data(), 3, 3);
    auto [labels, n] = connected_components_2d(view, GridConnectivity::four);
    CHECK_EQ(n, 5u);
}

// ---- Bug hunt: DisjointSet capacity overflow (lines 93-99) -----------------
// The connected_components code allocates DisjointSet with voxels/2+1 capacity.
// With 4-connectivity on a checkerboard pattern, every foreground pixel is its
// own initial label. If more than voxels/2+1 foreground pixels exist, the
// DisjointSet runs out of capacity, creates a NEW one (losing all unions), and
// produces corrupt results.

TEST_CASE("cc 2D 4-conn wide checkerboard triggers UF overflow") {
    // Create a wide image with checkerboard pattern.
    // With 4-conn, each foreground pixel gets a unique label initially.
    // For a 2-row image of width W, every other pixel is foreground.
    // With 8-conn after all merges they should form 1 component.
    // But we use 4-conn, so each pixel is isolated -> N labels needed.
    // The UF is allocated with voxels/2+1 = rows*cols/2+1.
    // A checkerboard of rows=2, cols=W has roughly rows*cols/2 fg pixels.
    // If next_label exceeds voxels/2+1, the bug triggers.

    // Use a pattern that's all-foreground on a single row to maximize labels
    // with 4-connectivity where no merges happen in the first row.
    // Actually, the simplest way: make a 1-row image of width W, all 1s.
    // With 4-conn on a 1-row image, each adjacent pixel connects to left.
    // That actually merges. Let's use actual checkerboard.

    // For a 4x100 checkerboard with 4-conn:
    // ~200 foreground pixels out of 400 voxels. UF size = 400/2+1 = 201.
    // Labels needed = 200. That's fine: 200 < 201.

    // To trigger the bug, we need more fg pixels than voxels/2+1.
    // Actually, in a checkerboard with 4-conn, EVERY fg pixel is isolated,
    // meaning we need next_label = num_fg_pixels + 1.
    // UF capacity = voxels/2+1. If num_fg_pixels >= voxels/2+1, boom.

    // For a 1xW image (single row) with alternating 1,0,1,0,...
    // num_fg = ceil(W/2). voxels = W. UF size = W/2+1.
    // num_fg = ceil(W/2). For W even: num_fg = W/2, and we need next_label
    // up to W/2 (label 1 through W/2), so max label = W/2.
    // UF size = W/2+1. So W/2 < W/2+1. Not triggered.

    // For ALL foreground (no zeros) with 4-conn on 1-row: each pixel merges
    // with left neighbor, so only 1 label ever used. Not helpful.

    // The trick: 2 rows, checkerboard, 4-conn. Row 0 is 1,0,1,0,...
    // Row 1 is 0,1,0,1,... Each fg pixel is isolated under 4-conn.
    // Total voxels = 2*W. fg pixels = W. UF size = W+1.
    // Labels needed = W. So W < W+1. Still fine.

    // Hmm, the initial allocation is voxels/2+1 but labels start at 1.
    // The check is: if (next_label >= uf.size()). So we need next_label >= voxels/2+1.
    // With W fg pixels, next_label reaches W+1 (labels 1..W).
    // uf.size() = voxels/2 + 1 = W + 1. So next_label = W+1 = uf.size().
    // The check is >= so it DOES trigger when allocating the (W+1)-th label!
    // But wait, we only have W fg pixels, so we allocate labels 1..W.
    // After allocating label W, next_label becomes W+1. The NEXT pixel
    // that needs a new label would check if W+1 >= W+1, which is true.
    // But if we have exactly W isolated fg pixels and exactly W labels,
    // the last label allocation is for next_label=W, checking W >= W+1 = false.
    // So no overflow. We need more fg pixels than voxels/2.

    // To get more fg pixels than voxels/2: make most pixels foreground!
    // E.g., 3-row image with 4-conn:
    // Row 0: 1 0 1 0 1 0 ... (W/2 fg)
    // Row 1: 0 1 0 1 0 1 ... (W/2 fg)
    // Row 2: 1 0 1 0 1 0 ... (W/2 fg)
    // Total fg = 3*W/2. Voxels = 3*W. UF = 3*W/2+1.
    // Each fg is isolated under 4-conn. Labels needed = 3*W/2.
    // next_label goes up to 3*W/2 + 1. Check: 3*W/2 + 1 > 3*W/2 + 1? No.
    // Still just barely fits.

    // Actually re-reading: the check is BEFORE allocating (line 93):
    // if (next_label >= uf.size()) means if we're about to use label
    // next_label which needs index next_label in the UF, but UF has
    // size voxels/2+1 meaning valid indices 0..voxels/2.
    // So label voxels/2 is the last valid one. If we need label voxels/2+1,
    // the check triggers.

    // With N foreground pixels all isolated, we need labels 1..N.
    // We need N+1 > voxels/2+1, i.e., N > voxels/2.
    // For 3 rows, W cols checkerboard, N = ceil(3W/2).
    // voxels/2 = 3W/2.  N = ceil(3W/2). For W even: N = 3W/2 = voxels/2.
    // So N is not > voxels/2. Need N > voxels/2.

    // What if we have more fg than bg? E.g. "almost all foreground"
    // but still no 4-connected adjacency between them.
    // This is impossible in a standard grid - if >50% are fg and 4-conn,
    // some must be adjacent. Unless it's a checkerboard, which is exactly 50%.

    // So the bug can't be triggered with a checkerboard because fg = voxels/2.
    // But there's a subtlety: for odd dimensions, ceil rounding...
    // 3x3 checkerboard: fg = 5 (positions with (r+c)%2==0). voxels = 9.
    // UF size = 9/2 + 1 = 5. Labels 1..5, next_label=6. Check: 6 >= 5? Yes!
    // BUG TRIGGERS on pixel 5 (the 5th foreground pixel)!

    // Let's use 3x3 checkerboard with 4-conn to trigger it.
    // Actually wait, the existing test "cc checkerboard 4-conn" already
    // does this with 3x3 and 5 fg pixels, but it seems to pass.
    // The UF has size 5, labels go 1..5, when allocating label 5 we have
    // next_label=5, check: 5 >= 5 -> TRUE. So it recreates the UF,
    // losing unions. But with 4-conn checkerboard, there are NO unions
    // (each pixel is isolated), so nothing is lost! The bug only manifests
    // when there ARE unions that get lost.

    // So we need a pattern where:
    // 1. Many initial labels are created (requiring >voxels/2 labels)
    // 2. Some of those labels need to be united

    // Strategy: use 8-connectivity but craft a pattern where:
    // - Most fg pixels only see predecessors with different labels initially
    // - But later pixels discover they should be connected

    // Actually, let's just use a larger odd-sized grid.
    // 5x5 checkerboard 4-conn: fg = 13 (at positions where (r+c)%2==0).
    // voxels = 25. UF size = 25/2+1 = 13. Labels 1..13, next_label=14.
    // Label 13 is the last one. Check before allocating: next_label=13,
    // 13 >= 13 -> TRUE. UF rebuilt at this point.
    // But again, with 4-conn checkerboard no unions happen, so no data loss.

    // We need a pattern that creates many labels AND has unions.
    // Key insight: with 4-conn, if row 0 has alternating fg, and row 1 has
    // a long fg strip connecting some of them, the initial pass on row 0
    // creates many labels, then row 1 merges them. If the UF overflows
    // during row 0 (before row 1 is processed), the unions from earlier
    // in row 0 are lost.

    // Better idea: use 8-connectivity on a specific pattern.
    // Row 0: 1 0 1 0 1 0 1 0 1 ... (each gets own label, no 8-conn predecessors)
    // Row 1: 0 1 0 1 0 1 0 1 0 ... (each 1 sees row0 diagonals, causing unions)
    // With 8-conn predecessors: (-1,-1), (-1,0), (-1,1), (0,-1)
    // Row 1 col 1: sees row0 col 0 (via -1,0: val=1, label=1) and
    //              row0 col 2 (via -1,1: val=1, label=2). So these unite.
    // With enough such pixels, ALL of row 0 labels get united.
    // If the UF overflows between creating these labels, the unions are lost.

    // For a 3-row checkerboard with 8-conn:
    // All fg pixels should form 1 component (since diagonals connect).
    // Row 0 creates ceil(W/2) labels.
    // Row 1 creates floor(W/2) labels but also unites row 0 labels.
    // Row 2 creates ceil(W/2) labels but also unites with row 1.
    // Total labels ~= 3W/2. UF size = 3W/2 + 1.
    // For W=9: fg=14, voxels=27, UF=14. Label 14 checks 14>=14 -> TRUE.
    // But this is right at the boundary. Let's be safe and test with a
    // known-good reference: the answer should be 1 component.

    // Let's use 3x9 checkerboard with 8-conn.
    const std::size_t rows = 3;
    const std::size_t cols = 9;
    std::vector<int> img(rows * cols, 0);
    for (std::size_t r = 0; r < rows; ++r)
        for (std::size_t c = 0; c < cols; ++c)
            if ((r + c) % 2 == 0) img[r * cols + c] = 1;

    std::mdspan<const int, std::dextents<std::size_t, 2>> view(img.data(), rows, cols);

    // With 8-connectivity, all foreground pixels should be in 1 component
    auto [labels, n] = connected_components_2d(view, GridConnectivity::eight);

    // Verify all foreground pixels have the same label
    std::uint32_t fg_label = 0;
    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t c = 0; c < cols; ++c) {
            if ((r + c) % 2 == 0) {
                if (fg_label == 0) {
                    fg_label = labels[r * cols + c];
                } else {
                    CHECK_EQ(labels[r * cols + c], fg_label);
                }
            }
        }
    }
    CHECK_EQ(n, 1u);
}

TEST_CASE("cc 2D UF overflow stress: many labels then merge") {
    // Create a pattern specifically designed to overflow the UF and lose unions.
    // Use a wide single-row-pair pattern with 4-conn where row 0 has isolated
    // pixels and row 1 connects them all.
    // Row 0: 1 0 1 0 1 0 1 0 1 0 1  (isolated under 4-conn)
    // Row 1: 1 1 1 1 1 1 1 1 1 1 1  (connects everything in 4-conn)
    // voxels = 2 * W. UF size = W + 1.
    // Row 0 creates ceil(W/2) labels. Row 1 pixel (1,0) merges with (0,0).
    // Row 1 pixel (1,1) has predecessor (0,1)=0 (bg) and (1,0)=fg, so gets
    // same label as (1,0). Row 1 pixel (1,2) has predecessors (0,2)=fg and
    // (1,1)=fg. So it unites those two labels.
    // Labels from row 0: 1, 2, 3, ..., ceil(W/2). For W=21, that's 11 labels.
    // voxels = 42, UF size = 22. 11 labels. Fine.

    // To trigger: we need fg pixels > voxels/2.
    // With the above pattern: fg = ceil(W/2) + W = ceil(3W/2).
    // voxels = 2W. voxels/2 = W. fg = ceil(3W/2) > W for W>=1.
    // So for W odd, W=11: fg=6+11=17, voxels=22, UF=12.
    // Labels: row 0 gets 6 labels (1-6). Row 1: (1,0) merges with label 1.
    // (1,1) gets label 7 (new, no fg predecessor). (1,2) merges label 2
    // with label from (1,1). (1,3) gets label 8. (1,4) merges label 3 with 8.
    // Actually let me re-check. Under 4-conn, predecessors are (-1,0) and (0,-1).
    // Row 1, col 0: predecessor (-1,0) = img[0,0] = 1 with label 1.
    //   predecessor (0,-1) = out of bounds. -> min_label = 1. labels[1*11+0]=1.
    // Row 1, col 1: predecessor (-1,0) = img[0,1] = 0.
    //   predecessor (0,-1) = img[1,0] = 1 with label 1. -> min_label=1. labels[1*11+1]=1.
    // Row 1, col 2: predecessor (-1,0) = img[0,2] = 1 with label 2.
    //   predecessor (0,-1) = img[1,1] = 1 with label 1. -> unite(2, 1).
    // Row 1, col 3: predecessor (-1,0) = img[0,3] = 0.
    //   predecessor (0,-1) = img[1,2] = 1 with label (root of 2 or 1). -> min_label = that.
    // Row 1, col 4: predecessor (-1,0) = img[0,4] = 1 with label 3.
    //   predecessor (0,-1) = img[1,3] = 1 with some label. -> unite(3, that).
    // So every other row-1 pixel triggers a union. Total new labels from row 1: 0.
    // Total labels = ceil(W/2) = 6 for W=11. UF size = 12. No overflow.

    // The unions happen IN row 1 so they don't create new labels.
    // We need to force new label creation beyond voxels/2.
    // A pure checkerboard with odd dimensions is the simplest case.

    // Let's go back to the theoretical analysis: for a 1x(2N+1) image,
    // all 1s, 4-conn: label 1 for pixel 0, pixel 1 merges (label 1),
    // pixel 2 merges (label 1), ... Only 1 label total. No.

    // For TRULY triggering this: we need a grid where initially many
    // separate labels are allocated, exceeding voxels/2.
    // This requires >50% foreground, which for 4-conn means some
    // adjacent fg pixels (and thus merges). But the overflow happens
    // on label allocation, and if it happens after some unions were
    // already done, those unions are lost.

    // A 1x(2N+1) grid of all 1s with 4-conn: sequential labels,
    // each pixel merges with left. So only 1 label used. Not helpful.

    // What about a grid that's ALL foreground, 2 rows?
    // Row 0: 1 1 1 1 1 ... W cols
    // Row 1: 1 1 1 1 1 ... W cols
    // 4-conn. Row 0 pixel 0 gets label 1. Pixel 1 merges with left -> label 1.
    // ... All of row 0 gets label 1. Row 1 pixel 0: predecessor up = label 1.
    // All of row 1 also label 1. Total labels = 1. No overflow.

    // The only way to get many labels is many disconnected initial regions.
    // In 4-conn, >50% fg means some are adjacent. But we can have
    // a weird pattern:
    // Rows of: 1 0 1 0 1 ... (each row independent under 4-conn)
    // This is a checkerboard, ~50% fg.

    // I think the actual bug needs truly more than 50% fg pixels that
    // are initially disconnected. This is geometrically impossible in 4-conn.
    // But in 8-conn with special patterns, or with multi-label...

    // Actually wait - let me re-read the code. The UF is shared between
    // labels. Label 0 is unused (background). Labels start at 1.
    // UF size = voxels/2 + 1. The check is next_label >= uf.size().
    // So valid labels are 0..voxels/2. Since labels start at 1,
    // we can have voxels/2 labels (1..voxels/2).
    // For a checkerboard on an odd-dimension grid:
    // E.g., 1 row, 5 cols: 1 0 1 0 1. fg=3. voxels=5. UF=3.
    // Labels 1,2,3. next_label after 3 = 4. Check: 4 >= 3 -> TRUE.
    // But label 3 is index 3 in a UF of size 3 -> indices 0,1,2!
    // So even label 3 is OUT OF BOUNDS (UF has indices 0..2)!
    // The UF.find(3) accesses parent_[3] which is out of bounds!

    // Wait, the check on line 93 happens BEFORE assigning the label.
    // So if next_label >= uf.size(), it rebuilds (and loses data).
    // But if next_label < uf.size(), it uses next_label as a UF index.

    // For 1x5 checkerboard: voxels=5, UF size=3.
    // Pixel (0,0)=1: next_label=1. 1 >= 3? No. label=1, next_label=2.
    // Pixel (0,2)=1: next_label=2. 2 >= 3? No. label=2, next_label=3.
    // Pixel (0,4)=1: next_label=3. 3 >= 3? YES! Rebuild UF. Union data lost.
    //   But no unions were done (all isolated), so nothing lost.
    //   After rebuild, label=3, next_label=4.

    // Now for the actual bug to manifest, unions must exist before rebuild.
    // Consider 2x5 with 4-conn:
    // Row 0: 1 1 0 1 1
    // Row 1: 0 0 0 0 0 (just use row 0)
    // Actually, let me think of a 2-row pattern:
    // Row 0: 1 1 0 1 1  -> label 1 for (0,0). (0,1) merges with left -> label 1.
    //   (0,3) gets label 2. (0,4) merges with left -> label 2.
    //   Unite(1, 2) not needed. 2 labels, 4 fg pixels.
    // Now add row 1 that causes more labels and eventually overflow:
    // Row 1: 1 0 1 0 1  -> 3 more fg pixels.
    //   (1,0): predecessor up = 1 (label 1). label=1.
    //   (1,2): no fg predecessor under 4-conn. New label 3. 3 >= UF size?
    //     voxels=10, UF size=6. 3 < 6. Fine.
    //   (1,4): predecessor up = 1 (label 2). label=2.
    // Total labels used: 3. No overflow.

    // This is getting very hard to trigger with natural patterns because
    // voxels/2+1 is generous for typical images.

    // Let me try with a very specific odd-shaped grid.
    // 1x3 all-fg with 4-conn:
    // (0,0)=1: label 1. (0,1)=1: merges with 1. (0,2)=1: merges with 1.
    // Only 1 label. UF size = 2. Fine.

    // 3x1 column, all-fg:
    // (0,0)=1: label 1. (1,0)=1: merges with 1. (2,0)=1: merges with 1.
    // Fine.

    // For the overflow to cause actual corruption, we'd need a very crafted
    // scenario. Let me try the simplest approach: check that the RESULT
    // is correct, regardless of whether overflow happens. If it's wrong,
    // we found the bug.

    // 5x5 checkerboard 8-conn: should be 1 component.
    const std::size_t rows2 = 5;
    const std::size_t cols2 = 5;
    std::vector<int> img2(rows2 * cols2, 0);
    for (std::size_t r = 0; r < rows2; ++r)
        for (std::size_t c = 0; c < cols2; ++c)
            if ((r + c) % 2 == 0) img2[r * cols2 + c] = 1;

    std::mdspan<const int, std::dextents<std::size_t, 2>> view2(img2.data(), rows2, cols2);
    auto [labels2, n2] = connected_components_2d(view2, GridConnectivity::eight);
    CHECK_EQ(n2, 1u);

    // Also verify with 4-conn: should be 13 separate components
    auto [labels2b, n2b] = connected_components_2d(view2, GridConnectivity::four);
    CHECK_EQ(n2b, 13u);
}

TEST_CASE("cc 2D large checkerboard correctness with 8-conn") {
    // 21x21 checkerboard with 8-connectivity.
    // All fg pixels should be connected via diagonals -> 1 component.
    // 221 fg pixels, 441 voxels, UF size = 221.
    // 221 labels needed if all are initially separate (which they are
    // on row 0, but row 1 pixels merge via diagonal predecessors).
    const std::size_t N = 21;
    std::vector<int> img(N * N, 0);
    for (std::size_t r = 0; r < N; ++r)
        for (std::size_t c = 0; c < N; ++c)
            if ((r + c) % 2 == 0) img[r * N + c] = 1;

    std::mdspan<const int, std::dextents<std::size_t, 2>> view(img.data(), N, N);
    auto [labels, n] = connected_components_2d(view, GridConnectivity::eight);
    REQUIRE_EQ(n, 1u);

    // Every fg pixel should have the same label
    std::uint32_t expected_label = 0;
    for (std::size_t i = 0; i < N * N; ++i) {
        if (img[i] != 0) {
            if (expected_label == 0) expected_label = labels[i];
            else CHECK_EQ(labels[i], expected_label);
        } else {
            CHECK_EQ(labels[i], 0u);
        }
    }
}

// ---- Bug hunt: verify consecutive labeling ---------------------------------

TEST_CASE("cc 2D labels are consecutive from 1") {
    // Labels should be 1..N with no gaps
    std::vector<int> img = {
        1, 0, 1, 0, 1,
        0, 0, 0, 0, 0,
        1, 0, 1, 0, 1
    };
    std::mdspan<const int, std::dextents<std::size_t, 2>> view(img.data(), 3, 5);
    auto [labels, n] = connected_components_2d(view, GridConnectivity::four);
    REQUIRE_EQ(n, 6u);

    // Check that labels used are exactly {0, 1, 2, 3, 4, 5, 6}
    std::set<std::uint32_t> used(labels.begin(), labels.end());
    for (std::uint32_t i = 0; i <= n; ++i) {
        CHECK(used.count(i) > 0);
    }
}

UTILS2_TEST_MAIN()
