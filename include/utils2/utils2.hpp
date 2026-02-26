#pragma once

// Utils2: Composable C++26 single-header utility libraries
// Include individual headers for minimal compile times,
// or include this header for everything.

// Tier 1: Foundation
#include "vec.hpp"
#include "hash.hpp"
#include "json.hpp"
#include "mdspan_util.hpp"
#include "log.hpp"
#include "connectivity.hpp"

// Tier 2: Data Structures
#include "lru_cache.hpp"
#include "lock_pool.hpp"
#include "priority_queue.hpp"
#include "spatial_index.hpp"
#include "disjoint_set.hpp"
#include "grid_store.hpp"

// Tier 3: Algorithms
#include "interpolate.hpp"
#include "connected_components.hpp"
#include "distance_transform.hpp"
#include "pathfinding.hpp"
#include "compositing.hpp"
#include "morphology.hpp"
#include "statistics.hpp"

// Tier 4: I/O & Systems
#include "thread_pool.hpp"
#include "tiered_cache.hpp"
#include "disk_store.hpp"
#include "zarr.hpp"
#include "tiff.hpp"
#include "timer.hpp"
#if __has_include(<curl/curl.h>)
#include "http_fetch.hpp"
#endif

// Tier 5: Domain-Specific
#include "surface.hpp"
#include "mesh_flatten.hpp"
#include "cost_functions.hpp"
#include "argparse.hpp"
