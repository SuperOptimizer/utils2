cmake_minimum_required(VERSION 3.28)
project(utils2
    VERSION 0.1.0
    DESCRIPTION "Composable C++26 single-header utility libraries"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Options
option(UTILS2_BUILD_TESTS "Build tests" ON)
option(UTILS2_BUILD_BENCHMARKS "Build benchmarks" OFF)
option(UTILS2_BUILD_PYTHON "Build Python bindings" OFF)
option(UTILS2_COVERAGE "Build with code coverage" OFF)
set(UTILS2_SANITIZER "" CACHE STRING "Sanitizer to enable (address, thread, undefined, memory, type, numerical, or empty)")
set_property(CACHE UTILS2_SANITIZER PROPERTY STRINGS "" "address" "thread" "undefined" "memory" "type" "numerical" "address,undefined")

# Sanitizer support
if(UTILS2_SANITIZER)
    if(NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        message(FATAL_ERROR "Sanitizers require GCC or Clang")
    endif()
    # Type and numerical sanitizers are clang-only
    if(UTILS2_SANITIZER MATCHES "type|numerical")
        if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            message(FATAL_ERROR "TypeSanitizer and NumericalSanitizer require Clang")
        endif()
    endif()
    add_compile_options(-fsanitize=${UTILS2_SANITIZER} -fno-omit-frame-pointer)
    add_link_options(-fsanitize=${UTILS2_SANITIZER})
    if(UTILS2_SANITIZER MATCHES "address")
        add_compile_definitions(ASAN_OPTIONS=detect_leaks=1)
    endif()
endif()

# Header-only library target
add_library(utils2 INTERFACE)
target_include_directories(utils2 INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_features(utils2 INTERFACE cxx_std_26)

# Optional: find curl for http_fetch.hpp
find_package(CURL QUIET)
if(CURL_FOUND)
    target_link_libraries(utils2 INTERFACE CURL::libcurl)
    target_compile_definitions(utils2 INTERFACE UTILS2_HAS_CURL=1)
endif()

# Individual header targets (for granular dependency management)
# Each header can be used independently
set(UTILS2_HEADERS
    vec
    hash
    mdspan_util
    log
    connectivity
    lru_cache
    lock_pool
    priority_queue
    spatial_index
    disjoint_set
    grid_store
    interpolate
    connected_components
    distance_transform
    pathfinding
    compositing
    morphology
    statistics
    thread_pool
    tiered_cache
    disk_store
    zarr
    tiff
    http_fetch
    timer
    compression
    surface
    mesh_flatten
    cost_functions
    argparse
)

foreach(header ${UTILS2_HEADERS})
    add_library(utils2_${header} INTERFACE)
    target_include_directories(utils2_${header} INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    )
    target_compile_features(utils2_${header} INTERFACE cxx_std_26)
endforeach()

# Link curl for http_fetch and zarr (which optionally includes http_fetch)
if(CURL_FOUND)
    target_link_libraries(utils2_http_fetch INTERFACE CURL::libcurl)
    target_link_libraries(utils2_zarr INTERFACE CURL::libcurl)
endif()

# Optional: find compression libraries for compression.hpp
find_library(BLOSC2_LIB blosc2 QUIET)
find_library(ZSTD_LIB zstd QUIET)
find_library(LZ4_LIB lz4 QUIET)
find_library(ZLIB_LIB z QUIET)
set(UTILS2_COMPRESSION_LIBS "")
if(BLOSC2_LIB)
    list(APPEND UTILS2_COMPRESSION_LIBS ${BLOSC2_LIB})
endif()
if(ZSTD_LIB)
    list(APPEND UTILS2_COMPRESSION_LIBS ${ZSTD_LIB})
endif()
if(LZ4_LIB)
    list(APPEND UTILS2_COMPRESSION_LIBS ${LZ4_LIB})
endif()
if(ZLIB_LIB)
    list(APPEND UTILS2_COMPRESSION_LIBS ${ZLIB_LIB})
endif()
if(UTILS2_COMPRESSION_LIBS)
    target_link_libraries(utils2 INTERFACE ${UTILS2_COMPRESSION_LIBS})
    target_link_libraries(utils2_compression INTERFACE ${UTILS2_COMPRESSION_LIBS})
    target_link_libraries(utils2_zarr INTERFACE ${UTILS2_COMPRESSION_LIBS})
endif()

# Tests
if(UTILS2_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Benchmarks
if(UTILS2_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

# Code coverage
if(UTILS2_COVERAGE)
    if(NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        message(FATAL_ERROR "Code coverage requires GCC or Clang")
    endif()
    if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
        message(WARNING "Code coverage results are best with -DCMAKE_BUILD_TYPE=Debug")
    endif()

    find_program(GCOVR gcovr REQUIRED)

    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        set(GCOV_EXECUTABLE "llvm-cov gcov")
    else()
        set(GCOV_EXECUTABLE "gcov")
    endif()

    add_custom_target(coverage
        COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/coverage
        COMMAND ${GCOVR}
            --root ${CMAKE_SOURCE_DIR}
            --filter "include/utils2/"
            --exclude "tests/"
            --print-summary
            --html-details ${CMAKE_BINARY_DIR}/coverage/index.html
            --txt ${CMAKE_BINARY_DIR}/coverage/coverage.txt
            --gcov-executable ${GCOV_EXECUTABLE}
            ${CMAKE_BINARY_DIR}
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMENT "Running tests and generating coverage report"
        VERBATIM
    )
endif()

# Install
include(GNUInstallDirs)
install(DIRECTORY include/utils2/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/utils2)
install(TARGETS utils2 EXPORT utils2Targets)
install(EXPORT utils2Targets
    FILE utils2Config.cmake
    NAMESPACE utils2::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/utils2
)
