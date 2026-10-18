cmake_minimum_required(VERSION 3.16)
project(GEMM_Benchmark LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# =============================================================================
# Compiler flags
# =============================================================================
# -O3:           Full optimization (vectorization, inlining, unrolling)
# -march=native: Enable all ISA extensions available on this CPU (AVX2, FMA, etc.)
# -ffast-math:   Allow FP reassociation for auto-vectorization (we control precision
#                ourselves via intrinsics, so this is safe)
# -funroll-loops: Aggressive loop unrolling
# =============================================================================
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -ffast-math -funroll-loops -DNDEBUG")

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Include path
include_directories(${CMAKE_SOURCE_DIR}/include)

# =============================================================================
# GEMM library — all kernel implementations
# =============================================================================
add_library(gemm_kernels STATIC
    src/gemm_naive.cpp
    src/gemm_reordered.cpp
    src/gemm_blocked.cpp
    src/gemm_avx2.cpp
)

# Ensure AVX2/FMA intrinsics are available for the AVX2 source
# (redundant with -march=native but explicit is better)
set_source_files_properties(src/gemm_avx2.cpp PROPERTIES
    COMPILE_FLAGS "-mavx2 -mfma"
)

# =============================================================================
# Benchmark executable
# =============================================================================
add_executable(bench bench/bench.cpp)
target_link_libraries(bench gemm_kernels)
