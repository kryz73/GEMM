# High-Performance GEMM in C++

Progressive optimization of single-precision General Matrix Multiplication (GEMM) kernels, from textbook triple-loop to AVX2/FMA micro-kernel — benchmarked against theoretical CPU peak.

**Target:** 12th Gen Intel i5-12450H (Alder Lake) — single P-core  
**Theoretical Peak:** 70.4 SP GFLOPS (4.4 GHz × 2 FMA × 8 floats)

## Optimization Levels

| Level | Kernel | Key Idea | Expected GFLOPS |
|-------|--------|----------|-----------------|
| 0 | Naive (i-j-k) | Textbook triple loop | ~1-2 |
| 1 | Reordered (i-k-j) | Fix B's column-strided access pattern | ~3-6 |
| 2 | Blocked (tiled) | 64×64×64 cache tiling for L1 residency | ~5-12 |
| 3 | AVX2/FMA 6×16 | Packed micro-kernel with Goto's algorithm | ~30-55+ |

## Architecture

```
Naive → Reordered → Blocked → AVX2/FMA
 1 GFLOPS   5 GFLOPS   10 GFLOPS   50+ GFLOPS
```

### Level 3 Details (AVX2 Micro-Kernel)

The highest-performance kernel uses [Goto's algorithm](https://www.cs.utexas.edu/~flame/pubs/GotoTOMS_revision.pdf) (the same approach used by BLIS and OpenBLAS):

1. **3-level cache blocking:** NC (L3) → KC (L2) → MC (L1)
2. **Matrix packing:** A and B are packed into contiguous, aligned buffers that eliminate TLB misses and enable aligned SIMD loads
3. **6×16 register tile:** 12 YMM accumulators + 1 broadcast + 2 B-loads = 15/16 registers used. Each k-step produces 12 FMA instructions = 96 FLOPs

## Build & Run

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Run with default sizes (128, 256, 512, 1024, 2048)
./bench

# Run with custom sizes
./bench 512 1024 2048 4096
```

## Project Structure

```
GEMM/
├── CMakeLists.txt
├── README.md
├── include/
│   └── gemm.h              # Kernel declarations
├── src/
│   ├── gemm_naive.cpp       # Level 0: i-j-k triple loop
│   ├── gemm_reordered.cpp   # Level 1: i-k-j loop interchange
│   ├── gemm_blocked.cpp     # Level 2: Cache tiling
│   └── gemm_avx2.cpp        # Level 3: AVX2/FMA micro-kernel
└── bench/
    └── bench.cpp            # Benchmark harness
```

## Why Each Optimization Matters

### Loop Reordering (Level 1)
The naive i-j-k loop accesses `B[k][j]` with stride-N in the innermost (k) loop — every multiply fetches a new cache line. Swapping to i-k-j makes both B and C accesses stride-1, enabling hardware prefetch.

### Cache Blocking (Level 2)
Without tiling, the entire B matrix (4 MB for N=1024) is streamed through cache M times. With 64×64 tiles, the working set (A tile + B tile + C tile ≈ 48 KiB) fits in the 48 KiB L1d cache.

### AVX2 Micro-Kernel (Level 3)
Three critical innovations:
- **SIMD FMA:** Each `vfmadd231ps` computes 8 multiply-adds in one cycle. The 6×16 tile fires 12 FMAs per k-step.
- **Register blocking:** 12 accumulators stay in registers for the entire k-loop — no memory traffic for C until the tile is complete.
- **Packing:** Eliminates TLB thrashing, ensures alignment, and makes access patterns perfectly sequential for the prefetcher.
