#pragma once
// =============================================================================
// gemm.h — GEMM kernel declarations
//
// C = alpha * A * B + beta * C
// A is M×K, B is K×N, C is M×N (row-major)
//
// All matrices stored in row-major order:
//   A[i][j] = A[i * lda + j]   where lda >= K
//   B[i][j] = B[i * ldb + j]   where ldb >= N
//   C[i][j] = C[i * ldc + j]   where ldc >= N
// =============================================================================

#include <cstddef>

// ---------------------------------------------------------------------------
// Level 0: Naive triple-loop (i-j-k order)
// Expected: ~1-2 GFLOPS — bottlenecked by column-strided B access
// ---------------------------------------------------------------------------
void gemm_naive(
    int M, int N, int K,
    float alpha,
    const float* __restrict__ A, int lda,
    const float* __restrict__ B, int ldb,
    float beta,
    float* __restrict__ C, int ldc);

// ---------------------------------------------------------------------------
// Level 1: Loop-reordered (i-k-j) — row-major friendly
// Expected: ~3-5 GFLOPS — exploits spatial locality in B and C
// ---------------------------------------------------------------------------
void gemm_reordered(
    int M, int N, int K,
    float alpha,
    const float* __restrict__ A, int lda,
    const float* __restrict__ B, int ldb,
    float beta,
    float* __restrict__ C, int ldc);

// ---------------------------------------------------------------------------
// Level 2: Cache-blocked (tiled) — i-k-j order within tiles
// Expected: ~5-10 GFLOPS — temporal locality in L1/L2 for B tile
// ---------------------------------------------------------------------------
void gemm_blocked(
    int M, int N, int K,
    float alpha,
    const float* __restrict__ A, int lda,
    const float* __restrict__ B, int ldb,
    float beta,
    float* __restrict__ C, int ldc);

// ---------------------------------------------------------------------------
// Level 3: Blocked + packed + AVX2/FMA micro-kernel (6×16 register tile)
// Expected: ~30-55+ GFLOPS — approaching theoretical single-core peak
// ---------------------------------------------------------------------------
void gemm_avx2(
    int M, int N, int K,
    float alpha,
    const float* __restrict__ A, int lda,
    const float* __restrict__ B, int ldb,
    float beta,
    float* __restrict__ C, int ldc);
