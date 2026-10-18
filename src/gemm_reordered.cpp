// =============================================================================
// gemm_reordered.cpp — Level 1: Loop-reordered GEMM (i-k-j)
//
// Key insight: swap the j and k loops to fix B's access pattern.
//
// With i-k-j ordering:
//   - Inner loop (j): B[k][j] is stride-1 → sequential cache-line access ✓
//   - Inner loop (j): C[i][j] is stride-1 → sequential ✓
//   - A[i][k] is loop-invariant w.r.t. j → hoisted into a register ✓
//
// This single reordering typically yields 3-5× speedup over naive because
// the memory subsystem can now prefetch B rows effectively. The CPU's
// hardware prefetcher detects the sequential pattern and stays ahead.
//
// Still limited because:
//   - No data reuse across i iterations (B is re-streamed for every row of A)
//   - No SIMD utilization — scalar FP ops only
//   - No register blocking — only 1 element of C computed at a time
// =============================================================================

#include "gemm.h"

void gemm_reordered(
    int M, int N, int K,
    float alpha,
    const float* __restrict__ A, int lda,
    const float* __restrict__ B, int ldb,
    float beta,
    float* __restrict__ C, int ldc)
{
    // Scale C by beta
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j)
            C[i * ldc + j] *= beta;

    // i-k-j accumulation
    for (int i = 0; i < M; ++i) {
        for (int k = 0; k < K; ++k) {
            // A[i,k] is invariant in j — compiler hoists to register
            float a_ik = alpha * A[i * lda + k];

            // Inner loop: sequential access to both B row and C row
            // B[k,j]: stride-1 ✓   C[i,j]: stride-1 ✓
            for (int j = 0; j < N; ++j) {
                C[i * ldc + j] += a_ik * B[k * ldb + j];
            }
        }
    }
}
