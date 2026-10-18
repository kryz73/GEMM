// =============================================================================
// gemm_naive.cpp — Level 0: Textbook triple-loop GEMM
//
// C[i][j] = alpha * sum_k( A[i][k] * B[k][j] ) + beta * C[i][j]
//
// This is the classic i-j-k loop ordering. For row-major matrices:
//   - A[i][k] access: stride-1 over k → good spatial locality
//   - B[k][j] access: stride-N over k → BAD — cache-line thrashing
//   - C[i][j] access: stride-1 over j (innermost k doesn't touch j) → reused
//
// The catastrophic B access pattern means every multiply fetches a new cache
// line from B, leading to ~1-2 GFLOPS on a 4+ GHz core.
// =============================================================================

#include "gemm.h"

void gemm_naive(
    int M, int N, int K,
    float alpha,
    const float* __restrict__ A, int lda,
    const float* __restrict__ B, int ldb,
    float beta,
    float* __restrict__ C, int ldc)
{
    // Scale C by beta first (handles beta=0 case cleanly)
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j)
            C[i * ldc + j] *= beta;

    // Accumulate alpha * A * B
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                // A[i,k]: sequential in k ✓
                // B[k,j]: stride-N jump per k iteration ✗ ← the bottleneck
                sum += A[i * lda + k] * B[k * ldb + j];
            }
            C[i * ldc + j] += alpha * sum;
        }
    }
}
