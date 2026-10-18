// =============================================================================
// gemm_blocked.cpp — Level 2: Cache-blocked (tiled) GEMM
//
// PROBLEM with Level 1: When N and K are large, iterating k=0..K-1 for each
// row i means the B matrix is streamed M times through cache. For a 1024×1024
// matrix, B is 4 MB — it won't fit in L1 (40 KiB) or even L2 (1.25 MiB per
// P-core on Alder Lake). Each pass through B evicts the previous one.
//
// SOLUTION: Tile all three loops so that the working set fits in cache.
//
// Tile sizes chosen for i5-12450H P-core:
//   - L1d = 48 KiB (per P-core), L2 = 1.25 MiB
//   - Block B_tile = BK × BN × 4 bytes should fit in L1
//   - BK=64, BN=64: B_tile = 16 KiB ✓ (fits in L1 with room for A and C)
//   - BM=64: A_tile = BM × BK × 4 = 16 KiB ✓
//   - C_tile = BM × BN × 4 = 16 KiB ✓
//   - Total working set ≈ 48 KiB → snugly fits L1d
//
// The tiled loop structure is:
//   for (ii in 0..M step BM)          // tile rows of A and C
//     for (kk in 0..K step BK)        // tile cols of A / rows of B
//       for (jj in 0..N step BN)      // tile cols of B and C
//         micro_kernel(ii, jj, kk)    // i-k-j within tile
//
// The kk loop is SECOND (not innermost) so that the same B tile is reused
// across multiple A row-tiles, and C tiles accumulate across K-tiles.
// =============================================================================

#include "gemm.h"
#include <algorithm>

// Tile sizes — tuned for 48 KiB L1d
static constexpr int BM = 64;
static constexpr int BN = 64;
static constexpr int BK = 64;

void gemm_blocked(
    int M, int N, int K,
    float alpha,
    const float* __restrict__ A, int lda,
    const float* __restrict__ B, int ldb,
    float beta,
    float* __restrict__ C, int ldc)
{
    // Scale C by beta (full matrix, once)
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j)
            C[i * ldc + j] *= beta;

    // Blocked i-k-j loop
    for (int ii = 0; ii < M; ii += BM) {
        const int mb = std::min(BM, M - ii);  // handle edge tiles

        for (int kk = 0; kk < K; kk += BK) {
            const int kb = std::min(BK, K - kk);

            for (int jj = 0; jj < N; jj += BN) {
                const int nb = std::min(BN, N - jj);

                // ---- Micro-tile: i-k-j within the block ----
                // A tile: rows [ii, ii+mb), cols [kk, kk+kb)
                // B tile: rows [kk, kk+kb), cols [jj, jj+nb)
                // C tile: rows [ii, ii+mb), cols [jj, jj+nb)
                for (int i = ii; i < ii + mb; ++i) {
                    for (int k = kk; k < kk + kb; ++k) {
                        float a_ik = alpha * A[i * lda + k];

                        for (int j = jj; j < jj + nb; ++j) {
                            C[i * ldc + j] += a_ik * B[k * ldb + j];
                        }
                    }
                }
            }
        }
    }
}
