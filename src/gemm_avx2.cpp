// =============================================================================
// gemm_avx2.cpp — Level 3: AVX2/FMA micro-kernel with packing
//
// ARCHITECTURE: Goto's algorithm (BLIS-style)
//   1. Pack B into column-panels (KC × NR-wide, contiguous)
//   2. Pack A into row-panels (MR-high × KC, contiguous)
//   3. Drive a register-tiled micro-kernel over packed data
//
// MICRO-KERNEL DESIGN for AVX2 (256-bit = 8 floats):
//   - 16 YMM registers available (ymm0-ymm15)
//   - We want to maximize FMA throughput with register accumulation
//   - Register tile: MR × NR = 6 × 16
//     * 6 rows × 16 cols = 12 accumulators (6×2 ymm regs for C)
//     * 1 ymm for broadcasting A[i,k] (reused across cols)
//     * 2 ymm for loading B[k,j..j+7] and B[k,j+8..j+15]
//     * Total: 12 + 1 + 2 = 15 ymm registers ← fits perfectly
//   - Each iteration of k: 6 broadcasts × 2 FMAs = 12 FMA ops
//     producing 12 × 8 = 96 FLOPs per k step
//   - At 4.4 GHz, 2 FMA/cycle throughput (port 0+1):
//     Theoretical: 4.4e9 × 2 × 8 = 70.4 GFLOPS peak
//
// PACKING:
//   Eliminates TLB misses and unaligned loads in the micro-kernel.
//   - Pack B: KC × NR block stored contiguously, NR-wide strips
//   - Pack A: MR × KC block stored contiguously, MR-high strips
//   The micro-kernel then accesses memory purely sequentially.
//
// BLOCKING PARAMETERS (tuned for i5-12450H):
//   MC = 72   (multiple of MR=6)  — A panel fits in L2
//   KC = 256  — A panel: MC×KC×4 = 72 KiB, B panel in L3
//   NC = 4096 — B panel: KC×NC×4 = 4 MiB, fits in L3 (12 MiB)
// =============================================================================

#include "gemm.h"

#include <immintrin.h>
#include <cstring>
#include <cstdlib>
#include <algorithm>

// Register tile dimensions
static constexpr int MR = 6;   // rows of C in registers (6 broadcasts)
static constexpr int NR = 16;  // cols of C in registers (2 × 8-wide ymm)

// Cache blocking parameters
static constexpr int MC = 72;    // L2 block for A rows (multiple of MR)
static constexpr int KC = 256;   // Shared dimension block
static constexpr int NC = 4096;  // L3 block for B cols (multiple of NR)

// Aligned allocation helper
static float* aligned_alloc_floats(size_t n) {
    void* ptr = nullptr;
    if (posix_memalign(&ptr, 32, n * sizeof(float)) != 0) return nullptr;
    return static_cast<float*>(ptr);
}

// =============================================================================
// Pack a panel of B into column-major NR-wide strips.
//
// Original B layout (row-major):
//   B[k][j] = B[k * ldb + j], for k in [0,kc), j in [0,nc)
//
// Packed layout:  for each NR-wide column strip jr:
//   packed[jr * kc * NR + k * NR + jj] = B[k][jr*NR + jj]
//
// This makes the micro-kernel's B access perfectly sequential.
// =============================================================================
static void pack_B(const float* B, int ldb, int kc, int nc, float* packed_B) {
    for (int jr = 0; jr < nc; jr += NR) {
        const int nr = std::min(NR, nc - jr);
        for (int k = 0; k < kc; ++k) {
            // Copy NR elements from B row k, starting at column jr
            for (int jj = 0; jj < nr; ++jj) {
                packed_B[jj] = B[k * ldb + jr + jj];
            }
            // Zero-pad if nr < NR (edge case)
            for (int jj = nr; jj < NR; ++jj) {
                packed_B[jj] = 0.0f;
            }
            packed_B += NR;
        }
    }
}

// =============================================================================
// Pack a panel of A into row-major MR-high strips.
//
// Original A layout (row-major):
//   A[i][k] = A[i * lda + k], for i in [0,mc), k in [0,kc)
//
// Packed layout:  for each MR-high row strip ir:
//   packed[ir * kc * MR + k * MR + ii] = A[ir*MR + ii][k]
//
// Note the transpose within each strip: A is transposed from row-major
// to column-major within MR-high panels, so the micro-kernel can broadcast
// each A element with a simple sequential load.
// =============================================================================
static void pack_A(const float* A, int lda, int mc, int kc, float* packed_A) {
    for (int ir = 0; ir < mc; ir += MR) {
        const int mr = std::min(MR, mc - ir);
        for (int k = 0; k < kc; ++k) {
            for (int ii = 0; ii < mr; ++ii) {
                packed_A[ii] = A[(ir + ii) * lda + k];
            }
            for (int ii = mr; ii < MR; ++ii) {
                packed_A[ii] = 0.0f;
            }
            packed_A += MR;
        }
    }
}

// =============================================================================
// AVX2/FMA micro-kernel: 6×16 register tile
//
// Computes: C[0..5][0..15] += A_panel[0..5][k] * B_panel[k][0..15]
//           for k in [0, kc)
//
// Register allocation (ymm0-ymm15):
//   ymm0-ymm1:   B[k][0..7] and B[k][8..15]  (loaded each k)
//   ymm2:        broadcast of A[i][k]          (loaded each i, each k)
//   ymm3-ymm14:  12 accumulators for C[0..5][0..7] and C[0..5][8..15]
//                 ymm[3+2*i]   = C[i][0..7]
//                 ymm[3+2*i+1] = C[i][8..15]
//   (ymm15 free as spare)
//
// Inner loop body per k iteration:
//   Load B[k][0:8] → ymm0,  B[k][8:16] → ymm1
//   For i=0..5:
//     broadcast A[i][k] → ymm2
//     ymm[3+2i]   = fma(ymm2, ymm0, ymm[3+2i])     // C[i][0:8]  += A*B
//     ymm[3+2i+1] = fma(ymm2, ymm1, ymm[3+2i+1])   // C[i][8:16] += A*B
//   → 12 FMA instructions per k, each producing 8 FLOPs = 96 FLOPs/k
// =============================================================================
static void micro_kernel_6x16(
    int kc,
    const float* __restrict__ packed_A,  // MR × kc, column-major within strip
    const float* __restrict__ packed_B,  // kc × NR, row of NR elements per k
    float* __restrict__ C, int ldc)
{
    // Initialize 12 accumulators to zero
    __m256 c00 = _mm256_setzero_ps();  __m256 c01 = _mm256_setzero_ps();
    __m256 c10 = _mm256_setzero_ps();  __m256 c11 = _mm256_setzero_ps();
    __m256 c20 = _mm256_setzero_ps();  __m256 c21 = _mm256_setzero_ps();
    __m256 c30 = _mm256_setzero_ps();  __m256 c31 = _mm256_setzero_ps();
    __m256 c40 = _mm256_setzero_ps();  __m256 c41 = _mm256_setzero_ps();
    __m256 c50 = _mm256_setzero_ps();  __m256 c51 = _mm256_setzero_ps();

    for (int k = 0; k < kc; ++k) {
        // Load NR=16 elements of packed B for this k
        __m256 b0 = _mm256_load_ps(packed_B + 0);   // B[k][0:8]
        __m256 b1 = _mm256_load_ps(packed_B + 8);   // B[k][8:16]

        // For each of the MR=6 rows of A:
        //   broadcast A[row][k] and FMA into two C accumulators
        __m256 a;

        a = _mm256_broadcast_ss(packed_A + 0);  // A[0][k]
        c00 = _mm256_fmadd_ps(a, b0, c00);
        c01 = _mm256_fmadd_ps(a, b1, c01);

        a = _mm256_broadcast_ss(packed_A + 1);  // A[1][k]
        c10 = _mm256_fmadd_ps(a, b0, c10);
        c11 = _mm256_fmadd_ps(a, b1, c11);

        a = _mm256_broadcast_ss(packed_A + 2);  // A[2][k]
        c20 = _mm256_fmadd_ps(a, b0, c20);
        c21 = _mm256_fmadd_ps(a, b1, c21);

        a = _mm256_broadcast_ss(packed_A + 3);  // A[3][k]
        c30 = _mm256_fmadd_ps(a, b0, c30);
        c31 = _mm256_fmadd_ps(a, b1, c31);

        a = _mm256_broadcast_ss(packed_A + 4);  // A[4][k]
        c40 = _mm256_fmadd_ps(a, b0, c40);
        c41 = _mm256_fmadd_ps(a, b1, c41);

        a = _mm256_broadcast_ss(packed_A + 5);  // A[5][k]
        c50 = _mm256_fmadd_ps(a, b0, c50);
        c51 = _mm256_fmadd_ps(a, b1, c51);

        packed_A += MR;  // next column of A panel
        packed_B += NR;  // next row of B panel
    }

    // Store accumulators back to C (additive — C already scaled by beta)
    // Using unaligned stores since C's column alignment isn't guaranteed
    _mm256_storeu_ps(C + 0 * ldc + 0, _mm256_add_ps(_mm256_loadu_ps(C + 0 * ldc + 0), c00));
    _mm256_storeu_ps(C + 0 * ldc + 8, _mm256_add_ps(_mm256_loadu_ps(C + 0 * ldc + 8), c01));

    _mm256_storeu_ps(C + 1 * ldc + 0, _mm256_add_ps(_mm256_loadu_ps(C + 1 * ldc + 0), c10));
    _mm256_storeu_ps(C + 1 * ldc + 8, _mm256_add_ps(_mm256_loadu_ps(C + 1 * ldc + 8), c11));

    _mm256_storeu_ps(C + 2 * ldc + 0, _mm256_add_ps(_mm256_loadu_ps(C + 2 * ldc + 0), c20));
    _mm256_storeu_ps(C + 2 * ldc + 8, _mm256_add_ps(_mm256_loadu_ps(C + 2 * ldc + 8), c21));

    _mm256_storeu_ps(C + 3 * ldc + 0, _mm256_add_ps(_mm256_loadu_ps(C + 3 * ldc + 0), c30));
    _mm256_storeu_ps(C + 3 * ldc + 8, _mm256_add_ps(_mm256_loadu_ps(C + 3 * ldc + 8), c31));

    _mm256_storeu_ps(C + 4 * ldc + 0, _mm256_add_ps(_mm256_loadu_ps(C + 4 * ldc + 0), c40));
    _mm256_storeu_ps(C + 4 * ldc + 8, _mm256_add_ps(_mm256_loadu_ps(C + 4 * ldc + 8), c41));

    _mm256_storeu_ps(C + 5 * ldc + 0, _mm256_add_ps(_mm256_loadu_ps(C + 5 * ldc + 0), c50));
    _mm256_storeu_ps(C + 5 * ldc + 8, _mm256_add_ps(_mm256_loadu_ps(C + 5 * ldc + 8), c51));
}

// =============================================================================
// Edge micro-kernel: handles MR/NR edge cases with scalar fallback
// =============================================================================
static void micro_kernel_edge(
    int mr, int nr, int kc,
    const float* __restrict__ packed_A,
    const float* __restrict__ packed_B,
    float* __restrict__ C, int ldc)
{
    // Simple scalar fallback for edge tiles
    for (int k = 0; k < kc; ++k) {
        for (int i = 0; i < mr; ++i) {
            float a_val = packed_A[k * MR + i];
            for (int j = 0; j < nr; ++j) {
                C[i * ldc + j] += a_val * packed_B[k * NR + j];
            }
        }
    }
}

// =============================================================================
// Main GEMM driver: Goto's algorithm with 3 levels of blocking
//
// Loop structure (outermost to innermost):
//   jc loop (NC blocks of B columns) — L3 blocking for B panel
//     pc loop (KC blocks of shared dim) — pack B panel
//       ic loop (MC blocks of A rows) — L2 blocking, pack A panel
//         jr loop (NR micro-tiles of B) — micro-kernel col index
//           ir loop (MR micro-tiles of A) — micro-kernel row index
//             micro_kernel_6x16(...)    — register-level computation
// =============================================================================
void gemm_avx2(
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

    // Pre-multiply alpha into A during packing (avoids per-element multiply)
    // For simplicity we'll pack alpha*A below.

    // Allocate packing buffers (reused across loop iterations)
    float* packed_B = aligned_alloc_floats(KC * NC);
    float* packed_A = aligned_alloc_floats(MC * KC);

    if (!packed_B || !packed_A) {
        free(packed_B);
        free(packed_A);
        return;  // allocation failure
    }

    // === jc loop: partition B columns into NC-wide panels ===
    for (int jc = 0; jc < N; jc += NC) {
        int nc = std::min(NC, N - jc);

        // === pc loop: partition shared dimension K into KC-deep slices ===
        for (int pc = 0; pc < K; pc += KC) {
            int kc = std::min(KC, K - pc);

            // Pack B[pc:pc+kc, jc:jc+nc] → packed_B
            // This B panel lives in L3 and is reused across all ic iterations
            pack_B(B + pc * ldb + jc, ldb, kc, nc, packed_B);

            // === ic loop: partition A rows into MC-high panels ===
            for (int ic = 0; ic < M; ic += MC) {
                int mc = std::min(MC, M - ic);

                // Pack alpha * A[ic:ic+mc, pc:pc+kc] → packed_A
                // We bake alpha into the packing to avoid extra multiply
                pack_A(A + ic * lda + pc, lda, mc, kc, packed_A);

                // Apply alpha during packing
                {
                    int packed_a_size = mc * kc;  // conceptually
                    // Round up to account for MR padding
                    int num_mr_panels = (mc + MR - 1) / MR;
                    int total = num_mr_panels * kc * MR;
                    for (int idx = 0; idx < total; ++idx)
                        packed_A[idx] *= alpha;
                }

                // === jr loop: iterate over NR-wide micro-tiles of B ===
                for (int jr = 0; jr < nc; jr += NR) {
                    int nr = std::min(NR, nc - jr);

                    // Pointer to the packed B strip for column jr
                    const float* b_panel = packed_B + jr * kc;
                    // Note: pack_B stores jr-th strip at offset (jr/NR)*kc*NR
                    // Since jr steps by NR: b_panel = packed_B + (jr/NR)*kc*NR

                    // === ir loop: iterate over MR-high micro-tiles of A ===
                    for (int ir = 0; ir < mc; ir += MR) {
                        int mr = std::min(MR, mc - ir);

                        const float* a_panel = packed_A + ir * kc;
                        // Note: pack_A stores ir-th strip at offset (ir/MR)*kc*MR

                        float* c_block = C + (ic + ir) * ldc + (jc + jr);

                        if (mr == MR && nr == NR) {
                            // Fast path: full 6×16 micro-kernel
                            micro_kernel_6x16(kc, a_panel, b_panel, c_block, ldc);
                        } else {
                            // Edge case: partial tile
                            micro_kernel_edge(mr, nr, kc, a_panel, b_panel, c_block, ldc);
                        }
                    }
                }
            }
        }
    }

    free(packed_B);
    free(packed_A);
}
