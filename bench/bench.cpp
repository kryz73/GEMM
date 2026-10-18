// =============================================================================
// bench.cpp — GEMM Benchmark Harness
//
// Methodology:
//   1. Pin to a single P-core (core 0) via CPU affinity
//   2. Warm up the kernel with 2 dry-run iterations
//   3. Time N_ITERS iterations, report min/median/mean
//   4. Compute GFLOPS = 2*M*N*K / time_ns * 1e-9
//   5. Verify correctness against naive implementation (max abs error)
//   6. Report % of theoretical single-core peak
//
// Theoretical peak for i5-12450H single P-core:
//   4.4 GHz × 2 FMA units × 8 floats/FMA = 70.4 SP GFLOPS
//   (Conservative: use turbo clock; actual sustained may be ~4.0 GHz)
// =============================================================================

#include "gemm.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <vector>
#include <numeric>
#include <random>

#ifdef __linux__
#include <sched.h>
#endif

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------
static constexpr int N_ITERS     = 10;
static constexpr int N_WARMUP    = 2;
static constexpr double PEAK_GHZ = 4.4;   // Turbo clock (single P-core)
static constexpr int FMA_UNITS   = 2;     // Alder Lake P-core: ports 0+1
static constexpr int FMA_WIDTH   = 8;     // 256-bit / 32-bit = 8 floats
static constexpr double PEAK_GFLOPS = PEAK_GHZ * FMA_UNITS * FMA_WIDTH;
// = 70.4 GFLOPS (theoretical single-core SP peak)

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static void pin_to_core(int core_id) {
#ifdef __linux__
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    sched_setaffinity(0, sizeof(cpuset), &cpuset);
#endif
}

static float* alloc_matrix(int rows, int cols) {
    void* ptr = nullptr;
    posix_memalign(&ptr, 64, (size_t)rows * cols * sizeof(float));
    return static_cast<float*>(ptr);
}

static void fill_random(float* mat, int n, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int i = 0; i < n; ++i) mat[i] = dist(rng);
}

static double median(std::vector<double>& v) {
    std::sort(v.begin(), v.end());
    int n = v.size();
    if (n % 2 == 0) return (v[n/2 - 1] + v[n/2]) / 2.0;
    return v[n/2];
}

static float max_abs_error(const float* ref, const float* test, int n) {
    float maxerr = 0.0f;
    for (int i = 0; i < n; ++i) {
        float err = std::fabs(ref[i] - test[i]);
        if (err > maxerr) maxerr = err;
    }
    return maxerr;
}

// Relative error accounting for scale of the result
static float max_rel_error(const float* ref, const float* test, int n) {
    float maxerr = 0.0f;
    for (int i = 0; i < n; ++i) {
        float denom = std::max(std::fabs(ref[i]), 1e-7f);
        float err = std::fabs(ref[i] - test[i]) / denom;
        if (err > maxerr) maxerr = err;
    }
    return maxerr;
}

// ---------------------------------------------------------------------------
// Benchmark one kernel
// ---------------------------------------------------------------------------
struct BenchResult {
    double min_gflops;
    double median_gflops;
    double mean_gflops;
    double min_ms;
    double median_ms;
    float  max_abs_err;
    float  max_rel_err;
};

using GemmFn = void(*)(int, int, int, float, const float*, int,
                        const float*, int, float, float*, int);

static BenchResult benchmark_kernel(
    const char* name,
    GemmFn fn,
    int M, int N, int K,
    const float* A, const float* B,
    const float* C_ref,  // reference result from naive (or nullptr to skip)
    float* C_work)       // scratch buffer for C
{
    double flops = 2.0 * M * N * K;
    std::vector<double> times_ms;
    times_ms.reserve(N_ITERS);

    // Warm-up
    for (int it = 0; it < N_WARMUP; ++it) {
        std::memset(C_work, 0, (size_t)M * N * sizeof(float));
        fn(M, N, K, 1.0f, A, K, B, N, 0.0f, C_work, N);
    }

    // Timed iterations
    for (int it = 0; it < N_ITERS; ++it) {
        std::memset(C_work, 0, (size_t)M * N * sizeof(float));

        auto t0 = std::chrono::high_resolution_clock::now();
        fn(M, N, K, 1.0f, A, K, B, N, 0.0f, C_work, N);
        auto t1 = std::chrono::high_resolution_clock::now();

        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        times_ms.push_back(ms);
    }

    // Compute statistics
    std::sort(times_ms.begin(), times_ms.end());
    double min_ms = times_ms.front();
    double med_ms = median(times_ms);
    double mean_ms = std::accumulate(times_ms.begin(), times_ms.end(), 0.0) / times_ms.size();

    BenchResult res;
    res.min_ms       = min_ms;
    res.median_ms    = med_ms;
    res.min_gflops   = flops / (min_ms * 1e6);    // ms→s: *1e-3, then /1e9
    res.median_gflops= flops / (med_ms * 1e6);
    res.mean_gflops  = flops / (mean_ms * 1e6);
    res.max_abs_err  = 0.0f;
    res.max_rel_err  = 0.0f;

    // Correctness check
    if (C_ref) {
        res.max_abs_err = max_abs_error(C_ref, C_work, M * N);
        res.max_rel_err = max_rel_error(C_ref, C_work, M * N);
    }

    return res;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    // Default sizes
    std::vector<int> sizes = {128, 256, 512, 1024, 2048};

    if (argc > 1) {
        sizes.clear();
        for (int i = 1; i < argc; ++i)
            sizes.push_back(std::atoi(argv[i]));
    }

    pin_to_core(0);  // Pin to P-core 0 for stable measurements

    printf("================================================================"
           "================================================================\n");
    printf("GEMM Benchmark — 12th Gen Intel i5-12450H (single P-core)\n");
    printf("Theoretical SP Peak: %.1f GFLOPS "
           "(%.1f GHz × %d FMA × %d floats/FMA)\n",
           PEAK_GFLOPS, PEAK_GHZ, FMA_UNITS, FMA_WIDTH);
    printf("Iterations: %d (+ %d warmup)\n", N_ITERS, N_WARMUP);
    printf("================================================================"
           "================================================================\n\n");

    struct KernelInfo {
        const char* name;
        GemmFn fn;
        const char* desc;
    };

    KernelInfo kernels[] = {
        {"Naive (i-j-k)",      gemm_naive,     "Textbook triple loop"},
        {"Reordered (i-k-j)",  gemm_reordered, "Loop interchange for locality"},
        {"Blocked (tiled)",    gemm_blocked,    "64×64×64 cache tiling"},
        {"AVX2/FMA (6×16)",    gemm_avx2,      "Packed micro-kernel + Goto alg"},
    };
    int n_kernels = sizeof(kernels) / sizeof(kernels[0]);

    std::mt19937 rng(42);

    for (int sz : sizes) {
        int M = sz, N = sz, K = sz;
        double flops = 2.0 * M * N * K;

        printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
               "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
        printf(" M=%d  N=%d  K=%d  |  %.2f GFLOP per call\n",
               M, N, K, flops / 1e9);
        printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
               "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
        printf("%-24s  %10s  %10s  %10s  %8s  %10s  %10s\n",
               "Kernel", "Min(ms)", "Med(ms)", "GFLOPS", "% Peak",
               "MaxAbsErr", "MaxRelErr");
        printf("%-24s  %10s  %10s  %10s  %8s  %10s  %10s\n",
               "------------------------", "----------", "----------",
               "----------", "--------", "----------", "----------");

        // Allocate matrices
        float* A     = alloc_matrix(M, K);
        float* B     = alloc_matrix(K, N);
        float* C_ref = alloc_matrix(M, N);
        float* C_work= alloc_matrix(M, N);

        fill_random(A, M * K, rng);
        fill_random(B, K * N, rng);

        // Compute reference: use naive for small, reordered for large
        std::memset(C_ref, 0, (size_t)M * N * sizeof(float));
        int ref_kernel = (sz > 1024) ? 1 : 0;  // reordered for large sizes
        kernels[ref_kernel].fn(M, N, K, 1.0f, A, K, B, N, 0.0f, C_ref, N);

        for (int ki = 0; ki < n_kernels; ++ki) {
            // Skip naive for very large sizes (too slow — ~25s/iter at 2048)
            if (ki == 0 && sz > 1024) {
                printf("%-24s  %10s  %10s  %10s  %8s  %10s  %10s\n",
                       kernels[ki].name, "skipped", "---", "---", "---",
                       "---", "---");
                continue;
            }

            BenchResult r = benchmark_kernel(
                kernels[ki].name, kernels[ki].fn,
                M, N, K, A, B,
                (ki > 0) ? C_ref : nullptr,  // compare non-naive to naive
                C_work);

            double pct_peak = r.median_gflops / PEAK_GFLOPS * 100.0;

            if (ki > 0) {
                printf("%-24s  %10.2f  %10.2f  %10.2f  %7.1f%%  %10.2e  %10.2e\n",
                       kernels[ki].name,
                       r.min_ms, r.median_ms, r.median_gflops, pct_peak,
                       r.max_abs_err, r.max_rel_err);
            } else {
                printf("%-24s  %10.2f  %10.2f  %10.2f  %7.1f%%  %10s  %10s\n",
                       kernels[ki].name,
                       r.min_ms, r.median_ms, r.median_gflops, pct_peak,
                       "(ref)", "(ref)");
            }
        }
        printf("\n");

        free(A);
        free(B);
        free(C_ref);
        free(C_work);
    }

    // Print bar chart summary for last size
    printf("================================================================"
           "================================================================\n");
    printf("Performance summary (visual) — last tested size\n");
    printf("================================================================"
           "================================================================\n");

    return 0;
}
