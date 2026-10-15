// Runtime plan for column-major matrices: A is lda x k, B is ldb x n, C is ldc x n.
// No input read crosses the logical rows; output padding is not written.
// Select the smallest available K-plan covering k and its matching encoder sign flips.
// Recompute ring fields for the runtime cut count and check table premises before allocating operands.
// Included after the kernel and CRT constants by l8rt.cu.
#pragma once
#include "../../../CUDA-Common/Common/host_options.h"
#include <vector>
#include <algorithm>
#include <cstdlib>
#include "sample_hash.cuh" // the check behind the reuse promise
#include <cstring>
#include <stdexcept>
#include <string>

#ifndef L8_S12_DEFAULTS
#define L8_S12_DEFAULTS 0 // 1 enables the tuned defaults; 0 uses the baseline settings.
#endif
#ifndef L8_PACE_MIN_WORK
#define L8_PACE_MIN_WORK 40000 // Full-round work per cluster required to enable pacing by default.
#endif
#ifndef L8_KPLAN4096
#define L8_KPLAN4096 0 // 1 selects the K4096 table for k <= 4096; 0 selects the K8192 table.
#endif
#if L8_KPLAN4096
#include "l8k_tables_k4096.h"
#endif
#ifndef L8_EXTRA_FX_KNOB
#define L8_EXTRA_FX_KNOB 0 // L8_ENC_EXTRA_FX selects FastEncoder::extra_fx.
#endif

namespace l8rt {

inline U128 mulmod_u(U128 a, unsigned b, U128 p) {
    return (a % p) * b % p;
} // a < p < 2^125, b < 2^8: no overflow
// the CRT constants live in __constant__ memory (one copy per module): the plan that uploaded them last. Plans of different
// K plans differ in lambda0^-1, so run uploads its own when another plan was the last (single host thread per module).
inline const void *&crt_owner() {
    static const void *o = nullptr;
    return o;
}

struct Plan {
    int m = 0, n = 0, k = 0, bits = 53, kplan = 0;
    long long lda = 0, ldb = 0, ldc = 0;
    L8kMod hm[L8K_NMOD];
    L8Shape sh{};
    U128 P = 1;
#if L8_FAST_ENC
    l8e::FastEncoder enc;
#else
    l8e::Encoder enc;
#endif
    int *ea = nullptr, *eb = nullptr, *bad = nullptr, *hbad = nullptr;
    unsigned char *pa = nullptr, *pb = nullptr;
    unsigned short *resid = nullptr;
    size_t bytesA = 0, bytesB = 0, resid_bytes = 0;
    unsigned block_tiles = 0;
    L8kMod *dmods = nullptr;
    CUtensorMap tmA, tmB;
    KernelMetrics *km = nullptr;
    Metrics *mm = nullptr;
    f64_impl::U128 *crt_state = nullptr;
    cudaLaunchAttribute attr{};
    cudaLaunchConfig_t cfg{};
    cudaEvent_t ev[4] = {nullptr, nullptr, nullptr, nullptr};
    cudaEvent_t crt_ev[2] = {nullptr, nullptr}; // the CRT kernel after a split GEMM, timed apart
    float last_crt_ms = 0; // that kernel's time in the last timed call (0 with whole tiles)
    int crt_chunks = 16;   // column ranges per (tile, rank) of l8k_crt_after
    int accs = 0;
    // the sides the last call encoded, for run(..., reuse); the caller promises the contents did not change
    const double *lastA = nullptr;
    const double *lastB = nullptr;
    unsigned long long reused_a = 0, reused_b = 0;
    // a side asked to be reused is sampled on every such call; it is
    // skipped only when the pointer is the last encoded one and the sample hash is the stored one, otherwise it is encoded again
    // (a changed hash under the same pointer counts as a broken promise).
    static constexpr int kSamples = 4096;
    unsigned long long *dsh = nullptr, *hsh = nullptr;
    unsigned long long hashA = 0, hashB = 0, violations = 0;
    bool validA = false, validB = false;
    unsigned hwl[L8K_NMOD][4];
    unsigned long long hmagic = 0;
    unsigned h2wl[L8K_NMOD][2] = {}, h2fx[L8K_NMOD] = {}; // crt2_dev.cuh constants (L8_CRT2)
    unsigned long long h2c[4] = {};
    unsigned *dpace = nullptr,
             *dpace_to = nullptr; // the pace entries (one per cluster) and the waits that gave up

    void init(int m_, int n_, int k_, long long lda_, long long ldb_, long long ldc_,
              const awe_engine_options *options = nullptr) {
        if (m_ < 1 || n_ < 1 || k_ < 1) {
            throw std::runtime_error("shape: m, n, k must be positive");
        }
        if (lda_ < m_ || ldb_ < k_ || ldc_ < m_) {
            throw std::runtime_error("shape: lda >= m, ldb >= k and ldc >= m are required");
        }
        if (k_ > ModulusSet::table_max_k) {
            throw std::runtime_error(AWE_GEOMETRY_TABLE_ERROR);
        }
        m = m_;
        n = n_;
        k = k_;
        lda = lda_;
        ldb = ldb_;
        ldc = ldc_;
        kplan = k <= 8192 ? 8192 : 16384;
        const L8kMod *base = kplan == 8192 ? l8k_mods59_k8192 : l8k_mods59_k16384;
        const unsigned *flips = kplan == 8192 ? l8k_flips59_k8192 : l8k_flips59_k16384;
        accs = kplan == 8192 ? L8K_ACCS59_K8192 : L8K_ACCS59_K16384;
#if L8_KPLAN4096
        // Select the K4096 table together with its encoder sign flips.
        // The generator checks the per-accumulator bounds at that plan's K.
        if (k <= 4096) {
            kplan = 4096;
            base = l8k_mods59_k4096;
            flips = l8k_flips59_k4096;
            accs = L8K_ACCS59_K4096;
        }
#endif
        const int cuts = l8e::cuts_of(k);
        // the ring fields for this cut count (make_l8k_tables.py's running sums)
        memcpy(hm, base, sizeof(hm));
        int gbase = 0, pbase = 0, gt = 0;
        for (int q = 0; q < L8K_NMOD; ++q) {
            gt += (cuts + L8K_FTS / hm[q].planes - 1) / (L8K_FTS / hm[q].planes);
        }
        for (int q = 0; q < L8K_NMOD; ++q) {
            const int cpg = L8K_FTS / hm[q].planes, groups = (cuts + cpg - 1) / cpg;
            hm[q].cpg = cpg;
            hm[q].groups = groups;
            hm[q].gbase = gbase;
            hm[q].pbase = pbase;
            hm[q].gt = gt;
            gbase += groups;
            pbase += hm[q].planes;
        }
        l8k_check_tables(hm, L8K_NMOD, k, cuts, L8K_FTS, kBoxTiles,
                         false); // P1..P4 at the run-time k and cuts
        P = 1;
        for (int q = 0; q < L8K_NMOD; ++q) {
            P *= (unsigned)hm[q].p;
        }
        if (P >> 125) {
            throw std::runtime_error("the CRT assumes P < 2^125");
        }
        { // exactness for every input: P > 2 k (2^bits - 1)^2 (|sum| <= k (2^bits - 1)^2)
            const U128 qm = (U128(1) << bits) - 1;
            if (!(P / 2 / U128(unsigned(k)) / qm >= qm)) {
                throw std::runtime_error(
                    "the modulus product does not cover 2 k (2^bits - 1)^2 at this k");
            }
        }

        enc.set_flips_for_k(kplan);
        enc.init(bits, m, n, k, false);
#if L8_FAST_ENC
        // Unset: the encoder defaults, or the L8_S12_DEFAULTS choices below when enabled.
        // the encoder's blocks per SM (L8_ENC_MINB: 0 = the default 3, or 3..6) and B's rows 112..127 of
        // each half left unwritten (L8_ENC_SKIP_BPAD=1; the N = 224 x cta_group 2 MMA reads 112 rows per CTA).
        if (const char *e = awe_option_env(options, "L8_ENC_MINB")) {
            enc.minb = atoi(e);
        }
        if (const char *e = awe_option_env(options, "L8_ENC_SKIP_BPAD")) {
            enc.skip_b_pad = atoi(e) != 0;
        }
        // L8_PDL=1 = a call that encodes both sides runs FastEncoder::chain (shift A, shift B, encode A,
        // encode B with programmatic dependent launch between them; MINB 3 or 6, otherwise chain's plain launches).
        if (const char *e = awe_option_env(options, "L8_PDL")) {
            enc.pdl = atoi(e) != 0;
        }
        // L8_ENC_SHIFT_FORM=1 = shift A's block form chosen by m (2x4 below 768 rows, 4x4 up to 2,048);
        // The default is 8x1. The source test_chain.cu checks reported byte equality on RTX 4070 Ti.
        // Their shape list is not retained here; other shapes are intended to preserve bytes,
        // not established by that observation.
        if (const char *e = awe_option_env(options, "L8_ENC_SHIFT_FORM")) {
            enc.shift_form = atoi(e);
        }
#if L8_EXTRA_FX_KNOB
        // L8_ENC_EXTRA_FX selects FastEncoder::extra_fx: 0, 256, 512, or 768.
        // The 256 form pairs the one-plane moduli; invalid values throw.
        if (const char *e = awe_option_env(options, "L8_ENC_EXTRA_FX")) {
            enc.extra_fx = atoi(e);
        }
#endif
#if L8_S12_DEFAULTS
        // The default settings can be overridden by the corresponding environment variables.
        // Choose encoder occupancy from the number of blocks on each side.
        // Small grids use fewer resident blocks; larger grids use the higher occupancy form.
        if (!awe_option_env(options, "L8_ENC_SKIP_BPAD")) {
            enc.skip_b_pad = true;
        }
        if (!awe_option_env(options, "L8_PDL")) {
            enc.pdl = true;
        }
        if (!awe_option_env(options, "L8_ENC_MINB")) {
            int dev = 0, sms = 0;
            ck(cudaGetDevice(&dev));
            ck(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
            const long long ga = 6LL * enc.geo.cuts * enc.tiles(0) * 2,
                            gb = 6LL * enc.geo.cuts * enc.tiles(1) * 2;
            enc.minb = (ga <= 3LL * sms && gb <= 3LL * sms) ? 3 : 6;
        }
#endif
#endif
        if (enc.geo.planes != L8K_PLANES59 || enc.geo.cuts != cuts) {
            throw std::runtime_error("encoder geometry differs from the engine's");
        }
        for (int q = 0; q < L8K_NMOD; ++q) {
            if (enc.flips[0][q] != flips[q] || enc.flips[1][q] != 0u) {
                throw std::runtime_error("encoder sign flips differ from the table's");
            }
        }

        sh.m = m;
        sh.n = n;
        sh.cuts = cuts;
        sh.tile_rows = (m + Side - 1) / Side;
        sh.tile_cols = (n + Cols - 1) / Cols;
        sh.tiles = sh.tile_rows * sh.tile_cols;
        { // Split the moduli of a tile across clusters when the grid cannot fill the device.
            // Choose the smallest split count within 5% of the best cost estimate.
            // The estimate is rounds(nc) / nc + 0.15 for nc > 1, accounting for the separate CRT.
            // L8_NCHUNK=1 selects whole tiles; a number forces a modulus split count.
            int nc = 1;
            const char *ev = awe_option_env(options, "L8_NCHUNK");
            if ((!ev || std::string(ev) == "auto") && sh.tiles < 2 * GridClusters) {
                auto cost = [&](int c) {
                    return double((sh.tiles * c + GridClusters - 1) / GridClusters) / c +
                           (c > 1 ? 0.15 : 0.0);
                };
                double best = cost(1);
                for (int c = 2; c <= kParts; ++c) {
                    best = std::min(best, cost(c));
                }
                while (cost(nc) > best * 1.05) {
                    ++nc;
                }
            }
            if (ev && std::string(ev) != "auto") {
                nc = std::max(1, std::min(kParts, atoi(ev))); // an explicit number
            }
            // from 2 tiles per cluster up the full rounds stay whole tiles and only the rem = tiles %
            // clusters tiles of the last round are split, into nct = min(kParts, clusters / rem) ranges (one round of rem x nct items); the
            // split applies when nct >= 2 (rem <= clusters / 2). 4096^3: 304 tiles = 4 rounds + 8 -> 296 whole + 8 x 9 items. L8_TAIL=0
            // An explicit L8_NCHUNK selects the all-split form (whole = 0).
            int whole = 0;
            const char *et = awe_option_env(options, "L8_TAIL");
            if ((!ev || std::string(ev) == "auto") && sh.tiles >= 2 * GridClusters &&
                !(et && std::string(et) == "0")) {
                const int rem = sh.tiles % GridClusters,
                          nct = rem > 0 ? std::min(kParts, GridClusters / rem) : 1;
                if (nct >= 2) {
                    nc = nct;
                    whole = sh.tiles - rem;
                }
            }
            sh.nchunk = nc;
            sh.whole = nc > 1 ? whole : sh.tiles;
        }
        {
            const int items =
                sh.nchunk > 1 ? sh.whole + (sh.tiles - sh.whole) * sh.nchunk : sh.tiles;
            sh.clusters = items < GridClusters ? items : GridClusters;
        }
        sh.ldc = ldc;
        sh.plane = (unsigned long long)sh.tiles * Cols * Side;
        // L8_ITEM_ORDER=1 = the split tiles' items chunk-major (with L8_NCHUNK=21 every
        // tile's modulus 0 before any tile's modulus 1); unset / 0 = tile-major. No effect with whole tiles only.
        sh.item_order = 0;
        if (const char *e = awe_option_env(options, "L8_ITEM_ORDER")) {
            sh.item_order = atoi(e) != 0 ? 1 : 0;
        }
        // L8_PACE=<S>, S >= 0 = the pace of the A-side supply (l8k_kernel_rt.cuh's L8Shape); unset or
        // negative = off (no memset, no wait: serial baseline)
        sh.pace_s = -1;
        sh.pace = nullptr;
        sh.pace_timeouts = nullptr;
        if (const char *e = awe_option_env(options, "L8_PACE")) {
            sh.pace_s = atoi(e) < 0 ? -1 : atoi(e);
        }
#if L8_S12_DEFAULTS
        // Enable pacing only when the cluster has enough full-round work.
        // The threshold uses full-round work per cluster, measured in part-cuts.
        else if (sh.nchunk == 1 &&
                 (long long)(sh.tiles / GridClusters) * kParts * cuts >= L8_PACE_MIN_WORK) {
            sh.pace_s = 1;
        }
#endif
        block_tiles = unsigned(2 * cuts * L8K_PLANES59);
        bytesA = enc.packed_side_bytes(0);
        bytesB = enc.packed_side_bytes(1);
        if (bytesA != size_t(sh.tile_rows) * block_tiles * TileBytes ||
            bytesB != size_t(sh.tile_cols) * block_tiles * TileBytes) {
            throw std::runtime_error("packed sizes differ from the engine's");
        }

        ck(cudaMalloc(&ea, size_t(m) * 4));
        ck(cudaMalloc(&eb, size_t(n) * 4));
        ck(cudaMalloc(&bad, 4));
        ck(cudaMallocHost(&hbad, 4));
        ck(cudaMalloc(&pa, bytesA));
        ck(cudaMalloc(&pb, bytesB));
        resid_bytes = size_t(sh.plane) * kParts * sizeof(ResT);
        ck(cudaMalloc(&resid, resid_bytes));
        ck(cudaMalloc(&crt_state, 16));
        // L8_PINGPONG=1 overlaps payout with the next part's MMAs; 0 uses serial payout.
        // The kernel puts a one-accumulator modulus at TMEM place accb (0 / 1) and adds the product's
        // accumulator bit, so it needs every product of a one-accumulator modulus on accumulator 0: checked here on the table it runs.
        sh.pingpong = 0;
        if (const char *e = awe_option_env(options, "L8_PINGPONG")) {
            sh.pingpong = atoi(e) != 0 ? 1 : 0;
        }
        // Prefetch and proportional CRT column assignment default on unless explicitly disabled.
        sh.crt_pf = L8_CRT_PF;
        sh.crt_prop = L8_CRT_PROP;
#if L8_CRT_SWITCH
        if (const char *e = awe_option_env(options, "L8_CRT_PF")) {
            sh.crt_pf = atoi(e) != 0 ? 1 : 0;
        }
        if (const char *e = awe_option_env(options, "L8_CRT_PROP")) {
            sh.crt_prop = atoi(e) != 0 ? 1 : 0;
        }
#endif
        sh.crt2 = 0;
        if (sh.pingpong) {
            for (int i = 0; i < L8K_NMOD; ++i) {
                for (int j = 0; j < hm[i].nprod; ++j) {
                    if (hm[i].nacc == 1 && ((hm[i].prod[j] >> 4) & 1u)) {
                        throw std::runtime_error(
                            "L8_PINGPONG: a one-accumulator modulus has a product on accumulator 1");
                    }
                }
            }
        }
        ck(cudaMalloc(&dmods, sizeof(hm)));
        ck(cudaMemcpy(dmods, hm, sizeof(hm), cudaMemcpyHostToDevice));
        // tensor maps over the packed operands (rows of 256 bytes); box 256 x (2 plane tiles / 256)
        memset(&tmA, 0, sizeof tmA);
        memset(&tmB, 0, sizeof tmB);
        {
            typedef CUresult (*EncodeTiled)(
                CUtensorMap *, CUtensorMapDataType, cuuint32_t, void *, const cuuint64_t *,
                const cuuint64_t *, const cuuint32_t *, const cuuint32_t *, CUtensorMapInterleave,
                CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
            void *fn = nullptr;
            cudaDriverEntryPointQueryResult qr;
            ck(cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &fn, cudaEnableDefault, &qr));
            if (!fn || qr != cudaDriverEntryPointSuccess) {
                throw std::runtime_error("cuTensorMapEncodeTiled not found");
            }
            auto make = [&](CUtensorMap *t, void *base_, size_t bytes) {
                const cuuint64_t gdim[2] = {256, cuuint64_t(bytes / 256)};
                const cuuint64_t gs[1] = {256};
                const cuuint32_t box[2] = {256, cuuint32_t(kBoxTiles * TileBytes / 256)};
                const cuuint32_t es[2] = {1, 1};
                CUresult r = ((EncodeTiled)fn)(
                    t, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, base_, gdim, gs, box, es,
                    CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
                    CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
                if (r != CUDA_SUCCESS) {
                    throw std::runtime_error("cuTensorMapEncodeTiled failed " +
                                             std::to_string(int(r)));
                }
            };
            make(&tmA, pa, bytesA);
            make(&tmB, pb, bytesB);
        }
        // the CRT constants of crt_cols_rt: W_q lambda0_q^-1 mod P in 4 words (make_l8k_src.py's setup, without the LUT) and the
        // Barrett magic of P's top word
        {
            for (int q = 0; q < L8K_NMOD; ++q) {
                const unsigned p = unsigned(hm[q].p);
                const U128 base_q = P / p;
                unsigned inv = 1;
                while (unsigned(base_q % p) * inv % p != 1) {
                    ++inv;
                }
                if (base_q * p != P) {
                    throw std::runtime_error("p does not divide P");
                }
                inv = inv * hm[q].lam0inv % p;
                const U128 w = mulmod_u(base_q, inv, P);
                for (int j = 0; j < 4; ++j) {
                    hwl[q][j] = unsigned(w >> (32 * j));
                }
            }
            const unsigned long long pt1 = (unsigned long long)(P >> 96) + 1;
            hmagic = (unsigned long long)((U128(1) << 64) / pt1);
            ck(cudaMemcpyToSymbol(crt_wl, hwl, sizeof(hwl)));
            ck(cudaMemcpyToSymbol(crt_pt_magic, &hmagic, sizeof(hmagic)));
#if L8_CRT2
            { // Enable the two-group CRT only when its constants cover this modulus product.
                int pq[L8K_NMOD];
                unsigned lq[L8K_NMOD];
                for (int q = 0; q < L8K_NMOD; ++q) {
                    pq[q] = hm[q].p;
                    lq[q] = hm[q].lam0inv;
                }
                int on = 1;
#if L8_CRT_SWITCH
                if (const char *e = awe_option_env(options, "L8_CRT2")) {
                    on = atoi(e) != 0 ? 1 : 0;
                }
#endif
                if (on && !crt2_make(pq, lq, h2wl, h2fx, h2c)) {
                    on = 0;
                }
                if (on && U128(h2c[0]) * U128(h2c[1]) != P) {
                    on = 0;
                }
#if !L8_CRT_SWITCH
                if (!on) {
                    throw std::runtime_error(
                        "L8_CRT2: the build's two-group split does not cover this P (crt2_split.h)");
                }
#endif
                sh.crt2 = on;
                if (on) {
                    ck(cudaMemcpyToSymbol(crt2_wl, h2wl, sizeof(h2wl)));
                    ck(cudaMemcpyToSymbol(crt2_fx, h2fx, sizeof(h2fx)));
                    ck(cudaMemcpyToSymbol(crt2_c, h2c, sizeof(h2c)));
                }
            }
#endif
            crt_owner() = this;
        }
        ck(cudaMalloc(&km, size_t(sh.tiles) * 2 * sizeof(KernelMetrics)));
        ck(cudaMalloc(&mm, size_t(sh.tiles) * kParts * 2 * sizeof(Metrics)));
        if (sh.pace_s >= 0) {
            ck(cudaMalloc(&dpace, GridClusters * sizeof(unsigned)));
            ck(cudaMalloc(&dpace_to, sizeof(unsigned)));
            ck(cudaMemset(dpace_to, 0, sizeof(unsigned)));
            sh.pace = dpace;
            sh.pace_timeouts = dpace_to;
        }
        attr.id = cudaLaunchAttributeClusterDimension;
        attr.val.clusterDim = {2, 1, 1};
        cfg.gridDim = dim3(2 * sh.clusters);
        cfg.blockDim = dim3(384);
        cfg.attrs = &attr;
        cfg.numAttrs = 1;
        cudaFuncAttributes ka{};
        ck(cudaFuncGetAttributes(&ka, l8k_gemm_kernel_rt<true>));
        if (ka.sharedSizeBytes + kShared > 227 * 1024) {
            throw std::runtime_error("Shared capacity");
        }
        ck(cudaFuncSetAttribute(l8k_gemm_kernel_rt<true>,
                                cudaFuncAttributeMaxDynamicSharedMemorySize, kShared));
        cfg.dynamicSmemBytes = kShared;
        for (auto &e : ev) {
            ck(cudaEventCreate(&e));
        }
        for (auto &e : crt_ev) {
            ck(cudaEventCreate(&e));
        }
        if (const char *e = awe_option_env(options, "L8_CRT_CHUNKS")) {
            crt_chunks = std::max(1, std::min(112, atoi(e)));
        }
        ck(cudaMalloc(&dsh, 16));
        ck(cudaMallocHost(&hsh, 16));
    }

    // 0 = ok, 2 = an input is not finite (checked after the call; C then holds no product).
    // reuse bit 1 (A) / 2 (B): when the pointer is the one the last call encoded, that side's shift and encoding are skipped
    // and its exponents and packed planes are used again (the caller promises the values did not change; lda / ldb / m / n / k are
    // the plan's own). A call that finds a non-finite input forgets both sides.
    int run(const double *A, const double *B, double *C, cudaStream_t st, float *stage_ms,
            unsigned reuse = 0) {
        ck(cudaEventRecord(ev[0],
                           st)); // the sample hash and its sync are inside the first stage's time
        bool skipA = false, skipB = false;
        if (reuse & 3u) { // sample the sides asked to be reused (one sync), then decide
            if (reuse & 1u) {
                ck(shash::hash_sample<double>(A, lda, m, k, kSamples, dsh, st));
            }
            if (reuse & 2u) {
                ck(shash::hash_sample<double>(B, ldb, k, n, kSamples, dsh + 1, st));
            }
            ck(cudaMemcpyAsync(hsh, dsh, 16, cudaMemcpyDeviceToHost, st));
            ck(cudaStreamSynchronize(st));
            if (reuse & 1u) {
                if (A == lastA && validA && hsh[0] == hashA) {
                    skipA = true;
                } else {
                    if (A == lastA && validA) {
                        ++violations;
                    }
                    hashA = hsh[0];
                    validA = true;
                }
            }
            if (reuse & 2u) {
                if (B == lastB && validB && hsh[1] == hashB) {
                    skipB = true;
                } else {
                    if (B == lastB && validB) {
                        ++violations;
                    }
                    hashB = hsh[1];
                    validB = true;
                }
            }
        }
        if (!(reuse & 1u)) {
            validA = false;
        }
        if (!(reuse & 2u)) {
            validB = false;
        }
        if (crt_owner() != this) { // another plan (another K plan) uploaded its constants last
            ck(cudaMemcpyToSymbolAsync(crt_wl, hwl, sizeof(hwl), 0, cudaMemcpyHostToDevice, st));
            ck(cudaMemcpyToSymbolAsync(crt_pt_magic, &hmagic, sizeof(hmagic), 0,
                                       cudaMemcpyHostToDevice, st));
#if L8_CRT2
            if (sh.crt2) { // this plan's table (lam0inv) differs from another K plan's
                ck(cudaMemcpyToSymbolAsync(crt2_wl, h2wl, sizeof(h2wl), 0, cudaMemcpyHostToDevice,
                                           st));
                ck(cudaMemcpyToSymbolAsync(crt2_fx, h2fx, sizeof(h2fx), 0, cudaMemcpyHostToDevice,
                                           st));
                ck(cudaMemcpyToSymbolAsync(crt2_c, h2c, sizeof(h2c), 0, cudaMemcpyHostToDevice,
                                           st));
            }
#endif
            crt_owner() = this;
        }
        ck(cudaMemsetAsync(bad, 0, 4, st));
        if (sh.pace_s >= 0) {
            ck(cudaMemsetAsync(dpace, 0, GridClusters * sizeof(unsigned),
                               st)); // every launch starts at part 0
        }
#if L8_FAST_ENC
        if (enc.pdl && !skipA &&
            !skipB) { // no event may sit between chain's four kernels, so the first stage ends before them
            ck(cudaEventRecord(ev[1], st));
            enc.chain(A, lda, B, ldb, ea, eb, bad, pa, pb, st);
        } else
#endif
        {
            if (!skipA) {
                enc.shift(A, ea, bad, 0, lda, st);
            }
            if (!skipB) {
                enc.shift(B, eb, bad, 1, ldb, st);
            }
            ck(cudaEventRecord(ev[1], st));
            if (!skipA) {
                enc.launch(A, lda, ea, pa, 0, st);
            }
            if (!skipB) {
                enc.launch(B, ldb, eb, pb, 1, st);
            }
        }
        ck(cudaEventRecord(ev[2], st));
        cfg.stream = st;
        ck(cudaLaunchKernelEx(&cfg, l8k_gemm_kernel_rt<true>, (const unsigned char *)pa,
                              (const unsigned char *)pb, resid, (const L8kMod *)dmods, block_tiles,
                              km, mm, crt_state, (const f64_impl::U128 *)nullptr, C,
                              (const int *)ea, (const int *)eb, P, tmA, tmB, sh));
        ck(cudaGetLastError());
        if (sh.nchunk > 1) { // Reconstruct split tiles after GEMM, timing the CRT column ranges.
            ck(cudaEventRecord(crt_ev[0], st));
            l8k_crt_after<<<(sh.tiles - sh.whole) * 2 * crt_chunks, 256, 0, st>>>(
                resid, C, (const int *)ea, (const int *)eb, P, sh,
                crt_chunks); // the split tiles only
            ck(cudaGetLastError());
            ck(cudaEventRecord(crt_ev[1], st));
        }
        ck(cudaEventRecord(ev[3], st));
        lastA = A;
        lastB = B;
        reused_a += skipA;
        reused_b += skipB;
        if (stage_ms) {
            ck(cudaEventSynchronize(ev[3]));
            ck(cudaEventElapsedTime(&stage_ms[0], ev[0], ev[1]));
            ck(cudaEventElapsedTime(&stage_ms[1], ev[1], ev[2]));
            ck(cudaEventElapsedTime(&stage_ms[2], ev[2], ev[3]));
            last_crt_ms = 0;
            if (sh.nchunk > 1) {
                ck(cudaEventElapsedTime(&last_crt_ms, crt_ev[0], crt_ev[1]));
            }
            ck(cudaMemcpy(hbad, bad, 4, cudaMemcpyDeviceToHost));
            if (*hbad) {
                lastA = lastB = nullptr;
                validA = validB = false;
                return 2;
            }
        }
        return 0;
    }

    unsigned pace_timeouts() const { // the waits that gave up, over every call so far
        unsigned h = 0;
        if (dpace_to) {
            ck(cudaMemcpy(&h, dpace_to, sizeof(unsigned), cudaMemcpyDeviceToHost));
        }
        return h;
    }

    void destroy() {
        if (crt_owner() == this) {
            crt_owner() = nullptr;
        }
        lastA = lastB = nullptr;
        validA = validB = false;
        cudaFree(dsh);
        if (hsh) {
            cudaFreeHost(hsh);
        }
        dsh = hsh = nullptr;
        cudaFree(dpace);
        cudaFree(dpace_to);
        dpace = dpace_to = nullptr;
        for (auto &e : ev) {
            if (e) {
                cudaEventDestroy(e);
                e = nullptr;
            }
        }
        for (auto &e : crt_ev) {
            if (e) {
                cudaEventDestroy(e);
                e = nullptr;
            }
        }
        cudaFree(ea);
        cudaFree(eb);
        cudaFree(bad);
        if (hbad) {
            cudaFreeHost(hbad);
        }
        cudaFree(pa);
        cudaFree(pb);
        cudaFree(resid);
        cudaFree(crt_state);
        cudaFree(dmods);
        cudaFree(km);
        cudaFree(mm);
        ea = eb = bad = hbad = nullptr;
        pa = pb = nullptr;
        resid = nullptr;
        crt_state = nullptr;
        dmods = nullptr;
        km = nullptr;
        mm = nullptr;
        enc.release();
    }
};

} // namespace l8rt
