#pragma once
// Inline PTX primitives adapted from faces/tmem_face_2sm_k96_burst.cu.
// B layout, two-CTA allocation and scale initialization follow TMEM_LAYOUT_SPEC.md.
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

constexpr int Instructions = 1024, Width = 256, Depth = 96, MaxStages = 8;
__device__ unsigned sa(const void *p) {
    return unsigned(__cvta_generic_to_shared(p));
}
__device__ void st(unsigned a, unsigned v) {
    asm volatile("tcgen05.st.sync.aligned.32x32b.x1.b32 [%0], {%1};" ::"r"(a), "r"(v) : "memory");
}
__device__ unsigned ld(unsigned a) {
    unsigned v;
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x1.b32 {%0}, [%1]; tcgen05.wait::ld.sync.aligned;"
                 : "=r"(v)
                 : "r"(a)
                 : "memory");
    return v;
}
__device__ void done(unsigned b) {
    asm volatile(
        "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;" ::
            "r"(b),
        "h"((unsigned short)3)
        : "memory");
}
__device__ void await(unsigned b, unsigned phase) {
    asm volatile(
        "{ .reg .pred p; Lwait: mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%0], %1; @!p bra Lwait; }" ::
            "r"(b),
        "r"(phase)
        : "memory");
}
__device__ unsigned long long descriptor(unsigned ba) {
    return ((ba >> 4) & 0x3fffull) | ((((ba + 32) >> 4) & 0x3fffull) << 16) | (64ull << 32) |
           (1ull << 46) | (1ull << 52) | (2ull << 61);
}
__device__ void mma(unsigned tm, unsigned long long desc, bool acc) {
    asm volatile(
        "{.reg .pred p; setp.ne.u32 p,%5,0; tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.block16 [%0], [%1], %2, %3, [%4], [%6], p;}" ::
            "r"(tm),
        "r"(tm + 288), "l"(desc), "r"(0x90400480u), "r"(tm + 256), "r"(int(acc)), "r"(tm + 272)
        : "memory");
}
struct Measure {
    unsigned long long cycles, ns;
};
void ck(cudaError_t e) {
    if (e != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(e));
    }
}
unsigned code(int row, int k, int stage, bool b) {
    unsigned z = unsigned(row * 73 + k * 31 + stage * 97 + (row * k) % 29 + (b ? 19 : 7));
    return (1 + z % 4) | ((z / 5 % 2) * 8);
}
int value(unsigned code) {
    return int(code & 7) * ((code & 8) ? -1 : 1);
}
void file(const std::string &path, void *data, size_t bytes, bool write) {
    FILE *f = fopen(path.c_str(), write ? "wb" : "rb");
    if (!f) {
        throw std::runtime_error("cannot open " + path);
    }
    size_t n = write ? fwrite(data, 1, bytes, f) : fread(data, 1, bytes, f);
    fclose(f);
    if (n != bytes) {
        throw std::runtime_error("short file " + path);
    }
}

__device__ void done_leader(unsigned b) {
    asm volatile(
        "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0],%1;" ::
            "r"(b),
        "h"((unsigned short)1)
        : "memory");
}
