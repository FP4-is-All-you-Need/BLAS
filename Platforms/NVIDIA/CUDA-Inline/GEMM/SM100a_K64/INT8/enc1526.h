#pragma once
#include <cstdint>
#if defined(__CUDACC__)
#define E_HD __host__ __device__ __forceinline__
#else
#define E_HD inline
#endif

namespace enc1526 {

// nibble position p (0..7) of an output word holds element PERM[p] of the 8 input elements
constexpr int PERM[8] = {0, 4, 1, 5, 2, 6, 3, 7};

E_HD uint32_t perm(uint32_t x, uint32_t y, uint32_t s) { // __byte_perm semantics
#if defined(__CUDA_ARCH__)
    return __byte_perm(x, y, s);
#else
    uint64_t v = (uint64_t(y) << 32) | x;
    uint32_t r = 0;
    for (int i = 0; i < 4; ++i) {
        uint32_t sel = (s >> (4 * i)) & 7;
        r |= uint32_t((v >> (8 * sel)) & 0xFF) << (8 * i);
    }
    return r;
#endif
}

// e[0..7]: table entries of the 8 elements. out[0..5]: one word per face.
E_HD void transpose_pack(const uint32_t (&e)[8], uint32_t (&out)[6]) {
    // bytes b of e0..e3 into one word (and of e4..e7): 4 + 3 prmt per 4 elements
    uint32_t lo[3], hi[3];
    {
        uint32_t t01 = perm(e[0], e[1], 0x5140), u01 = perm(e[0], e[1], 0x7362);
        uint32_t t23 = perm(e[2], e[3], 0x5140), u23 = perm(e[2], e[3], 0x7362);
        lo[0] = perm(t01, t23, 0x5410);
        lo[1] = perm(t01, t23, 0x7632);
        lo[2] = perm(u01, u23, 0x5410);
    }
    {
        uint32_t t01 = perm(e[4], e[5], 0x5140), u01 = perm(e[4], e[5], 0x7362);
        uint32_t t23 = perm(e[6], e[7], 0x5140), u23 = perm(e[6], e[7], 0x7362);
        hi[0] = perm(t01, t23, 0x5410);
        hi[1] = perm(t01, t23, 0x7632);
        hi[2] = perm(u01, u23, 0x5410);
    }
// byte i of lo[b] = (f_{2b+1} << 4 | f_{2b}) of element i; of hi[b] = of element 4+i
#pragma unroll
    for (int b = 0; b < 3; ++b) {
        out[2 * b] = (lo[b] & 0x0F0F0F0Fu) | ((hi[b] & 0x0F0F0F0Fu) << 4);
        out[2 * b + 1] = ((lo[b] >> 4) & 0x0F0F0F0Fu) | (hi[b] & 0xF0F0F0F0u);
    }
}

// x8: 8 int8 in a 64-bit word (element j = byte j). lut: 256 entries indexed by (uint8)x.
E_HD void encode8(uint64_t x8, const uint32_t *lut, uint32_t (&out)[6]) {
    uint32_t e[8];
#pragma unroll
    for (int j = 0; j < 8; ++j)
        e[j] = lut[(x8 >> (8 * j)) & 0xFF];
    transpose_pack(e, out);
}

// ---- host helpers ----------------------------------------------------------------------
inline uint32_t code_of(int v) {
    int m = v < 0 ? -v : v;
    uint32_t c = m == 6 ? 5u : uint32_t(m); // 0,1,2,3,4,6 -> 0,1,2,3,4,5
    return (v < 0 ? 8u : 0u) | c;
}
inline int value_of(uint32_t c) {
    int m = c & 7;
    int v = m == 5 ? 6 : m;
    return (c & 8) ? -v : v;
}
} // namespace enc1526
