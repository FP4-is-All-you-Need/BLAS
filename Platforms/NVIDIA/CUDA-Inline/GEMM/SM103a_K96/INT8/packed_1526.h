#pragma once
#include <stdint.h>
#ifdef __CUDACC__
#define AT1526 __host__ __device__ __forceinline__
#else
#define AT1526 inline
#endif
namespace integer_planes {
// Input codes 0..5 and 9..13: twice E2M1 value = 0,+-1,+-2,+-3,+-4,+-6.
// No negative zero. The caller guarantees all differences belong to this set.
AT1526 uint32_t decode(uint32_t p) {
    const uint32_t s = (p >> 3) & 0x11111111u;
    const uint32_t six = (p & (p >> 2)) & 0x11111111u; // code magnitude 5 means integer 6
    const uint32_t v = p + six;
    return (v ^ (s * 7u)) + s;
}
AT1526 uint32_t add(uint32_t a, uint32_t b) {
    return ((a & 0x77777777u) + (b & 0x77777777u)) ^ ((a ^ b) & 0x88888888u);
}
AT1526 uint32_t neg(uint32_t a) { return ((~a & 0x77777777u) + 0x11111111u) ^ (~a & 0x88888888u); }
AT1526 uint32_t encode(uint32_t v) {
    const uint32_t s = (v >> 3) & 0x11111111u;
    const uint32_t mag = (v ^ (s * 15u)) + s;
    const uint32_t six = ((mag >> 1) & (mag >> 2)) & 0x11111111u;
    return (mag - six) | (s << 3);
}
struct Diff {
    uint32_t d01, d12, d02;
};
AT1526 Diff form(uint32_t a0, uint32_t a1, uint32_t a2) {
    const auto x = decode(a0), y = decode(a1), z = decode(a2);
    const auto ny = neg(y), nz = neg(z);
    return {encode(add(x, ny)), encode(add(y, nz)), encode(add(x, nz))};
}
} // namespace integer_planes
#undef AT1526
