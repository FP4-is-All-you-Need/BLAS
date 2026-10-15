

#pragma once
#include <stdint.h>
#ifdef __CUDACC__
#define GEN_HD __host__ __device__
#else
#define GEN_HD
#endif
GEN_HD inline uint64_t gen_sm64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}
GEN_HD inline int32_t gen_q(uint64_t s, uint64_t idx) {
    return int32_t(gen_sm64((s << 40) ^ idx) >> 40) - (1 << 23);
}
GEN_HD inline float gen_val(uint64_t s, uint64_t idx) {
    return float(gen_q(s, idx)) * (1.0f / 8388608.0f);
}
