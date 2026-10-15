#pragma once
#include <cstdint>
#include <cuda_runtime.h>
namespace v5b {
struct Ctx;
Ctx *create();
void destroy(Ctx *c);
// 0 = ok; 1 = unsupported shape (too many tiles for the kernel's 64 tiles a pair); 2 = CUDA error; 3 = no tensor-map
// entry point
int set_shape(Ctx *c, int m, int n, int k);
int issue(Ctx *c, cudaStream_t s, const int8_t *A, long long lda, const int8_t *B, long long ldb, bool prof,
          cudaEvent_t *ev, int32_t *Cd = nullptr, int form = 0, bool reuse_b = false);
int32_t *out(Ctx *c);
int mpad(Ctx *c);
const unsigned long long *flag(Ctx *c);
const unsigned *stat_dev(Ctx *c);
// after the call (synchronizes): [0..2] A's shifted blocks / leftovers / max |shift|, [3..5] B's, [6] the segment units
// (80 = 1,536 k / 35 = 672 k), [7] the fallback flag, [8] / [9] the raw +-103 of A / B
int last_stats(Ctx *c, long long *out10);
} // namespace v5b
