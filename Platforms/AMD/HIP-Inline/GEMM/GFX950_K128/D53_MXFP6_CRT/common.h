#pragma once
// Shared definitions for the MXFP6 DGEMM engine. Headers that also run on the host (table checks,
// quantization, CRT) mark their functions with AWE_MXFP6_HD so the CPU test can call the same code.
#include <cstdint>
#include <stdexcept>
#include <string>

#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#define AWE_MXFP6_HD __host__ __device__
// Forced inlining keeps the instruction order of these device functions.
#define AWE_MXFP6_HD_INLINE __host__ __device__ __forceinline__
#else
#define AWE_MXFP6_HD
#define AWE_MXFP6_HD_INLINE inline
#endif

namespace awe_mxfp6 {

#if defined(__HIPCC__)
inline void hip_check(hipError_t e) {
    if (e != hipSuccess) {
        throw std::runtime_error(hipGetErrorString(e));
    }
}

// Device memory released on scope exit. No copies, so a buffer cannot be freed twice.
struct Memory {
    void *p = nullptr;
    Memory() = default;
    explicit Memory(size_t bytes) {
        hip_check(hipMalloc(&p, bytes));
    }
    // Release paths cannot throw, so a failed hipFree is not reported.
    ~Memory() {
        if (p) {
            (void)hipFree(p);
        }
    }
    Memory(const Memory &) = delete;
    Memory &operator=(const Memory &) = delete;
    Memory(Memory &&other) noexcept : p(other.p) {
        other.p = nullptr;
    }
    Memory &operator=(Memory &&other) noexcept {
        if (this != &other) {
            if (p) {
                (void)hipFree(p);
            }
            p = other.p;
            other.p = nullptr;
        }
        return *this;
    }
    template <class T> T *as() const {
        return static_cast<T *>(p);
    }
};
#endif

inline void require(bool ok, const char *why) {
    if (!ok) {
        throw std::invalid_argument(why);
    }
}

} // namespace awe_mxfp6
