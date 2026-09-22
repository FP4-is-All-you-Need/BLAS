#include <awe_blas.hpp>
#include <array>
#include <cstdio>

struct device_buffer {
    void* pointer = nullptr;
    device_buffer() = default;
    device_buffer(const device_buffer&) = delete;
    device_buffer& operator=(const device_buffer&) = delete;
    ~device_buffer() { if (pointer) cudaFree(pointer); }
    cudaError_t allocate(size_t bytes) { return bytes ? cudaMalloc(&pointer, bytes) : cudaSuccess; }
};
#define CUDA_CHECK(call) do { auto e = (call); if (e != cudaSuccess) { \
    std::fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(e)); return 1; } } while (0)
#define AWE_CHECK(call) do { auto s = (call); if (s != AWE_SUCCESS) { \
    std::fprintf(stderr, "%s: %s\n", #call, awe_status_string(s)); return 1; } } while (0)

int main() {
    // Row-major A is 2 by 3; B is 3 by 2; C is 2 by 2.
    const std::array<int8_t, 6> host_a{1, 2, 3, 4, 5, 6};
    const std::array<int8_t, 6> host_b{7, 8, 9, 10, 11, 12};
    const std::array<int64_t, 4> expected{58, 64, 139, 154};
    std::array<int64_t, 4> host_c{};
    awe_gemm_desc desc;
    awe_gemm_desc_init(&desc);
    desc.layout = AWE_ROW_MAJOR;
    desc.m = 2; desc.n = 2; desc.k = 3;
    desc.lda = 3; desc.ldb = 2; desc.ldc = 2;
    awe_options options;
    awe_options_init(&options);
    options.method = AWE_METHOD_AWE;
    awe::plan plan;
    AWE_CHECK(plan.create(desc, &options));
    size_t bytes = 0;
    AWE_CHECK(plan.workspace_size(bytes));
    device_buffer a, b, c, workspace;
    CUDA_CHECK(a.allocate(sizeof(host_a)));
    CUDA_CHECK(b.allocate(sizeof(host_b)));
    CUDA_CHECK(c.allocate(sizeof(host_c)));
    CUDA_CHECK(workspace.allocate(bytes));
    CUDA_CHECK(cudaMemcpy(a.pointer, host_a.data(), sizeof(host_a), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b.pointer, host_b.data(), sizeof(host_b), cudaMemcpyHostToDevice));
    AWE_CHECK(plan.execute(a.pointer, b.pointer, c.pointer, workspace.pointer, bytes));
    CUDA_CHECK(cudaMemcpy(host_c.data(), c.pointer, sizeof(host_c), cudaMemcpyDeviceToHost));
    if (host_c != expected) { std::fprintf(stderr, "Output mismatch\n"); return 1; }
    std::puts("C++ example PASS: [58, 64; 139, 154]");
    return 0;
}
