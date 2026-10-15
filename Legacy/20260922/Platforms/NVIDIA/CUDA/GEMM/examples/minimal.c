#include <awe_blas.h>
#include <inttypes.h>
#include <stdio.h>

#define CUDA_CHECK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
    fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(e)); goto cleanup; } } while (0)
#define AWE_CHECK(call) do { awe_status s = (call); if (s != AWE_SUCCESS) { \
    fprintf(stderr, "%s: %s\n", #call, awe_status_string(s)); goto cleanup; } } while (0)

int main(void) {
    /* Column-major A is 2 by 3; B is 3 by 2; C is 2 by 2. */
    const int8_t host_a[] = {1, 4, 2, 5, 3, 6};
    const int8_t host_b[] = {7, 9, 11, 8, 10, 12};
    const int64_t expected[] = {58, 139, 64, 154};
    int64_t host_c[4] = {0};
    int8_t *a = NULL, *b = NULL;
    int64_t *c = NULL;
    void *workspace = NULL;
    size_t bytes = 0;
    int result = 1;
    awe_plan *plan = NULL;
    awe_gemm_desc desc;
    awe_options options;
    awe_plan_info info = {0};
    awe_execution_stats stats = {0};
    awe_gemm_desc_init(&desc);
    desc.m = 2; desc.n = 2; desc.k = 3;
    desc.lda = 2; desc.ldb = 3; desc.ldc = 2;
    awe_options_init(&options);
    options.method = AWE_METHOD_AWE;
    options.profiling = 1;
    info.struct_size = sizeof(info);
    stats.struct_size = sizeof(stats);
    AWE_CHECK(awe_plan_create(&plan, &desc, &options));
    AWE_CHECK(awe_plan_get_workspace_size(plan, &bytes));
    CUDA_CHECK(cudaMalloc((void **)&a, sizeof(host_a)));
    CUDA_CHECK(cudaMalloc((void **)&b, sizeof(host_b)));
    CUDA_CHECK(cudaMalloc((void **)&c, sizeof(host_c)));
    if (bytes) CUDA_CHECK(cudaMalloc(&workspace, bytes));
    CUDA_CHECK(cudaMemcpy(a, host_a, sizeof(host_a), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b, host_b, sizeof(host_b), cudaMemcpyHostToDevice));
    AWE_CHECK(awe_gemm_i8(plan, a, b, c, workspace, bytes, 0, &stats));
    CUDA_CHECK(cudaMemcpy(host_c, c, sizeof(host_c), cudaMemcpyDeviceToHost));
    AWE_CHECK(awe_plan_get_info(plan, &info));
    for (int i = 0; i < 4; ++i) {
        if (host_c[i] != expected[i]) { fprintf(stderr, "Mismatch at %d\n", i); goto cleanup; }
    }
    printf("C example PASS: [%" PRId64 ", %" PRId64 "; %" PRId64 ", %" PRId64 "]\n",
           host_c[0], host_c[2], host_c[1], host_c[3]);
    printf("input range [%d,%d], guaranteed K %d, payout period %d, total %.6f ms\n",
           info.input_min, info.input_max, info.guaranteed_k, info.payout_period, stats.total_ms);
    result = 0;
cleanup:
    awe_plan_destroy(plan);
    if (workspace) cudaFree(workspace);
    if (c) cudaFree(c);
    if (b) cudaFree(b);
    if (a) cudaFree(a);
    return result;
}
