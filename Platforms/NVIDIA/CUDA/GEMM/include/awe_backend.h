#ifndef AWE_BACKEND_H
#define AWE_BACKEND_H
#include "awe_blas.h"
#ifdef __cplusplus
extern "C" {
#endif
/* Backend extension boundary; implemented by src/ozaki2_backend.cu.
 * The core dispatches AWE_METHOD_OZAKI2 here, with the public descriptor,
 * options, caller-owned workspace, and synchronous execution semantics. */
awe_status awe_ozaki2_create(void** state,const awe_gemm_desc*,const awe_options*);
void awe_ozaki2_destroy(void* state);
awe_status awe_ozaki2_info(const void* state,awe_plan_info*);
awe_status awe_ozaki2_execute(void* state,const void*,const void*,void*,void*,size_t,
 cudaStream_t,awe_execution_stats*);
#ifdef __cplusplus
}
#endif
#endif
