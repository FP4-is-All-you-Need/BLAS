#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>

extern "C" float emulated_zgemm(void* a,void* b,void* c,int n,int reps,int* version){
 cublasHandle_t handle=nullptr;void* workspace=nullptr;cudaEvent_t first=nullptr,last=nullptr;
 float result=-1;cudaError_t cu_status=cudaSuccess;cublasStatus_t bl_status=CUBLAS_STATUS_SUCCESS;
#define CUBLAS_TRY(expr) do{bl_status=(expr);if(bl_status!=CUBLAS_STATUS_SUCCESS){std::fprintf(stderr,"%s status=%d\n",#expr,int(bl_status));goto done;}}while(0)
#define CUDA_TRY(expr) do{cu_status=(expr);if(cu_status!=cudaSuccess){std::fprintf(stderr,"%s: %s\n",#expr,cudaGetErrorString(cu_status));goto done;}}while(0)
 {
 const size_t bytes=size_t(4)<<30;
 cuDoubleComplex one=make_cuDoubleComplex(1,0),zero=make_cuDoubleComplex(0,0);
 CUBLAS_TRY(cublasCreate(&handle));CUBLAS_TRY(cublasGetVersion(handle,version));
 CUDA_TRY(cudaMalloc(&workspace,bytes));CUBLAS_TRY(cublasSetWorkspace(handle,workspace,bytes));
 CUBLAS_TRY(cublasSetMathMode(handle,CUBLAS_FP64_EMULATED_FIXEDPOINT_MATH));
 CUBLAS_TRY(cublasSetEmulationStrategy(handle,CUBLAS_EMULATION_STRATEGY_EAGER));
 CUBLAS_TRY(cublasSetFixedPointEmulationMantissaControl(handle,CUDA_EMULATION_MANTISSA_CONTROL_DYNAMIC));
 CUBLAS_TRY(cublasZgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,n,n,n,&one,(cuDoubleComplex*)a,n,(cuDoubleComplex*)b,n,&zero,(cuDoubleComplex*)c,n));
 CUDA_TRY(cudaDeviceSynchronize());CUDA_TRY(cudaEventCreate(&first));CUDA_TRY(cudaEventCreate(&last));
 CUDA_TRY(cudaEventRecord(first));
 for(int r=0;r<reps;++r)CUBLAS_TRY(cublasZgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,n,n,n,&one,(cuDoubleComplex*)a,n,(cuDoubleComplex*)b,n,&zero,(cuDoubleComplex*)c,n));
 CUDA_TRY(cudaEventRecord(last));CUDA_TRY(cudaEventSynchronize(last));CUDA_TRY(cudaEventElapsedTime(&result,first,last));result/=reps;
 }
done:
 if(first)cudaEventDestroy(first);if(last)cudaEventDestroy(last);if(workspace)cudaFree(workspace);if(handle)cublasDestroy(handle);
 return result;
}
