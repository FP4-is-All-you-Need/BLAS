#include "awe_blas.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <vector>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
void cu(cudaError_t e){if(e)throw std::runtime_error(cudaGetErrorString(e));}
void aw(awe_status s){if(s)throw std::runtime_error(awe_status_string(s));}
void bl(cublasStatus_t s){if(s)throw std::runtime_error("cuBLAS reference failed");}
struct Buf{void* p=nullptr;explicit Buf(size_t n){if(n)cu(cudaMalloc(&p,n));}~Buf(){if(p)cudaFree(p);}template<class T>T* as(){return static_cast<T*>(p);}};
__device__ unsigned mix(unsigned x){x^=x>>16;x*=0x7feb352dU;x^=x>>15;x*=0x846ca68bU;x^=x>>16;return x;}
__global__ void fill(int8_t* p,size_t count,unsigned seed,int distribution){for(size_t z=size_t(blockIdx.x)*blockDim.x+threadIdx.x;z<count;z+=size_t(blockDim.x)*gridDim.x){unsigned h=mix(unsigned(z)^seed);int v=int(h%255)-127;if(distribution==1){int sum=0;for(int j=0;j<12;++j){h=mix(h+0x9e3779b9U);sum+=h&255;}v=__float2int_rn((sum-1530)*0.61233f);v=max(-127,min(127,v));}p[z]=v;}}
__global__ void compare(const int64_t* a,const int32_t* b,size_t count,unsigned long long* bad){for(size_t z=size_t(blockIdx.x)*blockDim.x+threadIdx.x;z<count;z+=size_t(blockDim.x)*gridDim.x)if(a[z]!=b[z])atomicAdd(bad,1ull);}
int main(int argc,char**argv){try{
 int n=argc>1?atoi(argv[1]):8192,k=argc>2?atoi(argv[2]):16384,distribution=argc>3?atoi(argv[3]):0,reps=5;if(n<128||n%128||k<128||k%128||k>131071)return 2;
 cudaDeviceProp prop;cu(cudaGetDeviceProperties(&prop,0));printf("{\"kind\":\"environment\",\"gpu\":\"%s\",\"cublasLt\":%zu,\"distribution\":\"%s\",\"seed_A\":71983,\"seed_B\":91923,\"n\":%d,\"k\":%d}\n",prop.name,cublasLtGetVersion(),distribution?"12 discrete uniform summands, clipped approximate normal":"hash uniform -127..127",n,k);
 size_t input=size_t(n)*k,count=size_t(n)*n;Buf a(input),b(input),c(count*8),ref(count*4),bad(8),refwork(32ul<<20);fill<<<4096,256>>>(a.as<int8_t>(),input,71983,distribution);fill<<<4096,256>>>(b.as<int8_t>(),input,91923,distribution);cu(cudaDeviceSynchronize());
 cublasHandle_t h;bl(cublasCreate(&h));bl(cublasSetWorkspace(h,refwork.p,32ul<<20));int one=1,zero=0;
 auto native=[&](){bl(cublasGemmEx(h,CUBLAS_OP_N,CUBLAS_OP_N,n,n,k,&one,a.p,CUDA_R_8I,n,b.p,CUDA_R_8I,k,&zero,ref.p,CUDA_R_32I,n,CUBLAS_COMPUTE_32I,CUBLAS_GEMM_DEFAULT_TENSOR_OP));};native();cu(cudaDeviceSynchronize());cudaEvent_t begin,end;cu(cudaEventCreate(&begin));cu(cudaEventCreate(&end));std::vector<double> native_times;for(int r=0;r<reps;++r){cu(cudaEventRecord(begin));native();cu(cudaEventRecord(end));cu(cudaEventSynchronize(end));float x;cu(cudaEventElapsedTime(&x,begin,end));native_times.push_back(x);}std::sort(native_times.begin(),native_times.end());printf("{\"kind\":\"native_int8_reference\",\"ms\":%.6f,\"integer_bound\":%lld}\n",native_times[reps/2],16384ll*k);
 for(auto method:{AWE_METHOD_AWE,AWE_METHOD_RADIX13})for(int streams:{1,3})for(int profiling:{0,1}){
  awe_gemm_desc d;awe_gemm_desc_init(&d);d.m=n;d.n=n;d.k=k;d.lda=n;d.ldb=k;d.ldc=n;awe_options o;awe_options_init(&o);o.method=method;o.stream_count=streams;o.profiling=profiling;
  awe_plan* p=nullptr;aw(awe_plan_create(&p,&d,&o));size_t bytes;aw(awe_plan_get_workspace_size(p,&bytes));Buf work(bytes);awe_execution_stats st{};st.struct_size=sizeof(st);aw(awe_gemm_i8(p,a.as<int8_t>(),b.as<int8_t>(),c.as<int64_t>(),work.p,bytes,nullptr,&st));
  std::vector<double> times;for(int r=0;r<(profiling?1:reps);++r){aw(awe_gemm_i8(p,a.as<int8_t>(),b.as<int8_t>(),c.as<int64_t>(),work.p,bytes,nullptr,&st));times.push_back(st.total_ms);cu(cudaMemset(bad.p,0,8));compare<<<4096,256>>>(c.as<int64_t>(),ref.as<int32_t>(),count,bad.as<unsigned long long>());unsigned long long failures;cu(cudaMemcpy(&failures,bad.p,8,cudaMemcpyDeviceToHost));if(failures)throw std::runtime_error("all-output INT8 reference mismatch");}
  std::sort(times.begin(),times.end());printf("{\"kind\":\"awe_benchmark\",\"n\":%d,\"k\":%d,\"distribution\":%d,\"method\":%d,\"streams\":%d,\"profiling\":%d,\"faces\":%d,\"median_ms\":%.6f,\"selection_ms\":%.6f,\"encode_ms\":%.6f,\"gemm_ms\":%.6f,\"reconstruction_ms\":%.6f,\"wait_ms\":%.6f,\"outputs\":%zu,\"mismatch\":0}\n",n,k,distribution,int(method),streams,profiling,st.face_count,times[times.size()/2],st.selection_ms,st.encode_ms,st.gemm_ms,st.reconstruction_ms,st.wait_ms,count);fflush(stdout);awe_plan_destroy(p);
 }
 cudaEventDestroy(begin);cudaEventDestroy(end);cublasDestroy(h);return 0;
 }catch(std::exception const&e){fprintf(stderr,"%s\n",e.what());return 1;}}
