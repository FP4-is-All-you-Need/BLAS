#define main standalone_benchmark_main
#include "bench_i8.cu"
#undef main
#include <gemmul8.hpp>
#include <cuda_fp8.h>
__global__ void as_double(const int8_t* x,double* y,size_t count){for(size_t z=size_t(blockIdx.x)*blockDim.x+threadIdx.x;z<count;z+=size_t(blockDim.x)*gridDim.x)y[z]=x[z];}
__global__ void as_fp8(const int8_t* a,const int8_t* b,__nv_fp8_e4m3* a8,__nv_fp8_e4m3* b8,int n,int k){for(size_t z=size_t(blockIdx.x)*blockDim.x+threadIdx.x;z<size_t(n)*k;z+=size_t(blockDim.x)*gridDim.x){int i=z/k,q=z%k;a8[z]=__nv_fp8_e4m3(float(a[i+size_t(q)*n]));b8[z]=__nv_fp8_e4m3(float(b[z]));}}
template<class T> __global__ void compare_typed(const T* a,const int* b,size_t count,unsigned long long* bad){for(size_t z=size_t(blockIdx.x)*blockDim.x+threadIdx.x;z<count;z+=size_t(blockDim.x)*gridDim.x)if(double(a[z])!=double(b[z]))atomicAdd(bad,1ull);}
int main(int argc,char**argv){try{
 int n=argc>1?atoi(argv[1]):8192,k=argc>2?atoi(argv[2]):16384,distribution=argc>3?atoi(argv[3]):0;if(n<128||n%128||k<128||k%128||k>131071)return 2;
 size_t input=size_t(n)*k,count=size_t(n)*n;Buf a(input),b(input),ref(count*4),a64(input*8),b64(input*8),out(count*8),bad(8);fill<<<4096,256>>>(a.as<int8_t>(),input,71983,distribution);fill<<<4096,256>>>(b.as<int8_t>(),input,91923,distribution);cu(cudaDeviceSynchronize());
 cublasHandle_t h;bl(cublasCreate(&h));int one_i=1,zero_i=0;bl(cublasGemmEx(h,CUBLAS_OP_N,CUBLAS_OP_N,n,n,k,&one_i,a.p,CUDA_R_8I,n,b.p,CUDA_R_8I,k,&zero_i,ref.p,CUDA_R_32I,n,CUBLAS_COMPUTE_32I,CUBLAS_GEMM_DEFAULT_TENSOR_OP));cu(cudaDeviceSynchronize());
 cudaEvent_t begin,end;cu(cudaEventCreate(&begin));cu(cudaEventCreate(&end));
 printf("{\"kind\":\"comparison_environment\",\"n\":%d,\"k\":%d,\"distribution\":%d,\"seed_A\":71983,\"seed_B\":91923,\"gemmul8\":\"%s\",\"commit\":\"91315445ea07294e71b4671861b20f42cbf06f12\",\"input_conversion_timed\":true}\n",n,k,distribution,GEMMUL8_VERSION_STRING);
 double one=1,zero=0;
 for(int mods:{2,3,6,9})for(bool fast:{false,true}){
  auto size=gemmul8::gemm<double>(h,CUBLAS_OP_N,CUBLAS_OP_N,n,n,k,&one,a64.as<double>(),n,b64.as<double>(),k,&zero,out.as<double>(),n,mods,fast,nullptr);Buf work{size_t(size[0])};
  auto run=[&](){as_double<<<4096,256>>>(a.as<int8_t>(),a64.as<double>(),input);as_double<<<4096,256>>>(b.as<int8_t>(),b64.as<double>(),input);gemmul8::gemm<double>(h,CUBLAS_OP_N,CUBLAS_OP_N,n,n,k,&one,a64.as<double>(),n,b64.as<double>(),k,&zero,out.as<double>(),n,mods,fast,work.p);};
  run();cu(cudaDeviceSynchronize());std::vector<double> times;for(int r=0;r<5;++r){cu(cudaEventRecord(begin));run();cu(cudaEventRecord(end));cu(cudaEventSynchronize(end));float t;cu(cudaEventElapsedTime(&t,begin,end));times.push_back(t);}std::sort(times.begin(),times.end());cu(cudaMemset(bad.p,0,8));compare_typed<<<4096,256>>>(out.as<double>(),ref.as<int>(),count,bad.as<unsigned long long>());unsigned long long errors;cu(cudaMemcpy(&errors,bad.p,8,cudaMemcpyDeviceToHost));printf("{\"kind\":\"gemmul8\",\"moduli\":%d,\"fast\":%d,\"median_ms\":%.6f,\"outputs\":%zu,\"mismatch\":%llu}\n",mods,fast,times[2],count,errors);fflush(stdout);
 }

 {awe_gemm_desc d;awe_gemm_desc_init(&d);d.m=n;d.n=n;d.k=k;d.lda=n;d.ldb=k;d.ldc=n;awe_options o;awe_options_init(&o);o.method=AWE_METHOD_AWE;awe_plan* p=nullptr;aw(awe_plan_create(&p,&d,&o));size_t bytes;aw(awe_plan_get_workspace_size(p,&bytes));Buf work_awe(bytes);
  auto sizes=gemmul8::gemm<double>(h,CUBLAS_OP_N,CUBLAS_OP_N,n,n,k,&one,a64.as<double>(),n,b64.as<double>(),k,&zero,out.as<double>(),n,6,true,nullptr);Buf work_gem{size_t(sizes[0])};
  auto gem=[&](){as_double<<<4096,256>>>(a.as<int8_t>(),a64.as<double>(),input);as_double<<<4096,256>>>(b.as<int8_t>(),b64.as<double>(),input);gemmul8::gemm<double>(h,CUBLAS_OP_N,CUBLAS_OP_N,n,n,k,&one,a64.as<double>(),n,b64.as<double>(),k,&zero,out.as<double>(),n,6,true,work_gem.p);};
  auto awe=[&](){aw(awe_gemm_i8(p,a.as<int8_t>(),b.as<int8_t>(),out.as<int64_t>(),work_awe.p,bytes,nullptr,nullptr));};
  awe();gem();cu(cudaDeviceSynchronize());std::vector<double> at,gt;
  for(int r=0;r<7;++r)for(int j=0;j<2;++j){bool is_awe=(r+j)%2;cu(cudaEventRecord(begin));if(is_awe)awe();else gem();cu(cudaEventRecord(end));cu(cudaEventSynchronize(end));float t;cu(cudaEventElapsedTime(&t,begin,end));(is_awe?at:gt).push_back(t);cu(cudaMemset(bad.p,0,8));if(is_awe)compare_typed<<<4096,256>>>(out.as<int64_t>(),ref.as<int>(),count,bad.as<unsigned long long>());else compare_typed<<<4096,256>>>(out.as<double>(),ref.as<int>(),count,bad.as<unsigned long long>());unsigned long long errors;cu(cudaMemcpy(&errors,bad.p,8,cudaMemcpyDeviceToHost));if(errors)throw std::runtime_error("paired all-output mismatch");}
  std::sort(at.begin(),at.end());std::sort(gt.begin(),gt.end());printf("{\"kind\":\"paired_same_process\",\"awe_ms\":%.6f,\"gemmul8_ms\":%.6f,\"gemmul8_over_awe\":%.6f,\"moduli\":6,\"fast\":true,\"repetitions\":7,\"outputs\":%zu,\"mismatch_each_repetition\":0}\n",at[3],gt[3],gt[3]/at[3],count);awe_plan_destroy(p);
 }
 {Buf a8(input),b8(input),fout(count*4),scratch(64ul<<20),scale(4);float fone=1,fzero=0;cu(cudaMemcpy(scale.p,&fone,4,cudaMemcpyHostToDevice));
 cublasLtHandle_t handle;cublasLtMatmulDesc_t op;cublasLtMatrixLayout_t la,lb,lc;cublasLtMatmulPreference_t pref;bl(cublasLtCreate(&handle));bl(cublasLtMatmulDescCreate(&op,CUBLAS_COMPUTE_32F,CUDA_R_32F));cublasOperation_t ta=CUBLAS_OP_T;
 bl(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSA,&ta,sizeof(ta)));bl(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,&scale.p,sizeof(void*)));bl(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,&scale.p,sizeof(void*)));
 bl(cublasLtMatrixLayoutCreate(&la,CUDA_R_8F_E4M3,k,n,k));bl(cublasLtMatrixLayoutCreate(&lb,CUDA_R_8F_E4M3,k,n,k));bl(cublasLtMatrixLayoutCreate(&lc,CUDA_R_32F,n,n,n));bl(cublasLtMatmulPreferenceCreate(&pref));size_t scratch_bytes=64ul<<20;bl(cublasLtMatmulPreferenceSetAttribute(pref,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&scratch_bytes,sizeof(scratch_bytes)));cublasLtMatmulHeuristicResult_t algo{};int returned=0;auto status=cublasLtMatmulAlgoGetHeuristic(handle,op,la,lb,lc,lc,pref,1,&algo,&returned);
 if(status==CUBLAS_STATUS_SUCCESS&&returned&&algo.state==CUBLAS_STATUS_SUCCESS){auto run=[&](){as_fp8<<<4096,256>>>(a.as<int8_t>(),b.as<int8_t>(),a8.as<__nv_fp8_e4m3>(),b8.as<__nv_fp8_e4m3>(),n,k);bl(cublasLtMatmul(handle,op,&fone,a8.p,la,b8.p,lb,&fzero,fout.p,lc,fout.p,lc,&algo.algo,scratch.p,scratch_bytes,nullptr));};run();cu(cudaDeviceSynchronize());std::vector<double> times;for(int r=0;r<5;++r){cu(cudaEventRecord(begin));run();cu(cudaEventRecord(end));cu(cudaEventSynchronize(end));float t;cu(cudaEventElapsedTime(&t,begin,end));times.push_back(t);}std::sort(times.begin(),times.end());cu(cudaMemset(bad.p,0,8));compare_typed<<<4096,256>>>(fout.as<float>(),ref.as<int>(),count,bad.as<unsigned long long>());unsigned long long errors;cu(cudaMemcpy(&errors,bad.p,8,cudaMemcpyDeviceToHost));printf("{\"kind\":\"cublaslt_fp8_approximate\",\"quantization_included\":true,\"median_ms\":%.6f,\"outputs\":%zu,\"mismatch\":%llu}\n",times[2],count,errors);}else printf("{\"kind\":\"cublaslt_fp8_approximate\",\"status\":\"not_supported\"}\n");
 cublasLtMatmulPreferenceDestroy(pref);cublasLtMatrixLayoutDestroy(la);cublasLtMatrixLayoutDestroy(lb);cublasLtMatrixLayoutDestroy(lc);cublasLtMatmulDescDestroy(op);cublasLtDestroy(handle);
 }
 cublasDestroy(h);cudaEventDestroy(begin);cudaEventDestroy(end);return 0;
 }catch(std::exception const& e){fprintf(stderr,"%s\n",e.what());return 1;}}
