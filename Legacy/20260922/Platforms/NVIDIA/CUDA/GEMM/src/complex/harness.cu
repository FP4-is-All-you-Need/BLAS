#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <string>

static std::string harness_error;
extern "C" const char* harness_last_error(){return harness_error.c_str();}
#define CK(expr) do{cudaError_t s_=(expr);if(s_!=cudaSuccess){harness_error=cudaGetErrorString(s_);return 1;}}while(0)

static __host__ __device__ unsigned long long splitmix64(unsigned long long x){
 x += 0x9E3779B97F4A7C15ULL;
 unsigned long long z = x;
 z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
 z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
 return z ^ (z >> 31);
}

__global__ void fill_kernel(double2* p, long long count, unsigned long long seed, int shift){
 long long stride = (long long)blockDim.x * gridDim.x;
 double scale = ldexp(1.0, -shift);
 unsigned long long bound = 1ULL << shift;
 unsigned long long span = 2ULL * bound + 1ULL;
 for(long long z = (long long)blockIdx.x * blockDim.x + threadIdx.x; z < count; z += stride){
  unsigned long long h0 = splitmix64(seed ^ (unsigned long long)(2*z));
  unsigned long long h1 = splitmix64(seed ^ (unsigned long long)(2*z+1));
  long long re = (long long)(h0 % span) - (long long)bound;
  long long im = (long long)(h1 % span) - (long long)bound;
  p[z].x = (double)re * scale;
  p[z].y = (double)im * scale;
 }
}

__global__ void mismatch_kernel(const long long* a, const long long* b, long long words, unsigned long long* bad){
 long long stride = (long long)blockDim.x * gridDim.x;
 for(long long z = (long long)blockIdx.x * blockDim.x + threadIdx.x; z < words; z += stride)
  if(a[z] != b[z]) atomicAdd(bad, 1ULL);
}

__global__ void maxabs_kernel(const double* a, const double* b, long long count, unsigned long long* best){
 long long stride = (long long)blockDim.x * gridDim.x;
 for(long long z = (long long)blockIdx.x * blockDim.x + threadIdx.x; z < count; z += stride){
  double d = fabs(a[z] - b[z]);
  if(!(d >= 0.0)) d = INFINITY;
  unsigned long long bits = __double_as_longlong(d);
  atomicMax(best, bits);
 }
}

__global__ void d2f_kernel(const double* src, float* dst, long long count){
 long long stride = (long long)blockDim.x * gridDim.x;
 for(long long z = (long long)blockIdx.x * blockDim.x + threadIdx.x; z < count; z += stride) dst[z] = (float)src[z];
}

__global__ void f2d_kernel(const float* src, double* dst, long long count){
 long long stride = (long long)blockDim.x * gridDim.x;
 for(long long z = (long long)blockIdx.x * blockDim.x + threadIdx.x; z < count; z += stride) dst[z] = (double)src[z];
}

static int grid(long long n){ long long g = (n + 255) / 256; return (int)(g > 65535 ? 65535 : (g < 1 ? 1 : g)); }

extern "C" void* h_alloc(size_t bytes){ void* p=nullptr; cudaError_t s=cudaMalloc(&p,bytes); if(s!=cudaSuccess){harness_error=cudaGetErrorString(s);return nullptr;} return p; }
extern "C" void  h_free(void* p){ if(p) cudaFree(p); }
extern "C" int   h_sync(){ CK(cudaDeviceSynchronize()); return 0; }
extern "C" int   h_memset(void* p, int value, size_t bytes){ CK(cudaMemset(p,value,bytes)); return 0; }

extern "C" int h_fill(void* p, long long complex_count, unsigned long long seed, int shift){
 fill_kernel<<<grid(complex_count),256>>>((double2*)p, complex_count, seed, shift);
 CK(cudaGetLastError()); CK(cudaDeviceSynchronize()); return 0;
}

extern "C" int h_copy_out(const void* device, void* host, size_t bytes){
 CK(cudaMemcpy(host, device, bytes, cudaMemcpyDeviceToHost)); return 0;
}

extern "C" long long h_mismatch(const void* a, const void* b, long long words){
 unsigned long long* bad=nullptr;
 if(cudaMalloc(&bad,8)!=cudaSuccess){harness_error="mismatch counter allocation";return -1;}
 if(cudaMemset(bad,0,8)!=cudaSuccess){cudaFree(bad);harness_error="mismatch counter reset";return -1;}
 mismatch_kernel<<<grid(words),256>>>((const long long*)a,(const long long*)b,words,bad);
 unsigned long long value=0;
 if(cudaDeviceSynchronize()!=cudaSuccess||cudaMemcpy(&value,bad,8,cudaMemcpyDeviceToHost)!=cudaSuccess){cudaFree(bad);harness_error="mismatch readback";return -1;}
 cudaFree(bad); return (long long)value;
}

extern "C" double h_max_abs_diff(const void* a, const void* b, long long count){
 unsigned long long* best=nullptr;
 if(cudaMalloc(&best,8)!=cudaSuccess){harness_error="max allocation";return -1.0;}
 if(cudaMemset(best,0,8)!=cudaSuccess){cudaFree(best);harness_error="max reset";return -1.0;}
 maxabs_kernel<<<grid(count),256>>>((const double*)a,(const double*)b,count,best);
 unsigned long long bits=0;
 if(cudaDeviceSynchronize()!=cudaSuccess||cudaMemcpy(&bits,best,8,cudaMemcpyDeviceToHost)!=cudaSuccess){cudaFree(best);harness_error="max readback";return -1.0;}
 cudaFree(best);
 double value; memcpy(&value,&bits,8); return value;
}

extern "C" int h_d2f(const void* src, void* dst, long long count){
 d2f_kernel<<<grid(count),256>>>((const double*)src,(float*)dst,count);
 CK(cudaGetLastError()); CK(cudaDeviceSynchronize()); return 0;
}

extern "C" int h_f2d(const void* src, void* dst, long long count){
 f2d_kernel<<<grid(count),256>>>((const float*)src,(double*)dst,count);
 CK(cudaGetLastError()); CK(cudaDeviceSynchronize()); return 0;
}

extern "C" int h_device_info(char* name, int length, int* major, int* minor, size_t* total_bytes){
 cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop,0));
 snprintf(name,length,"%s",prop.name); *major=prop.major; *minor=prop.minor; *total_bytes=prop.totalGlobalMem; return 0;
}
