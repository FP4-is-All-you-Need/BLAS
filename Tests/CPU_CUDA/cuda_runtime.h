#pragma once
// CPU-only stand-in used exclusively by test_api.py; no CUDA module is loaded.
#include <cstdlib>
#include <cstring>
struct dim3 { unsigned x,y,z; dim3(unsigned a=1,unsigned b=1,unsigned c=1):x(a),y(b),z(c){} };
struct double2 {double x,y;};
using cudaStream_t=void*; using cudaEvent_t=void*; using cudaError_t=int;
constexpr int cudaSuccess=0,cudaStreamNonBlocking=1,cudaMemoryTypeDevice=2;
struct cudaDeviceProp { int major=10,minor=3; };
struct cudaPointerAttributes { int type=2,device=0; };
extern int fake_device, fake_minor;
inline int cudaGetDevice(int *d){*d=fake_device;return 0;}
inline int cudaGetDeviceProperties(cudaDeviceProp *p,int){p->minor=fake_minor;return 0;}
inline const char *cudaGetErrorString(int){return "CPU stub failure";}
inline int cudaGetLastError(){return 0;}
inline int cudaPointerGetAttributes(cudaPointerAttributes *p,const void*){p->device=fake_device;return 0;}
inline int cudaMalloc(void **p,size_t n){*p=std::malloc(n);return *p ? 0 : 1;}
inline int cudaFree(void *p){std::free(p);return 0;}
inline int cudaStreamCreateWithFlags(void **p,int){*p=std::malloc(1);return 0;}
inline int cudaStreamDestroy(void *p){std::free(p);return 0;}
inline int cudaStreamSynchronize(void*){return 0;}
inline int cudaEventCreate(void **p){*p=std::malloc(1);return 0;}
inline int cudaEventDestroy(void *p){std::free(p);return 0;}
inline int cudaEventRecord(void*,void*){return 0;}
inline int cudaEventSynchronize(void*){return 0;}

constexpr int cudaMemcpyDeviceToDevice=3;
inline int cudaMemcpyAsync(void *dst,const void *src,size_t n,int,void*){std::memcpy(dst,src,n);return 0;}
