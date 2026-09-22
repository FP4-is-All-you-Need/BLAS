#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>
#include "awe_blas.h"
#include "awe_backend.h"
#define CU(x) do{auto e_=(x);if(e_!=cudaSuccess){printf("CUDA FAIL %s: %s\n",#x,cudaGetErrorString(e_));exit(1);}}while(0)

__global__ void fill(double* x,size_t n,unsigned long long seed,int mode){
 for(size_t z=size_t(blockIdx.x)*blockDim.x+threadIdx.x;z<n;z+=size_t(blockDim.x)*gridDim.x){
  unsigned long long h=z*0x9e3779b97f4a7c15ULL+seed;
  h^=h>>33;h*=0xff51afd7ed558ccdULL;h^=h>>33;h*=0xc4ceb9fe1a85ec53ULL;h^=h>>33;
  double u=double(h>>11)*(1.0/9007199254740992.0);
  if(mode==0) x[z]=2.0*u-1.0;
  else { unsigned long long g=h*0x2545f4914f6cdd1dULL;
         x[z]=(2.0*u-1.0)*ldexp(1.0,int(g%40)-20); }
 }
}
int main(int argc,char** argv){
 int n=argc>1?atoi(argv[1]):8192;
 int mode=argc>2?atoi(argv[2]):0;
 int bits=argc>3?atoi(argv[3]):53;
 int r1=argc>4?atoi(argv[4]):1;
 char b[16];
 snprintf(b,sizeof(b),"%d",bits);setenv("AWE_FP64_BITS",b,1);
 snprintf(b,sizeof(b),"%d",r1);setenv("AWE_F64_R1",b,1);
 setenv("AWE_F64_GROUP","21",1);
 awe_gemm_desc d{};d.struct_size=sizeof(d);d.m=d.n=d.k=n;d.lda=d.ldb=d.ldc=n;
 d.trans_a=AWE_OP_N;d.trans_b=AWE_OP_N;d.data_type=AWE_F64_F64;d.layout=AWE_COLUMN_MAJOR;
 d.alpha=1;d.beta=0;
 awe_options o{};o.struct_size=sizeof(o);o.method=AWE_METHOD_OZAKI2;o.stream_count=1;
 o.buffer_count=3;o.k_chunk=0;o.profiling=0;o.heuristic_index=0;o.matmul_workspace_bytes=64u<<20;
 void* st=nullptr;
 if(awe_ozaki2_create(&st,&d,&o)!=AWE_SUCCESS){printf("{\"error\":\"create failed\",\"bits\":%d}\n",bits);return 2;}
 awe_plan_info info{};info.struct_size=sizeof(info);awe_ozaki2_info(st,&info);
 double *A,*B,*C;void* W=nullptr;
 CU(cudaMalloc(&A,size_t(n)*n*8));CU(cudaMalloc(&B,size_t(n)*n*8));CU(cudaMalloc(&C,size_t(n)*n*8));
 if(info.workspace_bytes)CU(cudaMalloc(&W,info.workspace_bytes));
 fill<<<256,256>>>(A,size_t(n)*n,1111+mode,mode);
 fill<<<256,256>>>(B,size_t(n)*n,2222+mode,mode);
 CU(cudaDeviceSynchronize());
 awe_execution_stats s{};s.struct_size=sizeof(s);
 if(awe_ozaki2_execute(st,A,B,C,W,info.workspace_bytes,0,&s)!=AWE_SUCCESS){printf("{\"error\":\"execute\"}\n");return 2;}
 CU(cudaDeviceSynchronize());

 std::vector<int> ri,ci{0,n/2,n-1};
 for(int t=0;t<11;++t) ri.push_back(int((long long)t*(n-1)/10));
 char name[128];snprintf(name,sizeof(name),"acc_n%d_mode%d_bits%d_r1%d.bin",n,mode,bits,r1);
 FILE* f=fopen(name,"wb");
 int nr=int(ri.size()),nc=int(ci.size());
 fwrite(&n,4,1,f);fwrite(&nr,4,1,f);fwrite(&nc,4,1,f);fwrite(&bits,4,1,f);
 fwrite(ri.data(),4,nr,f);fwrite(ci.data(),4,nc,f);
 std::vector<double> buf(n);
 for(int t=0;t<nr;++t){
  CU(cudaMemcpy2D(buf.data(),8,A+ri[t],size_t(n)*8,8,n,cudaMemcpyDeviceToHost));
  fwrite(buf.data(),8,n,f);
 }
 for(int t=0;t<nc;++t){
  CU(cudaMemcpy(buf.data(),B+size_t(ci[t])*n,size_t(n)*8,cudaMemcpyDeviceToHost));
  fwrite(buf.data(),8,n,f);
 }
 std::vector<double> got(size_t(nr)*nc);
 for(int a=0;a<nr;++a)for(int c2=0;c2<nc;++c2)
  CU(cudaMemcpy(&got[size_t(a)*nc+c2],C+size_t(ci[c2])*n+ri[a],8,cudaMemcpyDeviceToHost));
 fwrite(got.data(),8,got.size(),f);
 fclose(f);
 printf("{\"kind\":\"dump\",\"file\":\"%s\",\"n\":%d,\"mode\":%d,\"bits\":%d,\"r1\":%d,"
        "\"faces\":%d,\"positions\":%d,\"total_ms\":%.6f}\n",name,n,mode,bits,r1,info.face_count,nr*nc,s.total_ms);
 cudaFree(A);cudaFree(B);cudaFree(C);if(W)cudaFree(W);awe_ozaki2_destroy(st);
 return 0;
}
