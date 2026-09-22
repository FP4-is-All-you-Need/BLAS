#include "awe_blas.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <cmath>
#include <cstdlib>
#define CU(x) do{auto r=(x);if(r){fprintf(stderr,"CUDA %d line%d\n",int(r),__LINE__);return 1;}}while(0)
#define AW(x) do{auto r=(x);if(r){fprintf(stderr,"AWE %d line%d\n",int(r),__LINE__);return 1;}}while(0)
int main(){int cases=0;size_t checked=0;for(int bits:{24,53})for(int streams:{1,3})for(int row:{0,1})for(int ta:{0,1})for(int tb:{0,1}){
 char q[16];sprintf(q,"%d",bits);setenv("AWE_FP64_BITS",q,1);int m=17,n=19,k=129;int lda=(row?(ta?m:k):(ta?k:m))+3,ldb=(row?(tb?k:n):(tb?n:k))+5,ldc=(row?n:m)+7;
 size_t na=size_t(row?(ta?k:m):(ta?m:k))*lda,nb=size_t(row?(tb?n:k):(tb?k:n))*ldb,nc=size_t(row?m:n)*ldc;
 std::vector<double>a(na,999),b(nb,999),c(nc,777),ref(nc,777);
 auto idx=[&](int i,int j,int ld){return row?size_t(i)*ld+j:size_t(j)*ld+i;};
 for(int i=0;i<m;++i)for(int z=0;z<k;++z)a[idx(ta?z:i,ta?i:z,lda)]=((i*19+z*11)%37-18)*0.25;
 for(int z=0;z<k;++z)for(int j=0;j<n;++j)b[idx(tb?j:z,tb?z:j,ldb)]=((j*23+z*7)%41-20)*0.125;
 for(int i=0;i<m;++i)for(int j=0;j<n;++j){double v=0;for(int z=0;z<k;++z)v+=a[idx(ta?z:i,ta?i:z,lda)]*b[idx(tb?j:z,tb?z:j,ldb)];ref[idx(i,j,ldc)]=v;}
 double *da,*db,*dc;void* w;CU(cudaMalloc(&da,na*8));CU(cudaMalloc(&db,nb*8));CU(cudaMalloc(&dc,nc*8));CU(cudaMemcpy(da,a.data(),na*8,cudaMemcpyHostToDevice));CU(cudaMemcpy(db,b.data(),nb*8,cudaMemcpyHostToDevice));CU(cudaMemcpy(dc,c.data(),nc*8,cudaMemcpyHostToDevice));
 awe_gemm_desc d;awe_gemm_desc_init(&d);d.m=m;d.n=n;d.k=k;d.lda=lda;d.ldb=ldb;d.ldc=ldc;d.layout=(awe_layout)row;d.trans_a=(awe_operation)ta;d.trans_b=(awe_operation)tb;d.data_type=AWE_F64_F64;
 awe_options o;awe_options_init(&o);o.method=AWE_METHOD_OZAKI2;o.stream_count=streams;o.profiling=1;awe_plan* p;AW(awe_plan_create(&p,&d,&o));size_t bytes;AW(awe_plan_get_workspace_size(p,&bytes));CU(cudaMalloc(&w,bytes));AW(awe_gemm_f64(p,da,db,dc,w,bytes,nullptr,nullptr));CU(cudaMemcpy(c.data(),dc,nc*8,cudaMemcpyDeviceToHost));for(size_t z=0;z<nc;++z)if(c[z]!=ref[z]){fprintf(stderr,"mismatch case%d at%zu %.17g %.17g\n",cases,z,c[z],ref[z]);return 1;}checked+=nc;++cases;
 a[0]=NAN;CU(cudaMemcpy(da,a.data(),na*8,cudaMemcpyHostToDevice));if(awe_gemm_f64(p,da,db,dc,w,bytes,nullptr,nullptr)!=AWE_INPUT_OUT_OF_RANGE)return 1;
 awe_plan_destroy(p);CU(cudaFree(w));CU(cudaFree(da));CU(cudaFree(db));CU(cudaFree(dc));}
 printf("{\"status\":\"PASS\",\"cases\":%d,\"outputs_and_padding\":%zu,\"nonfinite_rejections\":%d}\n",cases,checked,cases);return 0;}
