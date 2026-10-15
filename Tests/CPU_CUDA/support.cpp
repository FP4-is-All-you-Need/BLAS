#include "cuda_runtime.h"
#include <cassert>
extern "C" void support_transpose_nk(dim3 g,dim3 b,cudaStream_t,const double *a,size_t n,size_t k,double *x){
    assert(b.x==32&&b.y==8&&g.x==(n+31)/32&&g.y==(k+31)/32);
    for(size_t j=0;j<k;++j)for(size_t i=0;i<n;++i)x[j+i*k]=a[i+j*n];
}
extern "C" void support_tril_copy(dim3 g,dim3 b,cudaStream_t,const double *a,size_t n,double *x){
    assert(g.x==4096&&b.x==256);
    for(size_t j=0;j<n;++j)for(size_t i=0;i<n;++i)x[i+j*n]=i>=j?a[i+j*n]:0;
}
extern "C" void support_transpose_into(dim3 g,dim3 b,cudaStream_t,const double2 *a,size_t n,size_t k,double2 *x,size_t ld,size_t row,int conj){
    assert(b.x==32&&b.y==8&&g.x==(n+31)/32&&g.y==(k+31)/32&&(conj==0||conj==1)&&ld==k&&row==0);
    for(size_t j=0;j<k;++j)for(size_t i=0;i<n;++i)x[j+i*ld]={a[i+j*n].x,conj ? -a[i+j*n].y : a[i+j*n].y};
}
extern "C" void support_matrix_copy(dim3 g,dim3 b,cudaStream_t,const double2 *a,size_t n,double2 *x,int sym,int herm){
    assert(g.x==4096&&b.x==256);
    for(size_t j=0;j<n;++j)for(size_t i=0;i<n;++i) {
        auto z=i>=j?a[i+j*n]:sym?a[j+i*n]:double2{0,0};
        if(herm && i<j)z.y=-z.y;
        if(herm && i==j)z.y=0;
        x[i+j*n]=z;
    }
}

extern "C" void support_commit(dim3,dim3,cudaStream_t,const double2 *p,size_t n,double2 *c,int lower,int xxt,int herm) {
    for(size_t j=0;j<n;++j)for(size_t i=0;i<n;++i)if(!lower || i>=j) {
        auto z=p[i+j*n];
        if(xxt){auto v=p[j+i*n];z.x+=v.x;z.y+=herm?-v.y:v.y;}
        if(herm && i==j)z.y=0;
        c[i+j*n]=z;
    }
}

extern "C" void support_symm_full(dim3 g,dim3 b,cudaStream_t,const double *a,size_t n,double *x) {
    assert(g.x==4096 && b.x==256);
    for(size_t j=0;j<n;++j)for(size_t i=0;i<n;++i)x[i+j*n]=i>=j?a[i+j*n]:a[j+i*n];
}
extern "C" void support_xxt_lower(dim3 g,dim3 b,cudaStream_t,const double *a,size_t n,double *c) {
    assert(g.x==4096 && b.x==256);
    for(size_t j=0;j<n;++j)for(size_t i=j;i<n;++i)c[i+j*n]=a[i+j*n]+a[j+i*n];
}
extern "C" void support_float_transpose(dim3 g,dim3 b,cudaStream_t,const float *a,size_t n,size_t k,float *x,size_t ld,size_t row) {
    assert(b.x==32 && b.y==8 && ld==k && row==0 && g.x==(n+31)/32 && g.y==(k+31)/32);
    for(size_t q=0;q<k;++q)for(size_t i=0;i<n;++i)x[q+i*k]=a[i+q*n];
}
extern "C" void support_float_matrix(dim3 g,dim3 b,cudaStream_t,const float *a,size_t n,float *x,int sym) {
    assert(g.x==4096 && b.x==256);
    for(size_t j=0;j<n;++j)for(size_t i=0;i<n;++i)x[i+j*n]=i>=j?a[i+j*n]:sym?a[j+i*n]:0;
}
extern "C" void support_float_commit(dim3 g,dim3 b,cudaStream_t,const float *a,size_t n,float *c,int lower,int xxt) {
    assert(g.x==4096 && b.x==256);
    for(size_t j=0;j<n;++j)for(size_t i=0;i<n;++i)if(!lower || i>=j)c[i+j*n]=xxt?a[i+j*n]+a[j+i*n]:a[i+j*n];
}

extern "C" void support_copy_lower(dim3 g, dim3 b, cudaStream_t, const double *a, size_t n, double *c) {
    assert(g.x == 4096 && b.x == 256);
    for (size_t j = 0; j < n; ++j) for (size_t i = j; i < n; ++i) c[i+j*n] = a[i+j*n];
}
