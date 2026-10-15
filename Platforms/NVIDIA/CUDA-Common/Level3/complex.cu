// Frozen Level-3 support kernels; see Docs/Provenance/level3_support.json.
#include <cuda_runtime.h>
#include <cstdint>
namespace {
__device__ double2 cj(double2 z){return make_double2(z.x,-z.y);}
__global__ void transpose_into(const double2* A,size_t n,size_t k,double2* Y,size_t ldo,size_t row0,int conjugate) {
  __shared__ double2 t[32][33];size_t i0=size_t(blockIdx.x)*32,p0=size_t(blockIdx.y)*32;
  for(int y=threadIdx.y;y<32;y+=8){size_t i=i0+threadIdx.x,p=p0+y;if(i<n&&p<k)t[y][threadIdx.x]=A[i+p*n];}
  __syncthreads();
  for(int y=threadIdx.y;y<32;y+=8){size_t p=p0+threadIdx.x,i=i0+y;if(i<n&&p<k){double2 z=t[threadIdx.x][y];Y[row0+p+i*ldo]=conjugate?cj(z):z;}}
}
__global__ void matrix_copy(const double2* A,size_t n,double2* X,int symmetric,int hermitian) {
  for(size_t t=size_t(blockIdx.x)*blockDim.x+threadIdx.x;t<n*n;t+=size_t(gridDim.x)*blockDim.x){size_t i=t%n,j=t/n;double2 z=make_double2(0,0);
    if(i>=j)z=A[t];else if(symmetric){z=A[j+i*n];if(hermitian)z=cj(z);}if(hermitian&&i==j)z.y=0.0;X[t]=z;}
}
__global__ void commit(const double2* P,size_t n,double2* C,int lower,int xxt,int hermitian) {
  for(size_t t=size_t(blockIdx.x)*blockDim.x+threadIdx.x;t<n*n;t+=size_t(gridDim.x)*blockDim.x){size_t i=t%n,j=t/n;
    if(!lower||i>=j){double2 z=P[t];if(xxt){double2 v=P[j+i*n];if(hermitian)v=cj(v);z=make_double2(__dadd_rn(z.x,v.x),__dadd_rn(z.y,v.y));}if(hermitian&&i==j)z.y=0.0;C[t]=z;}}
}
}
extern "C" void support_transpose_into(dim3 grid,dim3 block,cudaStream_t stream,const double2* A,size_t n,size_t k,double2* Y,size_t ldo,size_t row0,int conjugate) { transpose_into<<<grid,block,0,stream>>>(A,n,k,Y,ldo,row0,conjugate); }
extern "C" void support_matrix_copy(dim3 grid,dim3 block,cudaStream_t stream,const double2* A,size_t n,double2* X,int symmetric,int hermitian) { matrix_copy<<<grid,block,0,stream>>>(A,n,X,symmetric,hermitian); }
extern "C" void support_commit(dim3 grid,dim3 block,cudaStream_t stream,const double2* P,size_t n,double2* C,int lower,int xxt,int hermitian) { commit<<<grid,block,0,stream>>>(P,n,C,lower,xxt,hermitian); }
