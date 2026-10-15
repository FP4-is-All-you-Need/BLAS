#include "awe_blas.h"
#include "awe_backend.h"
#include "tables.hpp"
#include <cublasLt.h>
#include <algorithm>
#include <climits>
#include <cstring>
#include <memory>
#include <vector>
#include <new>
#include <cstdint>
#include <cstdlib>
#include <cerrno>

namespace {
struct failure { awe_status status; };
void cu(cudaError_t s){if(s!=cudaSuccess)throw failure{AWE_CUDA_ERROR};}
void lt(cublasStatus_t s){if(s!=CUBLAS_STATUS_SUCCESS)throw failure{s==CUBLAS_STATUS_NOT_SUPPORTED?AWE_NOT_SUPPORTED:AWE_CUDA_ERROR};}
size_t aligned(size_t n){return (n+255)&~size_t(255);}
struct Event {
 cudaEvent_t e=nullptr;
 Event(){cu(cudaEventCreate(&e));}
 ~Event(){if(e)cudaEventDestroy(e);}
 Event(const Event&)=delete;
};
void rec(Event& e,cudaStream_t s){cu(cudaEventRecord(e.e,s));}
void wait(cudaStream_t s,Event& e){cu(cudaStreamWaitEvent(s,e.e,0));}
float ms(Event& a,Event& b){float v;cu(cudaEventElapsedTime(&v,a.e,b.e));return v;}
struct SlotEvents {Event encoded,gemmed,free;};
struct Stages {Event enc0,enc1,wait0,gem0,gem1,out0,out1;};
__device__ int read_a(const int8_t* p,int row,int q,long long ld,int trans){return p[trans?q+row*ld:row+q*ld];}
__device__ int read_b(const int8_t* p,int col,int q,long long ld,int trans){return p[trans?col+q*ld:q+col*ld];}
__device__ bool is_fp4(int v){v=abs(v);return v<=4||v==6||v==8||v==12;}
/* Inspect every logical input once; padding and inactive rows are excluded. */
__global__ void inspect_inputs(const int8_t* a,const int8_t* b,int m,int n,int k,long long lda,long long ldb,int ta,int tb,unsigned* flags){
 unsigned local=0;
 for(size_t z=size_t(blockIdx.x)*blockDim.x+threadIdx.x;z<size_t(m+n)*k;z+=size_t(blockDim.x)*gridDim.x){
  int q=z%k,r=z/k;int v=r<m?read_a(a,r,q,lda,ta):read_b(b,r-m,q,ldb,tb);
  if(!is_fp4(v))local=1;
 }
 if(local)atomicOr(flags,local);
}
__global__ void encode_face(const int8_t* x,uint8_t* y,const uint32_t* lut,int rows,int k,int padded_rows,int kp,int start,long long ld,int trans,int side,int shift){
 for(size_t z=size_t(blockIdx.x)*blockDim.x+threadIdx.x;z<size_t(padded_rows)*kp/2;z+=size_t(blockDim.x)*gridDim.x){
  int row=z/(kp/2),q=2*(z%(kp/2));unsigned packed=0;
  for(int t=0;t<2;++t){int64_t at=int64_t(start)+q+t;int v=0;if(row<rows&&at<k)v=side?read_b(x,row,at,ld,trans):read_a(x,row,at,ld,trans);packed|=((lut[v+128]>>shift)&15)<<(4*t);}
  y[z]=packed;
 }
}
/* Coalesced transpose while packing a face; inherited TN layout is retained. */
__global__ void encode_transposed(const int8_t* x,uint8_t* y,const uint32_t* lut,int rows,int k,int padded_rows,int kp,int start,long long ld,int shift){
 __shared__ int8_t tile[32][33];__shared__ uint32_t cache[256];
 int tx=threadIdx.x,ty=threadIdx.y;cache[ty*32+tx]=lut[ty*32+tx];__syncthreads();
 int tiles_k=kp/32;size_t tiles=size_t(padded_rows/32)*tiles_k;
 for(size_t t=blockIdx.x;t<tiles;t+=gridDim.x){int rr=(t/tiles_k)*32,qq=(t%tiles_k)*32;
  for(int j=0;j<32;j+=8){int row=rr+tx;int64_t q=int64_t(start)+qq+ty+j;tile[ty+j][tx]=(row<rows&&q<k)?x[row+q*ld]:0;}
  __syncthreads();
  for(int j=0;j<2;++j){int row=ty+8*(tx/16)+j*16,q=2*(tx%16);unsigned a=(cache[int(tile[q][row])+128]>>shift)&15,b=(cache[int(tile[q+1][row])+128]>>shift)&15;y[size_t(rr+row)*(kp/2)+(qq+q)/2]=a|(b<<4);}
  __syncthreads();
 }
}
__global__ void encode_contiguous(const int8_t* x,uint8_t* y,const uint32_t* lut,size_t vectors,int shift){
 __shared__ uint32_t cache[256];cache[threadIdx.x]=lut[threadIdx.x];__syncthreads();
 for(size_t z=size_t(blockIdx.x)*blockDim.x+threadIdx.x;z<vectors;z+=size_t(blockDim.x)*gridDim.x){
  uint4 in=reinterpret_cast<const uint4*>(x)[z];unsigned words[4]={in.x,in.y,in.z,in.w},lo=0,hi=0;
  for(int j=0;j<4;++j){unsigned v=words[j]^0x80808080u,p=0;for(int t=0;t<4;++t)p|=((cache[(v>>(8*t))&255]>>shift)&15)<<(4*t);if(j<2)lo|=p<<(16*j);else hi|=p<<(16*(j-2));}
  reinterpret_cast<uint2*>(y)[z]=make_uint2(lo,hi);
 }
}
void encode_dispatch(const int8_t* x,uint8_t* y,const uint32_t* lut,int rows,int k,int padded_rows,int kp,int start,long long ld,int trans,int side,int shift,cudaStream_t stream){
 if(side?trans:!trans){encode_transposed<<<std::min(size_t(4096),size_t(padded_rows/32)*(kp/32)),dim3(32,8),0,stream>>>(x,y,lut,rows,k,padded_rows,kp,start,ld,shift);}
 else if(rows==padded_rows&&start==0&&k==kp&&ld==k&&reinterpret_cast<uintptr_t>(x)%16==0){size_t vectors=size_t(rows)*k/16;encode_contiguous<<<std::min(size_t(4096),(vectors+255)/256),256,0,stream>>>(x,y,lut,vectors,shift);}
 else encode_face<<<std::min(size_t(4096),(size_t(padded_rows)*kp/2+255)/256),256,0,stream>>>(x,y,lut,rows,k,padded_rows,kp,start,ld,trans,side,shift);
}
__global__ void initialize_output(int64_t* c,int m,int n,long long ld){for(size_t z=size_t(blockIdx.x)*blockDim.x+threadIdx.x;z<size_t(m)*n;z+=size_t(blockDim.x)*gridDim.x)c[z%m+(z/m)*ld]=0;}
__global__ void initialize_i32(int32_t* c,int m,int n,long long ld){for(size_t z=size_t(blockIdx.x)*blockDim.x+threadIdx.x;z<size_t(m)*n;z+=size_t(blockDim.x)*gridDim.x)c[z%m+(z/m)*ld]=0;}
__global__ void store_i32(const int64_t* x,int32_t* c,int m,int n,int mp,long long ld){for(size_t z=size_t(blockIdx.x)*blockDim.x+threadIdx.x;z<size_t(m)*n;z+=size_t(blockDim.x)*gridDim.x)c[z%m+(z/m)*ld]=int32_t(x[z%m+(z/m)*mp]);}
__global__ void accumulate(const float* d,int64_t* c,int m,int n,int mp,long long ld,int weight){
 for(size_t z=size_t(blockIdx.x)*blockDim.x+threadIdx.x;z<size_t(m)*n;z+=size_t(blockDim.x)*gridDim.x){int i=z%m,j=z/m;c[i+size_t(j)*ld]+=int64_t(__float2int_rn(d[i+size_t(j)*mp]))*weight;}
}
/* Four adjacent outputs per thread; no row/column division on dense output. */
__global__ void accumulate_contiguous(const float* d,int64_t* c,size_t vectors,int weight){
 for(size_t z=size_t(blockIdx.x)*blockDim.x+threadIdx.x;z<vectors;z+=size_t(blockDim.x)*gridDim.x){
  float4 v=reinterpret_cast<const float4*>(d)[z];
  longlong2 a=reinterpret_cast<const longlong2*>(c)[2*z];
  longlong2 b=reinterpret_cast<const longlong2*>(c)[2*z+1];
  a.x+=int64_t(__float2int_rn(v.x))*weight;a.y+=int64_t(__float2int_rn(v.y))*weight;
  b.x+=int64_t(__float2int_rn(v.z))*weight;b.y+=int64_t(__float2int_rn(v.w))*weight;
  reinterpret_cast<longlong2*>(c)[2*z]=a;reinterpret_cast<longlong2*>(c)[2*z+1]=b;
 }
}
/* Inspect and encode all direct faces during one input traversal. */
__global__ void encode_all_faces(const int8_t* x,uint8_t* y,const uint32_t* lut,int rows,int k,int padded_rows,int kp,long long ld,int trans,int side,unsigned* flags){
 __shared__ uint32_t cache[256];__shared__ uint32_t single[256];
 cache[threadIdx.x]=lut[threadIdx.x];single[threadIdx.x]=lut[512+threadIdx.x];__syncthreads();
 size_t plane=size_t(padded_rows)*kp/2;int invalid=0;
 for(size_t z=size_t(blockIdx.x)*blockDim.x+threadIdx.x;z<plane;z+=size_t(blockDim.x)*gridDim.x){
  int row=z/(kp/2),q=2*(z%(kp/2));int v[2]={0,0};
  if(row<rows){if(q<k)v[0]=side?read_b(x,row,q,ld,trans):read_a(x,row,q,ld,trans);if(q+1<k)v[1]=side?read_b(x,row,q+1,ld,trans):read_a(x,row,q+1,ld,trans);}
  uint32_t a=cache[v[0]+128],b=cache[v[1]+128],sa=single[v[0]+128],sb=single[v[1]+128];
  invalid|=(sa==255||sb==255);
  #pragma unroll
  for(int f=0;f<6;++f)y[size_t(f)*plane+z]=((a>>(4*f))&15)|(((b>>(4*f))&15)<<4);
  y[6*plane+z]=(sa&15)|((sb&15)<<4);
 }
 int any=__syncthreads_or(invalid);if(threadIdx.x==0&&any)atomicOr(flags,1u);
}
__global__ void encode_all_transposed(const int8_t* x,uint8_t* y,const uint32_t* lut,int rows,int k,int padded_rows,int kp,long long ld,unsigned* flags){
 __shared__ int8_t tile[32][33];__shared__ uint32_t cache[256],single[256];
 int tx=threadIdx.x,ty=threadIdx.y,lane=ty*32+tx;cache[lane]=lut[lane];single[lane]=lut[512+lane];__syncthreads();
 int tiles_k=kp/32;size_t tiles=size_t(padded_rows/32)*tiles_k,plane=size_t(padded_rows)*kp/2;int invalid=0;
 for(size_t t=blockIdx.x;t<tiles;t+=gridDim.x){int rr=(t/tiles_k)*32,qq=(t%tiles_k)*32;
  for(int j=0;j<32;j+=8){int row=rr+tx,q=qq+ty+j;tile[ty+j][tx]=(row<rows&&q<k)?x[row+size_t(q)*ld]:0;}
  __syncthreads();
  for(int j=0;j<2;++j){int row=ty+8*(tx/16)+j*16,q=2*(tx%16);int va=int(tile[q][row])+128,vb=int(tile[q+1][row])+128;
   uint32_t a=cache[va],b=cache[vb],sa=single[va],sb=single[vb];invalid|=(sa==255||sb==255);
   size_t z=size_t(rr+row)*(kp/2)+(qq+q)/2;
   #pragma unroll
   for(int f=0;f<6;++f)y[size_t(f)*plane+z]=((a>>(4*f))&15)|(((b>>(4*f))&15)<<4);
   y[6*plane+z]=(sa&15)|((sb&15)<<4);
  }
  __syncthreads();
 }
 int any=__syncthreads_or(invalid);if(lane==0&&any)atomicOr(flags,1u);
}
/* Eight input values per thread produce aligned words in every plane. */
__global__ void encode_all_contiguous(const int8_t* x,uint8_t* y,const uint32_t* lut,size_t vectors,unsigned* flags){
 __shared__ uint32_t cache[256],single[256];
 cache[threadIdx.x]=lut[threadIdx.x];single[threadIdx.x]=lut[512+threadIdx.x];__syncthreads();int invalid=0;
 for(size_t z=size_t(blockIdx.x)*blockDim.x+threadIdx.x;z<vectors;z+=size_t(blockDim.x)*gridDim.x){
  uint2 in=reinterpret_cast<const uint2*>(x)[z];uint32_t words[2]={in.x^0x80808080u,in.y^0x80808080u};uint32_t packed[7]={};
  #pragma unroll
  for(int j=0;j<8;++j){unsigned index=(words[j/4]>>(8*(j%4)))&255;uint32_t a=cache[index],b=single[index];invalid|=b==255;
   #pragma unroll
   for(int f=0;f<6;++f)packed[f]|=((a>>(4*f))&15)<<(4*j);
   packed[6]|=(b&15)<<(4*j);
  }
  #pragma unroll
  for(int f=0;f<7;++f)reinterpret_cast<uint32_t*>(y)[size_t(f)*vectors+z]=packed[f];
 }
 int any=__syncthreads_or(invalid);if(threadIdx.x==0&&any)atomicOr(flags,1u);
}
void encode_all_dispatch(const int8_t* x,uint8_t* y,const uint32_t* lut,int rows,int k,int padded_rows,int kp,long long ld,int trans,int side,unsigned* flags,cudaStream_t stream){
 if(side?trans:!trans)encode_all_transposed<<<std::min(size_t(4096),size_t(padded_rows/32)*(kp/32)),dim3(32,8),0,stream>>>(x,y,lut,rows,k,padded_rows,kp,ld,flags);
 else if(rows==padded_rows&&k==kp&&ld==k&&reinterpret_cast<uintptr_t>(x)%8==0){size_t vectors=size_t(rows)*k/8;encode_all_contiguous<<<std::min(size_t(4096),(vectors+255)/256),256,0,stream>>>(x,y,lut,vectors,flags);}
 else encode_all_faces<<<std::min(size_t(4096),(size_t(padded_rows)*kp/2+255)/256),256,0,stream>>>(x,y,lut,rows,k,padded_rows,kp,ld,trans,side,flags);
}
template<int NF> __global__ void restore_fused(const float* d,int64_t* c,int m,int n,int mp,long long ld,size_t plane){
 const int coeff[6]={33,165,891,-5,-27,-135};
 for(size_t z=size_t(blockIdx.x)*blockDim.x+threadIdx.x;z<size_t(m)*n;z+=size_t(blockDim.x)*gridDim.x){
  int row=z%m,col=z/m;size_t at=row+size_t(col)*mp;int64_t value=0;
  #pragma unroll
  for(int f=0;f<NF;++f)value+=int64_t(__float2int_rn(d[size_t(f)*plane+at]))*(NF==1?1:coeff[f]);
  c[row+size_t(col)*ld]=value;
 }
}
template<int NF> __global__ void restore_fused_contiguous(const float* d,int64_t* c,size_t vectors,size_t plane){
 const int coeff[6]={33,165,891,-5,-27,-135};
 for(size_t z=size_t(blockIdx.x)*blockDim.x+threadIdx.x;z<vectors;z+=size_t(blockDim.x)*gridDim.x){
  longlong2 a=make_longlong2(0,0),b=make_longlong2(0,0);
  #pragma unroll
  for(int f=0;f<NF;++f){float4 v=reinterpret_cast<const float4*>(d+size_t(f)*plane)[z];int weight=NF==1?1:coeff[f];
   a.x+=int64_t(__float2int_rn(v.x))*weight;a.y+=int64_t(__float2int_rn(v.y))*weight;
   b.x+=int64_t(__float2int_rn(v.z))*weight;b.y+=int64_t(__float2int_rn(v.w))*weight;
  }
  reinterpret_cast<longlong2*>(c)[2*z]=a;reinterpret_cast<longlong2*>(c)[2*z+1]=b;
 }
}

int blocks(size_t n){return int(std::min(size_t(4096),(n+255)/256));}
size_t extent(int rows,int cols,int64_t ld,size_t item){return rows&&cols?(size_t(cols-1)*ld+rows)*item:0;}
bool overlaps(const void* a,size_t an,const void* b,size_t bn){auto x=reinterpret_cast<uintptr_t>(a),y=reinterpret_cast<uintptr_t>(b);return an&&bn&&x<y+bn&&y<x+an;}
}

struct awe_plan {
 awe_gemm_desc d{};awe_options o{};awe_plan_info info{};awe_execution_stats last{};bool row=false;void* ozaki=nullptr;
 bool fused=false;int tile_n=0;int m=0,n=0,k=0,mp=0,np=0,kp=0;size_t bytes=0,scale_bytes=0,ab=0,bb=0,db=0;
 cublasLtHandle_t handle=nullptr;cublasLtMatmulDesc_t op=nullptr;cublasLtMatrixLayout_t la=nullptr,lb=nullptr,lc=nullptr;
 cublasLtMatmulAlgo_t algo{};
 cudaStream_t streams[3]{};
 std::vector<std::unique_ptr<SlotEvents>> slots;
 std::vector<std::unique_ptr<Stages>> stages;
 std::unique_ptr<Event> begin,end,selection_end,ready;
 unsigned* host_flags=nullptr;uint32_t* host_lut=nullptr;
 ~awe_plan(){
#ifdef AWE_HAS_OZAKI2
  if(ozaki)awe_ozaki2_destroy(ozaki);
#endif
  for(auto s:streams)if(s)cudaStreamDestroy(s);
  if(host_flags)cudaFreeHost(host_flags);if(host_lut)cudaFreeHost(host_lut);
  if(la)cublasLtMatrixLayoutDestroy(la);if(lb)cublasLtMatrixLayoutDestroy(lb);if(lc)cublasLtMatrixLayoutDestroy(lc);
  if(op)cublasLtMatmulDescDestroy(op);if(handle)cublasLtDestroy(handle);
 }
};

namespace {
void execute_fused(awe_plan* p,const int8_t* a,const int8_t* b,void* cc,void* workspace,cudaStream_t stream,awe_execution_stats* stats){
 auto es=p->streams[0],gs=p->o.stream_count==3?p->streams[1]:es,os=p->o.stream_count==3?p->streams[2]:es;
 rec(*p->begin,stream);wait(es,*p->begin);wait(gs,*p->begin);wait(os,*p->begin);
 auto cursor=static_cast<unsigned char*>(workspace);auto lut=reinterpret_cast<uint32_t*>(cursor);cursor+=aligned(3072);auto flags=reinterpret_cast<unsigned*>(cursor);cursor+=256;
 void* scales=cursor;cursor+=aligned(p->scale_bytes);void* scratch=cursor;cursor+=aligned(p->o.matmul_workspace_bytes);
 auto encoded_a=cursor;cursor+=aligned(7*p->ab);auto encoded_b=cursor;cursor+=aligned(7*p->bb);auto results=cursor;
 auto c=static_cast<int64_t*>(cc);long long ld=p->d.ldc;
 if(p->d.data_type==AWE_I8_I32){c=reinterpret_cast<int64_t*>(results+p->o.buffer_count*aligned(6*p->db));ld=p->mp;}
 lt(cublasLtMatmulDescSetAttribute(p->op,CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,&scales,sizeof(scales)));
 lt(cublasLtMatmulDescSetAttribute(p->op,CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,&scales,sizeof(scales)));
 cu(cudaMemcpyAsync(lut,p->host_lut,3072,cudaMemcpyHostToDevice,es));cu(cudaMemsetAsync(scales,0x40,p->scale_bytes,es));cu(cudaMemsetAsync(flags,0,4,es));
 if(p->o.profiling)rec(p->stages[0]->enc0,es);
 encode_all_dispatch(a,encoded_a,lut,p->m,p->k,p->mp,p->kp,p->d.lda,p->d.trans_a,0,flags,es);
 encode_all_dispatch(b,encoded_b,lut,p->n,p->k,p->np,p->kp,p->d.ldb,p->d.trans_b,1,flags,es);
 if(p->o.profiling)rec(p->stages[0]->enc1,es);
 rec(*p->ready,es);cu(cudaMemcpyAsync(p->host_flags,flags,4,cudaMemcpyDeviceToHost,es));rec(*p->selection_end,es);cu(cudaEventSynchronize(p->selection_end->e));
 int nf=*p->host_flags?6:1;p->info.face_count=nf;p->info.weights[0]=1;p->info.weights[1]=nf==1?0:5;p->info.weights[2]=nf==1?0:27;
 size_t ix=0;int tile=0;
 for(int col=0;col<p->np;col+=p->tile_n,++tile){
  auto& ev=*p->slots[tile%p->o.buffer_count];auto out=reinterpret_cast<float*>(results+(tile%p->o.buffer_count)*aligned(6*p->db));
  Stages* first=p->o.profiling?p->stages[size_t(tile)*6].get():nullptr;
  for(int f=0;f<nf;++f,++ix){
   Stages* t=p->o.profiling?p->stages[size_t(tile)*6+f].get():nullptr;
   if(t)rec(t->wait0,gs);if(f==0){wait(gs,*p->ready);if(tile>=p->o.buffer_count)wait(gs,ev.free);}if(t)rec(t->gem0,gs);
   int plane=nf==1?6:f;float one=1,zero=0;auto d=out+size_t(f)*(p->db/4);
   lt(cublasLtMatmul(p->handle,p->op,&one,encoded_a+size_t(plane)*p->ab,p->la,encoded_b+size_t(plane)*p->bb+size_t(col)*p->kp/2,p->lb,&zero,d,p->lc,d,p->lc,&p->algo,scratch,p->o.matmul_workspace_bytes,gs));
   if(t)rec(t->gem1,gs);
  }
  rec(ev.gemmed,gs);wait(os,ev.gemmed);if(first)rec(first->out0,os);
  int columns=std::min(p->tile_n,p->n-col);size_t count=size_t(p->m)*columns;auto dest=c+size_t(col)*ld;
  if(p->m==p->mp&&ld==p->m&&count%4==0&&reinterpret_cast<uintptr_t>(dest)%16==0){
   if(nf==6)restore_fused_contiguous<6><<<blocks(count/4),256,0,os>>>(out,dest,count/4,p->db/4);
   else restore_fused_contiguous<1><<<blocks(count/4),256,0,os>>>(out,dest,count/4,p->db/4);
  }else{
   if(nf==6)restore_fused<6><<<blocks(count),256,0,os>>>(out,dest,p->m,columns,p->mp,ld,p->db/4);
   else restore_fused<1><<<blocks(count),256,0,os>>>(out,dest,p->m,columns,p->mp,ld,p->db/4);
  }
  if(first)rec(first->out1,os);rec(ev.free,os);
 }
 if(p->d.data_type==AWE_I8_I32)store_i32<<<blocks(size_t(p->m)*p->n),256,0,os>>>(c,static_cast<int32_t*>(cc),p->m,p->n,p->mp,p->d.ldc);
 rec(*p->end,os);wait(stream,*p->end);cu(cudaGetLastError());cu(cudaEventSynchronize(p->end->e));
 stats->total_ms=ms(*p->begin,*p->end);stats->selection_ms=ms(*p->ready,*p->selection_end);stats->face_count=nf;stats->chunk_count=1;
 if(p->o.profiling){stats->encode_ms=ms(p->stages[0]->enc0,p->stages[0]->enc1);
  for(int t=0;t<tile;++t){auto& first=*p->stages[size_t(t)*6];stats->reconstruction_ms+=ms(first.out0,first.out1);
   for(int f=0;f<nf;++f){auto& e=*p->stages[size_t(t)*6+f];stats->gemm_ms+=ms(e.gem0,e.gem1);stats->wait_ms+=ms(e.wait0,e.gem0);}
  }
 }
}
}

extern "C" void awe_gemm_desc_init(awe_gemm_desc* d){if(!d)return;*d={};d->struct_size=sizeof(*d);d->alpha=1;d->lda=d->ldb=d->ldc=1;}
extern "C" void awe_options_init(awe_options* o){if(!o)return;*o={};o->struct_size=sizeof(*o);o->method=AWE_METHOD_AUTO;o->stream_count=3;o->buffer_count=3;o->matmul_workspace_bytes=64ul<<20;}
extern "C" const char* awe_status_string(awe_status s){switch(s){case AWE_SUCCESS:return "success";case AWE_INVALID_ARGUMENT:return "invalid argument";case AWE_NOT_SUPPORTED:return "not supported";case AWE_INSUFFICIENT_WORKSPACE:return "insufficient workspace";case AWE_CUDA_ERROR:return "CUDA or cuBLAS error";case AWE_INPUT_OUT_OF_RANGE:return "input out of range";default:return "internal error";}}
extern "C" awe_status awe_plan_create(awe_plan** out,const awe_gemm_desc* desc,const awe_options* options){
 if(!out)return AWE_INVALID_ARGUMENT;*out=nullptr;
 try{
  if(!desc||desc->struct_size!=sizeof(*desc))return AWE_INVALID_ARGUMENT;
  auto p=std::make_unique<awe_plan>();p->d=*desc;awe_options_init(&p->o);
  if(options){if(options->struct_size!=sizeof(*options))return AWE_INVALID_ARGUMENT;p->o=*options;}
  auto& d=p->d;auto& o=p->o;
  if(d.m<0||d.n<0||d.k<0||d.m>(1<<20)||d.n>(1<<20)||d.k>INT_MAX||d.trans_a<AWE_OP_N||d.trans_a>AWE_OP_T||d.trans_b<AWE_OP_N||d.trans_b>AWE_OP_T||d.data_type<AWE_I8_I64||d.data_type>AWE_I8_I32||o.method<AWE_METHOD_AUTO||o.method>AWE_METHOD_OZAKI2||o.k_chunk<0||o.buffer_count<1||o.buffer_count>(o.method==AWE_METHOD_OZAKI2?11:8)||(o.stream_count!=1&&o.stream_count!=3)||o.heuristic_index<0||o.heuristic_index>=16)return AWE_INVALID_ARGUMENT;
  if(d.layout<AWE_COLUMN_MAJOR||d.layout>AWE_ROW_MAJOR||d.alpha!=1||d.beta!=0)return AWE_NOT_SUPPORTED;
  if(d.data_type==AWE_F64_F64)return AWE_NOT_SUPPORTED;
  if(d.data_type==AWE_I8_I32&&d.k>131071)return AWE_INPUT_OUT_OF_RANGE;
  p->last.struct_size=sizeof(p->last);p->info.input_min=-128;p->info.input_max=127;
  if(o.method!=AWE_METHOD_OZAKI2&&d.data_type!=AWE_F64_F64&&d.layout==AWE_ROW_MAJOR){p->row=true;std::swap(d.m,d.n);std::swap(d.lda,d.ldb);std::swap(d.trans_a,d.trans_b);d.layout=AWE_COLUMN_MAJOR;}
  if(d.lda<std::max(int64_t(1),d.layout==AWE_ROW_MAJOR?(d.trans_a==AWE_OP_N?d.k:d.m):(d.trans_a==AWE_OP_N?d.m:d.k))||d.ldb<std::max(int64_t(1),d.layout==AWE_ROW_MAJOR?(d.trans_b==AWE_OP_N?d.n:d.k):(d.trans_b==AWE_OP_N?d.k:d.n))||d.ldc<std::max(int64_t(1),d.layout==AWE_ROW_MAJOR?d.n:d.m)||d.lda>INT_MAX||d.ldb>INT_MAX||d.ldc>INT_MAX)return AWE_INVALID_ARGUMENT;
  p->info.struct_size=sizeof(p->info);p->info.method=o.method;p->info.stream_count=o.stream_count;
  if(o.method==AWE_METHOD_OZAKI2||(o.method==AWE_METHOD_AUTO&&d.data_type==AWE_F64_F64)){
#ifdef AWE_HAS_OZAKI2
   awe_status s=awe_ozaki2_create(&p->ozaki,&d,&o);if(s!=AWE_SUCCESS)return s;s=awe_ozaki2_info(p->ozaki,&p->info);if(s!=AWE_SUCCESS)return s;p->bytes=p->info.workspace_bytes;*out=p.release();return AWE_SUCCESS;
#else
   return AWE_NOT_SUPPORTED;
#endif
  }
  if(d.data_type!=AWE_I8_I64&&d.data_type!=AWE_I8_I32)return AWE_NOT_SUPPORTED;
  if(o.k_chunk>116480)return AWE_INVALID_ARGUMENT;
  p->m=int(d.m);p->n=int(d.n);p->k=int(d.k);p->mp=(p->m+127)/128*128;p->np=(p->n+127)/128*128;
  p->info.method=o.method==AWE_METHOD_RADIX13?AWE_METHOD_RADIX13:AWE_METHOD_AWE;
  p->info.face_count=o.method==AWE_METHOD_RADIX13?9:6;
  p->info.weights[0]=1;p->info.weights[1]=o.method==AWE_METHOD_RADIX13?13:5;p->info.weights[2]=o.method==AWE_METHOD_RADIX13?169:27;
  p->info.guaranteed_k=116480;p->info.algorithm_id=-1;
  if(!p->m||!p->n){*out=p.release();return AWE_SUCCESS;}
  int low_priority=0,high_priority=0;cu(cudaDeviceGetStreamPriorityRange(&low_priority,&high_priority));
  for(int i=0;i<o.stream_count;++i)cu(cudaStreamCreateWithPriority(&p->streams[i],cudaStreamNonBlocking,(o.stream_count==3&&i==2)?high_priority:low_priority));
  p->begin=std::make_unique<Event>();p->end=std::make_unique<Event>();p->selection_end=std::make_unique<Event>();p->ready=std::make_unique<Event>();
  if(!p->k){*out=p.release();return AWE_SUCCESS;}
  int chunk=o.k_chunk?o.k_chunk:16384;p->kp=(std::min(p->k,chunk)+127)/128*128;
  const char* policy=std::getenv("AWE_FUSION");
  if(policy&&std::strcmp(policy,"0")&&std::strcmp(policy,"1"))return AWE_INVALID_ARGUMENT;
  p->fused=o.method!=AWE_METHOD_RADIX13&&p->k<=p->kp&&(!policy||std::strcmp(policy,"0"));
  p->tile_n=p->np;
  if(p->fused){const char* value=std::getenv("AWE_FUSED_TILE_N");if(value){char* end=nullptr;errno=0;long requested=std::strtol(value,&end,10);if(errno||end==value||*end||requested<0||requested>(1<<20)||requested%128)return AWE_INVALID_ARGUMENT;if(requested)p->tile_n=std::min(p->np,int(requested));}p->np=(p->n+p->tile_n-1)/p->tile_n*p->tile_n;}
  p->info.payout_period=p->kp;p->info.padded_m=p->mp;p->info.padded_n=p->np;p->info.padded_k=p->kp;
  p->scale_bytes=size_t(std::max(p->mp,p->np))*((p->kp+63)/64)*4;
  p->ab=size_t(p->mp)*p->kp/2;p->bb=size_t(p->np)*p->kp/2;p->db=size_t(p->mp)*(p->fused?p->tile_n:p->np)*4;
  p->bytes=aligned(3*256*sizeof(uint32_t))+256+aligned(p->scale_bytes)+aligned(o.matmul_workspace_bytes)+o.buffer_count*(aligned(p->ab)+aligned(p->bb)+aligned(p->db));
  if(p->fused)p->bytes=aligned(3072)+256+aligned(p->scale_bytes)+aligned(o.matmul_workspace_bytes)+aligned(7*p->ab)+aligned(7*p->bb)+o.buffer_count*aligned(6*p->db);
  if(d.data_type==AWE_I8_I32)p->bytes+=aligned(size_t(p->mp)*p->np*8);
  p->info.workspace_bytes=p->bytes;
  cu(cudaMallocHost(&p->host_flags,sizeof(unsigned)));cu(cudaMallocHost(&p->host_lut,3*256*sizeof(uint32_t)));
  memcpy(p->host_lut,awe_six_lut,1024);memcpy(p->host_lut+256,radix13_lut,1024);memcpy(p->host_lut+512,single_lut,1024);
  lt(cublasLtCreate(&p->handle));lt(cublasLtMatmulDescCreate(&p->op,CUBLAS_COMPUTE_32F,CUDA_R_32F));
  cublasOperation_t trans=CUBLAS_OP_T,normal=CUBLAS_OP_N;auto scale=CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
  lt(cublasLtMatmulDescSetAttribute(p->op,CUBLASLT_MATMUL_DESC_TRANSA,&trans,sizeof(trans)));lt(cublasLtMatmulDescSetAttribute(p->op,CUBLASLT_MATMUL_DESC_TRANSB,&normal,sizeof(normal)));
  lt(cublasLtMatmulDescSetAttribute(p->op,CUBLASLT_MATMUL_DESC_A_SCALE_MODE,&scale,sizeof(scale)));lt(cublasLtMatmulDescSetAttribute(p->op,CUBLASLT_MATMUL_DESC_B_SCALE_MODE,&scale,sizeof(scale)));
  lt(cublasLtMatrixLayoutCreate(&p->la,CUDA_R_4F_E2M1,p->kp,p->mp,p->kp));lt(cublasLtMatrixLayoutCreate(&p->lb,CUDA_R_4F_E2M1,p->kp,p->fused?p->tile_n:p->np,p->kp));lt(cublasLtMatrixLayoutCreate(&p->lc,CUDA_R_32F,p->mp,p->fused?p->tile_n:p->np,p->mp));
  struct QueryStorage {void* p=nullptr;QueryStorage(){cu(cudaMalloc(&p,256));}~QueryStorage(){if(p)cudaFree(p);}} query_storage;
  lt(cublasLtMatmulDescSetAttribute(p->op,CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,&query_storage.p,sizeof(void*)));
  lt(cublasLtMatmulDescSetAttribute(p->op,CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,&query_storage.p,sizeof(void*)));
  cublasLtMatmulPreference_t pref=nullptr;lt(cublasLtMatmulPreferenceCreate(&pref));
  auto s=cublasLtMatmulPreferenceSetAttribute(pref,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&o.matmul_workspace_bytes,sizeof(size_t));if(s!=CUBLAS_STATUS_SUCCESS){cublasLtMatmulPreferenceDestroy(pref);lt(s);}
  cublasLtMatmulHeuristicResult_t results[16]{};int count=0;s=cublasLtMatmulAlgoGetHeuristic(p->handle,p->op,p->la,p->lb,p->lc,p->lc,pref,16,results,&count);cublasLtMatmulPreferenceDestroy(pref);lt(s);
  int valid=0;bool found=false;for(int i=0;i<count;++i)if(results[i].state==CUBLAS_STATUS_SUCCESS){if(valid++==o.heuristic_index){p->algo=results[i].algo;found=true;break;}}
  if(!found)return AWE_NOT_SUPPORTED;size_t written=0;lt(cublasLtMatmulAlgoConfigGetAttribute(&p->algo,CUBLASLT_ALGO_CONFIG_ID,&p->info.algorithm_id,sizeof(int),&written));
  for(int i=0;i<o.buffer_count;++i)p->slots.emplace_back(new SlotEvents);
  if(o.profiling){size_t steps=p->fused?size_t(p->np/p->tile_n)*6:((size_t(p->k)+p->kp-1)/p->kp)*p->info.face_count;for(size_t i=0;i<steps;++i)p->stages.emplace_back(new Stages);}
  *out=p.release();return AWE_SUCCESS;
 }catch(failure f){return f.status;}catch(std::bad_alloc const&){return AWE_INTERNAL_ERROR;}catch(...){return AWE_INTERNAL_ERROR;}
}
extern "C" awe_status awe_plan_get_workspace_size(const awe_plan* p,size_t* bytes){if(!p||!bytes)return AWE_INVALID_ARGUMENT;*bytes=p->bytes;return AWE_SUCCESS;}
extern "C" awe_status awe_plan_get_info(const awe_plan* p,awe_plan_info* info){if(!p||!info||info->struct_size!=sizeof(*info))return AWE_INVALID_ARGUMENT;
#ifdef AWE_HAS_OZAKI2
 if(p->ozaki)return awe_ozaki2_info(p->ozaki,info);
#endif
 *info=p->info;return AWE_SUCCESS;}
extern "C" awe_status awe_plan_get_last_stats(const awe_plan* p,awe_execution_stats* stats){if(!p||!stats||stats->struct_size!=sizeof(*stats))return AWE_INVALID_ARGUMENT;*stats=p->last;return AWE_SUCCESS;}
extern "C" void awe_plan_destroy(awe_plan* p){delete p;}
extern "C" awe_status awe_plan_execute(awe_plan* p,const void* aa,const void* bb,void* cc,void* workspace,size_t bytes,cudaStream_t stream,awe_execution_stats* stats){
 if(!p||(stats&&stats->struct_size!=sizeof(*stats)))return AWE_INVALID_ARGUMENT;
#ifdef AWE_HAS_OZAKI2
 if(p->ozaki){auto result=awe_ozaki2_execute(p->ozaki,aa,bb,cc,workspace,bytes,stream,&p->last);if(stats)*stats=p->last;return result;}
#endif
 awe_execution_stats* supplied_stats=stats;stats=&p->last;*stats={};stats->struct_size=sizeof(*stats);
 struct CopyStats{awe_execution_stats* to;const awe_execution_stats* from;~CopyStats(){if(to)*to=*from;}} copy_stats{supplied_stats,stats};
 if(p->row)std::swap(aa,bb);
 if(!p->m||!p->n)return AWE_SUCCESS;
 if(!cc||(p->k&&(!aa||!bb)))return AWE_INVALID_ARGUMENT;
 if(bytes<p->bytes)return AWE_INSUFFICIENT_WORKSPACE;
 if(p->bytes&&(!workspace||reinterpret_cast<uintptr_t>(workspace)%256))return AWE_INVALID_ARGUMENT;
 auto& d=p->d;
 size_t an=extent(d.trans_a==AWE_OP_N?p->m:p->k,d.trans_a==AWE_OP_N?p->k:p->m,d.lda,1),bn=extent(d.trans_b==AWE_OP_N?p->k:p->n,d.trans_b==AWE_OP_N?p->n:p->k,d.ldb,1),cn=extent(p->m,p->n,d.ldc,d.data_type==AWE_I8_I32?4:8);
 if(overlaps(aa,an,cc,cn)||overlaps(bb,bn,cc,cn)||overlaps(workspace,p->bytes,aa,an)||overlaps(workspace,p->bytes,bb,bn)||overlaps(workspace,p->bytes,cc,cn))return AWE_INVALID_ARGUMENT;
 try{
  cudaStreamCaptureStatus capture;cu(cudaStreamIsCapturing(stream,&capture));if(capture!=cudaStreamCaptureStatusNone)return AWE_NOT_SUPPORTED;
  if(p->fused){execute_fused(p,static_cast<const int8_t*>(aa),static_cast<const int8_t*>(bb),cc,workspace,stream,stats);return AWE_SUCCESS;}
  auto es=p->streams[0],gs=p->o.stream_count==3?p->streams[1]:es,os=p->o.stream_count==3?p->streams[2]:es;
  rec(*p->begin,stream);wait(es,*p->begin);wait(gs,*p->begin);wait(os,*p->begin);
  auto c=static_cast<int64_t*>(cc);size_t count=size_t(p->m)*p->n;long long output_ld=d.ldc;
  if(!p->k){if(d.data_type==AWE_I8_I32)initialize_i32<<<blocks(count),256,0,os>>>(static_cast<int32_t*>(cc),p->m,p->n,d.ldc);else initialize_output<<<blocks(count),256,0,os>>>(c,p->m,p->n,d.ldc);rec(*p->end,os);wait(stream,*p->end);cu(cudaEventSynchronize(p->end->e));if(stats)stats->total_ms=ms(*p->begin,*p->end);return AWE_SUCCESS;}
  auto cursor=static_cast<unsigned char*>(workspace);auto lut=reinterpret_cast<uint32_t*>(cursor);cursor+=aligned(3072);auto flags=reinterpret_cast<unsigned*>(cursor);cursor+=256;void* scales=cursor;cursor+=aligned(p->scale_bytes);void* scratch=cursor;cursor+=aligned(p->o.matmul_workspace_bytes);
  lt(cublasLtMatmulDescSetAttribute(p->op,CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,&scales,sizeof(scales)));lt(cublasLtMatmulDescSetAttribute(p->op,CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,&scales,sizeof(scales)));
  cu(cudaMemsetAsync(flags,0,sizeof(unsigned),es));
  inspect_inputs<<<blocks(size_t(p->m+p->n)*p->k),256,0,es>>>(static_cast<const int8_t*>(aa),static_cast<const int8_t*>(bb),p->m,p->n,p->k,d.lda,d.ldb,d.trans_a,d.trans_b,flags);
  cu(cudaMemcpyAsync(p->host_flags,flags,sizeof(unsigned),cudaMemcpyDeviceToHost,es));rec(*p->selection_end,es);cu(cudaEventSynchronize(p->selection_end->e));
  bool radix=p->o.method==AWE_METHOD_RADIX13;int nf=radix?9:(*p->host_flags?6:1);
  p->info.face_count=nf;p->info.weights[0]=1;p->info.weights[1]=nf==1?0:(radix?13:5);p->info.weights[2]=nf==1?0:(radix?169:27);
  cu(cudaMemcpyAsync(lut,p->host_lut,3072,cudaMemcpyHostToDevice,es));cu(cudaMemsetAsync(scales,0x40,p->scale_bytes,es));rec(*p->ready,es);wait(gs,*p->ready);
  if(d.data_type==AWE_I8_I32){c=reinterpret_cast<int64_t*>(cursor+p->o.buffer_count*(aligned(p->ab)+aligned(p->bb)+aligned(p->db)));output_ld=p->mp;}
  initialize_output<<<blocks(count),256,0,os>>>(c,p->m,p->n,output_ld);
  int coeff[6]={33,165,891,-5,-27,-135},pow13[5]={1,13,169,2197,28561};
  size_t stride=aligned(p->ab)+aligned(p->bb)+aligned(p->db),ix=0;
  for(int64_t start=0;start<p->k;start+=p->kp)for(int f=0;f<nf;++f,++ix){
   size_t slot=ix%p->o.buffer_count;auto& ev=*p->slots[slot];auto s=cursor+slot*stride;auto a=s;auto b=s+aligned(p->ab);auto out=reinterpret_cast<float*>(b+aligned(p->bb));
   if(ix>=size_t(p->o.buffer_count))wait(es,ev.free);
   Stages* t=p->o.profiling?p->stages[ix].get():nullptr;if(t)rec(t->enc0,es);
   const uint32_t* table=lut+(radix?256:nf==1?512:0);int fa=radix?f/3:f,fb=radix?f%3:f;
   encode_dispatch(static_cast<const int8_t*>(aa),a,table,p->m,p->k,p->mp,p->kp,int(start),d.lda,d.trans_a,0,fa*4,es);
   encode_dispatch(static_cast<const int8_t*>(bb),b,table,p->n,p->k,p->np,p->kp,int(start),d.ldb,d.trans_b,1,fb*4,es);
   if(t)rec(t->enc1,es);rec(ev.encoded,es);if(t)rec(t->wait0,gs);wait(gs,ev.encoded);if(t)rec(t->gem0,gs);
   float one=1,zero=0;lt(cublasLtMatmul(p->handle,p->op,&one,a,p->la,b,p->lb,&zero,out,p->lc,out,p->lc,&p->algo,scratch,p->o.matmul_workspace_bytes,gs));
   if(t)rec(t->gem1,gs);rec(ev.gemmed,gs);wait(os,ev.gemmed);if(t)rec(t->out0,os);
   int weight=radix?pow13[fa+fb]:nf==1?1:coeff[f];if(p->m==p->mp&&output_ld==p->m&&count%4==0&&reinterpret_cast<uintptr_t>(c)%16==0)
    accumulate_contiguous<<<blocks(count/4),256,0,os>>>(out,c,count/4,weight);
   else accumulate<<<blocks(count),256,0,os>>>(out,c,p->m,p->n,p->mp,output_ld,weight);
   if(t)rec(t->out1,os);rec(ev.free,os);
  }
  if(d.data_type==AWE_I8_I32)store_i32<<<blocks(count),256,0,os>>>(c,static_cast<int32_t*>(cc),p->m,p->n,p->mp,d.ldc);
  rec(*p->end,os);wait(stream,*p->end);cu(cudaGetLastError());cu(cudaEventSynchronize(p->end->e));
  if(stats){stats->total_ms=ms(*p->begin,*p->end);stats->selection_ms=ms(*p->begin,*p->selection_end);stats->face_count=nf;stats->chunk_count=(p->k+p->kp-1ll)/p->kp;
   if(p->o.profiling)for(size_t i=0;i<ix;++i){auto& t=*p->stages[i];stats->encode_ms+=ms(t.enc0,t.enc1);stats->gemm_ms+=ms(t.gem0,t.gem1);stats->reconstruction_ms+=ms(t.out0,t.out1);stats->wait_ms+=ms(t.wait0,t.gem0);}
  }
  return AWE_SUCCESS;
 }catch(failure f){for(auto s:p->streams)if(s)cudaStreamSynchronize(s);return f.status;}catch(...){for(auto s:p->streams)if(s)cudaStreamSynchronize(s);return AWE_INTERNAL_ERROR;}
}
extern "C" awe_status awe_gemm_i8(awe_plan* p,const int8_t* a,const int8_t* b,int64_t* c,void* w,size_t bytes,cudaStream_t s,awe_execution_stats* t){if(!p||p->d.data_type!=AWE_I8_I64)return AWE_INVALID_ARGUMENT;return awe_plan_execute(p,a,b,c,w,bytes,s,t);}
extern "C" awe_status awe_gemm_f64(awe_plan* p,const double* a,const double* b,double* c,void* w,size_t bytes,cudaStream_t s,awe_execution_stats* t){return AWE_NOT_SUPPORTED;}

extern "C" awe_status awe_gemm_i8_i32(awe_plan* p,const int8_t* a,const int8_t* b,int32_t* c,void* w,size_t bytes,cudaStream_t s,awe_execution_stats* t){if(!p||p->d.data_type!=AWE_I8_I32)return AWE_INVALID_ARGUMENT;return awe_plan_execute(p,a,b,c,w,bytes,s,t);}
