#include "awe_backend.h"
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>
#include <cstdlib>
#include <cmath>

namespace f64_impl {
using U64 = unsigned long long;
using U128 = __uint128_t;
struct Failure { awe_status status; };
void cu(cudaError_t s) { if (s != cudaSuccess) throw Failure{AWE_CUDA_ERROR}; }
void lt(cublasStatus_t s) { if (s != CUBLAS_STATUS_SUCCESS) throw Failure{AWE_CUDA_ERROR}; }
size_t align256(size_t n) { return (n+255)&~size_t(255); }
int blocks(size_t n) { return int(std::min<size_t>(4096,(n+255)/256)); }
struct WitnessFace { int coefficient; unsigned char a[241], b[241]; };
struct Witness { int p, faces, safe; WitnessFace f[8]; };
struct Choice { int faces, count; U128 product; int indices[32]; };
#include "tables_generated.h"

struct DeviceFace {
 int p;
 unsigned char a[256],b[256];
 U128 crt[241];
};
struct Event {
 cudaEvent_t e=nullptr;
 explicit Event(bool timing=true) { cu(cudaEventCreateWithFlags(&e,timing?cudaEventDefault:cudaEventDisableTiming)); }
 ~Event(){ if(e) cudaEventDestroy(e); }
};
void record(Event& e,cudaStream_t s){ cu(cudaEventRecord(e.e,s)); }
void wait(cudaStream_t s,Event& e){ cu(cudaStreamWaitEvent(s,e.e,0)); }
float elapsed(Event& a,Event& b){ float ms=0;cu(cudaEventElapsedTime(&ms,a.e,b.e));return ms; }
struct Slot {
 Event available{false},encoded{false},computed{false};
 size_t a=0,b=0,g=0;
};
struct Timing { Event e0,e1,w0,g0,g1,r0,r1; };

__device__ __forceinline__ long long quantize_one(const double* x,int o,int q,int64_t ld,
                                                  bool trans,bool right,bool row,int shift){
 int i=right?q:o,j=right?o:q;if(trans){int t=i;i=j;j=t;}
 double z=x[row?size_t(i)*ld+j:size_t(j)*ld+i];
 return isfinite(z)?__double2ll_rn(ldexp(z,shift)):0ll;
}

__global__ void quantize(const double* x,long long* out,int* exponents,int* invalid,int outer,int k,int64_t ld,bool trans,bool right,bool row,int bits,int store=1){
 __shared__ double maximum[256];int o=blockIdx.x;double v=0;
 for(int q=threadIdx.x;q<k;q+=256){int i=right?q:o,j=right?o:q;if(trans){int t=i;i=j;j=t;}double z=x[row?size_t(i)*ld+j:size_t(j)*ld+i];if(!isfinite(z))atomicExch(invalid,1);else v=fmax(v,fabs(z));}
 maximum[threadIdx.x]=v;__syncthreads();for(int s=128;s;s/=2){if(threadIdx.x<s)maximum[threadIdx.x]=fmax(maximum[threadIdx.x],maximum[threadIdx.x+s]);__syncthreads();}
 int shift=maximum[0]==0?0:bits-1-ilogb(maximum[0]);if(!threadIdx.x)exponents[o]=shift;
 if(!store)return;
 for(int q=threadIdx.x;q<k;q+=256)out[size_t(o)*k+q]=quantize_one(x,o,q,ld,trans,right,row,shift);
}
__global__ void encode_f64(const long long* x,unsigned char* y,int outer,int k,int opad,int kp,int off,int p,const unsigned char* lut){
 size_t count=size_t(opad)*kp/2;
 for(size_t z=size_t(blockIdx.x)*blockDim.x+threadIdx.x;z<count;z+=size_t(blockDim.x)*gridDim.x){int o=z/(kp/2),q=2*(z%(kp/2));unsigned packed=0;
 for(int t=0;t<2;++t)if(o<outer&&q+t<k-off){int r=x[size_t(o)*k+off+q+t]%p;if(r<0)r+=p;packed|=unsigned(lut[r])<<(4*t);}y[z]=packed;}
}
__device__ double round_scaled(U128 x,bool neg,int exponent){
 if(!x)return 0.;U64 hi=x>>64,lo=x;int top=hi?127-__clzll(hi):63-__clzll(lo);int drop=max(0,top-52);U64 mant=x>>drop;
 if(drop){U128 rem=x-((U128)mant<<drop),half=U128(1)<<(drop-1);if(rem>half||(rem==half&&(mant&1)))++mant;}
 double v=ldexp(double(mant),drop+exponent);return neg?-v:v;
}

struct EncPlane { const unsigned char* lut; unsigned char* out; };

struct EncMod   { int p; int nf; int plane0; int c1, c2; unsigned bias, M;
                  const unsigned* packed; };
struct FoldFace { const float* g; int coeff; };
struct FoldMod  { int p; int nf; int face0; unsigned short* t; int d1; unsigned bias2, M; };

#define AWE_CMAX_MOD 32
#define AWE_CMAX_FACE 128
__constant__ EncMod   c_encmod[2*AWE_CMAX_MOD];
__constant__ EncPlane c_encplane[2*AWE_CMAX_FACE];
__constant__ FoldMod  c_foldmod[AWE_CMAX_MOD];
__constant__ FoldFace c_foldface[AWE_CMAX_FACE];

__device__ __forceinline__ int barrett(unsigned t,int p,unsigned M){
 unsigned q=(unsigned)((( unsigned long long)t*M)>>32);
 unsigned r=t-q*(unsigned)p;
 if(r>=(unsigned)p)r-=(unsigned)p;
 return (int)r;
}

__device__ __forceinline__ int reduce_quantized(long long v,const EncMod& m){
 int v0=(int)(v&0x3FFFF),v1=(int)((v>>18)&0x3FFFF);long long v2=v>>36;
 long long t=(long long)v0+(long long)v1*m.c1+v2*m.c2+(long long)m.bias;
 return barrett((unsigned)t,m.p,m.M);
}
struct CrtMod   { const unsigned short* t; const U128* crtw; };

template<int MODE,int FUSE=0>
__global__ void encode_f64_group_t(const long long* x,int outer,int k,int opad,int kp,int off,
                                   int nmod,int mod0,int side,
                                   const EncMod* mods,const EncPlane* planes,
                                   const double* xf=nullptr,const int* shifts=nullptr,
                                   int64_t ld=0,int trans=0,int right=0,int row=0){
 size_t count=size_t(opad)*kp/2;
 for(size_t z=size_t(blockIdx.x)*blockDim.x+threadIdx.x;z<count;z+=size_t(blockDim.x)*gridDim.x){
  int o=z/(kp/2),q=2*(z%(kp/2));
  bool ok0=(o<outer&&q+0<k-off),ok1=(o<outer&&q+1<k-off);
  long long v0,v1;
  if(FUSE){
   const int shift=shifts[o];
   v0=ok0?quantize_one(xf,o,off+q+0,ld,trans!=0,right!=0,row!=0,shift):0;
   v1=ok1?quantize_one(xf,o,off+q+1,ld,trans!=0,right!=0,row!=0,shift):0;
  }else{
   v0=ok0?x[size_t(o)*k+off+q+0]:0;v1=ok1?x[size_t(o)*k+off+q+1]:0;
  }
  for(int m=0;m<nmod;++m){
   const EncMod em = (MODE&2) ? c_encmod[side*AWE_CMAX_MOD+mod0+m] : mods[m];
   int r0=0,r1=0;
   if(ok0)r0=reduce_quantized(v0,em);
   if(ok1)r1=reduce_quantized(v1,em);
   if(MODE&1){
    unsigned w0=ok0?em.packed[r0]:0u, w1=ok1?em.packed[r1]:0u;
    for(int j=0;j<em.nf;++j){
     unsigned char* out = (MODE&2) ? c_encplane[side*AWE_CMAX_FACE+em.plane0+j].out
                                   : planes[em.plane0+j].out;
     out[z]=(unsigned char)(((w0>>(4*j))&15u)|(((w1>>(4*j))&15u)<<4));
    }
   } else {
    for(int j=0;j<em.nf;++j){
     const EncPlane pl = (MODE&2) ? c_encplane[side*AWE_CMAX_FACE+em.plane0+j]
                                  : planes[em.plane0+j];
     unsigned packed=0;
     if(ok0)packed|=unsigned(pl.lut[r0]);
     if(ok1)packed|=unsigned(pl.lut[r1])<<4;
     pl.out[z]=(unsigned char)packed;
    }
   }
  }
 }
}

template<int MODE>
__global__ void fold_mods_t(int nmod,int mod0,const FoldMod* mods,const FoldFace* faces,
                            size_t count,int first_chunk){
 for(size_t z=size_t(blockIdx.x)*blockDim.x+threadIdx.x;z<count;z+=size_t(blockDim.x)*gridDim.x){
  for(int m=0;m<nmod;++m){
   const FoldMod fm = (MODE&2) ? c_foldmod[mod0+m] : mods[m];
   int p=fm.p;long long acc=0;
   for(int j=0;j<fm.nf;++j){
    const FoldFace f = (MODE&2) ? c_foldface[fm.face0+j] : faces[fm.face0+j];
    acc+=(long long)f.coeff*(long long)__float2int_rn(f.g[z]);
   }
   if(!first_chunk)acc+=(long long)fm.t[z];

   int a0=(int)(acc&0xFFFF);long long a1=acc>>16;
   long long tt=(long long)a0+a1*fm.d1+(long long)fm.bias2;
   int t=barrett((unsigned)tt,p,fm.M);
   fm.t[z]=(unsigned short)t;
  }
 }
}

__global__ void crt_store(int nmod,const CrtMod* mods,double* out,int m,int n,int mp,
                          int64_t ld,bool row,U128 P,const int* ea,const int* eb){
 for(size_t z=size_t(blockIdx.x)*blockDim.x+threadIdx.x;z<size_t(m)*n;z+=size_t(blockDim.x)*gridDim.x){
  int i=z%m,j=z/m;size_t at=size_t(i)+size_t(j)*mp;U128 u=0;
  for(int q=0;q<nmod;++q){u+=mods[q].crtw[mods[q].t[at]];if(u>=P)u-=P;}
  bool neg=u>P/2;
  out[row?size_t(i)*ld+j:size_t(j)*ld+i]=round_scaled(neg?P-u:u,neg,-ea[i]-eb[j]);
 }
}
__global__ void update_crt(const float* g,U128* sum,size_t count,const DeviceFace* f,U128 P){
 for(size_t z=size_t(blockIdx.x)*blockDim.x+threadIdx.x;z<count;z+=size_t(blockDim.x)*gridDim.x){int r=__float2int_rn(g[z])%f->p;if(r<0)r+=f->p;U128 s=sum[z]+f->crt[r];if(s>=P)s-=P;sum[z]=s;}
}
__global__ void store_output(const U128* sum,double* out,int m,int n,int mp,int64_t ld,bool row,U128 P,const int* ea,const int* eb){
 for(size_t z=size_t(blockIdx.x)*blockDim.x+threadIdx.x;z<size_t(m)*n;z+=size_t(blockDim.x)*gridDim.x){int i=z%m,j=z/m;U128 u=sum?sum[i+size_t(j)*mp]:0;bool neg=u>P/2;out[row?size_t(i)*ld+j:size_t(j)*ld+i]=round_scaled(neg?P-u:u,neg,sum?-ea[i]-eb[j]:0);}
}
U128 multiply_mod(U128 a,int b,U128 p){U128 s=0;for(;b;b>>=1){if(b&1){s+=a;if(s>=p)s-=p;}a+=a;if(a>=p)a-=p;}return s;}

struct Engine {
 awe_gemm_desc d{};awe_options o{};awe_plan_info info{};
 U128 product=1;int bits=53;size_t qa_offset=0,qb_offset=0,ea_offset=0,eb_offset=0,bad_offset=0;int* host_bad=nullptr;int mp=0,np=0,kp=0,face_count=0;bool state32=true;
 size_t bytes=0,count=0,table_offset=0,sum_offset=0,scale_offset=0,scratch_offset=0,scale_bytes=0;
 DeviceFace* host_faces=nullptr;

 struct HostGroup { int p,first,nf; };
 std::vector<HostGroup> groups;
 std::vector<int> face_coeff;
 int r1=1,group_mods=4,max_planes=0,encmode=0,fuseq=0;
 size_t encmod_offset=0,encplane_offset=0,foldmod_offset=0,foldface_offset=0,
        crtmod_offset=0,crtw_offset=0,packed_offset=0,plane_a_offset=0,plane_b_offset=0,
        plane_g_offset=0,resid_offset=0;
 U128* host_crtw=nullptr;
 EncMod* host_encmod=nullptr; unsigned* host_packed=nullptr; EncPlane* host_encplane=nullptr;
 FoldMod* host_foldmod=nullptr; FoldFace* host_foldface=nullptr; CrtMod* host_crtmod=nullptr;
 cublasLtHandle_t handle=nullptr;cublasLtMatmulDesc_t op=nullptr;
 cublasLtMatrixLayout_t la=nullptr,lb=nullptr,lc=nullptr;cublasLtMatmulAlgo_t algo{};
 cudaStream_t streams[3]{};
 std::vector<std::unique_ptr<Slot>> slots;
 std::vector<std::unique_ptr<Timing>> timings;
 std::unique_ptr<Event> begin,end,ready,setup_end,output_begin;
 ~Engine(){
  for(auto s:streams)if(s)cudaStreamSynchronize(s);
  for(auto s:streams)if(s)cudaStreamDestroy(s);
  if(host_faces)cudaFreeHost(host_faces);if(host_bad)cudaFreeHost(host_bad);
  if(host_packed)cudaFreeHost(host_packed);if(host_crtw)cudaFreeHost(host_crtw);if(host_encmod)cudaFreeHost(host_encmod);
  if(host_encplane)cudaFreeHost(host_encplane);if(host_foldmod)cudaFreeHost(host_foldmod);
  if(host_foldface)cudaFreeHost(host_foldface);if(host_crtmod)cudaFreeHost(host_crtmod);
  if(la)cublasLtMatrixLayoutDestroy(la);if(lb)cublasLtMatrixLayoutDestroy(lb);
  if(lc)cublasLtMatrixLayoutDestroy(lc);if(op)cublasLtMatmulDescDestroy(op);
  if(handle)cublasLtDestroy(handle);
 }
 size_t reserve(size_t n){size_t offset=bytes; if(n>SIZE_MAX-255-bytes)throw Failure{AWE_INVALID_ARGUMENT};bytes+=align256(n);return offset;}
 void init(const awe_gemm_desc& desc,const awe_options& options){
  d=desc;o=options;info.struct_size=sizeof(info);info.method=AWE_METHOD_OZAKI2;
  info.stream_count=o.stream_count;info.algorithm_id=-1;info.input_min=-128;info.input_max=127;
  if(d.data_type!=AWE_F64_F64)throw Failure{AWE_NOT_SUPPORTED};
  if(d.m<0||d.n<0||d.k<0||d.m>(1<<20)||d.n>(1<<20)||d.k>INT_MAX ||
     (o.stream_count!=1&&o.stream_count!=3)||o.buffer_count<1||o.buffer_count>8||
     o.k_chunk<0||o.heuristic_index<0||o.heuristic_index>=16)throw Failure{AWE_INVALID_ARGUMENT};
  if(d.alpha!=1||d.beta!=0||(d.layout!=AWE_ROW_MAJOR&&d.layout!=AWE_COLUMN_MAJOR))throw Failure{AWE_NOT_SUPPORTED};
  if(d.data_type==AWE_I8_I32&&d.k>131071)throw Failure{AWE_INPUT_OUT_OF_RANGE};
  if(d.trans_a<AWE_OP_N||d.trans_a>AWE_OP_T||d.trans_b<AWE_OP_N||d.trans_b>AWE_OP_T)throw Failure{AWE_INVALID_ARGUMENT};
  const bool row=d.layout==AWE_ROW_MAJOR;
  auto minld=[&](int64_t r,int64_t c){return std::max<int64_t>(1,row?c:r);};
  if(d.lda<minld(d.trans_a==AWE_OP_N?d.m:d.k,d.trans_a==AWE_OP_N?d.k:d.m)||
     d.ldb<minld(d.trans_b==AWE_OP_N?d.k:d.n,d.trans_b==AWE_OP_N?d.n:d.k)||
     d.ldc<minld(d.m,d.n))throw Failure{AWE_INVALID_ARGUMENT};
  mp=(int(d.m)+127)/128*128;np=(int(d.n)+127)/128*128;
  info.padded_m=mp;info.padded_n=np;
  if(!d.m||!d.n)return;
  for(int i=0;i<o.stream_count;++i)cu(cudaStreamCreateWithFlags(&streams[i],cudaStreamNonBlocking));
  begin.reset(new Event);end.reset(new Event);ready.reset(new Event(false));
  setup_end.reset(new Event);output_begin.reset(new Event);
  if(!d.k)return;
  const char* setting=std::getenv("AWE_FP64_BITS");if(setting){char* end=nullptr;long q=strtol(setting,&end,10);if(!end||*end||q<8||q>55)throw Failure{AWE_INVALID_ARGUMENT};bits=int(q);}

  const char* r1env=std::getenv("AWE_F64_R1");if(r1env)r1=std::atoi(r1env)?1:0;
  const char* genv=std::getenv("AWE_F64_GROUP");if(genv){int g=std::atoi(genv);if(g>0)group_mods=g;}

  const char* fenv=std::getenv("AWE_F64_FUSEQ");if(fenv)fuseq=std::atoi(fenv)?1:0;
  const char* eenv=std::getenv("AWE_F64_ENCMODE");
  if(eenv){
   int v=std::atoi(eenv);
   const char* allow=std::getenv("AWE_F64_UNSAFE_CONSTANT");
   if(v>=2&&!(allow&&std::atoi(allow))) v=v&1;
   if(v>=0&&v<=3)encmode=v;
  }
  if(2*bits+1+(64-__builtin_clzll(d.k))>=127)throw Failure{AWE_NOT_SUPPORTED};
  const Choice* chosen=nullptr;U128 needed=U128(d.k)<<(2*bits+1);
  for(auto& c:choices)if(c.product>needed){chosen=&c;break;}
  if(!chosen)throw Failure{AWE_NOT_SUPPORTED};
  product=chosen->product;face_count=chosen->faces;state32=false;
  int safe=INT_MAX;for(int i=0;i<chosen->count;++i)safe=std::min(safe,witnesses[chosen->indices[i]].safe);
  safe=safe/128*128;info.guaranteed_k=safe;
  if(o.k_chunk>safe)throw Failure{AWE_INVALID_ARGUMENT};
  int chunk=o.k_chunk?o.k_chunk:std::min(16384,safe);
  kp=(int(std::min<int64_t>(d.k,chunk))+127)/128*128;
  info.face_count=face_count;info.payout_period=kp;info.padded_k=kp;
  count=size_t(mp)*np;
  cu(cudaMallocHost(&host_faces,size_t(face_count)*sizeof(DeviceFace)));
  std::memset(host_faces,0,size_t(face_count)*sizeof(DeviceFace));
  int at=0;
  for(int i=0;i<chosen->count;++i){
   const auto& w=witnesses[chosen->indices[i]];U128 base=product/w.p;
   int inv=1;while((base%w.p)*inv%w.p!=1)++inv;
   U128 weight=base*inv;
   for(int f=0;f<w.faces;++f){
    auto& out=host_faces[at];face_coeff.push_back(w.f[f].coefficient);++at;out.p=w.p;
    for(int r=0;r<w.p;++r){out.a[r]=w.f[f].a[r];out.b[r]=w.f[f].b[r];}
    for(int r=0;r<w.p;++r){int v=(w.f[f].coefficient*r)%w.p;if(v<0)v+=w.p;
     out.crt[r]=multiply_mod(weight,v,product);
    }
   }
  }

  {
   int at2=0;
   for(int i=0;i<chosen->count;++i){
    const auto& w=witnesses[chosen->indices[i]];
    groups.push_back(HostGroup{w.p,at2,w.faces});at2+=w.faces;
   }
   if(at2!=face_count)throw Failure{AWE_NOT_SUPPORTED};
   if(group_mods>int(groups.size()))group_mods=int(groups.size());
   for(size_t g=0;g<groups.size();g+=group_mods){
    int planes=0;for(size_t j=g;j<groups.size()&&j<g+group_mods;++j)planes+=groups[j].nf;
    max_planes=std::max(max_planes,planes);
   }
   const size_t nmod=groups.size();
   cu(cudaMallocHost(&host_crtw,nmod*241*sizeof(U128)));
   std::memset(host_crtw,0,nmod*241*sizeof(U128));
   for(size_t q=0;q<nmod;++q){
    U128 base=product/groups[q].p;int inv=1;while((base%groups[q].p)*inv%groups[q].p!=1)++inv;
    U128 weight=base*inv;
    for(int t=0;t<groups[q].p;++t)host_crtw[q*241+t]=multiply_mod(weight,t,product);
   }
   cu(cudaMallocHost(&host_encmod,2*nmod*sizeof(EncMod)));
   cu(cudaMallocHost(&host_packed,2*nmod*241*sizeof(unsigned)));
   std::memset(host_packed,0,2*nmod*241*sizeof(unsigned));
   for(size_t q=0;q<nmod;++q){
    const auto& G=groups[q];
    if(G.nf>8)throw Failure{AWE_NOT_SUPPORTED};
    if(nmod>AWE_CMAX_MOD||face_count>AWE_CMAX_FACE)throw Failure{AWE_NOT_SUPPORTED};
    for(int r=0;r<G.p;++r){
     unsigned wa=0,wb=0;
     for(int j=0;j<G.nf;++j){
      wa|=unsigned(host_faces[G.first+j].a[r]&15)<<(4*j);
      wb|=unsigned(host_faces[G.first+j].b[r]&15)<<(4*j);
     }
     host_packed[(0*nmod+q)*241+r]=wa; host_packed[(1*nmod+q)*241+r]=wb;
    }
   }
   cu(cudaMallocHost(&host_encplane,2*size_t(face_count)*sizeof(EncPlane)));
   cu(cudaMallocHost(&host_foldmod,nmod*sizeof(FoldMod)));
   cu(cudaMallocHost(&host_foldface,size_t(face_count)*sizeof(FoldFace)));
   cu(cudaMallocHost(&host_crtmod,nmod*sizeof(CrtMod)));
  }
  table_offset=reserve(size_t(face_count)*sizeof(DeviceFace));sum_offset=reserve(count*16);
  qa_offset=reserve(size_t(d.m)*d.k*8);qb_offset=reserve(size_t(d.n)*d.k*8);ea_offset=reserve(size_t(d.m)*4);eb_offset=reserve(size_t(d.n)*4);bad_offset=reserve(4);cu(cudaMallocHost(&host_bad,4));
  scale_bytes=size_t(std::max(mp,np))*((kp+63)/64)*4;scale_offset=reserve(scale_bytes);
  scratch_offset=reserve(o.matmul_workspace_bytes);
  for(int i=0;i<o.buffer_count;++i){auto s=std::make_unique<Slot>();s->a=reserve(size_t(mp)*kp/2);s->b=reserve(size_t(np)*kp/2);s->g=reserve(count*4);slots.push_back(std::move(s));}
  if(r1){
   const size_t nmod=groups.size();
   encmod_offset=reserve(2*nmod*sizeof(EncMod));
   packed_offset=reserve(2*nmod*241*sizeof(unsigned));
   encplane_offset=reserve(2*size_t(face_count)*sizeof(EncPlane));
   foldmod_offset=reserve(nmod*sizeof(FoldMod));
   foldface_offset=reserve(size_t(face_count)*sizeof(FoldFace));
   crtmod_offset=reserve(nmod*sizeof(CrtMod));
   crtw_offset=reserve(nmod*241*sizeof(U128));
   plane_a_offset=reserve(size_t(face_count)*(size_t(mp)*kp/2));
   plane_b_offset=reserve(size_t(face_count)*(size_t(np)*kp/2));
   plane_g_offset=reserve(size_t(face_count)*count*4);
   resid_offset=reserve(nmod*count*2);
  }
  info.workspace_bytes=bytes;
  lt(cublasLtCreate(&handle));lt(cublasLtMatmulDescCreate(&op,CUBLAS_COMPUTE_32F,CUDA_R_32F));
  cublasOperation_t t=CUBLAS_OP_T,n=CUBLAS_OP_N;auto scale=CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
  lt(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSA,&t,sizeof(t)));
  lt(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSB,&n,sizeof(n)));
  lt(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_A_SCALE_MODE,&scale,sizeof(scale)));
  lt(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_B_SCALE_MODE,&scale,sizeof(scale)));
  lt(cublasLtMatrixLayoutCreate(&la,CUDA_R_4F_E2M1,kp,mp,kp));
  lt(cublasLtMatrixLayoutCreate(&lb,CUDA_R_4F_E2M1,kp,np,kp));
  lt(cublasLtMatrixLayoutCreate(&lc,CUDA_R_32F,mp,np,mp));

  struct QueryStorage {void* p=nullptr;QueryStorage(){cu(cudaMalloc(&p,256));}~QueryStorage(){if(p)cudaFree(p);}} query;
  lt(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,&query.p,sizeof(void*)));
  lt(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,&query.p,sizeof(void*)));
  cublasLtMatmulPreference_t pref=nullptr;lt(cublasLtMatmulPreferenceCreate(&pref));
  auto status=cublasLtMatmulPreferenceSetAttribute(pref,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&o.matmul_workspace_bytes,sizeof(size_t));
  if(status!=CUBLAS_STATUS_SUCCESS){cublasLtMatmulPreferenceDestroy(pref);lt(status);}
  cublasLtMatmulHeuristicResult_t candidates[16]{};int number=0;
  status=cublasLtMatmulAlgoGetHeuristic(handle,op,la,lb,lc,lc,pref,16,candidates,&number);
  cublasLtMatmulPreferenceDestroy(pref);lt(status);bool found=false;int valid=0;
  for(int i=0;i<number;++i)if(candidates[i].state==CUBLAS_STATUS_SUCCESS&&valid++==o.heuristic_index){algo=candidates[i].algo;found=true;break;}
  if(!found)throw Failure{AWE_NOT_SUPPORTED};
  size_t written=0;lt(cublasLtMatmulAlgoConfigGetAttribute(&algo,CUBLASLT_ALGO_CONFIG_ID,&info.algorithm_id,sizeof(int),&written));
  if(o.profiling){size_t per=r1?((groups.size()+group_mods-1)/group_mods):size_t(face_count);size_t steps=((size_t(d.k)+kp-1)/kp)*per;for(size_t i=0;i<steps;++i)timings.emplace_back(new Timing);}
 }
 size_t extent(int64_t r,int64_t c,int64_t ld,size_t item)const{
  if(d.layout==AWE_ROW_MAJOR)std::swap(r,c);
  if(!r||!c)return 0;
  __uint128_t n=(__uint128_t(c-1)*ld+r)*item;
  if(n>SIZE_MAX)throw Failure{AWE_INVALID_ARGUMENT};return size_t(n);
 }
 bool overlaps(const void* a,size_t an,const void* b,size_t bn)const{
  uintptr_t aa=reinterpret_cast<uintptr_t>(a),bb=reinterpret_cast<uintptr_t>(b);
  if(an>UINTPTR_MAX-aa||bn>UINTPTR_MAX-bb)throw Failure{AWE_INVALID_ARGUMENT};
  return an&&bn&&aa<bb+bn&&bb<aa+an;
 }
 void execute(const void* a,const void* b,void* c,void* work,size_t size,cudaStream_t caller,awe_execution_stats* stats){
  if(stats){*stats={};stats->struct_size=sizeof(*stats);stats->face_count=face_count;stats->chunk_count=kp?int((d.k+kp-1)/kp):0;}
  if(!d.m||!d.n)return;
  if(!c||(d.k&&(!a||!b)))throw Failure{AWE_INVALID_ARGUMENT};
  if(size<bytes)throw Failure{AWE_INSUFFICIENT_WORKSPACE};
  if(bytes&&(!work||reinterpret_cast<uintptr_t>(work)%256))throw Failure{AWE_INVALID_ARGUMENT};
  size_t an=extent(d.trans_a==AWE_OP_N?d.m:d.k,d.trans_a==AWE_OP_N?d.k:d.m,d.lda,8);
  size_t bn=extent(d.trans_b==AWE_OP_N?d.k:d.n,d.trans_b==AWE_OP_N?d.n:d.k,d.ldb,8);
  size_t cn=extent(d.m,d.n,d.ldc,d.data_type==AWE_I8_I32?4:8);
  if(overlaps(a,an,c,cn)||overlaps(b,bn,c,cn)||overlaps(work,bytes,a,an)||overlaps(work,bytes,b,bn)||overlaps(work,bytes,c,cn))throw Failure{AWE_INVALID_ARGUMENT};
  cudaStreamCaptureStatus capturing;cu(cudaStreamIsCapturing(caller,&capturing));
  if(capturing!=cudaStreamCaptureStatusNone)throw Failure{AWE_NOT_SUPPORTED};
  auto es=streams[0],gs=o.stream_count==3?streams[1]:es,rs=o.stream_count==3?streams[2]:es;
  record(*begin,caller);wait(es,*begin);wait(gs,*begin);wait(rs,*begin);
  bool row=d.layout==AWE_ROW_MAJOR;auto raw=static_cast<unsigned char*>(work);
  void* sum=d.k?raw+sum_offset:nullptr;
  if(d.k){
   auto faces=reinterpret_cast<DeviceFace*>(raw+table_offset);void* scales=raw+scale_offset;
   lt(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,&scales,sizeof(scales)));
   lt(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,&scales,sizeof(scales)));
   cu(cudaMemcpyAsync(faces,host_faces,size_t(face_count)*sizeof(DeviceFace),cudaMemcpyHostToDevice,es));
   cu(cudaMemsetAsync(scales,0x40,scale_bytes,es));
   cu(cudaMemsetAsync(raw+bad_offset,0,4,es));

   const int qstore=(r1&&fuseq)?0:1;
   quantize<<<d.m,256,0,es>>>((const double*)a,(long long*)(raw+qa_offset),(int*)(raw+ea_offset),(int*)(raw+bad_offset),d.m,d.k,d.lda,d.trans_a==AWE_OP_T,false,row,bits,qstore);
   quantize<<<d.n,256,0,es>>>((const double*)b,(long long*)(raw+qb_offset),(int*)(raw+eb_offset),(int*)(raw+bad_offset),d.n,d.k,d.ldb,d.trans_b==AWE_OP_T,true,row,bits,qstore);
   cu(cudaMemcpyAsync(host_bad,raw+bad_offset,4,cudaMemcpyDeviceToHost,es));record(*setup_end,es);cu(cudaEventSynchronize(setup_end->e));if(*host_bad)throw Failure{AWE_INPUT_OUT_OF_RANGE};
   if(r1){

    const size_t nmod=groups.size();
    auto dmod=reinterpret_cast<EncMod*>(raw+encmod_offset);
    auto dplane=reinterpret_cast<EncPlane*>(raw+encplane_offset);
    auto dfmod=reinterpret_cast<FoldMod*>(raw+foldmod_offset);
    auto dfface=reinterpret_cast<FoldFace*>(raw+foldface_offset);
    auto dcmod=reinterpret_cast<CrtMod*>(raw+crtmod_offset);
    auto dcrtw=reinterpret_cast<U128*>(raw+crtw_offset);
    unsigned char* pa=raw+plane_a_offset;unsigned char* pb=raw+plane_b_offset;
    auto pg=reinterpret_cast<float*>(raw+plane_g_offset);
    auto pr=reinterpret_cast<unsigned short*>(raw+resid_offset);
    const size_t abytes=size_t(mp)*kp/2,bbytes=size_t(np)*kp/2;
    for(size_t q=0;q<nmod;++q){
     {
      const int gp=groups[q].p;
      const unsigned Mq=unsigned((1ull<<32)/unsigned(gp));
      const int c1=int((1u<<18)%unsigned(gp)),c2=int((1ull<<36)%(unsigned long long)gp);
      const int d1=int((1u<<16)%unsigned(gp));

      const unsigned bias =unsigned(gp)*(((1u<<28)+unsigned(gp)-1)/unsigned(gp)+1);
      const unsigned bias2=unsigned(gp)*(((1u<<28)+unsigned(gp)-1)/unsigned(gp)+1);
      {auto dpack=reinterpret_cast<unsigned*>(raw+packed_offset);
       host_encmod[q]     =EncMod{gp,groups[q].nf,groups[q].first,c1,c2,bias,Mq,dpack+(0*nmod+q)*241};
       host_encmod[nmod+q]=EncMod{gp,groups[q].nf,groups[q].first,c1,c2,bias,Mq,dpack+(1*nmod+q)*241};}
      host_foldmod[q]=FoldMod{gp,groups[q].nf,groups[q].first,pr+q*count,d1,bias2,Mq};
     }
     host_crtmod[q]=CrtMod{pr+q*count,dcrtw+q*241};
    }
    for(int f=0;f<face_count;++f){
     host_encplane[f]=EncPlane{faces[f].a,pa+size_t(f)*abytes};
     host_encplane[face_count+f]=EncPlane{faces[f].b,pb+size_t(f)*bbytes};
     host_foldface[f]=FoldFace{pg+size_t(f)*count,face_coeff[f]};
    }
    cu(cudaMemcpyAsync(dmod,host_encmod,2*nmod*sizeof(EncMod),cudaMemcpyHostToDevice,es));
    cu(cudaMemcpyAsync(raw+packed_offset,host_packed,2*nmod*241*sizeof(unsigned),cudaMemcpyHostToDevice,es));
    if(encmode&2){

     cu(cudaMemcpyToSymbolAsync(c_encmod,host_encmod,nmod*sizeof(EncMod),0,cudaMemcpyHostToDevice,es));
     cu(cudaMemcpyToSymbolAsync(c_encmod,host_encmod+nmod,nmod*sizeof(EncMod),
        AWE_CMAX_MOD*sizeof(EncMod),cudaMemcpyHostToDevice,es));
     cu(cudaMemcpyToSymbolAsync(c_encplane,host_encplane,size_t(face_count)*sizeof(EncPlane),0,
        cudaMemcpyHostToDevice,es));
     cu(cudaMemcpyToSymbolAsync(c_encplane,host_encplane+face_count,size_t(face_count)*sizeof(EncPlane),
        AWE_CMAX_FACE*sizeof(EncPlane),cudaMemcpyHostToDevice,es));
     cu(cudaMemcpyToSymbolAsync(c_foldmod,host_foldmod,nmod*sizeof(FoldMod),0,cudaMemcpyHostToDevice,es));
     cu(cudaMemcpyToSymbolAsync(c_foldface,host_foldface,size_t(face_count)*sizeof(FoldFace),0,
        cudaMemcpyHostToDevice,es));
    }
    cu(cudaMemcpyAsync(dplane,host_encplane,2*size_t(face_count)*sizeof(EncPlane),cudaMemcpyHostToDevice,es));
    cu(cudaMemcpyAsync(dfmod,host_foldmod,nmod*sizeof(FoldMod),cudaMemcpyHostToDevice,es));
    cu(cudaMemcpyAsync(dfface,host_foldface,size_t(face_count)*sizeof(FoldFace),cudaMemcpyHostToDevice,es));
    cu(cudaMemcpyAsync(dcmod,host_crtmod,nmod*sizeof(CrtMod),cudaMemcpyHostToDevice,es));
    cu(cudaMemcpyAsync(dcrtw,host_crtw,nmod*241*sizeof(U128),cudaMemcpyHostToDevice,es));
    record(*ready,es);wait(gs,*ready);wait(rs,*ready);
    Event pass_done(false);size_t serial=0;bool first_chunk=true;
    for(int64_t off=0;off<d.k;off+=kp){
     for(size_t g0=0;g0<nmod;g0+=group_mods){
      int ng=int(std::min<size_t>(group_mods,nmod-g0));
      Timing* t=o.profiling&&serial<timings.size()?timings[serial].get():nullptr;
      if(t)record(t->e0,es);
      {
       const int ga=blocks(size_t(mp)*kp/2),gb=blocks(size_t(np)*kp/2),m0=int(g0);
       auto A=(long long*)(raw+qa_offset);auto B=(long long*)(raw+qb_offset);
       auto EA=(const int*)(raw+ea_offset);auto EB=(const int*)(raw+eb_offset);
#define AWE_ENC_LAUNCH(M) \
       encode_f64_group_t<M,0><<<ga,256,0,es>>>(A,d.m,d.k,mp,kp,int(off),ng,m0,0,dmod+g0,dplane); \
       encode_f64_group_t<M,0><<<gb,256,0,es>>>(B,d.n,d.k,np,kp,int(off),ng,m0,1,dmod+nmod+g0,dplane+face_count)
#define AWE_ENC_LAUNCH_FUSED(M) \
       encode_f64_group_t<M,1><<<ga,256,0,es>>>(nullptr,d.m,d.k,mp,kp,int(off),ng,m0,0,dmod+g0,dplane, \
         (const double*)a,EA,d.lda,d.trans_a==AWE_OP_T,0,row); \
       encode_f64_group_t<M,1><<<gb,256,0,es>>>(nullptr,d.n,d.k,np,kp,int(off),ng,m0,1,dmod+nmod+g0,dplane+face_count, \
         (const double*)b,EB,d.ldb,d.trans_b==AWE_OP_T,1,row)
       if(fuseq){
        switch(encmode){case 1:AWE_ENC_LAUNCH_FUSED(1);break;case 2:AWE_ENC_LAUNCH_FUSED(2);break;
                        case 3:AWE_ENC_LAUNCH_FUSED(3);break;default:AWE_ENC_LAUNCH_FUSED(0);}
       }else{
        switch(encmode){case 1:AWE_ENC_LAUNCH(1);break;case 2:AWE_ENC_LAUNCH(2);break;
                        case 3:AWE_ENC_LAUNCH(3);break;default:AWE_ENC_LAUNCH(0);}
       }
#undef AWE_ENC_LAUNCH
#undef AWE_ENC_LAUNCH_FUSED
      }
      if(t)record(t->e1,es);
      Event encoded(false);record(encoded,es);wait(gs,encoded);
      if(t)record(t->w0,gs);if(t)record(t->g0,gs);
      float one=1,zero=0;
      for(int j=0;j<ng;++j)for(int f2=0;f2<groups[g0+j].nf;++f2){
       int fi=groups[g0+j].first+f2;
       lt(cublasLtMatmul(handle,op,&one,pa+size_t(fi)*abytes,la,pb+size_t(fi)*bbytes,lb,&zero,
          pg+size_t(fi)*count,lc,pg+size_t(fi)*count,lc,&algo,raw+scratch_offset,o.matmul_workspace_bytes,gs));
      }
      if(t)record(t->g1,gs);
      Event computed(false);record(computed,gs);wait(rs,computed);
      if(t)record(t->r0,rs);
      {const int fc2=first_chunk?1:0;const int m0=int(g0);
#define AWE_FOLD_LAUNCH(M) fold_mods_t<M><<<blocks(count),256,0,rs>>>(ng,m0,dfmod+g0,dfface,count,fc2)
       switch(encmode){case 2:AWE_FOLD_LAUNCH(2);break;case 3:AWE_FOLD_LAUNCH(3);break;
                       default:AWE_FOLD_LAUNCH(0);}
#undef AWE_FOLD_LAUNCH
      }
      if(t)record(t->r1,rs);
      record(pass_done,rs);wait(es,pass_done);wait(gs,pass_done);
      ++serial;
     }
     first_chunk=false;
    }
    record(*output_begin,rs);
    crt_store<<<blocks(size_t(d.m)*d.n),256,0,rs>>>(int(nmod),dcmod,(double*)c,d.m,d.n,mp,
      d.ldc,row,product,(int*)(raw+ea_offset),(int*)(raw+eb_offset));
    cu(cudaGetLastError());record(*end,rs);wait(caller,*end);cu(cudaEventSynchronize(end->e));
    if(stats){
     stats->total_ms=elapsed(*begin,*end);stats->reconstruction_ms=elapsed(*output_begin,*end);
     stats->selection_ms=elapsed(*begin,*setup_end);

     stats->wait_ms=elapsed(*output_begin,*end);
     if(o.profiling){stats->encode_ms=0;
      for(size_t i=0;i<serial&&i<timings.size();++i){auto& tt=timings[i];
       stats->encode_ms+=elapsed(tt->e0,tt->e1);stats->gemm_ms+=elapsed(tt->g0,tt->g1);
       stats->reconstruction_ms+=elapsed(tt->r0,tt->r1);stats->wait_ms+=elapsed(tt->w0,tt->g0);}
     }
    }
    return;
   }
   cu(cudaMemsetAsync(sum,0,count*16,rs));record(*ready,rs);wait(es,*ready);
   for(auto& slot:slots)record(slot->available,rs);
   size_t serial=0;
   for(int64_t off=0;off<d.k;off+=kp)for(int f=0;f<face_count;++f,++serial){
    auto& slot=*slots[serial%slots.size()];Timing* t=o.profiling?timings[serial].get():nullptr;
    wait(es,slot.available);if(t)record(t->e0,es);
    encode_f64<<<blocks(size_t(mp)*kp/2),256,0,es>>>((long long*)(raw+qa_offset),raw+slot.a,d.m,d.k,mp,kp,off,host_faces[f].p,faces[f].a);
    encode_f64<<<blocks(size_t(np)*kp/2),256,0,es>>>((long long*)(raw+qb_offset),raw+slot.b,d.n,d.k,np,kp,off,host_faces[f].p,faces[f].b);
    if(t)record(t->e1,es);record(slot.encoded,es);if(t)record(t->w0,gs);
    wait(gs,slot.encoded);if(t)record(t->g0,gs);float one=1,zero=0;
    lt(cublasLtMatmul(handle,op,&one,raw+slot.a,la,raw+slot.b,lb,&zero,raw+slot.g,lc,raw+slot.g,lc,&algo,raw+scratch_offset,o.matmul_workspace_bytes,gs));
    if(t)record(t->g1,gs);record(slot.computed,gs);wait(rs,slot.computed);if(t)record(t->r0,rs);
    update_crt<<<blocks(count),256,0,rs>>>((float*)(raw+slot.g),(U128*)sum,count,faces+f,product);
    if(t)record(t->r1,rs);record(slot.available,rs);
   }
  }
  record(*output_begin,rs);
  store_output<<<blocks(size_t(d.m)*d.n),256,0,rs>>>((U128*)sum,(double*)c,d.m,d.n,mp,d.ldc,row,product,(int*)(raw+ea_offset),(int*)(raw+eb_offset));
  cu(cudaGetLastError());record(*end,rs);wait(caller,*end);cu(cudaEventSynchronize(end->e));
  if(stats){
   stats->total_ms=elapsed(*begin,*end);stats->reconstruction_ms=elapsed(*output_begin,*end);

   stats->selection_ms=d.k?elapsed(*begin,*setup_end):0;
   if(o.profiling&&d.k){stats->encode_ms=0;
    for(auto& t:timings){stats->encode_ms+=elapsed(t->e0,t->e1);stats->gemm_ms+=elapsed(t->g0,t->g1);stats->reconstruction_ms+=elapsed(t->r0,t->r1);stats->wait_ms+=elapsed(t->w0,t->g0);}
   }
  }
 }
 void drain(){for(auto s:streams)if(s)cudaStreamSynchronize(s);}
};
}

