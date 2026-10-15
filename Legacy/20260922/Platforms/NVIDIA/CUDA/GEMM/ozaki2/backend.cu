#include "awe_backend.h"
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

namespace {
using U64 = unsigned long long;
struct Failure { awe_status status; };
void cu(cudaError_t s) { if (s != cudaSuccess) throw Failure{AWE_CUDA_ERROR}; }
void lt(cublasStatus_t s) { if (s != CUBLAS_STATUS_SUCCESS) throw Failure{AWE_CUDA_ERROR}; }
size_t align256(size_t n) { return (n+255)&~size_t(255); }
int blocks(size_t n) { return int(std::min<size_t>(4096,(n+255)/256)); }
struct WitnessFace { int coefficient; unsigned char a[241], b[241]; };
struct Witness { int p, faces, safe; WitnessFace f[4]; };
struct Choice { int faces, count; U64 product; int indices[32]; };
#include "tables_generated.h"

struct DeviceFace {
 int p; unsigned reciprocal;
 unsigned char a[256],b[256]; // Indexed directly by the original signed INT8 byte.
 U64 crt[241];
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

__global__ void encode_i8(const int8_t* x,unsigned char* y,int outer,int k,int opad,
                         int kp,int off,int64_t ld,bool transpose,bool right,
                         bool rowmajor,const unsigned char* lut) {
 size_t count=size_t(opad)*kp/2;
 for(size_t z=size_t(blockIdx.x)*blockDim.x+threadIdx.x;z<count;z+=size_t(blockDim.x)*gridDim.x){
  int o=int(z/(kp/2)),q=2*int(z%(kp/2));unsigned packed=0;
  for(int t=0;t<2;++t)if(o<outer && q+t<k-off){
   int i=right?off+q+t:o,j=right?o:off+q+t;
   if(transpose){int tmp=i;i=j;j=tmp;}
   size_t at=rowmajor?size_t(i)*ld+j:size_t(j)*ld+i;
   packed|=unsigned(lut[static_cast<unsigned char>(x[at])])<<(4*t);
  }
  y[z]=static_cast<unsigned char>(packed);
 }
}
// Coalesced read/transpose/pack pattern shared with the core encoder.
__global__ void encode_transposed_i8(const int8_t* x,unsigned char* y,int outer,int k,
                                    int opad,int kp,int off,int64_t ld,const unsigned char* lut){
 __shared__ unsigned char tile[32][33];__shared__ unsigned char cache[256];
 int tx=threadIdx.x,ty=threadIdx.y;cache[ty*32+tx]=lut[ty*32+tx];__syncthreads();
 int kt=kp/32;size_t tiles=size_t(opad/32)*kt;
 for(size_t t=blockIdx.x;t<tiles;t+=gridDim.x){int rr=int(t/kt)*32,qq=int(t%kt)*32;
  for(int j=0;j<32;j+=8){int r=rr+tx,q=qq+ty+j;tile[ty+j][tx]=(r<outer&&q<k-off)?static_cast<unsigned char>(x[r+int64_t(off+q)*ld]):0;}
  __syncthreads();
  for(int j=0;j<2;++j){int r=ty+8*(tx/16)+j*16,q=2*(tx%16);y[size_t(rr+r)*(kp/2)+(qq+q)/2]=cache[tile[q][r]]|(cache[tile[q+1][r]]<<4);}
  __syncthreads();
 }
}
void encode_dispatch(const int8_t* x,unsigned char* y,int outer,int k,int opad,int kp,
                     int off,int64_t ld,bool trans,bool right,bool row,const unsigned char* lut,cudaStream_t stream){
 if(right?(trans!=row):(trans==row))
  encode_transposed_i8<<<std::min<size_t>(4096,size_t(opad/32)*(kp/32)),dim3(32,8),0,stream>>>(x,y,outer,k,opad,kp,off,ld,lut);
 else encode_i8<<<blocks(size_t(opad)*kp/2),256,0,stream>>>(x,y,outer,k,opad,kp,off,ld,trans,right,row,lut);
}
// One original input read produces all eleven packed planes for contiguous K.
template<int F> __global__ void encode_all(const int8_t* x,unsigned char* y,int outer,int k,int opad,int kp,int off,int64_t ld,const DeviceFace* faces,bool right){
 __shared__ U64 lut[256];int tx=threadIdx.x;U64 codes=0;
 #pragma unroll
 for(int f=0;f<F;++f)codes|=U64(right?faces[f].b[tx]:faces[f].a[tx])<<(4*f);
 lut[tx]=codes;__syncthreads();size_t per=size_t(opad)*kp/8;
 for(size_t z=size_t(blockIdx.x)*256+tx;z<per;z+=size_t(gridDim.x)*256){int o=z/(kp/8),q=8*(z%(kp/8));U64 v[8]{};
  if(o<outer){
   if(q+7<k-off && !(ld%8) && !(reinterpret_cast<uintptr_t>(x)%8)){U64 raw=*reinterpret_cast<const U64*>(x+size_t(o)*ld+off+q);
    #pragma unroll
    for(int t=0;t<8;++t)v[t]=lut[(raw>>(8*t))&255];
   }else{
    #pragma unroll
    for(int t=0;t<8;++t)if(q+t<k-off)v[t]=lut[static_cast<unsigned char>(x[size_t(o)*ld+off+q+t])];
   }
  }
  #pragma unroll
  for(int f=0;f<F;++f){unsigned packed=0;
   #pragma unroll
   for(int t=0;t<8;++t)packed|=unsigned((v[t]>>(4*f))&15)<<(4*t);
   reinterpret_cast<unsigned*>(y)[size_t(f)*per+z]=packed;
  }
 }
}
// Load contiguous outer rows once, transpose within each tile, emit all planes.
template<int F> __global__ void encode_all_transposed(const int8_t* x,unsigned char* y,int outer,int k,int opad,int kp,int off,int64_t ld,const DeviceFace* faces,bool right){
 __shared__ U64 lut[256];__shared__ unsigned char tile[64][33];int tx=threadIdx.x,ty=threadIdx.y,tid=ty*32+tx;U64 codes=0;
 #pragma unroll
 for(int f=0;f<F;++f)codes|=U64(right?faces[f].b[tid]:faces[f].a[tid])<<(4*f);
 lut[tid]=codes;__syncthreads();size_t per=size_t(opad)*kp/8;int kt=kp/64;
 for(size_t z=blockIdx.x;z<size_t(opad/32)*kt;z+=gridDim.x){int rr=z/kt*32,qq=z%kt*64;
  #pragma unroll
  for(int j=0;j<64;j+=8)tile[ty+j][tx]=(rr+tx<outer&&off+qq+ty+j<k)?static_cast<unsigned char>(x[rr+tx+size_t(off+qq+ty+j)*ld]):0;
  __syncthreads();int r=tid/8,q=(tid%8)*8;U64 v[8]{};
  #pragma unroll
  for(int t=0;t<8;++t)if(rr+r<outer&&off+qq+q+t<k)v[t]=lut[tile[q+t][r]];
  #pragma unroll
  for(int f=0;f<F;++f){unsigned packed=0;
   #pragma unroll
   for(int t=0;t<8;++t)packed|=unsigned((v[t]>>(4*f))&15)<<(4*t);
   reinterpret_cast<unsigned*>(y)[size_t(f)*per+size_t(rr+r)*(kp/8)+(qq+q)/8]=packed;
  }
  __syncthreads();
 }
}
__device__ __forceinline__ int residue(int x,int p,unsigned reciprocal){
 unsigned a=x<0?unsigned(-x):unsigned(x);unsigned q=__umulhi(a,reciprocal);int r=int(a-q*unsigned(p));if(r<0)r+=p;return x<0&&r?p-r:r;
}
template<class StateWord> __device__ __forceinline__ StateWord add_mod(StateWord a,StateWord b,StateWord P){StateWord s=a+b;if(s<a||s>=P)s-=P;return s;}
struct CrtInputs { const float* g[12]{}; int number=0; };
template<int F,class StateWord,class Out> __global__ void update_crt(CrtInputs input,StateWord* sum,size_t count,const DeviceFace* f,U64 modulus,bool first,bool last,Out* out,int m,int n,int mp,int64_t ld){
 __shared__ StateWord lut[F][256];int p[F];unsigned reciprocal[F];
 #pragma unroll
 for(int h=0;h<F;++h)if(h<input.number){p[h]=f[h].p;reciprocal[h]=f[h].reciprocal;if(threadIdx.x<p[h])lut[h][threadIdx.x]=StateWord(f[h].crt[threadIdx.x]);}
 __syncthreads();StateWord P=modulus;
 for(size_t z=(size_t(blockIdx.x)*blockDim.x+threadIdx.x)*4;z<count;z+=size_t(blockDim.x)*gridDim.x*4){
  StateWord a[4]{};
  if(!first){
   if constexpr(sizeof(StateWord)==4){uint4 t=*reinterpret_cast<uint4*>(sum+z);a[0]=t.x;a[1]=t.y;a[2]=t.z;a[3]=t.w;}
   else{ulonglong2 lo=*reinterpret_cast<ulonglong2*>(sum+z),hi=*reinterpret_cast<ulonglong2*>(sum+z+2);a[0]=lo.x;a[1]=lo.y;a[2]=hi.x;a[3]=hi.y;}
  }
  #pragma unroll
  for(int h=0;h<F;++h)if(h<input.number){float4 v=*reinterpret_cast<const float4*>(input.g[h]+z);float vv[4]={v.x,v.y,v.z,v.w};
   #pragma unroll
   for(int j=0;j<4;++j)a[j]=add_mod(a[j],lut[h][residue(__float2int_rn(vv[j]),p[h],reciprocal[h])],P);
  }
  if(last){
   if(m==mp&&ld==mp){
    if(z<size_t(m)*n){Out result[4];
     #pragma unroll
     for(int j=0;j<4;++j){U64 u=a[j];result[j]=static_cast<Out>(u>modulus/2?-int64_t(modulus-u):int64_t(u));}
     if constexpr(sizeof(Out)==4)*reinterpret_cast<uint4*>(out+z)=make_uint4(result[0],result[1],result[2],result[3]);
     else{*reinterpret_cast<ulonglong2*>(out+z)=make_ulonglong2(result[0],result[1]);*reinterpret_cast<ulonglong2*>(out+z+2)=make_ulonglong2(result[2],result[3]);}
    }
   }else{
    #pragma unroll
    for(int j=0;j<4;++j){int row=(z+j)%mp,col=(z+j)/mp;if(row<m&&col<n){U64 u=a[j];out[size_t(col)*ld+row]=static_cast<Out>(u>modulus/2?-int64_t(modulus-u):int64_t(u));}}
   }
  }else{
  if constexpr(sizeof(StateWord)==4)*reinterpret_cast<uint4*>(sum+z)=make_uint4(a[0],a[1],a[2],a[3]);
  else{*reinterpret_cast<ulonglong2*>(sum+z)=make_ulonglong2(a[0],a[1]);*reinterpret_cast<ulonglong2*>(sum+z+2)=make_ulonglong2(a[2],a[3]);}
  }
 }
}
template<class StateWord,class Out>
void launch_crt(const CrtInputs& input,StateWord* sum,size_t count,const DeviceFace* f,U64 modulus,bool first,bool last,Out* out,int m,int n,int mp,int64_t ld,cudaStream_t rs){
 if(input.number<=2)update_crt<2><<<blocks(count/4),256,0,rs>>>(input,sum,count,f,modulus,first,last,out,m,n,mp,ld);
 else if(input.number<=7)update_crt<7><<<blocks(count/4),256,0,rs>>>(input,sum,count,f,modulus,first,last,out,m,n,mp,ld);
 else update_crt<11><<<blocks(count/4),256,0,rs>>>(input,sum,count,f,modulus,first,last,out,m,n,mp,ld);
}
template<class T,class StateWord> __global__ void store_output(const StateWord* sum,T* out,int m,int n,int mp,
                                               int64_t ld,bool rowmajor,U64 P){
 for(size_t z=size_t(blockIdx.x)*blockDim.x+threadIdx.x;z<size_t(m)*n;z+=size_t(blockDim.x)*gridDim.x){
  int i=z%m,j=z/m;U64 u=sum?sum[i+size_t(j)*mp]:0;
  int64_t v=u>P/2?-int64_t(P-u):int64_t(u);
  out[rowmajor?size_t(i)*ld+j:size_t(j)*ld+i]=static_cast<T>(v);
 }
}

struct Engine {
 awe_gemm_desc d{};awe_options o{};awe_plan_info info{};
 U64 product=1;int mp=0,np=0,kp=0,face_count=0;bool state32=true;
 size_t all_a=0,all_b=0;
 size_t bytes=0,count=0,table_offset=0,sum_offset=0,scale_offset=0,scratch_offset=0,scale_bytes=0;
 DeviceFace* host_faces=nullptr;
 cublasLtHandle_t handle=nullptr;cublasLtMatmulDesc_t op=nullptr;
 cublasLtMatrixLayout_t la=nullptr,lb=nullptr,lc=nullptr;cublasLtMatmulAlgo_t algo{};
 cudaStream_t streams[3]{};
 std::vector<std::unique_ptr<Slot>> slots;
 std::vector<std::unique_ptr<Timing>> timings;
 std::unique_ptr<Event> begin,end,ready,setup_end,output_begin;
 ~Engine(){
  for(auto s:streams)if(s)cudaStreamSynchronize(s);
  for(auto s:streams)if(s)cudaStreamDestroy(s);
  if(host_faces)cudaFreeHost(host_faces);
  if(la)cublasLtMatrixLayoutDestroy(la);if(lb)cublasLtMatrixLayoutDestroy(lb);
  if(lc)cublasLtMatrixLayoutDestroy(lc);if(op)cublasLtMatmulDescDestroy(op);
  if(handle)cublasLtDestroy(handle);
 }
 size_t reserve(size_t n){size_t offset=bytes; if(n>SIZE_MAX-255-bytes)throw Failure{AWE_INVALID_ARGUMENT};bytes+=align256(n);return offset;}
 void init(const awe_gemm_desc& desc,const awe_options& options){
  d=desc;o=options;info.struct_size=sizeof(info);info.method=AWE_METHOD_OZAKI2;
  info.stream_count=o.stream_count;info.algorithm_id=-1;info.input_min=-128;info.input_max=127;
  if(d.data_type!=AWE_I8_I64&&d.data_type!=AWE_I8_I32)throw Failure{AWE_NOT_SUPPORTED};
  if(d.m<0||d.n<0||d.k<0||d.m>(1<<20)||d.n>(1<<20)||d.k>INT_MAX ||
     (o.stream_count!=1&&o.stream_count!=3)||o.buffer_count<1||o.buffer_count>11||
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
  int lowest=0,highest=0;cu(cudaDeviceGetStreamPriorityRange(&lowest,&highest));
  for(int i=0;i<o.stream_count;++i)cu(cudaStreamCreateWithPriority(&streams[i],cudaStreamNonBlocking,i==2?highest:lowest));
  begin.reset(new Event);end.reset(new Event);ready.reset(new Event(false));
  setup_end.reset(new Event);output_begin.reset(new Event);
  if(!d.k)return;
  const Choice* chosen=nullptr;U64 needed=U64(d.k)*32768ULL;
  for(auto& c:choices)if(c.product>needed){chosen=&c;break;}
  if(!chosen)throw Failure{AWE_NOT_SUPPORTED};
  product=chosen->product;face_count=chosen->faces;state32=product<=UINT32_MAX;
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
   const auto& w=witnesses[chosen->indices[i]];U64 base=product/w.p;
   int inv=1;while((base%w.p)*inv%w.p!=1)++inv;
   U64 weight=base*inv;
   for(int f=0;f<w.faces;++f){
    auto& out=host_faces[at++];out.p=w.p;out.reciprocal=unsigned(((1ULL<<32)+w.p-1)/w.p);
    for(int b=0;b<256;++b){int x=b<128?b:b-256;int r=x%w.p;if(r<0)r+=w.p;out.a[b]=w.f[f].a[r];out.b[b]=w.f[f].b[r];}
    for(int r=0;r<w.p;++r){int v=(w.f[f].coefficient*r)%w.p;if(v<0)v+=w.p;
     out.crt[r]=U64((__uint128_t(weight)*v)%product);
    }
   }
  }
  table_offset=reserve(size_t(face_count)*sizeof(DeviceFace));sum_offset=reserve(count*(state32?4:8));
  scale_bytes=size_t(std::max(mp,np))*((kp+63)/64)*4;scale_offset=reserve(scale_bytes);
  scratch_offset=reserve(o.matmul_workspace_bytes);
  all_a=reserve(size_t(face_count)*mp*kp/2);all_b=reserve(size_t(face_count)*np*kp/2);
  for(int i=0;i<o.buffer_count;++i){auto s=std::make_unique<Slot>();s->g=reserve(count*4);slots.push_back(std::move(s));}
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
  // The NVFP4 heuristic requires non-null scale addresses even without execution.
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
  if(o.profiling){size_t steps=((size_t(d.k)+kp-1)/kp)*face_count;for(size_t i=0;i<steps;++i)timings.emplace_back(new Timing);}
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
  size_t an=extent(d.trans_a==AWE_OP_N?d.m:d.k,d.trans_a==AWE_OP_N?d.k:d.m,d.lda,1);
  size_t bn=extent(d.trans_b==AWE_OP_N?d.k:d.n,d.trans_b==AWE_OP_N?d.n:d.k,d.ldb,1);
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
   cu(cudaMemsetAsync(scales,0x40,scale_bytes,es));record(*setup_end,es);
   record(*ready,rs);wait(es,*ready);
   for(auto& slot:slots)record(slot->available,rs);
   size_t serial=0;
   for(int64_t off=0;off<d.k;off+=kp){
    for(auto& slot:slots)wait(es,slot->available);
    Timing* first=o.profiling?timings[serial].get():nullptr;if(first)record(first->e0,es);
    bool ac=(d.trans_a==AWE_OP_T)!=row,bc=(d.trans_b==AWE_OP_T)==row;
    if(face_count==11&&ac)encode_all<11><<<blocks(size_t(mp)*kp/8),256,0,es>>>((const int8_t*)a,raw+all_a,d.m,d.k,mp,kp,off,d.lda,faces,false);
    else if(face_count==11)encode_all_transposed<11><<<std::min<size_t>(4096,size_t(mp/32)*(kp/64)),dim3(32,8),0,es>>>((const int8_t*)a,raw+all_a,d.m,d.k,mp,kp,off,d.lda,faces,false);
    else for(int f=0;f<face_count;++f)encode_dispatch((const int8_t*)a,raw+all_a+size_t(f)*mp*kp/2,d.m,d.k,mp,kp,off,d.lda,d.trans_a==AWE_OP_T,false,row,faces[f].a,es);
    if(face_count==11&&bc)encode_all<11><<<blocks(size_t(np)*kp/8),256,0,es>>>((const int8_t*)b,raw+all_b,d.n,d.k,np,kp,off,d.ldb,faces,true);
    else if(face_count==11)encode_all_transposed<11><<<std::min<size_t>(4096,size_t(np/32)*(kp/64)),dim3(32,8),0,es>>>((const int8_t*)b,raw+all_b,d.n,d.k,np,kp,off,d.ldb,faces,true);
    else for(int f=0;f<face_count;++f)encode_dispatch((const int8_t*)b,raw+all_b+size_t(f)*np*kp/2,d.n,d.k,np,kp,off,d.ldb,d.trans_b==AWE_OP_T,true,row,faces[f].b,es);
    int batch_start=0;
    for(int f=0;f<face_count;++f,++serial){
    auto& slot=*slots[serial%slots.size()];Timing* t=o.profiling?timings[serial].get():nullptr;
    wait(es,slot.available);if(t&&f)record(t->e0,es);
    if(t)record(t->e1,es);record(slot.encoded,es);if(t)record(t->w0,gs);
    wait(gs,slot.encoded);if(t)record(t->g0,gs);float one=1,zero=0;
    lt(cublasLtMatmul(handle,op,&one,raw+all_a+size_t(f)*mp*kp/2,la,raw+all_b+size_t(f)*np*kp/2,lb,&zero,raw+slot.g,lc,raw+slot.g,lc,&algo,raw+scratch_offset,o.matmul_workspace_bytes,gs));
    if(t)record(t->g1,gs);record(slot.computed,gs);
    int group=slots.size()>=8?7:slots.size()>=3?2:1;
    if(int(slots.size())>=face_count)group=face_count;
    bool finish=(f-batch_start+1==group||(group<face_count&&f==face_count-2)||f==face_count-1);
    if(finish){
     int number=f-batch_start+1,first_face=batch_start;
     wait(rs,slot.computed);if(t)record(t->r0,rs);
     CrtInputs inputs;inputs.number=number;
     for(int j=0;j<number;++j)inputs.g[j]=reinterpret_cast<float*>(raw+slots[(serial-number+1+j)%slots.size()]->g);
     bool initialize=off==0&&first_face==0,finish_output=off+kp>=d.k&&f==face_count-1&&!row;
     if(d.data_type==AWE_I8_I32){
      if(state32)launch_crt(inputs,static_cast<uint32_t*>(sum),count,faces+first_face,product,initialize,finish_output,static_cast<int32_t*>(c),int(d.m),int(d.n),mp,d.ldc,rs);
      else launch_crt(inputs,static_cast<U64*>(sum),count,faces+first_face,product,initialize,finish_output,static_cast<int32_t*>(c),int(d.m),int(d.n),mp,d.ldc,rs);
     }else{
      if(state32)launch_crt(inputs,static_cast<uint32_t*>(sum),count,faces+first_face,product,initialize,finish_output,static_cast<int64_t*>(c),int(d.m),int(d.n),mp,d.ldc,rs);
      else launch_crt(inputs,static_cast<U64*>(sum),count,faces+first_face,product,initialize,finish_output,static_cast<int64_t*>(c),int(d.m),int(d.n),mp,d.ldc,rs);
     }
     if(t)record(t->r1,rs);
     for(int j=0;j<number;++j)record(slots[(serial-j)%slots.size()]->available,rs);
     batch_start=f+1;
    }else if(t){record(t->r0,rs);record(t->r1,rs);}

   }
   }
  }
  record(*output_begin,rs);
  if(!d.k||row){
  if(d.data_type==AWE_I8_I32){
   if(state32)store_output<<<blocks(size_t(d.m)*d.n),256,0,rs>>>(static_cast<uint32_t*>(sum),static_cast<int32_t*>(c),int(d.m),int(d.n),mp,d.ldc,row,product);
   else store_output<<<blocks(size_t(d.m)*d.n),256,0,rs>>>(static_cast<U64*>(sum),static_cast<int32_t*>(c),int(d.m),int(d.n),mp,d.ldc,row,product);
  }else{
   if(state32)store_output<<<blocks(size_t(d.m)*d.n),256,0,rs>>>(static_cast<uint32_t*>(sum),static_cast<int64_t*>(c),int(d.m),int(d.n),mp,d.ldc,row,product);
   else store_output<<<blocks(size_t(d.m)*d.n),256,0,rs>>>(static_cast<U64*>(sum),static_cast<int64_t*>(c),int(d.m),int(d.n),mp,d.ldc,row,product);
  }
  }
  cu(cudaGetLastError());record(*end,rs);wait(caller,*end);cu(cudaEventSynchronize(end->e));
  if(stats){
   stats->total_ms=elapsed(*begin,*end);stats->reconstruction_ms=elapsed(*output_begin,*end);
   // Moduli are chosen from public dimensions at plan creation; no value scan.
   stats->selection_ms=0;
   if(o.profiling&&d.k){stats->encode_ms=elapsed(*begin,*setup_end);
    for(auto& t:timings){stats->encode_ms+=elapsed(t->e0,t->e1);stats->gemm_ms+=elapsed(t->g0,t->g1);stats->reconstruction_ms+=elapsed(t->r0,t->r1);stats->wait_ms+=elapsed(t->w0,t->g0);}
   }
  }
 }
 void drain(){for(auto s:streams)if(s)cudaStreamSynchronize(s);}
};
}

extern "C" awe_status awe_ozaki2_create(void** state,const awe_gemm_desc* d,const awe_options* o){
 if(!state||!d||!o||d->struct_size!=sizeof(*d)||o->struct_size!=sizeof(*o))return AWE_INVALID_ARGUMENT;
 *state=nullptr;try{auto e=std::make_unique<Engine>();e->init(*d,*o);*state=e.release();return AWE_SUCCESS;}
 catch(Failure f){return f.status;}catch(...){return AWE_INTERNAL_ERROR;}
}
extern "C" void awe_ozaki2_destroy(void* state){delete static_cast<Engine*>(state);}
extern "C" awe_status awe_ozaki2_info(const void* state,awe_plan_info* out){
 if(!state||!out||out->struct_size!=sizeof(*out))return AWE_INVALID_ARGUMENT;
 *out=static_cast<const Engine*>(state)->info;return AWE_SUCCESS;
}
extern "C" awe_status awe_ozaki2_execute(void* state,const void* a,const void* b,void* c,void* work,size_t bytes,cudaStream_t stream,awe_execution_stats* stats){
 if(!state||(stats&&stats->struct_size!=sizeof(*stats)))return AWE_INVALID_ARGUMENT;
 auto e=static_cast<Engine*>(state);try{e->execute(a,b,c,work,bytes,stream,stats);return AWE_SUCCESS;}
 catch(Failure f){e->drain();return f.status;}catch(...){e->drain();return AWE_INTERNAL_ERROR;}
}
