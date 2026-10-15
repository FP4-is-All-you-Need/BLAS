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
#include "i8_tables_generated.h"

struct DeviceFace {
 int p;
 unsigned char a[256],b[256];
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
template<class StateWord> __global__ void update_crt(const float* g,StateWord* sum,size_t count,const DeviceFace* f,U64 modulus){
 const StateWord P=StateWord(modulus);
 for(size_t z=size_t(blockIdx.x)*blockDim.x+threadIdx.x;z<count;z+=size_t(blockDim.x)*gridDim.x){
  int r=__float2int_rn(g[z])%f->p;if(r<0)r+=f->p;
  StateWord a=sum[z],b=StateWord(f->crt[r]),s=a+b;

  if(s<a || s>=P)s-=P;
  sum[z]=s;
 }
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
    auto& out=host_faces[at++];out.p=w.p;
    for(int b=0;b<256;++b){int x=b<128?b:b-256;int r=x%w.p;if(r<0)r+=w.p;out.a[b]=w.f[f].a[r];out.b[b]=w.f[f].b[r];}
    for(int r=0;r<w.p;++r){int v=(w.f[f].coefficient*r)%w.p;if(v<0)v+=w.p;
     out.crt[r]=U64((__uint128_t(weight)*v)%product);
    }
   }
  }
  table_offset=reserve(size_t(face_count)*sizeof(DeviceFace));sum_offset=reserve(count*(state32?4:8));
  scale_bytes=size_t(std::max(mp,np))*((kp+63)/64)*4;scale_offset=reserve(scale_bytes);
  scratch_offset=reserve(o.matmul_workspace_bytes);
  for(int i=0;i<o.buffer_count;++i){auto s=std::make_unique<Slot>();s->a=reserve(size_t(mp)*kp/2);s->b=reserve(size_t(np)*kp/2);s->g=reserve(count*4);slots.push_back(std::move(s));}
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
   cu(cudaMemsetAsync(sum,0,count*(state32?4:8),rs));record(*ready,rs);wait(es,*ready);
   for(auto& slot:slots)record(slot->available,rs);
   size_t serial=0;
   for(int64_t off=0;off<d.k;off+=kp)for(int f=0;f<face_count;++f,++serial){
    auto& slot=*slots[serial%slots.size()];Timing* t=o.profiling?timings[serial].get():nullptr;
    wait(es,slot.available);if(t)record(t->e0,es);
    encode_dispatch(static_cast<const int8_t*>(a),raw+slot.a,int(d.m),int(d.k),mp,kp,int(off),d.lda,d.trans_a==AWE_OP_T,false,row,faces[f].a,es);
    encode_dispatch(static_cast<const int8_t*>(b),raw+slot.b,int(d.n),int(d.k),np,kp,int(off),d.ldb,d.trans_b==AWE_OP_T,true,row,faces[f].b,es);
    if(t)record(t->e1,es);record(slot.encoded,es);if(t)record(t->w0,gs);
    wait(gs,slot.encoded);if(t)record(t->g0,gs);float one=1,zero=0;
    lt(cublasLtMatmul(handle,op,&one,raw+slot.a,la,raw+slot.b,lb,&zero,raw+slot.g,lc,raw+slot.g,lc,&algo,raw+scratch_offset,o.matmul_workspace_bytes,gs));
    if(t)record(t->g1,gs);record(slot.computed,gs);wait(rs,slot.computed);if(t)record(t->r0,rs);
    if(state32)update_crt<<<blocks(count),256,0,rs>>>(reinterpret_cast<float*>(raw+slot.g),static_cast<uint32_t*>(sum),count,faces+f,product);
    else update_crt<<<blocks(count),256,0,rs>>>(reinterpret_cast<float*>(raw+slot.g),static_cast<U64*>(sum),count,faces+f,product);
    if(t)record(t->r1,rs);record(slot.available,rs);
   }
  }
  record(*output_begin,rs);
  if(d.data_type==AWE_I8_I32){
   if(state32)store_output<<<blocks(size_t(d.m)*d.n),256,0,rs>>>(static_cast<uint32_t*>(sum),static_cast<int32_t*>(c),int(d.m),int(d.n),mp,d.ldc,row,product);
   else store_output<<<blocks(size_t(d.m)*d.n),256,0,rs>>>(static_cast<U64*>(sum),static_cast<int32_t*>(c),int(d.m),int(d.n),mp,d.ldc,row,product);
  }else{
   if(state32)store_output<<<blocks(size_t(d.m)*d.n),256,0,rs>>>(static_cast<uint32_t*>(sum),static_cast<int64_t*>(c),int(d.m),int(d.n),mp,d.ldc,row,product);
   else store_output<<<blocks(size_t(d.m)*d.n),256,0,rs>>>(static_cast<U64*>(sum),static_cast<int64_t*>(c),int(d.m),int(d.n),mp,d.ldc,row,product);
  }
  cu(cudaGetLastError());record(*end,rs);wait(caller,*end);cu(cudaEventSynchronize(end->e));
  if(stats){
   stats->total_ms=elapsed(*begin,*end);stats->reconstruction_ms=elapsed(*output_begin,*end);

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
