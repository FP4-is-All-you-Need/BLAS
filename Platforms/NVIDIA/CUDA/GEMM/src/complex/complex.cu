#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <stdexcept>
#include <string>

#define CU(x) do { auto cuda_result_=(x); if(cuda_result_!=cudaSuccess) throw std::runtime_error(std::string(#x)+": "+cudaGetErrorString(cuda_result_)); } while(0)
#define BL(x) do { auto blas_result_=(x); if(blas_result_!=CUBLAS_STATUS_SUCCESS) throw std::runtime_error(std::string(#x)+": "+std::to_string(int(blas_result_))); } while(0)
struct Buffer {
 void* p=nullptr; size_t bytes=0;
 void alloc(size_t n){ release(); bytes=n; CU(cudaMalloc(&p,n)); }
 void release(){ if(p)cudaFree(p); p=nullptr; bytes=0; }
 template<class T>T* as(){return static_cast<T*>(p);}
 ~Buffer(){if(p)cudaFree(p);}
};
struct Row {
 int p,root,faces,denom,ksafe;
 int coeff[6],ia[6],ib[6];
 unsigned char lut[241*6];
};
struct Config { int count; Row rows[32]; };
static thread_local std::string last_error;
extern "C" const char* error(){return last_error.c_str();}
__host__ __device__ static int posmod(long long a,int p){int r=a%p; return r<0?r+p:r;}
static int blocks(size_t n){return int(std::min<size_t>(65535,(n+255)/256));}

__host__ __device__ __forceinline__ int branch_residue(long long vr,long long vi,int tr,int ti,int p){
 return posmod((long long)tr*posmod(vr,p)+(long long)ti*posmod(vi,p),p);
}

__host__ __device__ __forceinline__ void branch_to_components(int x,int y,int p,int root,int& re,int& im){
 int inv2=(p+1)/2; re=posmod((long long)inv2*(x+y),p); im=posmod((long long)root*inv2*(y-x),p);
}

struct FaceIdx{int v[6];};
struct FaceCoeff{int v[6];};

struct Bar { int p; unsigned long long mu, bias; };

__host__ __device__ __forceinline__ int mod_p(long long v, const Bar& b){
 unsigned long long u = (unsigned long long)v + b.bias;
#ifdef __CUDA_ARCH__
 unsigned long long q = __umul64hi(u, b.mu);
#else
 unsigned long long q = (unsigned long long)(((__uint128_t)u * (__uint128_t)b.mu) >> 64);
#endif
 int r = (int)(u - q * (unsigned long long)b.p);
 if(r >= b.p) r -= b.p;
 return r;
}
static Bar make_bar(int p){
 Bar b; b.p = p;
 b.mu = ~0ull / (unsigned long long)p;
 b.bias = (unsigned long long)p * (((1ull << 55) + (unsigned long long)p - 1ull) / (unsigned long long)p);
 return b;
}

struct BranchCoef { int tr[4], ti[4]; };

struct Plan {
 cublasLtHandle_t h=nullptr; cublasLtMatmulDesc_t op=nullptr;
 cublasLtMatrixLayout_t a=nullptr,b=nullptr,d=nullptr;
 cublasLtMatmulPreference_t pref=nullptr; cublasLtMatmulAlgo_t algo{};
 Buffer workspace;
 void init(int m,int n,int k,void* sf) {
  BL(cublasLtCreate(&h)); BL(cublasLtMatmulDescCreate(&op,CUBLAS_COMPUTE_32F,CUDA_R_32F));
  cublasOperation_t t=CUBLAS_OP_T,nn=CUBLAS_OP_N;
  auto mode=CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
  BL(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSA,&t,sizeof(t)));
  BL(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSB,&nn,sizeof(nn)));
  BL(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_A_SCALE_MODE,&mode,sizeof(mode)));
  BL(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_B_SCALE_MODE,&mode,sizeof(mode)));
  BL(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,&sf,sizeof(sf)));
  BL(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,&sf,sizeof(sf)));
  BL(cublasLtMatrixLayoutCreate(&a,CUDA_R_4F_E2M1,k,m,k));
  BL(cublasLtMatrixLayoutCreate(&b,CUDA_R_4F_E2M1,k,n,k));
  BL(cublasLtMatrixLayoutCreate(&d,CUDA_R_32F,m,n,m));
  workspace.alloc(32ul<<20); BL(cublasLtMatmulPreferenceCreate(&pref));
  BL(cublasLtMatmulPreferenceSetAttribute(pref,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&workspace.bytes,sizeof(size_t)));
  cublasLtMatmulHeuristicResult_t result[16];int count=0;
  BL(cublasLtMatmulAlgoGetHeuristic(h,op,a,b,d,d,pref,16,result,&count));
  for(int i=0;i<count;++i)if(result[i].state==CUBLAS_STATUS_SUCCESS){algo=result[i].algo;return;}
  throw std::runtime_error("no NVFP4 algorithm");
 }
 void run(void*A,void*B,void*D,cudaStream_t st){float one=1,zero=0;BL(cublasLtMatmul(h,op,&one,A,a,B,b,&zero,D,d,D,d,&algo,workspace.p,workspace.bytes,st));}
 ~Plan(){if(pref)cublasLtMatmulPreferenceDestroy(pref);if(a)cublasLtMatrixLayoutDestroy(a);if(b)cublasLtMatrixLayoutDestroy(b);if(d)cublasLtMatrixLayoutDestroy(d);if(op)cublasLtMatmulDescDestroy(op);if(h)cublasLtDestroy(h);}
};

__global__ void encode(const double2* x,unsigned char* y,int outer,int k,int opad,int kp,
 int offset,bool b,int shift,int precision,const Row* row,int fi,int tr,int ti) {
 size_t count=size_t(opad)*kp/2;
 for(size_t z=size_t(blockIdx.x)*blockDim.x+threadIdx.x;z<count;z+=size_t(gridDim.x)*blockDim.x){
  int o=z/(kp/2),q=2*(z%(kp/2));unsigned val=0;
  for(int j=0;j<2;++j)if(o<outer && offset+q+j<k){
   size_t idx=b?size_t(o)*k+offset+q+j:o+size_t(offset+q+j)*outer;
   double2 v;if(precision==24){auto w=reinterpret_cast<const float2*>(x)[idx];v=make_double2(w.x,w.y);}else v=x[idx];
   long long vr=__double2ll_rn(ldexp(v.x,shift)),vi=__double2ll_rn(ldexp(v.y,shift));
   int r=branch_residue(vr,vi,tr,ti,row->p);
   val|=unsigned(row->lut[r*6+fi])<<(4*j);
  }
  y[z]=val;
 }
}

template<int F>
__global__ void encode_faces(const double2* x,unsigned char* y,size_t stride,int outer,int k,int opad,int kp,
 int offset,bool b,int shift,int precision,const Row* row,FaceIdx fi,int tr,int ti){
 size_t count=size_t(opad)*kp/2;
 for(size_t z=size_t(blockIdx.x)*blockDim.x+threadIdx.x;z<count;z+=size_t(gridDim.x)*blockDim.x){
  int o=z/(kp/2),q=2*(z%(kp/2));unsigned val[F];
  #pragma unroll
  for(int f=0;f<F;++f)val[f]=0;
  for(int j=0;j<2;++j)if(o<outer && offset+q+j<k){
   size_t idx=b?size_t(o)*k+offset+q+j:o+size_t(offset+q+j)*outer;
   double2 v;if(precision==24){auto w=reinterpret_cast<const float2*>(x)[idx];v=make_double2(w.x,w.y);}else v=x[idx];
   long long vr=__double2ll_rn(ldexp(v.x,shift)),vi=__double2ll_rn(ldexp(v.y,shift));
   int r=branch_residue(vr,vi,tr,ti,row->p);
   const unsigned char* row_lut=row->lut+r*6;
   #pragma unroll
   for(int f=0;f<F;++f)val[f]|=unsigned(row_lut[fi.v[f]])<<(4*j);
  }
  #pragma unroll
  for(int f=0;f<F;++f)y[size_t(f)*stride+z]=(unsigned char)val[f];
 }
}
__global__ void accumulate(const float* g,long long* sum,size_t count,int coeff){
 for(size_t z=size_t(blockIdx.x)*blockDim.x+threadIdx.x;z<count;z+=size_t(gridDim.x)*blockDim.x)
  sum[z]+=(long long)coeff*__float2ll_rn(g[z]);
}

template<int F>
__global__ void accumulate_faces(const float* g,size_t stride,long long* sum,size_t count,FaceCoeff c){
 for(size_t z=size_t(blockIdx.x)*blockDim.x+threadIdx.x;z<count;z+=size_t(gridDim.x)*blockDim.x){
  long long s=0;
  #pragma unroll
  for(int f=0;f<F;++f)s+=(long long)c.v[f]*__float2ll_rn(g[size_t(f)*stride+z]);
  sum[z]+=s;
 }
}

static void launch_encode_faces(int faces,int gr,cudaStream_t st,const double2* x,unsigned char* y,size_t stride,
 int outer,int k,int opad,int kp,int offset,bool b,int shift,int precision,const Row* row,FaceIdx fi,int tr,int ti){
#define AWE_ENCODE_CASE(F) case F: encode_faces<F><<<gr,256,0,st>>>(x,y,stride,outer,k,opad,kp,offset,b,shift,precision,row,fi,tr,ti); break;
 switch(faces){AWE_ENCODE_CASE(1)AWE_ENCODE_CASE(2)AWE_ENCODE_CASE(3)AWE_ENCODE_CASE(4)AWE_ENCODE_CASE(5)AWE_ENCODE_CASE(6)
  default: throw std::runtime_error("face count outside 1..6");}
#undef AWE_ENCODE_CASE
}
static void launch_accumulate_faces(int faces,int gr,cudaStream_t st,const float* g,size_t stride,long long* sum,size_t count,FaceCoeff c){
#define AWE_ACC_CASE(F) case F: accumulate_faces<F><<<gr,256,0,st>>>(g,stride,sum,count,c); break;
 switch(faces){AWE_ACC_CASE(1)AWE_ACC_CASE(2)AWE_ACC_CASE(3)AWE_ACC_CASE(4)AWE_ACC_CASE(5)AWE_ACC_CASE(6)
  default: throw std::runtime_error("face count outside 1..6");}
#undef AWE_ACC_CASE
}

template<int B,int F>
__global__ void encode_pair(const double2* x,unsigned char* y,size_t stride,int outer,int k,int opad,int kp,
 int offset,bool side_b,int shift,int precision,Bar bar,const unsigned* pack,BranchCoef bc){
 size_t count=size_t(opad)*kp/2;
 for(size_t z=size_t(blockIdx.x)*blockDim.x+threadIdx.x;z<count;z+=size_t(gridDim.x)*blockDim.x){
  int o=z/(kp/2),q=2*(z%(kp/2));unsigned val[B*F];
  #pragma unroll
  for(int t=0;t<B*F;++t)val[t]=0;
  for(int j=0;j<2;++j)if(o<outer && offset+q+j<k){
   size_t idx=side_b?size_t(o)*k+offset+q+j:o+size_t(offset+q+j)*outer;
   double2 v;if(precision==24){auto w=reinterpret_cast<const float2*>(x)[idx];v=make_double2(w.x,w.y);}else v=x[idx];
   long long vr=__double2ll_rn(ldexp(v.x,shift)),vi=__double2ll_rn(ldexp(v.y,shift));
   int al=mod_p(vr,bar),be=mod_p(vi,bar);
   #pragma unroll
   for(int br=0;br<B;++br){
    int r=mod_p((long long)(bc.tr[br]*al+bc.ti[br]*be),bar);
    unsigned pk=pack[r];
    #pragma unroll
    for(int f=0;f<F;++f)val[br*F+f]|=((pk>>(4*f))&0xFu)<<(4*j);
   }
  }
  #pragma unroll
  for(int t=0;t<B*F;++t)y[size_t(t)*stride+z]=(unsigned char)val[t];
 }
}
static void launch_encode_pair(int branches,int faces,int gr,cudaStream_t st,const double2* x,unsigned char* y,
 size_t stride,int outer,int k,int opad,int kp,int offset,bool side_b,int shift,int precision,
 Bar bar,const unsigned* pack,BranchCoef bc){
#define AWE_PAIR(B,F) case B*16+F: encode_pair<B,F><<<gr,256,0,st>>>(x,y,stride,outer,k,opad,kp,offset,side_b,shift,precision,bar,pack,bc); break;
#define AWE_PAIR_ROW(B) AWE_PAIR(B,1)AWE_PAIR(B,2)AWE_PAIR(B,3)AWE_PAIR(B,4)AWE_PAIR(B,5)AWE_PAIR(B,6)
 switch(branches*16+faces){AWE_PAIR_ROW(2)AWE_PAIR_ROW(3)AWE_PAIR_ROW(4)
  default: throw std::runtime_error("branch or face count outside the instantiated range");}
#undef AWE_PAIR_ROW
#undef AWE_PAIR
}

struct EncMod {
 Bar bar; int faces, slot_base, tr[4], ti[4]; const unsigned* pack;
};

__global__ void quantize_a(const double2* A,longlong2* qa,int m,int k,double scale,int precision){
 __shared__ double2 tile[32][33];
 int bx=blockIdx.x*32,by=blockIdx.y*32,tx=threadIdx.x,ty=threadIdx.y;
 for(int r=0;r<32;r+=8){
  int i=bx+tx,kk=by+ty+r;double2 v=make_double2(0,0);
  if(i<m&&kk<k){size_t idx=size_t(i)+size_t(kk)*m;
   if(precision==24){auto w=reinterpret_cast<const float2*>(A)[idx];v=make_double2(w.x,w.y);}else v=A[idx];}
  tile[ty+r][tx]=v;
 }
 __syncthreads();
 for(int r=0;r<32;r+=8){
  int kk=by+tx,i=bx+ty+r;
  if(i<m&&kk<k){double2 v=tile[tx][ty+r];
   qa[size_t(i)*k+kk]=make_longlong2(__double2ll_rn(v.x*scale),__double2ll_rn(v.y*scale));}
 }
}

__global__ void quantize_b(const double2* B,longlong2* qb,long long total,double scale,int precision){
 for(long long z=(long long)blockIdx.x*blockDim.x+threadIdx.x;z<total;z+=(long long)gridDim.x*blockDim.x){
  double2 v;
  if(precision==24){auto w=reinterpret_cast<const float2*>(B)[z];v=make_double2(w.x,w.y);}else v=B[z];
  qb[z]=make_longlong2(__double2ll_rn(v.x*scale),__double2ll_rn(v.y*scale));
 }
}

__global__ void encode_group(const longlong2* x,unsigned char* y,size_t stride,int outer,int k,int opad,int kp,
 int offset,int branches,int nmod,const EncMod* mods){
 size_t count=size_t(opad)*kp/2;
 for(size_t z=size_t(blockIdx.x)*blockDim.x+threadIdx.x;z<count;z+=size_t(gridDim.x)*blockDim.x){
  int o=z/(kp/2),q=2*(z%(kp/2));
  bool ok0=(o<outer&&offset+q+0<k),ok1=(o<outer&&offset+q+1<k);
  longlong2 v0=make_longlong2(0,0),v1=make_longlong2(0,0);
  if(ok0)v0=x[size_t(o)*k+offset+q+0];
  if(ok1)v1=x[size_t(o)*k+offset+q+1];
  for(int mi=0;mi<nmod;++mi){
   const EncMod& em=mods[mi];
   int a0=0,b0=0,a1=0,b1=0;
   if(ok0){a0=mod_p(v0.x,em.bar);b0=mod_p(v0.y,em.bar);}
   if(ok1){a1=mod_p(v1.x,em.bar);b1=mod_p(v1.y,em.bar);}
   for(int br=0;br<branches;++br){
    unsigned pk0=0,pk1=0;
    if(ok0)pk0=em.pack[mod_p((long long)(em.tr[br]*a0+em.ti[br]*b0),em.bar)];
    if(ok1)pk1=em.pack[mod_p((long long)(em.tr[br]*a1+em.ti[br]*b1),em.bar)];
    unsigned char* out=y+size_t(em.slot_base+br*em.faces)*stride+z;
    for(int f=0;f<em.faces;++f)
     out[size_t(f)*stride]=(unsigned char)(((pk0>>(4*f))&0xFu)|(((pk1>>(4*f))&0xFu)<<4));
   }
  }
 }
}

__global__ void combine_r2(const long long* t,int* residues,size_t count,Bar bar,int denom,int root,int method){
 for(size_t z=size_t(blockIdx.x)*blockDim.x+threadIdx.x;z<count;z+=size_t(gridDim.x)*blockDim.x){
  int p=bar.p,re,im;
  long long w0=t[z],w1=t[count+z];
  if(denom!=1){w0=w0<0?-((-w0)/denom):w0/denom;w1=w1<0?-((-w1)/denom):w1/denom;}
  int a=mod_p(w0,bar),b=mod_p(w1,bar);
  if(method==2){int inv2=(p+1)/2;re=mod_p((long long)inv2*(a+b),bar);im=mod_p((long long)root*inv2*(b-a),bar);}
  else{
   long long w2=t[2*count+z];if(denom!=1)w2=w2<0?-((-w2)/denom):w2/denom;
   if(method==3){int c=mod_p(w2,bar);re=mod_p((long long)(a-b),bar);im=mod_p((long long)(c-a-b),bar);}
   else{long long w3=t[3*count+z];if(denom!=1)w3=w3<0?-((-w3)/denom):w3/denom;
    re=mod_p((long long)(a-b),bar);im=mod_p(w2+w3,bar);}
  }
  residues[z]=re;residues[count+z]=im;
 }
}
__global__ void combine(const long long* t,int* residues,size_t count,const Row* r,int method){
 for(size_t z=size_t(blockIdx.x)*blockDim.x+threadIdx.x;z<count;z+=size_t(gridDim.x)*blockDim.x){
  int p=r->p,a=posmod(t[z]/r->denom,p),b=posmod(t[count+z]/r->denom,p),re,im;
  if(method==2){branch_to_components(a,b,p,r->root,re,im);}
  else if(method==3){int c=posmod(t[2*count+z]/r->denom,p);re=posmod(a-b,p);im=posmod(c-a-b,p);}
  else{re=posmod(a-b,p);im=posmod(t[2*count+z]/r->denom+t[3*count+z]/r->denom,p);}
  residues[z]=re;residues[count+z]=im;
 }
}
__device__ double round_exact(__int128 x,int shift,int precision){
 bool neg=x<0; __uint128_t a=neg?__uint128_t(-x):__uint128_t(x);
 unsigned long long hi=a>>64,lo=a;int top=hi?127-__clzll(hi):(lo?63-__clzll(lo):-1);
 int drop=max(0,top-(precision-1));unsigned long long mant=(unsigned long long)(a>>drop);
 if(drop){__uint128_t low=a&(((__uint128_t)1<<drop)-1),half=(__uint128_t)1<<(drop-1);if(low>half||(low==half&&(mant&1)))++mant;}
 double out=ldexp(double(mant),drop+shift);return neg?-out:out;
}
__global__ void restore(const int* r,double2* out,size_t count,int m,int n,int mp,
 const Config* cfg,const int* inv,int shift,int precision){
 for(size_t z=size_t(blockIdx.x)*blockDim.x+threadIdx.x;z<count;z+=size_t(gridDim.x)*blockDim.x){
  int i=z%mp,j=z/mp;if(i>=m||j>=n)continue;double v[2];
  for(int comp=0;comp<2;++comp){
   __uint128_t x=0,P=1;
   for(int q=0;q<cfg->count;++q){int p=cfg->rows[q].p;
    int a=posmod((long long)r[(size_t(q)*2+comp)*count+z]-int(x%p),p);
    int c=(a*inv[q])%p; x+=P*c;P*=p;
   }
   __int128 signed_x=x>P/2?-__int128(P-x):__int128(x);
   v[comp]=round_exact(signed_x,shift,precision);
  }
  if(precision==24)reinterpret_cast<float2*>(out)[size_t(j)*m+i]=make_float2(float(v[0]),float(v[1]));
  else out[size_t(j)*m+i]=make_double2(v[0],v[1]);
 }
}

__global__ void restore_r2(const int* r,double2* out,size_t count,int m,int n,int mp,
 int nmod,const __uint128_t* tab,const int* offs,__uint128_t P,int shift,int precision){
 for(size_t z=size_t(blockIdx.x)*blockDim.x+threadIdx.x;z<count;z+=size_t(gridDim.x)*blockDim.x){
  int i=z%mp,j=z/mp;if(i>=m||j>=n)continue;double v[2];
  for(int comp=0;comp<2;++comp){
   __uint128_t x=0;
   for(int q=0;q<nmod;++q){
    x+=tab[offs[q]+r[(size_t(q)*2+comp)*count+z]];
    if(x>=P)x-=P;
   }
   __int128 signed_x=x>P/2?-__int128(P-x):__int128(x);
   v[comp]=round_exact(signed_x,shift,precision);
  }
  if(precision==24)reinterpret_cast<float2*>(out)[size_t(j)*m+i]=make_float2(float(v[0]),float(v[1]));
  else out[size_t(j)*m+i]=make_double2(v[0],v[1]);
 }
}
struct Slot {Buffer a,b,g; cudaEvent_t available,encoded,computed;};

struct Group {Buffer a,b,g; cudaEvent_t available,encoded,computed;};
struct Engine {
 int m,n,k,mp,np,kp,method,shift,precision;size_t count;
 Config cfg;Buffer config,inverse,sf,totals,residues; Plan plan;
 std::array<Slot,3> slots;cudaStream_t enc,gemm,epi;cudaEvent_t start,stop;

 static constexpr int R1_GROUPS=3;
 std::array<Group,R1_GROUPS> groups; bool r1_ready=false; int fmax=0;
 size_t abytes=0,bbytes=0;

 std::array<Group,R1_GROUPS> groups2; bool r2_ready=false;
 std::vector<Bar> bar_host; std::vector<int> crt_offs_host;
 Buffer pack, crt_tab, crt_offs; __uint128_t Ptotal=1;
 std::vector<int> inverses_host;

 std::array<Group,2> egrp, ggrp; bool r3_ready=false, quant_ready=false;
 int r3_span=0, negrp=1, fmax_faces=0;
 Buffer qa, qb, encmods;
 std::vector<int> pass_start, pass_len;

 bool prof=false; std::vector<cudaEvent_t> pev; std::vector<int> plab; size_t pn=0;
 double stage[8]={0,0,0,0,0,0,0,0};
 void mark(int label,cudaStream_t s){
  if(!prof)return;
  if(pn>=pev.size()){cudaEvent_t e;CU(cudaEventCreate(&e));pev.push_back(e);plab.push_back(label);}
  plab[pn]=label;CU(cudaEventRecord(pev[pn],s));++pn;
 }
 Engine(int M,int N,int K,int meth,int sh,int prec,const Config& c):m(M),n(N),k(K),method(meth),shift(sh),precision(prec),cfg(c){
  if(c.count<1||c.count>32||M<1||N<1||K<1||!(meth==2||meth==3||meth==4)||!(prec==24||prec==53))throw std::runtime_error("invalid config");
  mp=(m+127)/128*128;np=(n+127)/128*128;kp=(k+127)/128*128;
  for(int q=0;q<c.count;++q)kp=std::min(kp,c.rows[q].ksafe/128*128);
  count=size_t(mp)*np;
  config.alloc(sizeof(Config));CU(cudaMemcpy(config.p,&cfg,sizeof(Config),cudaMemcpyHostToDevice));
  __uint128_t P=1;std::vector<int> inverses;
  for(int q=0;q<c.count;++q){int p=c.rows[q].p;int a=int(P%p),v=1;while((a*v)%p!=1&&v<p)++v;if(v==p)throw std::runtime_error("noncoprime moduli");inverses.push_back(v);if(P>(~(__uint128_t)0)/p)throw std::runtime_error("CRT exceeds 128 bits");P*=p;}
  inverse.alloc(inverses.size()*4);CU(cudaMemcpy(inverse.p,inverses.data(),inverses.size()*4,cudaMemcpyHostToDevice));
  inverses_host=inverses;Ptotal=P;
  sf.alloc(size_t(std::max(mp,np))*((kp+63)/64)*4);CU(cudaMemset(sf.p,0x40,sf.bytes));
  totals.alloc(4*count*8);residues.alloc(c.count*2*count*4);
  CU(cudaStreamCreateWithFlags(&enc,cudaStreamNonBlocking));CU(cudaStreamCreateWithFlags(&gemm,cudaStreamNonBlocking));CU(cudaStreamCreateWithFlags(&epi,cudaStreamNonBlocking));
  CU(cudaEventCreate(&start));CU(cudaEventCreate(&stop));
  abytes=size_t(mp)*kp/2;bbytes=size_t(np)*kp/2;
  for(auto& s:slots){s.a.alloc(abytes);s.b.alloc(bbytes);s.g.alloc(count*4);CU(cudaEventCreateWithFlags(&s.available,cudaEventDisableTiming));CU(cudaEventCreateWithFlags(&s.encoded,cudaEventDisableTiming));CU(cudaEventCreateWithFlags(&s.computed,cudaEventDisableTiming));CU(cudaEventRecord(s.available,epi));}
  plan.init(mp,np,kp,sf.p);CU(cudaDeviceSynchronize());
 }
 void ensure_r1(){
  if(r1_ready)return;
  int want=1;for(int q=0;q<cfg.count;++q)want=std::max(want,cfg.rows[q].faces);
  if(want<1||want>6)throw std::runtime_error("face count outside 1..6");
  for(auto& g:groups){
   g.a.alloc(abytes*size_t(want));g.b.alloc(bbytes*size_t(want));g.g.alloc(count*4*size_t(want));
   CU(cudaEventCreateWithFlags(&g.available,cudaEventDisableTiming));
   CU(cudaEventCreateWithFlags(&g.encoded,cudaEventDisableTiming));
   CU(cudaEventCreateWithFlags(&g.computed,cudaEventDisableTiming));
   CU(cudaEventRecord(g.available,epi));
  }
  CU(cudaDeviceSynchronize());fmax=want;r1_ready=true;
 }

 void branch_coefficients(const Row& r,BranchCoef& ca,BranchCoef& cb) const {
  for(int branch=0;branch<method;++branch){
   int ar=1,ai=0,br=1,bi=0;
   if(method==2){ai=bi=branch==0?r.root:-r.root;}
   else if(branch==1){ar=br=0;ai=bi=1;}
   else if(method==3&&branch==2){ai=bi=1;}
   else if(method==4&&branch==2){br=0;bi=1;}
   else if(method==4&&branch==3){ar=0;ai=1;}
   ca.tr[branch]=ar;ca.ti[branch]=ai;cb.tr[branch]=br;cb.ti[branch]=bi;
  }
  for(int branch=method;branch<4;++branch){ca.tr[branch]=ca.ti[branch]=cb.tr[branch]=cb.ti[branch]=0;}
 }
 void ensure_r2(){
  if(r2_ready)return;
  ensure_r1();
  if(Ptotal>((__uint128_t)1<<127))throw std::runtime_error("modulus product leaves no room for the CRT table");

  bar_host.clear();
  for(int q=0;q<cfg.count;++q){
   Bar b=make_bar(cfg.rows[q].p);bar_host.push_back(b);
   long long probe[]={0,1,-1,b.p,-b.p,b.p-1,-(long long)b.p+1,(1ll<<54),-(1ll<<54),(1ll<<54)-1,-((1ll<<54)-1),
                      (1ll<<53),-(1ll<<53),65535,-65535,58081,-58081};
   for(long long v:probe)if(mod_p(v,b)!=posmod(v,b.p))throw std::runtime_error("reciprocal residue disagrees with the remainder operator");
   unsigned long long s=0x9E3779B97F4A7C15ull^(unsigned long long)b.p;
   for(int t=0;t<4096;++t){
    s^=s<<13;s^=s>>7;s^=s<<17;
    long long v=(long long)(s%((1ull<<55)))-(1ll<<54);
    if(mod_p(v,b)!=posmod(v,b.p))throw std::runtime_error("reciprocal residue disagrees with the remainder operator");
   }
  }

  std::vector<unsigned> packed(size_t(cfg.count)*2*241,0u);
  for(int q=0;q<cfg.count;++q){const Row& r=cfg.rows[q];
   if(r.faces<1||r.faces>6)throw std::runtime_error("face count outside 1..6");
   for(int t=0;t<r.p;++t){unsigned wa=0,wb=0;
    for(int f=0;f<r.faces;++f){wa|=unsigned(r.lut[t*6+r.ia[f]])<<(4*f);wb|=unsigned(r.lut[t*6+r.ib[f]])<<(4*f);}
    packed[(size_t(q)*2+0)*241+t]=wa;packed[(size_t(q)*2+1)*241+t]=wb;}
  }
  pack.alloc(packed.size()*4);CU(cudaMemcpy(pack.p,packed.data(),packed.size()*4,cudaMemcpyHostToDevice));

  crt_offs_host.assign(cfg.count,0);size_t total=0;
  for(int q=0;q<cfg.count;++q){crt_offs_host[q]=int(total);total+=cfg.rows[q].p;}
  std::vector<__uint128_t> tab(total,0);
  for(int q=0;q<cfg.count;++q){int p=cfg.rows[q].p;
   __uint128_t Mq=Ptotal/p;int mq=int(Mq%p),y=1;
   while((mq*y)%p!=1&&y<p)++y;
   if(y==p)throw std::runtime_error("no inverse for the CRT table");
   __uint128_t step=0,cur=Mq;int e=y;
   while(e){if(e&1){step+=cur;if(step>=Ptotal)step-=Ptotal;}cur+=cur;if(cur>=Ptotal)cur-=Ptotal;e>>=1;}
   __uint128_t v=0;
   for(int t=0;t<p;++t){tab[crt_offs_host[q]+t]=v;v+=step;if(v>=Ptotal)v-=Ptotal;}
  }

  for(int trial=0;trial<256;++trial){
   __uint128_t x=0,Pc=1,y2=0;
   for(int q=0;q<cfg.count;++q){int p=cfg.rows[q].p;int res=(trial*7+q*13+1)%p;
    int a=posmod((long long)res-int(x%p),p);int c=(a*inverses_host[q])%p;x+=Pc*c;Pc*=p;
    y2+=tab[crt_offs_host[q]+res];if(y2>=Ptotal)y2-=Ptotal;}
   if(x!=y2)throw std::runtime_error("CRT table disagrees with the incremental reconstruction");
  }
  crt_tab.alloc(tab.size()*16);CU(cudaMemcpy(crt_tab.p,tab.data(),tab.size()*16,cudaMemcpyHostToDevice));
  crt_offs.alloc(crt_offs_host.size()*4);CU(cudaMemcpy(crt_offs.p,crt_offs_host.data(),crt_offs_host.size()*4,cudaMemcpyHostToDevice));

  size_t slots=size_t(method)*size_t(fmax);
  for(auto& g:groups2){
   g.a.alloc(abytes*slots);g.b.alloc(bbytes*slots);g.g.alloc(count*4*slots);
   CU(cudaEventCreateWithFlags(&g.available,cudaEventDisableTiming));
   CU(cudaEventCreateWithFlags(&g.encoded,cudaEventDisableTiming));
   CU(cudaEventCreateWithFlags(&g.computed,cudaEventDisableTiming));
   CU(cudaEventRecord(g.available,epi));
  }
  CU(cudaDeviceSynchronize());r2_ready=true;
 }

 void ensure_r3(int span){
  ensure_r2();
  int chunks=(k+kp-1)/kp;
  if(span<1||span>cfg.count)span=cfg.count;
  if(chunks>1)span=1;
  if(r3_ready&&span==r3_span)return;
  fmax_faces=1;for(int q=0;q<cfg.count;++q)fmax_faces=std::max(fmax_faces,cfg.rows[q].faces);
  if(!quant_ready){
   qa.alloc(size_t(m)*k*16);qb.alloc(size_t(n)*k*16);quant_ready=true;
  }
  pass_start.clear();pass_len.clear();
  for(int s=0;s<cfg.count;s+=span){pass_start.push_back(s);pass_len.push_back(std::min(span,cfg.count-s));}
  int maxslots=0;
  std::vector<EncMod> host(size_t(cfg.count)*2);
  const unsigned* packd=pack.as<unsigned>();
  for(size_t pi=0;pi<pass_start.size();++pi){
   int base=0;
   for(int g=0;g<pass_len[pi];++g){int q=pass_start[pi]+g;const Row& r=cfg.rows[q];
    BranchCoef ca{},cb{};branch_coefficients(r,ca,cb);
    for(int side=0;side<2;++side){
     EncMod& e=host[size_t(side)*cfg.count+q];
     e.bar=bar_host[q];e.faces=r.faces;e.slot_base=base;
     for(int t=0;t<4;++t){e.tr[t]=side?cb.tr[t]:ca.tr[t];e.ti[t]=side?cb.ti[t]:ca.ti[t];}
     e.pack=packd+(size_t(q)*2+size_t(side))*241;
    }
    base+=2*r.faces;
   }
   maxslots=std::max(maxslots,base);
  }
  encmods.alloc(host.size()*sizeof(EncMod));
  CU(cudaMemcpy(encmods.p,host.data(),host.size()*sizeof(EncMod),cudaMemcpyHostToDevice));
  negrp=std::min<int>(2,int(pass_start.size()));
  for(int i=0;i<2;++i){
   if(!r3_ready){
    CU(cudaEventCreateWithFlags(&egrp[i].available,cudaEventDisableTiming));
    CU(cudaEventCreateWithFlags(&egrp[i].encoded,cudaEventDisableTiming));
    CU(cudaEventCreateWithFlags(&egrp[i].computed,cudaEventDisableTiming));
    CU(cudaEventCreateWithFlags(&ggrp[i].available,cudaEventDisableTiming));
    CU(cudaEventCreateWithFlags(&ggrp[i].encoded,cudaEventDisableTiming));
    CU(cudaEventCreateWithFlags(&ggrp[i].computed,cudaEventDisableTiming));
   }
   if(i<negrp){egrp[i].a.alloc(abytes*size_t(maxslots));egrp[i].b.alloc(bbytes*size_t(maxslots));}
   else{egrp[i].a.release();egrp[i].b.release();}
   ggrp[i].g.alloc(count*4*size_t(method)*size_t(fmax_faces));
   CU(cudaEventRecord(egrp[i].available,epi));CU(cudaEventRecord(ggrp[i].available,epi));
  }
  CU(cudaDeviceSynchronize());r3_span=span;r3_ready=true;
 }
 float run(double2* A,double2* B,double2* D,bool overlap,bool profile=false){
  prof=profile;pn=0;for(auto&x:stage)x=0;
  cudaStream_t es=(overlap&&!prof)?enc:epi,gs=(overlap&&!prof)?gemm:epi;
  CU(cudaEventRecord(start,epi));CU(cudaStreamWaitEvent(es,start));CU(cudaStreamWaitEvent(gs,start));
  mark(6,epi);
  CU(cudaMemsetAsync(sf.p,0x40,sf.bytes,gs));size_t serial=0;
  mark(7,gs);
  for(int q=0;q<cfg.count;++q){auto&r=cfg.rows[q];Row*dr=&config.as<Config>()->rows[q];
   CU(cudaMemsetAsync(totals.p,0,totals.bytes,epi));
   mark(0,epi);
   for(int branch=0;branch<method;++branch){
    int ar=1,ai=0,br=1,bi=0;
    if(method==2){ai=bi=branch==0?r.root:-r.root;}
    else if(branch==1){ar=br=0;ai=bi=1;}
    else if(method==3&&branch==2){ai=bi=1;}
    else if(method==4&&branch==2){br=0;bi=1;}
    else if(method==4&&branch==3){ar=0;ai=1;}
    for(int off=0;off<k;off+=kp)for(int f=0;f<r.faces;++f){auto&s=slots[serial++%3];
     CU(cudaStreamWaitEvent(es,s.available));
     encode<<<blocks(s.a.bytes),256,0,es>>>(A,s.a.as<unsigned char>(),m,k,mp,kp,off,false,shift,precision,dr,r.ia[f],ar,ai);
     encode<<<blocks(s.b.bytes),256,0,es>>>(B,s.b.as<unsigned char>(),n,k,np,kp,off,true,shift,precision,dr,r.ib[f],br,bi);
     mark(1,es);
     CU(cudaEventRecord(s.encoded,es));CU(cudaStreamWaitEvent(gs,s.encoded));
     plan.run(s.a.p,s.b.p,s.g.p,gs);CU(cudaEventRecord(s.computed,gs));CU(cudaStreamWaitEvent(epi,s.computed));
     mark(2,gs);
     accumulate<<<blocks(count),256,0,epi>>>(s.g.as<float>(),totals.as<long long>()+branch*count,count,r.coeff[f]);
     mark(3,epi);
     CU(cudaEventRecord(s.available,epi));
    }
   }
   combine<<<blocks(count),256,0,epi>>>(totals.as<long long>(),residues.as<int>()+size_t(q)*2*count,count,dr,method);
   mark(4,epi);
  }
  restore<<<blocks(count),256,0,epi>>>(residues.as<int>(),D,count,m,n,mp,config.as<Config>(),inverse.as<int>(),-2*shift,precision);
  mark(5,epi);
  CU(cudaGetLastError());CU(cudaEventRecord(stop,epi));CU(cudaEventSynchronize(stop));float ms;CU(cudaEventElapsedTime(&ms,start,stop));
  if(prof)for(size_t i=1;i<pn;++i){float d;CU(cudaEventElapsedTime(&d,pev[i-1],pev[i]));stage[plab[i]]+=d;}
  return ms;
 }

 float run_r1(double2* A,double2* B,double2* D,bool overlap,bool profile=false){
  ensure_r1();
  prof=profile;pn=0;for(auto&x:stage)x=0;
  cudaStream_t es=(overlap&&!prof)?enc:epi,gs=(overlap&&!prof)?gemm:epi;
  CU(cudaEventRecord(start,epi));CU(cudaStreamWaitEvent(es,start));CU(cudaStreamWaitEvent(gs,start));
  mark(6,epi);
  CU(cudaMemsetAsync(sf.p,0x40,sf.bytes,gs));size_t serial=0;
  mark(7,gs);
  for(int q=0;q<cfg.count;++q){auto&r=cfg.rows[q];Row*dr=&config.as<Config>()->rows[q];

   CU(cudaMemsetAsync(totals.p,0,size_t(method)*count*8,epi));
   mark(0,epi);
   for(int branch=0;branch<method;++branch){
    int ar=1,ai=0,br=1,bi=0;
    if(method==2){ai=bi=branch==0?r.root:-r.root;}
    else if(branch==1){ar=br=0;ai=bi=1;}
    else if(method==3&&branch==2){ai=bi=1;}
    else if(method==4&&branch==2){br=0;bi=1;}
    else if(method==4&&branch==3){ar=0;ai=1;}
    FaceIdx fa{},fb{};FaceCoeff co{};
    for(int f=0;f<r.faces;++f){fa.v[f]=r.ia[f];fb.v[f]=r.ib[f];co.v[f]=r.coeff[f];}
    for(int off=0;off<k;off+=kp){auto&g=groups[serial++%R1_GROUPS];
     CU(cudaStreamWaitEvent(es,g.available));
     launch_encode_faces(r.faces,blocks(abytes),es,A,g.a.as<unsigned char>(),abytes,m,k,mp,kp,off,false,shift,precision,dr,fa,ar,ai);
     launch_encode_faces(r.faces,blocks(bbytes),es,B,g.b.as<unsigned char>(),bbytes,n,k,np,kp,off,true,shift,precision,dr,fb,br,bi);
     mark(1,es);
     CU(cudaEventRecord(g.encoded,es));CU(cudaStreamWaitEvent(gs,g.encoded));
     for(int f=0;f<r.faces;++f)
      plan.run(g.a.as<unsigned char>()+size_t(f)*abytes,g.b.as<unsigned char>()+size_t(f)*bbytes,g.g.as<float>()+size_t(f)*count,gs);
     CU(cudaEventRecord(g.computed,gs));CU(cudaStreamWaitEvent(epi,g.computed));
     mark(2,gs);
     launch_accumulate_faces(r.faces,blocks(count),epi,g.g.as<float>(),count,totals.as<long long>()+branch*count,count,co);
     mark(3,epi);
     CU(cudaEventRecord(g.available,epi));
    }
   }
   combine<<<blocks(count),256,0,epi>>>(totals.as<long long>(),residues.as<int>()+size_t(q)*2*count,count,dr,method);
   mark(4,epi);
  }
  restore<<<blocks(count),256,0,epi>>>(residues.as<int>(),D,count,m,n,mp,config.as<Config>(),inverse.as<int>(),-2*shift,precision);
  mark(5,epi);
  CU(cudaGetLastError());CU(cudaEventRecord(stop,epi));CU(cudaEventSynchronize(stop));float ms;CU(cudaEventElapsedTime(&ms,start,stop));
  if(prof)for(size_t i=1;i<pn;++i){float d;CU(cudaEventElapsedTime(&d,pev[i-1],pev[i]));stage[plab[i]]+=d;}
  return ms;
 }

 float run_r2(double2* A,double2* B,double2* D,bool overlap,bool profile=false){
  ensure_r2();
  prof=profile;pn=0;for(auto&x:stage)x=0;
  cudaStream_t es=(overlap&&!prof)?enc:epi,gs=(overlap&&!prof)?gemm:epi;
  CU(cudaEventRecord(start,epi));CU(cudaStreamWaitEvent(es,start));CU(cudaStreamWaitEvent(gs,start));
  mark(6,epi);
  CU(cudaMemsetAsync(sf.p,0x40,sf.bytes,gs));size_t serial=0;
  mark(7,gs);
  const unsigned* packd=pack.as<unsigned>();
  for(int q=0;q<cfg.count;++q){auto&r=cfg.rows[q];
   CU(cudaMemsetAsync(totals.p,0,size_t(method)*count*8,epi));
   mark(0,epi);
   BranchCoef ca{},cb{};branch_coefficients(r,ca,cb);
   FaceCoeff co{};for(int f=0;f<r.faces;++f)co.v[f]=r.coeff[f];
   for(int off=0;off<k;off+=kp){auto&g=groups2[serial++%R1_GROUPS];
    CU(cudaStreamWaitEvent(es,g.available));
    launch_encode_pair(method,r.faces,blocks(abytes),es,A,g.a.as<unsigned char>(),abytes,m,k,mp,kp,off,false,shift,precision,
                       bar_host[q],packd+(size_t(q)*2+0)*241,ca);
    launch_encode_pair(method,r.faces,blocks(bbytes),es,B,g.b.as<unsigned char>(),bbytes,n,k,np,kp,off,true,shift,precision,
                       bar_host[q],packd+(size_t(q)*2+1)*241,cb);
    mark(1,es);
    CU(cudaEventRecord(g.encoded,es));CU(cudaStreamWaitEvent(gs,g.encoded));
    for(int t=0;t<method*r.faces;++t)
     plan.run(g.a.as<unsigned char>()+size_t(t)*abytes,g.b.as<unsigned char>()+size_t(t)*bbytes,g.g.as<float>()+size_t(t)*count,gs);
    CU(cudaEventRecord(g.computed,gs));CU(cudaStreamWaitEvent(epi,g.computed));
    mark(2,gs);
    for(int branch=0;branch<method;++branch)
     launch_accumulate_faces(r.faces,blocks(count),epi,g.g.as<float>()+size_t(branch*r.faces)*count,count,
                             totals.as<long long>()+size_t(branch)*count,count,co);
    mark(3,epi);
    CU(cudaEventRecord(g.available,epi));
   }
   combine_r2<<<blocks(count),256,0,epi>>>(totals.as<long long>(),residues.as<int>()+size_t(q)*2*count,count,bar_host[q],r.denom,r.root,method);
   mark(4,epi);
  }
  restore_r2<<<blocks(count),256,0,epi>>>(residues.as<int>(),D,count,m,n,mp,cfg.count,
                                          crt_tab.as<__uint128_t>(),crt_offs.as<int>(),Ptotal,-2*shift,precision);
  mark(5,epi);
  CU(cudaGetLastError());CU(cudaEventRecord(stop,epi));CU(cudaEventSynchronize(stop));float ms;CU(cudaEventElapsedTime(&ms,start,stop));
  if(prof)for(size_t i=1;i<pn;++i){float d;CU(cudaEventElapsedTime(&d,pev[i-1],pev[i]));stage[plab[i]]+=d;}
  return ms;
 }

 float run_r3(double2* A,double2* B,double2* D,bool overlap,bool profile=false){
  ensure_r3(r3_span);
  prof=profile;pn=0;for(auto&x:stage)x=0;
  cudaStream_t es=(overlap&&!prof)?enc:epi,gs=(overlap&&!prof)?gemm:epi;
  CU(cudaEventRecord(start,epi));CU(cudaStreamWaitEvent(es,start));CU(cudaStreamWaitEvent(gs,start));
  mark(6,epi);
  CU(cudaMemsetAsync(sf.p,0x40,sf.bytes,gs));
  mark(7,gs);
  double scale=ldexp(1.0,shift);
  dim3 tb(32,8),ga((m+31)/32,(k+31)/32);
  quantize_a<<<ga,tb,0,es>>>(A,qa.as<longlong2>(),m,k,scale,precision);
  quantize_b<<<blocks(size_t(n)*k),256,0,es>>>(B,qb.as<longlong2>(),(long long)n*k,scale,precision);
  mark(6,es);
  const EncMod* encdev=encmods.as<EncMod>();
  size_t eserial=0,gserial=0;
  for(size_t pi=0;pi<pass_start.size();++pi){
   int s=pass_start[pi],len=pass_len[pi];
   auto& eg=egrp[eserial++%size_t(negrp)];
   CU(cudaStreamWaitEvent(es,eg.available));
   encode_group<<<blocks(abytes),256,0,es>>>(qa.as<longlong2>(),eg.a.as<unsigned char>(),abytes,m,k,mp,kp,0,method,len,encdev+s);
   encode_group<<<blocks(bbytes),256,0,es>>>(qb.as<longlong2>(),eg.b.as<unsigned char>(),bbytes,n,k,np,kp,0,method,len,encdev+cfg.count+s);
   mark(1,es);
   CU(cudaEventRecord(eg.encoded,es));CU(cudaStreamWaitEvent(gs,eg.encoded));
   for(int g=0;g<len;++g){int q=s+g;auto&r=cfg.rows[q];
    int slot_base=0;for(int h=0;h<g;++h)slot_base+=2*cfg.rows[s+h].faces;
    auto& gg=ggrp[gserial++%2];
    CU(cudaStreamWaitEvent(gs,gg.available));
    CU(cudaMemsetAsync(totals.p,0,size_t(method)*count*8,epi));
    mark(0,epi);
    for(int br=0;br<method;++br)for(int f=0;f<r.faces;++f){int slot=slot_base+br*r.faces+f;
     plan.run(eg.a.as<unsigned char>()+size_t(slot)*abytes,eg.b.as<unsigned char>()+size_t(slot)*bbytes,
              gg.g.as<float>()+size_t(br*r.faces+f)*count,gs);}
    CU(cudaEventRecord(gg.computed,gs));CU(cudaStreamWaitEvent(epi,gg.computed));
    mark(2,gs);
    FaceCoeff co{};for(int f=0;f<r.faces;++f)co.v[f]=r.coeff[f];
    for(int br=0;br<method;++br)
     launch_accumulate_faces(r.faces,blocks(count),epi,gg.g.as<float>()+size_t(br*r.faces)*count,count,
                             totals.as<long long>()+size_t(br)*count,count,co);
    mark(3,epi);
    CU(cudaEventRecord(gg.available,epi));
    combine_r2<<<blocks(count),256,0,epi>>>(totals.as<long long>(),residues.as<int>()+size_t(q)*2*count,count,bar_host[q],r.denom,r.root,method);
    mark(4,epi);
   }
   CU(cudaEventRecord(eg.available,epi));
  }
  restore_r2<<<blocks(count),256,0,epi>>>(residues.as<int>(),D,count,m,n,mp,cfg.count,
                                          crt_tab.as<__uint128_t>(),crt_offs.as<int>(),Ptotal,-2*shift,precision);
  mark(5,epi);
  CU(cudaGetLastError());CU(cudaEventRecord(stop,epi));CU(cudaEventSynchronize(stop));float ms;CU(cudaEventElapsedTime(&ms,start,stop));
  if(prof)for(size_t i=1;i<pn;++i){float d;CU(cudaEventElapsedTime(&d,pev[i-1],pev[i]));stage[plab[i]]+=d;}
  return ms;
 }
 ~Engine(){cudaDeviceSynchronize();for(auto&e:pev)cudaEventDestroy(e);if(r3_ready)for(int i=0;i<2;++i){cudaEventDestroy(egrp[i].available);cudaEventDestroy(egrp[i].encoded);cudaEventDestroy(egrp[i].computed);cudaEventDestroy(ggrp[i].available);cudaEventDestroy(ggrp[i].encoded);cudaEventDestroy(ggrp[i].computed);}if(r2_ready)for(auto&g:groups2){cudaEventDestroy(g.available);cudaEventDestroy(g.encoded);cudaEventDestroy(g.computed);}for(auto&s:slots){cudaEventDestroy(s.available);cudaEventDestroy(s.encoded);cudaEventDestroy(s.computed);}if(r1_ready)for(auto&g:groups){cudaEventDestroy(g.available);cudaEventDestroy(g.encoded);cudaEventDestroy(g.computed);}cudaEventDestroy(start);cudaEventDestroy(stop);cudaStreamDestroy(enc);cudaStreamDestroy(gemm);cudaStreamDestroy(epi);}
};
extern "C" void* create(int m,int n,int k,int method,int shift,int precision,const Config*c){try{return new Engine(m,n,k,method,shift,precision,*c);}catch(std::exception&e){last_error=e.what();return nullptr;}}
extern "C" float run(void*h,void*a,void*b,void*c,int overlap){try{return static_cast<Engine*>(h)->run((double2*)a,(double2*)b,(double2*)c,overlap,false);}catch(std::exception&e){last_error=e.what();return -1;}}

extern "C" float run_profiled(void*h,void*a,void*b,void*c){try{return static_cast<Engine*>(h)->run((double2*)a,(double2*)b,(double2*)c,false,true);}catch(std::exception&e){last_error=e.what();return -1;}}

extern "C" int prepare_r1(void*h){try{static_cast<Engine*>(h)->ensure_r1();return 0;}catch(std::exception&e){last_error=e.what();return 1;}}
extern "C" float run_r1(void*h,void*a,void*b,void*c,int overlap){try{return static_cast<Engine*>(h)->run_r1((double2*)a,(double2*)b,(double2*)c,overlap,false);}catch(std::exception&e){last_error=e.what();return -1;}}
extern "C" float run_r1_profiled(void*h,void*a,void*b,void*c){try{return static_cast<Engine*>(h)->run_r1((double2*)a,(double2*)b,(double2*)c,false,true);}catch(std::exception&e){last_error=e.what();return -1;}}

extern "C" unsigned long long r1_group_bytes(void*h){auto*e=static_cast<Engine*>(h);if(!e->r1_ready)return 0ull;
 return (unsigned long long)Engine::R1_GROUPS*(unsigned long long)e->fmax*((unsigned long long)e->abytes+(unsigned long long)e->bbytes+(unsigned long long)e->count*4ull);}

extern "C" int prepare_r2(void*h){try{static_cast<Engine*>(h)->ensure_r2();return 0;}catch(std::exception&e){last_error=e.what();return 1;}}
extern "C" float run_r2(void*h,void*a,void*b,void*c,int overlap){try{return static_cast<Engine*>(h)->run_r2((double2*)a,(double2*)b,(double2*)c,overlap,false);}catch(std::exception&e){last_error=e.what();return -1;}}
extern "C" float run_r2_profiled(void*h,void*a,void*b,void*c){try{return static_cast<Engine*>(h)->run_r2((double2*)a,(double2*)b,(double2*)c,false,true);}catch(std::exception&e){last_error=e.what();return -1;}}

extern "C" int prepare_r3(void*h,int span){try{static_cast<Engine*>(h)->ensure_r3(span);return 0;}catch(std::exception&e){last_error=e.what();return 1;}}
extern "C" int r3_span(void*h){return static_cast<Engine*>(h)->r3_span;}
extern "C" float run_r3(void*h,void*a,void*b,void*c,int overlap){try{return static_cast<Engine*>(h)->run_r3((double2*)a,(double2*)b,(double2*)c,overlap,false);}catch(std::exception&e){last_error=e.what();return -1;}}
extern "C" float run_r3_profiled(void*h,void*a,void*b,void*c){try{return static_cast<Engine*>(h)->run_r3((double2*)a,(double2*)b,(double2*)c,false,true);}catch(std::exception&e){last_error=e.what();return -1;}}
extern "C" unsigned long long r2_group_bytes(void*h){auto*e=static_cast<Engine*>(h);if(!e->r2_ready)return 0ull;
 return (unsigned long long)Engine::R1_GROUPS*(unsigned long long)e->method*(unsigned long long)e->fmax*((unsigned long long)e->abytes+(unsigned long long)e->bbytes+(unsigned long long)e->count*4ull);}

extern "C" int stage_times(void*h,double*out){try{auto*e=static_cast<Engine*>(h);for(int i=0;i<8;++i)out[i]=e->stage[i];return 0;}catch(std::exception&x){last_error=x.what();return 1;}}
extern "C" void destroy(void*h){delete static_cast<Engine*>(h);}

__global__ void reference(const double2*A,const double2*B,double2*C,int m,int n,int k,int shift,int precision){
 for(size_t z=size_t(blockIdx.x)*blockDim.x+threadIdx.x;z<size_t(m)*n;z+=size_t(gridDim.x)*blockDim.x){
  int i=z%m,j=z/m;__int128 re=0,im=0;
  for(int q=0;q<k;++q){auto a=A[i+size_t(q)*m],b=B[q+size_t(j)*k];
   long long ar=__double2ll_rn(ldexp(a.x,shift)),ai=__double2ll_rn(ldexp(a.y,shift)),br=__double2ll_rn(ldexp(b.x,shift)),bi=__double2ll_rn(ldexp(b.y,shift));
   re+=(__int128)ar*br-(__int128)ai*bi;im+=(__int128)ar*bi+(__int128)ai*br;
  }C[z]=make_double2(round_exact(re,-2*shift,precision),round_exact(im,-2*shift,precision));
 }
}
extern "C" int exact_reference(void*A,void*B,void*C,int m,int n,int k,int shift,int precision){try{reference<<<blocks(size_t(m)*n),256>>>((double2*)A,(double2*)B,(double2*)C,m,n,k,shift,precision);CU(cudaDeviceSynchronize());return 0;}catch(std::exception&e){last_error=e.what();return 1;}}
extern "C" float native(void*A,void*B,void*C,int m,int n,int k,int precision,int three,int reps){
 try{cublasHandle_t h;BL(cublasCreate(&h));BL(cublasSetMathMode(h,three>=2?CUBLAS_PEDANTIC_MATH:CUBLAS_DEFAULT_MATH));three%=2;
  cudaEvent_t a,b;CU(cudaEventCreate(&a));CU(cudaEventCreate(&b));
  cuDoubleComplex z1=make_cuDoubleComplex(1,0),z0=make_cuDoubleComplex(0,0);cuComplex c1=make_cuComplex(1,0),c0=make_cuComplex(0,0);
  auto call=[&](){if(precision==53){if(three)BL(cublasZgemm3m(h,CUBLAS_OP_N,CUBLAS_OP_N,m,n,k,&z1,(cuDoubleComplex*)A,m,(cuDoubleComplex*)B,k,&z0,(cuDoubleComplex*)C,m));else BL(cublasZgemm(h,CUBLAS_OP_N,CUBLAS_OP_N,m,n,k,&z1,(cuDoubleComplex*)A,m,(cuDoubleComplex*)B,k,&z0,(cuDoubleComplex*)C,m));}
   else{if(three)BL(cublasCgemm3m(h,CUBLAS_OP_N,CUBLAS_OP_N,m,n,k,&c1,(cuComplex*)A,m,(cuComplex*)B,k,&c0,(cuComplex*)C,m));else BL(cublasCgemm(h,CUBLAS_OP_N,CUBLAS_OP_N,m,n,k,&c1,(cuComplex*)A,m,(cuComplex*)B,k,&c0,(cuComplex*)C,m));}};
  call();CU(cudaEventRecord(a));for(int i=0;i<reps;++i)call();CU(cudaEventRecord(b));CU(cudaEventSynchronize(b));float ms;CU(cudaEventElapsedTime(&ms,a,b));cudaEventDestroy(a);cudaEventDestroy(b);cublasDestroy(h);return ms/reps;
 }catch(std::exception&e){last_error=e.what();return -1;}
}
extern "C" int cublas_version(){cublasHandle_t h;int v=0;if(cublasCreate(&h)==0){cublasGetVersion(h,&v);cublasDestroy(h);}return v;}
