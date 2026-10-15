#include "awe_blas.hpp"
#include <cuda_runtime.h>
#include <vector>
#include <random>
#include <cstdio>
#include <stdexcept>
#include <cstring>
#include <algorithm>
void check(awe_status s){if(s)throw std::runtime_error(awe_status_string(s));}
void cu(cudaError_t s){if(s)throw std::runtime_error(cudaGetErrorString(s));}
struct Buffer{void* p=nullptr;explicit Buffer(size_t n){if(n)cu(cudaMalloc(&p,n));}~Buffer(){if(p)cudaFree(p);}};
size_t idx(int i,int j,int ld,bool row){return row?size_t(i)*ld+j:i+size_t(j)*ld;}
void run(int m,int n,int k,awe_method method,bool row,bool ta,bool tb,bool i32,int streams,int chunk,int mode,int buffers=3){
 awe_gemm_desc d;awe_gemm_desc_init(&d);d.m=m;d.n=n;d.k=k;d.layout=row?AWE_ROW_MAJOR:AWE_COLUMN_MAJOR;d.trans_a=ta?AWE_OP_T:AWE_OP_N;d.trans_b=tb?AWE_OP_T:AWE_OP_N;d.data_type=i32?AWE_I8_I32:AWE_I8_I64;
 int ar=ta?k:m,ac=ta?m:k,br=tb?n:k,bc=tb?k:n;
 d.lda=std::max(1,row?ac:ar)+3;d.ldb=std::max(1,row?bc:br)+5;d.ldc=std::max(1,row?n:m)+7;
 size_t na=size_t(d.lda)*(row?ar:ac),nb=size_t(d.ldb)*(row?br:bc),nc=size_t(d.ldc)*(row?m:n);
 std::vector<int8_t>a(na,73),b(nb,61);std::mt19937 rng(81873+k+m);
 auto value=[&](int r){return int8_t(mode==1?-128:mode==2?(rng()%2?127:-127):mode==3?int(rng()%9)-4:mode==4?((rng()%10)?0:int(rng()%256)-128):mode==5?r-128:int(rng()%256)-128);};
 for(int i=0;i<m;++i)for(int q=0;q<k;++q)a[idx(ta?q:i,ta?i:q,d.lda,row)]=value(i);
 for(int j=0;j<n;++j)for(int q=0;q<k;++q)b[idx(tb?j:q,tb?q:j,d.ldb,row)]=value(j);
 Buffer da(na),db(nb),dc(nc*(i32?4:8));if(na)cu(cudaMemcpy(da.p,a.data(),na,cudaMemcpyHostToDevice));if(nb)cu(cudaMemcpy(db.p,b.data(),nb,cudaMemcpyHostToDevice));cu(cudaMemset(dc.p,0x5a,nc*(i32?4:8)));
 awe_options o;awe_options_init(&o);o.method=method;o.stream_count=streams;o.buffer_count=buffers;o.k_chunk=chunk;o.profiling=1;
 awe::plan p;check(p.create(d,&o));size_t bytes=0;check(p.workspace_size(bytes));Buffer workspace(bytes);awe_execution_stats st{};st.struct_size=sizeof(st);cudaStream_t caller;cu(cudaStreamCreateWithFlags(&caller,cudaStreamNonBlocking));
 check(p.execute(da.p,db.p,dc.p,workspace.p,bytes,caller,&st));
 std::vector<int64_t> result(nc);if(i32){std::vector<int32_t> v(nc);cu(cudaMemcpy(v.data(),dc.p,nc*4,cudaMemcpyDeviceToHost));std::copy(v.begin(),v.end(),result.begin());}else cu(cudaMemcpy(result.data(),dc.p,nc*8,cudaMemcpyDeviceToHost));
 for(int i=0;i<m;++i)for(int j=0;j<n;++j){int64_t ref=0;for(int q=0;q<k;++q)ref+=int(a[idx(ta?q:i,ta?i:q,d.lda,row)])*int(b[idx(tb?j:q,tb?q:j,d.ldb,row)]);size_t z=idx(i,j,d.ldc,row);if(ref!=result[z]){fprintf(stderr,"mismatch m=%d n=%d k=%d method=%d row=%d ta=%d tb=%d i=%d j=%d got=%lld ref=%lld\n",m,n,k,int(method),row,ta,tb,i,j,(long long)result[z],(long long)ref);throw std::runtime_error("exact CPU reference mismatch");}}
 int64_t sentinel=i32?int64_t(0x5a5a5a5a):int64_t(0x5a5a5a5a5a5a5a5a);for(int major=0;major<(row?m:n);++major)for(int minor=(row?n:m);minor<d.ldc;++minor)if(result[size_t(major)*d.ldc+minor]!=sentinel)throw std::runtime_error("output padding modified");
 awe_plan_info info{};info.struct_size=sizeof(info);check(awe_plan_get_info(p.get(),&info));awe_execution_stats last{};last.struct_size=sizeof(last);check(awe_plan_get_last_stats(p.get(),&last));if(last.total_ms!=st.total_ms)throw std::runtime_error("stats mismatch");
 if(k&&method!=AWE_METHOD_RADIX13&&mode==3&&info.face_count!=1)throw std::runtime_error("input-aware selection failed");
 if(bytes&&awe_plan_execute(p.get(),da.p,db.p,dc.p,workspace.p,bytes-1,caller,nullptr)!=AWE_INSUFFICIENT_WORKSPACE)throw std::runtime_error("workspace rejection failed");
 printf("{\"case\":\"cpu_full\",\"m\":%d,\"n\":%d,\"k\":%d,\"method\":%d,\"row\":%d,\"ta\":%d,\"tb\":%d,\"i32\":%d,\"streams\":%d,\"buffers\":%d,\"mode\":%d,\"faces\":%d,\"period\":%d,\"outputs\":%d,\"mismatch\":0,\"total_ms\":%.6f,\"selection_ms\":%.6f,\"encode_ms\":%.6f,\"gemm_ms\":%.6f,\"reconstruction_ms\":%.6f,\"wait_ms\":%.6f}\n",m,n,k,int(method),row,ta,tb,i32,streams,buffers,mode,info.face_count,info.payout_period,m*n,st.total_ms,st.selection_ms,st.encode_ms,st.gemm_ms,st.reconstruction_ms,st.wait_ms);fflush(stdout);cu(cudaStreamDestroy(caller));
}
int main(){try{
 for(auto method:{AWE_METHOD_AWE,AWE_METHOD_RADIX13}){
  run(256,256,1,method,false,false,false,false,3,0,5);
  for(bool row:{false,true})for(bool ta:{false,true})for(bool tb:{false,true})run(17,19,257,method,row,ta,tb,(row!=ta),3,128,0);
  for(int mode:{1,2,3,4})for(int streams:{1,3})run(31,7,131,method,false,false,false,false,streams,128,mode);
  run(3,5,116481,method,false,false,false,false,3,116480,1);
  run(2,3,131071,method,true,true,true,true,3,0,1);
  run(3,5,513,method,false,true,false,false,3,128,0,1);
  run(3,5,0,method,true,false,false,true,3,0,0);
 }
 // Single-period execution with padded output, every layout/transpose and buffer reuse.
 for(bool row:{false,true})for(bool ta:{false,true})for(bool tb:{false,true})for(int mode:{0,3})
  run(129,259,131,AWE_METHOD_AWE,row,ta,tb,(row!=ta),3,0,mode,1);
 awe_gemm_desc d;awe_gemm_desc_init(&d);d.m=1;d.n=1;d.k=131072;d.lda=1;d.ldb=d.k;d.data_type=AWE_I8_I32;awe_plan* p=nullptr;
 if(awe_plan_create(&p,&d,nullptr)!=AWE_INPUT_OUT_OF_RANGE)throw std::runtime_error("INT32 overflow policy failed");
 d.k=1;d.ldb=1;d.alpha=2;if(awe_plan_create(&p,&d,nullptr)!=AWE_NOT_SUPPORTED)throw std::runtime_error("alpha policy failed");
 d.alpha=1;d.m=0;d.n=0;if(awe_plan_create(&p,&d,nullptr)!=AWE_SUCCESS)throw std::runtime_error("empty create failed");check(awe_plan_execute(p,nullptr,nullptr,nullptr,nullptr,0,nullptr,nullptr));awe_plan_destroy(p);
 puts("{\"status\":\"PASS\",\"reference\":\"CPU INT64 all logical outputs\"}");return 0;
 }catch(std::exception const& e){fprintf(stderr,"%s\n",e.what());return 1;}}
