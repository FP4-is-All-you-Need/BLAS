#define awe_ozaki2_create awe_i8_create
#define awe_ozaki2_destroy awe_i8_destroy
#define awe_ozaki2_info awe_i8_info
#define awe_ozaki2_execute awe_i8_execute
#include "i8_backend.cuh"
#undef awe_ozaki2_create
#undef awe_ozaki2_destroy
#undef awe_ozaki2_info
#undef awe_ozaki2_execute
#include "f64_backend.cuh"
struct CombinedState {bool f64;void* inner;};
extern "C" awe_status awe_ozaki2_create(void** state,const awe_gemm_desc* d,const awe_options* o){
 if(!state||!d||!o||d->struct_size!=sizeof(*d)||o->struct_size!=sizeof(*o))return AWE_INVALID_ARGUMENT;*state=nullptr;
 try{auto s=std::make_unique<CombinedState>();s->f64=d->data_type==AWE_F64_F64;s->inner=nullptr;
 if(s->f64){auto e=std::make_unique<f64_impl::Engine>();e->init(*d,*o);s->inner=e.release();}
 else {auto rc=awe_i8_create(&s->inner,d,o);if(rc)return rc;}*state=s.release();return AWE_SUCCESS;
 }catch(f64_impl::Failure f){return f.status;}catch(...){return AWE_INTERNAL_ERROR;}}
extern "C" void awe_ozaki2_destroy(void* state){if(!state)return;auto s=(CombinedState*)state;if(s->f64)delete (f64_impl::Engine*)s->inner;else awe_i8_destroy(s->inner);delete s;}
extern "C" awe_status awe_ozaki2_info(const void* state,awe_plan_info* out){if(!state||!out||out->struct_size!=sizeof(*out))return AWE_INVALID_ARGUMENT;auto s=(const CombinedState*)state;if(s->f64){*out=((f64_impl::Engine*)s->inner)->info;return AWE_SUCCESS;}return awe_i8_info(s->inner,out);}
extern "C" awe_status awe_ozaki2_execute(void* state,const void* a,const void* b,void* c,void* w,size_t bytes,cudaStream_t stream,awe_execution_stats* stats){
 if(!state||(stats&&stats->struct_size!=sizeof(*stats)))return AWE_INVALID_ARGUMENT;auto s=(CombinedState*)state;if(!s->f64)return awe_i8_execute(s->inner,a,b,c,w,bytes,stream,stats);auto e=(f64_impl::Engine*)s->inner;
 try{e->execute(a,b,c,w,bytes,stream,stats);return AWE_SUCCESS;}catch(f64_impl::Failure f){e->drain();return f.status;}catch(...){e->drain();return AWE_INTERNAL_ERROR;}}
