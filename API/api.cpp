#include <AWE/awe.h>
#include "../Routines/Common/dispatch.hpp"
#include "../Platforms/NVIDIA/CUDA-Common/Common/host_options.h"
#include "../Routines/Common/engine_loader.hpp"
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <cstring>
#include <filesystem>
#include <memory>
#include <sstream>

namespace {
thread_local std::string error;
void ck(cudaError_t result) {
    if (result != cudaSuccess) throw awe::ApiError(AWE_RUNTIME_ERROR, cudaGetErrorString(result));
}
template<class F> awe_status protect(F f) noexcept {
    try { error.clear(); f(); return AWE_SUCCESS; }
    catch (const awe::ApiError &e) { error=e.what(); return e.status; }
    catch (const std::exception &e) { error=e.what(); return AWE_RUNTIME_ERROR; }
    catch (...) { error="unknown runtime error"; return AWE_RUNTIME_ERROR; }
}
std::string directory_of_api() {
    Dl_info info{};
    awe::require(dladdr(reinterpret_cast<void *>(&awe_plan_create), &info) && info.dli_fname,
                 AWE_RUNTIME_ERROR, "cannot locate libawe.so");
    return std::filesystem::canonical(info.dli_fname).parent_path().string();
}
template<class F> F symbol(const std::shared_ptr<awe::EngineModule> &m, const std::string &name) {
    return reinterpret_cast<F>(m->symbol(name.c_str()));
}
bool overlap(const void *a, size_t an, const void *b, size_t bn) {
    auto x=reinterpret_cast<uintptr_t>(a), y=reinterpret_cast<uintptr_t>(b);
    awe::require(x<=UINTPTR_MAX-an && y<=UINTPTR_MAX-bn, AWE_INVALID_ARGUMENT, "pointer range overflow");
    return x<y+bn && y<x+an;
}
}
struct awe_plan {
    awe_descriptor d{};
    awe_effective_options effective{};
    awe::Selection selection;
    std::shared_ptr<awe::EngineModule> module, support;
    void *context=nullptr, *scratch=nullptr, *copy=nullptr, *product=nullptr, *commit=nullptr;
    int device=0;
    cudaStream_t stream=nullptr;
    cudaEvent_t ready=nullptr;
    int (*run)(void *,const void *,const void *,void *,int,double *)=nullptr;
    void (*destroy)(void *)=nullptr;
    const char *(*last_error)()=nullptr;
    std::string info;
    ~awe_plan() {
        if (context) destroy(context);
        if (scratch) cudaFree(scratch);
        if (product) cudaFree(product);
        if (ready) cudaEventDestroy(ready);
        if (stream) cudaStreamDestroy(stream);
    }
};
extern "C" void awe_descriptor_init(awe_descriptor *d, awe_routine r) {
    if (!d) return;
    *d={}; d->struct_size=sizeof(*d); d->routine=r; d->backend=AWE_K96; d->alpha.real=1;
}
extern "C" const char *awe_last_error(void) { return error.c_str(); }
extern "C" awe_status awe_validate(const awe_descriptor *d) {
    return protect([&] { awe::require(d,AWE_INVALID_ARGUMENT,"null descriptor"); (void)awe::select(*d); });
}
extern "C" void awe_options_init(awe_options *options) {
    if (options) { *options={}; options->struct_size=sizeof(*options); }
}
extern "C" awe_status awe_plan_create(awe_plan **out, const awe_descriptor *d, const char *directory) {
    awe_options options; awe_options_init(&options);
    return awe_plan_create_with_options(out,d,&options,directory);
}
extern "C" awe_status awe_plan_create_with_options(awe_plan **out, const awe_descriptor *d, const awe_options *options, const char *directory) {
    if (out) *out=nullptr;
    return protect([&] {
        awe::require(out && d,AWE_INVALID_ARGUMENT,"null create argument");
        awe::require(options && (options->struct_size == sizeof(*options) ||
                     options->struct_size == offsetof(awe_options, engine_name) ||
                     options->struct_size == offsetof(awe_options, quantize_bits)),
                     AWE_NOT_SUPPORTED, "options size mismatch");
        const int bits = options->struct_size >= offsetof(awe_options, engine_name) ? options->quantize_bits : 0;
        const char *name = options->struct_size == sizeof(*options) ? options->engine_name : nullptr;
        auto selected = awe::select(*d, bits, name);
        const awe_engine_configuration config{sizeof(awe_engine_configuration),options->pacing,options->chunks,options->order,
                                             options->encoder_minb,options->pingpong,options->encoder};
        awe::require(awe_configuration_supported(&config),AWE_NOT_SUPPORTED,"only the measured default tuning is public");
        int device; ck(cudaGetDevice(&device));
        cudaDeviceProp prop{}; ck(cudaGetDeviceProperties(&prop,device));
        awe::require(prop.major==10 && prop.minor==(d->backend==AWE_K96 ? 3 : 0), AWE_NOT_SUPPORTED,
                     "this binary requires B300 sm_103a for K96 or B200 sm_100a for K64");
        auto dir=directory ? std::string(directory) : directory_of_api();
        auto module=awe::EngineModule::open(dir+"/libawe_"+selected.engine+".so");
        std::lock_guard<std::mutex> lock(module->mutex());
        auto p=std::make_unique<awe_plan>();
        p->d=*d; p->device=device; p->selection=selected; p->module=module;
        const std::string prefix=selected.operation==10 ? "t_" : selected.element_bytes==1 ? "i_" : selected.native_adapter ? (selected.complex ? "v_" : "u_") : selected.element_bytes==4 ? "f_" : selected.complex ? (selected.element_bytes==8 ? "c_" : "z_") : "d_";
        p->destroy=symbol<void(*)(void *)>(module,prefix+"destroy");
        p->last_error=symbol<const char *(*)()>(module,prefix+"last_error");
        p->run=symbol<int(*)(void *,const void *,const void *,void *,int,double *)>(module,prefix+"run");
        auto create=symbol<void *(*)(int,int,int,long long,long long,long long,int,const awe_engine_configuration *)>(module,prefix+"create_configured");
        const bool rank2 = selected.operation == 7 || selected.operation == 8;
        p->context=create(int(d->m),int(d->n),int(d->k),d->lda,rank2 && !selected.native_adapter ? d->k : d->ldb,d->ldc,
                          awe::create_operation(selected),&config);
        awe::require(p->context,AWE_RUNTIME_ERROR,p->last_error());
        awe::require(symbol<int(*)(void *,int)>(module,prefix+"poison")(p->context,0)>=0,
                     AWE_RUNTIME_ERROR,"cannot disable engine poisoning");
        awe_engine_effective ef{};
        awe::require(symbol<int(*)(void *,awe_engine_effective *)>(module,prefix+"effective_options")(p->context,&ef)==0,
                     AWE_RUNTIME_ERROR,"cannot query effective settings");
        p->effective={ef.pace,ef.chunks,ef.crt_chunks,ef.item_order,ef.encoder_minb,ef.pingpong,
                      ef.skip_b_padding,ef.pdl,ef.shift_form,ef.extra_fx,ef.encoder_fast,ef.moduli};
        if (selected.operation!=1 && !selected.native_adapter) {
            p->support=awe::EngineModule::open(dir+(selected.element_bytes==4 ? "/libawe_level3_float.so" : selected.complex ? "/libawe_level3_complex.so" : "/libawe_level3_real.so"));
            p->copy=p->support->symbol(selected.element_bytes==4 ? (rank2 || selected.operation==2 ? "support_float_transpose" : "support_float_matrix") : selected.complex ? ((selected.operation==2 || selected.operation==4 || selected.operation==9 || rank2) ? "support_transpose_into" : "support_matrix_copy") : ((selected.operation==2 || rank2) ? "support_transpose_nk" : selected.operation==5 ? "support_symm_full" : "support_tril_copy"));
            ck(cudaMalloc(&p->scratch,size_t(d->m)*size_t(d->k)*(selected.element_bytes ? selected.element_bytes : selected.complex ? 16 : 8)));
            ck(cudaStreamCreateWithFlags(&p->stream,cudaStreamNonBlocking));
            if (selected.complex) ck(cudaEventCreate(&p->ready));
            if (rank2 || (selected.general_adapter && selected.operation == 2)) {
                ck(cudaMalloc(&p->product, size_t(d->m)*size_t(d->n)*(selected.element_bytes ? selected.element_bytes : selected.complex ? 16 : 8)));
                p->commit=p->support->symbol(selected.element_bytes==4 ? "support_float_commit" : selected.complex ? "support_commit" : selected.general_adapter && selected.operation == 2 ? "support_copy_lower" : "support_xxt_lower");
            }
        }
        std::ostringstream info;
        info<<"{\"engine\":\""<<selected.engine<<"\",\"engine_name\":\""<<(name ? name : awe::public_engine_name(selected.engine))<<"\",\"backend\":"<<d->backend
            <<",\"quantize_bits\":"<<selected.bits<<",\"device\":"<<device
            <<",\"create_operation\":"<<awe::create_operation(selected)
            <<",\"native_adapter\":"<<(selected.native_adapter ? "true" : "false")
            <<",\"operation\":"<<selected.operation<<",\"moduli\":"<<p->effective.moduli
            <<",\"pace\":"<<p->effective.pace<<",\"chunks\":"<<p->effective.chunks
            <<",\"crt_chunks\":"<<p->effective.crt_chunks<<",\"item_order\":"<<p->effective.item_order
            <<",\"encoder_minb\":"<<p->effective.encoder_minb<<",\"pingpong\":"<<p->effective.pingpong
            <<",\"skip_b_padding\":"<<p->effective.skip_b_padding<<",\"pdl\":"<<p->effective.pdl
            <<",\"shift_form\":"<<p->effective.shift_form<<",\"extra_fx\":"<<p->effective.extra_fx
            <<",\"encoder_fast\":"<<p->effective.encoder_fast<<",\"poison\":0}";
        p->info=info.str();
        *out=p.release();
    });
}
namespace {
void check_device(const awe_plan *p) {
    awe::require(p,AWE_INVALID_ARGUMENT,"null plan");
    int device; ck(cudaGetDevice(&device));
    awe::require(device==p->device,AWE_INVALID_ARGUMENT,"active device differs from the plan device");
}
void check_pointer(const void *x, int device) {
    cudaPointerAttributes attr{}; ck(cudaPointerGetAttributes(&attr,x));
    awe::require(attr.type==cudaMemoryTypeDevice && attr.device==device, AWE_INVALID_ARGUMENT,
                 "matrix must be a device allocation on the plan device");
}
void execute(awe_plan *p,const void *a,const void *b,void *c) {
    check_device(p);
    const auto &d=p->d; auto s=p->selection;
    const bool rank=s.operation==2 || s.operation==4 || s.operation==9;
    const bool rank2=s.operation==7 || s.operation==8;
    awe::require(a && c && (rank ? b==nullptr : b!=nullptr),AWE_INVALID_ARGUMENT,"null matrix or non-null B for rank-k");
    const size_t width=s.element_bytes ? s.element_bytes : s.complex ? 16 : 8;
    const size_t ab=size_t(d.lda)*d.k*width, bb=size_t(d.ldb)*(rank2 ? d.k : d.n)*width, cb=size_t(d.ldc)*d.n*(s.element_bytes==1 ? sizeof(int32_t) : width);
    awe::require(!overlap(a,ab,c,cb) && (rank || !overlap(b,bb,c,cb)),AWE_INVALID_ARGUMENT,"output overlaps an input");
    check_pointer(a,p->device); check_pointer(c,p->device); if (!rank) check_pointer(b,p->device);
    std::lock_guard<std::mutex> lock(p->module->mutex());
    const void *ea=a, *eb=b;
    if (s.operation!=1 && !s.native_adapter) {
        std::lock_guard<std::mutex> support_lock(p->support->mutex());
        if (s.element_bytes==4) {
            if (rank || rank2)
                reinterpret_cast<void(*)(dim3,dim3,cudaStream_t,const float *,size_t,size_t,float *,size_t,size_t)>(p->copy)
                    (dim3((d.n+31)/32,(d.k+31)/32),dim3(32,8),p->stream,static_cast<const float *>(rank2 ? b : a),d.n,d.k,static_cast<float *>(p->scratch),d.k,0);
            else
                reinterpret_cast<void(*)(dim3,dim3,cudaStream_t,const float *,size_t,float *,int)>(p->copy)
                    (dim3(4096),dim3(256),p->stream,static_cast<const float *>(a),d.n,static_cast<float *>(p->scratch),int(s.operation==5));
        } else if (!s.complex && (rank || rank2)) {
            reinterpret_cast<void(*)(dim3,dim3,cudaStream_t,const double *,size_t,size_t,double *)>(p->copy)
                (dim3((d.n+31)/32,(d.k+31)/32),dim3(32,8),p->stream,static_cast<const double *>(rank2 ? b : a),d.n,d.k,static_cast<double *>(p->scratch));
        } else if (!s.complex) {
            reinterpret_cast<void(*)(dim3,dim3,cudaStream_t,const double *,size_t,double *)>(p->copy)
                (dim3(4096),dim3(256),p->stream,static_cast<const double *>(a),d.n,static_cast<double *>(p->scratch));
        } else if (rank || rank2) {
            reinterpret_cast<void(*)(dim3,dim3,cudaStream_t,const double2 *,size_t,size_t,double2 *,size_t,size_t,int)>(p->copy)
                (dim3((d.n+31)/32,(d.k+31)/32),dim3(32,8),p->stream,static_cast<const double2 *>(rank2 ? b : a),d.n,d.k,static_cast<double2 *>(p->scratch),d.k,0,int(s.operation==4 || s.operation==8 || s.operation==9));
        } else {
            reinterpret_cast<void(*)(dim3,dim3,cudaStream_t,const double2 *,size_t,double2 *,int,int)>(p->copy)
                (dim3(4096),dim3(256),p->stream,static_cast<const double2 *>(a),d.n,static_cast<double2 *>(p->scratch),int(s.operation==5 || s.operation==6),int(s.operation==6));
        }
        ck(cudaGetLastError());
        // Preserve the measured adapters' copy-before-engine synchronization.
        if (s.complex) { ck(cudaEventRecord(p->ready,p->stream)); ck(cudaEventSynchronize(p->ready)); }
        else ck(cudaStreamSynchronize(p->stream));
        if (rank || rank2) eb=p->scratch; else ea=p->scratch;
    }
    double stage[6]{};
    if (p->run(p->context,ea,eb,(rank2 && !s.native_adapter) || (s.general_adapter && rank) ? p->product : c,1,stage)!=0) throw awe::ApiError(AWE_RUNTIME_ERROR,p->last_error());
    if (s.general_adapter && rank) {
        // The general GEMM writes a full temporary; rank-k changes only the lower triangle.
        std::lock_guard<std::mutex> support_lock(p->support->mutex());
        reinterpret_cast<void(*)(dim3,dim3,cudaStream_t,const double *,size_t,double *)>(p->commit)
            (dim3(4096),dim3(256),p->stream,static_cast<const double *>(p->product),d.n,static_cast<double *>(c));
        ck(cudaGetLastError());
        ck(cudaStreamSynchronize(p->stream));
    }
    if (rank2 && !s.native_adapter) {
        std::lock_guard<std::mutex> support_lock(p->support->mutex());
        if (s.element_bytes==4) {
            reinterpret_cast<void(*)(dim3,dim3,cudaStream_t,const float *,size_t,float *,int,int)>(p->commit)
                (dim3(4096),dim3(256),p->stream,static_cast<const float *>(p->product),d.n,static_cast<float *>(c),1,1);
        } else if (s.complex) {
        reinterpret_cast<void(*)(dim3,dim3,cudaStream_t,const double2 *,size_t,double2 *,int,int,int)>(p->commit)
            (dim3(4096),dim3(256),p->stream,static_cast<const double2 *>(p->product),d.n,
             static_cast<double2 *>(c),1,1,int(s.operation==8));
        } else {
            reinterpret_cast<void(*)(dim3,dim3,cudaStream_t,const double *,size_t,double *)>(p->commit)
                (dim3(4096),dim3(256),p->stream,static_cast<const double *>(p->product),d.n,static_cast<double *>(c));
        }
        ck(cudaGetLastError());
        ck(cudaStreamSynchronize(p->stream));
    }
}
awe_status named(awe_plan *p,awe_routine routine,const void *a,const void *b,void *c) {
    return protect([&] { awe::require(p && p->d.routine==routine,AWE_INVALID_ARGUMENT,"routine does not match plan"); execute(p,a,b,c); });
}
}
extern "C" awe_status awe_execute(awe_plan *p,const void *a,const void *b,void *c) { return protect([&]{execute(p,a,b,c);}); }
extern "C" awe_status awe_dgemm(awe_plan *p,const double *a,const double *b,double *c) { return named(p,AWE_DGEMM,a,b,c); }
extern "C" awe_status awe_zgemm(awe_plan *p,const awe_complex *a,const awe_complex *b,awe_complex *c) { return named(p,AWE_ZGEMM,a,b,c); }
extern "C" awe_status awe_dsyrk(awe_plan *p,const double *a,double *c) { return named(p,AWE_DSYRK,a,nullptr,c); }
extern "C" awe_status awe_dtrmm(awe_plan *p,const double *a,const double *b,double *c) { return named(p,AWE_DTRMM,a,b,c); }
extern "C" awe_status awe_ztrmm(awe_plan *p,const awe_complex *a,const awe_complex *b,awe_complex *c) { return named(p,AWE_ZTRMM,a,b,c); }
extern "C" awe_status awe_zherk(awe_plan *p,const awe_complex *a,awe_complex *c) { return named(p,AWE_ZHERK,a,nullptr,c); }
extern "C" awe_status awe_plan_info(const awe_plan *p,char *buffer,size_t capacity,size_t *required) {
    return protect([&] {
        awe::require(p && required,AWE_INVALID_ARGUMENT,"null info argument");
        *required=p->info.size()+1;
        if (!buffer) { awe::require(capacity==0,AWE_INVALID_ARGUMENT,"capacity without buffer"); return; }
        awe::require(capacity>=*required,AWE_INVALID_ARGUMENT,"info buffer too small");
        std::memcpy(buffer,p->info.c_str(),*required);
    });
}
extern "C" awe_status awe_plan_destroy(awe_plan *p) {
    return protect([&] { if (!p) return; check_device(p); auto module=p->module;
        std::lock_guard<std::mutex> lock(module->mutex()); delete p; });
}

extern "C" awe_status awe_plan_effective_options(const awe_plan *p, awe_effective_options *out) {
    return protect([&] { awe::require(p && out,AWE_INVALID_ARGUMENT,"null effective-options argument"); *out=p->effective; });
}

extern "C" awe_status awe_dsymm(awe_plan *p, const double *a, const double *b, double *c) {
    return named(p, AWE_DSYMM, a, b, c);
}
extern "C" awe_status awe_dsyr2k(awe_plan *p, const double *a, const double *b, double *c) {
    return named(p, AWE_DSYR2K, a, b, c);
}
extern "C" awe_status awe_dtrsm(awe_plan *p, const double *a, const double *b, double *c) {
    return named(p, AWE_DTRSM, a, b, c);
}
extern "C" awe_status awe_zhemm(awe_plan *p, const awe_complex *a, const awe_complex *b, awe_complex *c) {
    return named(p, AWE_ZHEMM, a, b, c);
}
extern "C" awe_status awe_zher2k(awe_plan *p, const awe_complex *a, const awe_complex *b, awe_complex *c) {
    return named(p, AWE_ZHER2K, a, b, c);
}
extern "C" awe_status awe_zsymm(awe_plan *p, const awe_complex *a, const awe_complex *b, awe_complex *c) {
    return named(p, AWE_ZSYMM, a, b, c);
}
extern "C" awe_status awe_zsyrk(awe_plan *p, const awe_complex *a, awe_complex *c) {
    return named(p, AWE_ZSYRK, a, nullptr, c);
}
extern "C" awe_status awe_zsyr2k(awe_plan *p, const awe_complex *a, const awe_complex *b, awe_complex *c) {
    return named(p, AWE_ZSYR2K, a, b, c);
}
extern "C" awe_status awe_sgemm(awe_plan *p, const float *a, const float *b, float *c) {
    return named(p, AWE_SGEMM, a, b, c);
}
extern "C" awe_status awe_ssymm(awe_plan *p, const float *a, const float *b, float *c) {
    return named(p, AWE_SSYMM, a, b, c);
}
extern "C" awe_status awe_ssyrk(awe_plan *p, const float *a, float *c) {
    return named(p, AWE_SSYRK, a, nullptr, c);
}
extern "C" awe_status awe_ssyr2k(awe_plan *p, const float *a, const float *b, float *c) {
    return named(p, AWE_SSYR2K, a, b, c);
}
extern "C" awe_status awe_strmm(awe_plan *p, const float *a, const float *b, float *c) {
    return named(p, AWE_STRMM, a, b, c);
}
extern "C" awe_status awe_cgemm(awe_plan *p, const awe_complex_float *a, const awe_complex_float *b, awe_complex_float *c) {
    return named(p, AWE_CGEMM, a, b, c);
}
extern "C" awe_status awe_i8gemm(awe_plan *p, const int8_t *a, const int8_t *b, int32_t *c) {
    return named(p, AWE_I8GEMM, a, b, c);
}
