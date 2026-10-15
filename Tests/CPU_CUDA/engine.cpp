#include <complex>
#include <cstring>
static int last_create_operation = 0;
extern "C" int test_last_create_operation() { return last_create_operation; }
struct Plan {int m,n,k,lda,ldb,ldc,op; int poison=1;};
extern "C" void *d_create_operation(int m,int n,int k,long long lda,long long ldb,long long ldc,int op) {
    last_create_operation = op;
    return new Plan{m,n,k,int(lda),int(ldb),int(ldc),op};
}
extern "C" void *z_create_operation(int m,int n,int k,long long lda,long long ldb,long long ldc,int op) {
    return d_create_operation(m,n,k,lda,ldb,ldc,op);
}
extern "C" void d_destroy(void *p){delete static_cast<Plan*>(p);}
extern "C" void z_destroy(void *p){d_destroy(p);}
extern "C" const char *d_last_error(){return "CPU fake engine";}
extern "C" const char *z_last_error(){return d_last_error();}
extern "C" int d_poison(void *h,int on){auto p=static_cast<Plan*>(h);int old=p->poison;p->poison=on;return old;}
extern "C" int z_poison(void *h,int on){return d_poison(h,on);}
template<class T> int run(void *h,const void *a,const void *b,void *c) {
    auto p=static_cast<Plan*>(h);if(p->poison)return 1;auto A=static_cast<const T*>(a),B=static_cast<const T*>(b);auto C=static_cast<T*>(c);
    for(int j=0;j<p->n;++j)for(int i=0;i<p->m;++i)if((p->op!=2 && p->op!=4)||i>=j){
        T sum=0;for(int t=0;t<p->k;++t)sum+=A[i+t*p->lda]*B[t+j*p->ldb]; C[i+j*p->ldc]=sum;
    }return 0;
}
extern "C" int d_run(void *p,const void *a,const void *b,void *c,int,double*) {return run<double>(p,a,b,c);}
extern "C" int z_run(void *p,const void *a,const void *b,void *c,int,double*) {return run<std::complex<double>>(p,a,b,c);}
#include "../../Platforms/NVIDIA/CUDA-Common/Common/host_options.h"
extern "C" void *d_create_configured(int m,int n,int k,long long lda,long long ldb,long long ldc,int op,const awe_engine_configuration *) {
    return d_create_operation(m,n,k,lda,ldb,ldc,op);
}
extern "C" void *z_create_configured(int m,int n,int k,long long lda,long long ldb,long long ldc,int op,const awe_engine_configuration *) {
    return z_create_operation(m,n,k,lda,ldb,ldc,op);
}
extern "C" int d_effective_options(void *h,awe_engine_effective *out){*out={};out->pace=7;out->chunks=3;out->moduli=18;auto p=static_cast<Plan*>(h);out->pingpong=(p->op==1 || p->op==2);return 0;}
extern "C" int z_effective_options(void *p,awe_engine_effective *out){d_effective_options(p,out);out->pingpong=0;return 0;}
extern "C" void *f_create_configured(int m,int n,int k,long long lda,long long ldb,long long ldc,int op,const awe_engine_configuration *c) { if (op != 1) return nullptr; return d_create_configured(m,n,k,lda,ldb,ldc,op,c); }
extern "C" int f_run(void *p,const void *a,const void *b,void *c,int,double *) { return run<float>(p,a,b,c); }
extern "C" void f_destroy(void *p) { d_destroy(p); }
extern "C" int f_poison(void *p,int on) { return d_poison(p,on); }
extern "C" const char *f_last_error() { return d_last_error(); }
extern "C" int f_effective_options(void *p,awe_engine_effective *out) { return d_effective_options(p,out); }
extern "C" void *u_create_configured(int m,int n,int k,long long lda,long long ldb,long long ldc,int op,const awe_engine_configuration *c) { return d_create_configured(m,n,k,lda,ldb,ldc,op,c); }
extern "C" int u_run(void *h,const void *ap,const void *bp,void *cp,int,double *) {
    const auto &p=*static_cast<Plan *>(h); if(p.poison)return 1;
    const auto *a=static_cast<const float *>(ap), *b=static_cast<const float *>(bp); auto *c=static_cast<float *>(cp);
    for(int j=0;j<p.n;++j)for(int i=0;i<p.m;++i) {
        if((p.op==2 || p.op==7) && i<j)continue;
        float value=0;
        for(int q=0;q<p.k;++q) {
            if(p.op==2)value+=a[i+q*p.lda]*a[j+q*p.lda];
            else if(p.op==7)value+=a[i+q*p.lda]*b[j+q*p.ldb]+b[i+q*p.ldb]*a[j+q*p.lda];
            else if(p.op==3)value+=(i>=q ? a[i+q*p.lda] : 0)*b[q+j*p.ldb];
            else value+=(i>=q ? a[i+q*p.lda] : a[q+i*p.lda])*b[q+j*p.ldb];
        }
        c[i+j*p.ldc]=value;
    }
    return 0;
}
extern "C" void u_destroy(void *p) { d_destroy(p); }
extern "C" int u_poison(void *p,int on) { return d_poison(p,on); }
extern "C" const char *u_last_error() { return d_last_error(); }
extern "C" int u_effective_options(void *p,awe_engine_effective *out) { return d_effective_options(p,out); }

extern "C" void *c_create_configured(int m,int n,int k,long long lda,long long ldb,long long ldc,int op,const awe_engine_configuration *c) { return d_create_configured(m,n,k,lda,ldb,ldc,op,c); }
extern "C" int c_run(void *p,const void *a,const void *b,void *c,int,double *) { return run<std::complex<float>>(p,a,b,c); }
extern "C" void c_destroy(void *p) { d_destroy(p); }
extern "C" int c_poison(void *p,int on) { return d_poison(p,on); }
extern "C" const char *c_last_error() { return d_last_error(); }
extern "C" int c_effective_options(void *p,awe_engine_effective *out) { return d_effective_options(p,out); }

extern "C" void *i_create_configured(int m,int n,int k,long long lda,long long ldb,long long ldc,int op,const awe_engine_configuration *c) { return d_create_configured(m,n,k,lda,ldb,ldc,op,c); }
extern "C" int i_run(void *h,const void *ap,const void *bp,void *cp,int,double *) {
    const auto &p=*static_cast<Plan *>(h); if(p.poison)return 1;
    auto a=static_cast<const int8_t *>(ap),b=static_cast<const int8_t *>(bp);auto c=static_cast<int32_t *>(cp);
    for(int j=0;j<p.n;++j)for(int i=0;i<p.m;++i){int64_t sum=0;for(int q=0;q<p.k;++q)sum+=int64_t(a[i+q*p.lda])*b[q+j*p.ldb];c[i+j*p.ldc]=int32_t(sum);}
    return 0;
}
extern "C" void i_destroy(void *p) { d_destroy(p); }
extern "C" int i_poison(void *p,int on) { return d_poison(p,on); }
extern "C" const char *i_last_error() { return d_last_error(); }
extern "C" int i_effective_options(void *,awe_engine_effective *out) { *out={};out->chunks=1;out->encoder_minb=4;return 0; }

extern "C" void *v_create_configured(int m,int n,int k,long long lda,long long ldb,long long ldc,int op,const awe_engine_configuration *c) {return d_create_configured(m,n,k,lda,ldb,ldc,op,c);}
extern "C" int v_run(void *h,const void *ap,const void *bp,void *cp,int,double *) {
    const auto &p=*static_cast<Plan *>(h);if(p.poison)return 1;
    using Z=std::complex<double>;auto a=static_cast<const Z *>(ap),b=static_cast<const Z *>(bp);auto c=static_cast<Z *>(cp);
    bool rank=p.op==2 || p.op==9, rank2=p.op==7 || p.op==8, hermitian=p.op==6 || p.op==8 || p.op==9;
    for(int j=0;j<p.n;++j)for(int i=0;i<p.m;++i){
        if((rank || rank2) && i<j)continue;
        Z value{};
        for(int q=0;q<p.k;++q){
            if(rank)value+=a[i+q*p.lda]*(hermitian ? std::conj(a[j+q*p.lda]) : a[j+q*p.lda]);
            else if(rank2)value+=a[i+q*p.lda]*(hermitian ? std::conj(b[j+q*p.ldb]) : b[j+q*p.ldb])+b[i+q*p.ldb]*(hermitian ? std::conj(a[j+q*p.lda]) : a[j+q*p.lda]);
            else{Z x=p.op==3 ? (i>=q ? a[i+q*p.lda] : Z{}) : (i>=q ? a[i+q*p.lda] : (hermitian ? std::conj(a[q+i*p.lda]) : a[q+i*p.lda]));if(hermitian && i==q)x={x.real(),0};value+=x*b[q+j*p.ldb];}
        }
        if(hermitian && i==j)value={value.real(),0};
        c[i+j*p.ldc]=value;
    }return 0;
}
extern "C" void v_destroy(void *p){d_destroy(p);}
extern "C" int v_poison(void *p,int on){return d_poison(p,on);}
extern "C" const char *v_last_error(){return d_last_error();}
extern "C" int v_effective_options(void *p,awe_engine_effective *out){return d_effective_options(p,out);}
extern "C" void *t_create_configured(int m,int n,int k,long long lda,long long ldb,long long ldc,int op,const awe_engine_configuration *c){return d_create_configured(m,n,k,lda,ldb,ldc,op,c);}
extern "C" int t_run(void *h,const void *ap,const void *bp,void *cp,int,double *){
    auto p=static_cast<Plan*>(h);if(p->poison)return 1;auto a=static_cast<const double*>(ap),b=static_cast<const double*>(bp);auto c=static_cast<double*>(cp);
    for(int j=0;j<p->n;++j)for(int i=0;i<p->m;++i){double sum=b[i+j*p->ldb];for(int q=0;q<i;++q)sum-=a[i+q*p->lda]*c[q+j*p->ldc];c[i+j*p->ldc]=sum/a[i+i*p->lda];}return 0;
}
extern "C" void t_destroy(void *p){d_destroy(p);}
extern "C" int t_poison(void *p,int on){return d_poison(p,on);}
extern "C" const char *t_last_error(){return d_last_error();}
extern "C" int t_effective_options(void *p,awe_engine_effective *out){return d_effective_options(p,out);}
