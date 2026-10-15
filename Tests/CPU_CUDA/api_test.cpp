#include <AWE/awe.h>
#include "dispatch.hpp"
#include "../../Platforms/NVIDIA/CUDA-Common/Common/host_options.h"
#include <cassert>
#include <dlfcn.h>
#include <cmath>
#include <complex>
#include <cstring>
#include <string>
#include <vector>
int fake_device=0,fake_minor=3;
awe_descriptor desc(awe_routine r,int k=2) {
    awe_descriptor d;awe_descriptor_init(&d,r);d.m=d.n=d.lda=d.ldc=2;d.k=d.ldb=k;return d;
}
int main(int argc,char **argv){
    assert(argc==2);
    for(auto backend:{AWE_K96,AWE_K64})for(int k:{4096,4097,8254,8255}){
        auto d=desc(AWE_DGEMM,k);d.backend=backend;
        assert(awe_validate(&d)==((k==8255 || (backend==AWE_K64 && k>8192))?AWE_NOT_SUPPORTED:AWE_SUCCESS));
        if(k<=(backend==AWE_K64?8192:8254))assert(awe::select(d).moduli==(k<=(backend==AWE_K64?8192:4096)?17:18));
    }
    auto d=desc(AWE_DGEMM,8192);d.moduli=AWE_MODULI_17;assert(awe_validate(&d)==0);
    d.k=8193;assert(awe_validate(&d)==AWE_INVALID_ARGUMENT);d.ldb=8193;assert(awe_validate(&d)==AWE_NOT_SUPPORTED);
    d=desc(AWE_DGEMM,8193);d.backend=AWE_K64;assert(awe_validate(&d)==AWE_NOT_SUPPORTED);
    assert(std::string(awe_last_error()).find("select AWE_MODULI_21 explicitly")!=std::string::npos);
    d=desc(AWE_DGEMM,16384);d.backend=AWE_K64;d.moduli=AWE_MODULI_21;assert(awe_validate(&d)==0);
    d.backend=AWE_K96;assert(awe_validate(&d)==AWE_NOT_SUPPORTED);
    d=desc(AWE_ZGEMM);assert(awe_validate(&d)==AWE_SUCCESS);
    d=desc(AWE_DGEMM);d.alpha.real=2;assert(awe_validate(&d)==AWE_NOT_SUPPORTED);
    d=desc(AWE_DSYRK);d.triangle=AWE_UPPER;assert(awe_validate(&d)==AWE_NOT_SUPPORTED);
    d=desc(AWE_DGEMM);d.lda=INT64_MAX;assert(awe_validate(&d)==AWE_INVALID_ARGUMENT);
    d=desc(AWE_DGEMM);d.trans_b=AWE_TRANSPOSE;assert(awe_validate(&d)==AWE_NOT_SUPPORTED);
    d=desc(AWE_DGEMM);d.struct_size=0;assert(awe_validate(&d)==AWE_INVALID_ARGUMENT);
    awe_plan *p=reinterpret_cast<awe_plan*>(1);assert(awe_plan_create(&p,nullptr,argv[1])==AWE_INVALID_ARGUMENT&&!p);
    awe_options options; awe_options_init(&options);
    auto defaults=options; options.pacing=static_cast<awe_tuning_policy>(2); d=desc(AWE_DGEMM);
    assert(awe_plan_create_with_options(&p,&d,&options,argv[1])==AWE_NOT_SUPPORTED && !p);
    options = defaults;
    options.quantize_bits = 53;
    assert(awe_plan_create_with_options(&p, &d, &options, argv[1]) == 0);
    assert(awe_plan_destroy(p) == 0);
    options.quantize_bits = 54;
    assert(awe_plan_create_with_options(&p, &d, &options, argv[1]) == AWE_INVALID_ARGUMENT && !p);
    options.quantize_bits = 50;
    assert(awe_plan_create_with_options(&p, &d, &options, argv[1]) == AWE_NOT_SUPPORTED && !p);
    options.struct_size = offsetof(awe_options, quantize_bits);
    assert(awe_plan_create_with_options(&p, &d, &options, argv[1]) == 0);
    assert(awe_plan_destroy(p) == 0);
    // Old layouts must not read the new pointer, including old trailing padding.
    static_assert(offsetof(awe_options, quantize_bits) == 32);
    static_assert(offsetof(awe_options, engine_name) == 40);
    static_assert(sizeof(awe_options) == 48);
    options=defaults; options.struct_size=40; options.engine_name=reinterpret_cast<const char *>(1);
    assert(awe_plan_create_with_options(&p,&d,&options,argv[1])==0);
    assert(awe_plan_destroy(p)==0);
    options=defaults; options.engine_name="d53_m18_gemm";
    assert(awe_plan_create_with_options(&p,&d,&options,argv[1])==0);
    char choice_info[2048];size_t choice_size=0;
    assert(awe_plan_info(p,choice_info,sizeof(choice_info),&choice_size)==0);
    assert(std::strstr(choice_info,"\"engine_name\":\"d53_m18_gemm\""));
    assert(std::strstr(choice_info,"\"engine\":\"k96_d18\""));
    assert(awe_plan_destroy(p)==0);
    options.quantize_bits=51;
    assert(awe_plan_create_with_options(&p,&d,&options,argv[1])==AWE_NOT_SUPPORTED&&!p);
    options.quantize_bits=0;d.moduli=AWE_MODULI_17;
    assert(awe_plan_create_with_options(&p,&d,&options,argv[1])==AWE_NOT_SUPPORTED&&!p);
    d=desc(AWE_ZGEMM);
    assert(awe_plan_create_with_options(&p,&d,&options,argv[1])==AWE_NOT_SUPPORTED&&!p);
    d=desc(AWE_DGEMM);options.engine_name="d57_m20_trsm";
    assert(awe_plan_create_with_options(&p,&d,&options,argv[1])==AWE_NOT_SUPPORTED&&!p);
    options.engine_name="../untrusted";
    assert(awe_plan_create_with_options(&p,&d,&options,argv[1])==AWE_NOT_SUPPORTED&&!p);
    options=defaults;
    for (int routine = AWE_DSYMM; routine <= AWE_DTRSM; ++routine) {
        auto added = desc(static_cast<awe_routine>(routine));
        const bool general = routine == AWE_ZSYMM || routine == AWE_ZHEMM ||
                             routine == AWE_ZSYR2K || routine == AWE_ZHER2K || routine == AWE_ZSYRK || routine == AWE_DSYMM || routine == AWE_DSYR2K || (routine >= AWE_SGEMM && routine <= AWE_I8GEMM);
        assert(awe_validate(&added) == (general ? AWE_SUCCESS : AWE_NOT_SUPPORTED));
    }
    for (const char *name : {"s23_m11_syrk_k4096_a", "s23_m11_syrk_k4096_b"}) {
        d=desc(AWE_SSYRK,4096);
        auto chosen=awe::select(d,23,name);
        assert(chosen.moduli==11 && chosen.native_adapter);
        assert(awe::public_engine_name(chosen.engine)==std::string(name));
        ++d.k;++d.ldb;bool refused=false;
        try{(void)awe::select(d,23,name);}catch(const awe::ApiError &e){refused=e.status==AWE_NOT_SUPPORTED;}
        assert(refused);
        d=desc(AWE_SSYRK);options=defaults;options.engine_name=name;
        assert(awe_plan_create_with_options(&p,&d,&options,argv[1])==0);
        assert(awe_plan_destroy(p)==0);
    }
    d=desc(AWE_SSYRK,4096);
    assert(awe::select(d).engine=="k96_s23_syrk_k4096_a");
    assert(awe::select(d,23,"s23_m12_level3_fused").engine=="k96_s23_fused");
    for(auto row : {std::pair<awe_routine,int>{AWE_SGEMM,22},{AWE_SSYMM,21},{AWE_SSYR2K,22},{AWE_STRMM,21}}) {
        d=desc(row.first);options=defaults;options.quantize_bits=row.second;
        assert(awe_plan_create_with_options(&p,&d,&options,argv[1])==0);
        assert(awe_plan_destroy(p)==0);
    }
    for(auto row : {std::pair<awe_routine,const char *>{AWE_CGEMM,"c23_m10_gemm"},
                   {AWE_ZHERK,"z51_m19_herk_k4096"},{AWE_ZHERK,"z53_m20_herk"},
                   {AWE_ZSYRK,"z53_m20_syrk_trmm"}}) {
        d=desc(row.first);options=defaults;options.engine_name=row.second;
        assert(awe_plan_create_with_options(&p,&d,&options,argv[1])==0);
        assert(awe_plan_destroy(p)==0);
    }
    d=desc(AWE_ZHERK);assert(awe::select(d).engine=="k96_z51_herk_k4096");
    assert(awe::select(d,51,"z51_m19_herk").engine=="k96_z51_herk");
    d=desc(AWE_ZSYRK);assert(awe::select(d).bits==53);
    d=desc(AWE_CGEMM,4097);bool c23_refused=false;
    try{(void)awe::select(d,23);}catch(const awe::ApiError &e){c23_refused=e.status==AWE_NOT_SUPPORTED;}
    assert(c23_refused);
    // A rectangular rank-k input catches accidental N-by-N copies and strides.
    for (auto routine : {AWE_ZSYRK, AWE_ZHERK}) for (int width : (routine == AWE_ZSYRK ? std::vector<int>{50,52} : std::vector<int>{49,51})) {
        d=desc(routine,3); options=defaults; options.quantize_bits=width;
        assert(awe_plan_create_with_options(&p,&d,&options,argv[1])==0);
        awe_complex a[]={{1,2},{2,-1},{3,1},{4,-2},{-1,1},{2,3}}, c[4]{};
        c[2]={-99,13};
        assert(awe_execute(p,a,nullptr,c)==0);
        for(int j=0;j<2;++j)for(int i=j;i<2;++i) {
            std::complex<double> expected{};
            for(int q=0;q<3;++q) {
                std::complex<double> x(a[i+q*2].real,a[i+q*2].imag), y(a[j+q*2].real,a[j+q*2].imag);
                expected+=x*(routine==AWE_ZHERK ? std::conj(y) : y);
            }
            assert(c[i+j*2].real==expected.real() && c[i+j*2].imag==expected.imag());
        }
        assert(c[2].real==-99 && c[2].imag==13);
        assert(awe_plan_destroy(p)==0);
    }
    for (auto entry : {std::pair<int,int>{52,8958}, {50,15925}, {48,15187}}) {
        const auto routine=entry.first==48 ? AWE_ZGEMM : AWE_ZSYRK;
        d=desc(routine,entry.second);
        assert(awe::select(d,entry.first).bits==entry.first);
        ++d.k; ++d.ldb;
        bool rejected=false;
        try { (void)awe::select(d,entry.first); } catch(const awe::ApiError &e) { rejected=e.status==AWE_NOT_SUPPORTED; }
        assert(rejected);
    }
    for(auto routine : {AWE_DSYMM,AWE_DSYR2K}) {
        d=desc(routine, routine==AWE_DSYR2K ? 3 : 2); d.ldb=routine==AWE_DSYR2K ? 2 : d.k;
        assert(awe_plan_create(&p,&d,argv[1])==0);
        double a[]={1,2,3,4,5,6}, b[]={2,1,4,3,6,5}, c[]={0,0,-99,0}, expected[4]{};
        for(int j=0;j<2;++j)for(int i=0;i<2;++i)for(int q=0;q<d.k;++q) {
            if(routine==AWE_DSYR2K)expected[i+j*2]+=a[i+q*2]*b[j+q*2]+b[i+q*2]*a[j+q*2];
            else expected[i+j*2]+=(i>=q ? a[i+q*2] : a[q+i*2])*b[q+j*2];
        }
        if(routine==AWE_DSYMM)a[2]=NAN;
        assert(awe_execute(p,a,b,c)==0);
        for(int j=0;j<2;++j)for(int i=0;i<2;++i)assert(c[i+j*2]==(routine==AWE_DSYR2K && i<j ? -99 : expected[i+j*2]));
        assert(awe_plan_destroy(p)==0);
    }
    fake_minor=0;
    for (auto row : {std::pair<awe_routine,int>{AWE_SGEMM,2},{AWE_SSYRK,8192}}) {
        d=desc(row.first,row.second);d.backend=AWE_K64;options=defaults;
        assert(awe_plan_create_with_options(&p,&d,&options,argv[1])==AWE_NOT_SUPPORTED && !p);
        options.engine_name=row.first==AWE_SGEMM ? "s23_m9_gemm" : "s24_m12_level3";
        assert(awe_plan_create_with_options(&p,&d,&options,argv[1])==AWE_SUCCESS);
        assert(awe_plan_destroy(p)==0);
    }
    fake_minor=3;
    for (auto row : {std::pair<int,int>{51,17},{52,17},{53,18}}) {
        const auto name="d"+std::to_string(row.first)+"_m"+std::to_string(row.second)+"_gemm_crt";
        d=desc(AWE_DGEMM);options=defaults;options.engine_name=name.c_str();
        assert(awe_plan_create_with_options(&p,&d,&options,argv[1])==0);
        assert(awe_plan_info(p,choice_info,sizeof(choice_info),&choice_size)==0);
        assert(std::strstr(choice_info,name.c_str()));
        assert(awe_plan_destroy(p)==0);
        d=desc(AWE_DSYMM);
        assert(awe_plan_create_with_options(&p,&d,&options,argv[1])==AWE_NOT_SUPPORTED && !p);
    }
    d=desc(AWE_DGEMM);
    assert(awe::select(d,51).engine=="k96_d51_gemm_crt");
    assert(awe::select(d,52).engine=="k96_d52_gemm_crt");
    assert(awe::select(d,51,"d51_m17_level3").engine=="k96_d51_general");
    assert(awe::select(d,53).engine=="k96_d17");
    for(auto entry : {std::pair<int,int>{51,8720},{52,8560}}) {
        d=desc(AWE_DGEMM,entry.second);assert(awe::select(d,entry.first).bits==entry.first);
        ++d.k; ++d.ldb; bool rejected=false;
        try {(void)awe::select(d,entry.first);}catch(const awe::ApiError &e){rejected=e.status==AWE_NOT_SUPPORTED;}
        assert(rejected);
    }
    for(auto routine : {AWE_SGEMM,AWE_SSYMM,AWE_SSYRK,AWE_SSYR2K,AWE_STRMM}) {
        const auto widths=routine==AWE_SGEMM ? std::vector<int>{21,23} : routine==AWE_SSYRK ? std::vector<int>{23,24} :
                          routine==AWE_SSYR2K ? std::vector<int>{20,21} : routine==AWE_SSYMM ? std::vector<int>{20,22} : std::vector<int>{21,22,23};
        for(int width:widths) {
            const bool rank=routine==AWE_SSYRK, rank2=routine==AWE_SSYR2K;
            d=desc(routine,rank || rank2 ? 3 : 2); if(rank2)d.ldb=2;
            if(routine==AWE_SSYR2K && width==20)assert(awe::select(d,width).engine=="k96_s20_syr2k");
            if(routine==AWE_SSYMM && width==20)assert(awe::select(d,width).engine=="k96_s20_fused");
            options=defaults;options.quantize_bits=width;
            assert(awe_plan_create_with_options(&p,&d,&options,argv[1])==0);
            const auto selected=awe::select(d,width);
            if(routine==AWE_STRMM)assert(selected.operation==3);
            void *module=dlopen((std::string(argv[1])+"/libawe_"+selected.engine+".so").c_str(),RTLD_NOW);
            assert(module);
            const int actual=reinterpret_cast<int(*)()>(dlsym(module,"test_last_create_operation"))();
            assert(actual==awe::create_operation(selected));
            if(!selected.native_adapter)assert(actual==1);
            dlclose(module);
            float a[]={1,2,3,4,5,6},b[]={2,1,4,3,6,5},c[]={0,0,-99,0},expected[4]{};
            for(int j=0;j<2;++j)for(int i=0;i<2;++i)for(int q=0;q<d.k;++q) {
                if(rank)expected[i+j*2]+=a[i+q*2]*a[j+q*2];
                else if(rank2)expected[i+j*2]+=a[i+q*2]*b[j+q*2]+b[i+q*2]*a[j+q*2];
                else {float x=a[i+q*2];if(routine==AWE_SSYMM && i<q)x=a[q+i*2];if(routine==AWE_STRMM && i<q)x=0;expected[i+j*2]+=x*b[q+j*d.k];}
            }
            if(routine==AWE_SSYMM || routine==AWE_STRMM)a[2]=NAN;
            assert(awe_execute(p,a,rank ? nullptr : b,c)==0);
            for(int j=0;j<2;++j)for(int i=0;i<2;++i)assert(c[i+j*2]==((rank || rank2) && i<j ? -99 : expected[i+j*2]));
            assert(awe_plan_destroy(p)==0);
        }
    }

    fake_minor=0;d=desc(AWE_STRMM);d.backend=AWE_K64;
    options=defaults;options.engine_name="s23_m11_trmm_gemm";
    auto shared_trmm=awe::select(d,0,options.engine_name);
    assert(shared_trmm.operation==3 && !shared_trmm.native_adapter);
    assert(awe::create_operation(shared_trmm)==1);
    assert(awe_plan_create_with_options(&p,&d,&options,argv[1])==0);
    float triangular[]={1,2,NAN,4},rhs[]={2,1,4,3},answer[4]{};
    assert(awe_strmm(p,triangular,rhs,answer)==0);
    assert(answer[0]==2 && answer[1]==8 && answer[2]==4 && answer[3]==20);
    char route_info[2048];size_t route_size=0;
    assert(awe_plan_info(p,route_info,sizeof(route_info),&route_size)==0);
    assert(std::strstr(route_info,"\"engine_name\":\"s23_m11_trmm_gemm\""));
    assert(std::strstr(route_info,"\"operation\":3"));
    assert(std::strstr(route_info,"\"create_operation\":1"));
    assert(std::strstr(route_info,"\"native_adapter\":false"));
    assert(awe_plan_destroy(p)==0);
    d=desc(AWE_SSYRK);d.backend=AWE_K64;
    assert(awe_plan_create_with_options(&p,&d,&options,argv[1])==AWE_NOT_SUPPORTED);
    d=desc(AWE_STRMM,4097);d.m=d.n=d.lda=d.ldc=4097;d.backend=AWE_K64;
    assert(awe_plan_create_with_options(&p,&d,&options,argv[1])==AWE_NOT_SUPPORTED);
    d=desc(AWE_STRMM);
    assert(awe_plan_create_with_options(&p,&d,&options,argv[1])==AWE_NOT_SUPPORTED);
    fake_minor=3;
    for(auto routine : {AWE_DTRMM,AWE_ZTRMM}) {
        d=desc(routine);
        const auto selected=awe::select(d);
        assert(selected.operation==3 && awe::create_operation(selected)==3);
        assert(awe_plan_create(&p,&d,argv[1])==0);
        void *module=dlopen((std::string(argv[1])+"/libawe_"+selected.engine+".so").c_str(),RTLD_NOW);
        assert(module);
        assert(reinterpret_cast<int(*)()>(dlsym(module,"test_last_create_operation"))()==3);
        dlclose(module);
        assert(awe_plan_destroy(p)==0);
    }
    for(int width : {0,20,22}) {
        d=desc(AWE_CGEMM,3);d.n=3;d.lda=3;d.ldb=4;d.ldc=3;
        options=defaults;options.quantize_bits=width;
        assert(awe_plan_create_with_options(&p,&d,&options,argv[1])==0);
        std::complex<float> a[9], b[12], c[9], expected[9];
        for(int i=0;i<9;++i){a[i]={float(i%5-2),float(i%3-1)};c[i]={-99,-99};expected[i]=c[i];}
        for(int i=0;i<12;++i)b[i]={float(i%4-1),float(2-i%3)};
        for(int j=0;j<3;++j)for(int i=0;i<2;++i){expected[i+j*3]=0;for(int q=0;q<3;++q)expected[i+j*3]+=a[i+q*3]*b[q+j*4];}
        assert(awe_cgemm(p,reinterpret_cast<awe_complex_float *>(a),reinterpret_cast<awe_complex_float *>(b),reinterpret_cast<awe_complex_float *>(c))==0);
        for(int i=0;i<9;++i)assert(c[i]==expected[i]);
        assert(awe_cgemm(p,reinterpret_cast<awe_complex_float *>(a),reinterpret_cast<awe_complex_float *>(b),reinterpret_cast<awe_complex_float *>(a))==AWE_INVALID_ARGUMENT);
        assert(awe_plan_destroy(p)==0);
    }
    {
        d=desc(AWE_I8GEMM);d.m=2;d.n=3;d.k=5;d.lda=3;d.ldb=6;d.ldc=4;
        assert(awe_plan_create(&p,&d,argv[1])==AWE_SUCCESS);
        awe_effective_options ef{};assert(awe_plan_effective_options(p,&ef)==0);
        assert(ef.moduli==0 && ef.chunks==1 && ef.encoder_minb==4 && ef.pingpong==0);
        int8_t a[15],b[18];int32_t c[12];
        for(int i=0;i<15;++i)a[i]=int8_t(i%2 ? -128 : 127);
        for(int i=0;i<18;++i)b[i]=int8_t(i%3 ? 127 : -128);
        for(auto &x:c)x=-999;
        assert(awe_i8gemm(p,a,b,c)==AWE_SUCCESS);
        for(int j=0;j<3;++j)for(int i=0;i<4;++i){
            int64_t expected=0;for(int q=0;q<5;++q)expected+=int64_t(a[i%2+q*3])*b[q+j*6];
            assert(c[i+j*4]==(i<2 ? expected : -999));
        }
        // This input overlaps only the last output bytes; checking INT8 output width would miss it.
        assert(awe_i8gemm(p,reinterpret_cast<int8_t *>(c)+32,b,c)==AWE_INVALID_ARGUMENT);
        assert(awe_plan_destroy(p)==0);
        d.k=16384;d.ldb=d.k;assert(awe_validate(&d)==0);
        d.k=16385;d.ldb=d.k;assert(awe_validate(&d)==AWE_NOT_SUPPORTED);
        d=desc(AWE_I8GEMM);d.lda=int64_t(INT_MAX)+1;assert(awe_validate(&d)==AWE_INVALID_ARGUMENT);
        d=desc(AWE_I8GEMM);bool rejected=false;
        try{(void)awe::select(d,7);}catch(const awe::ApiError &e){rejected=e.status==AWE_NOT_SUPPORTED;}
        assert(rejected);
        assert(awe::select(d,8).bits==8);
    }
    awe_complex_float ca[4]{}, cb[4]{}, cc[4]{};
    int8_t ia[4]{}, ib[4]{};
    int32_t ic[4]{};
    assert(awe_cgemm(nullptr, ca, cb, cc) == AWE_INVALID_ARGUMENT);
    assert(awe_i8gemm(nullptr, ia, ib, ic) == AWE_INVALID_ARGUMENT);
    options=defaults; options.struct_size=0;
    assert(awe_plan_create_with_options(&p,&d,&options,argv[1])==AWE_NOT_SUPPORTED && !p);
    setenv("L8_PACE","123",1);setenv("ZL8_ENC","0",1);setenv("AWE_FP64_BITS","9",1);
    awe_engine_options frozen{4,true},legacy{};
    for(auto key:{"L8_PACE","ZL8_ENC","AWE_FP64_BITS"}) {
        assert(awe_option_env(&legacy,key)!=nullptr); assert(awe_option_env(&frozen,key)==nullptr);
    }
    assert(std::string(awe_option_env(&frozen,"ZL8_HERK"))=="1");
    assert(awe_option_env(&frozen,"L8_TRMM")==nullptr);
    for(auto r:{AWE_DGEMM,AWE_DSYRK,AWE_DTRMM,AWE_ZTRMM,AWE_ZHERK}){
        d=desc(r);assert(awe_plan_create(&p,&d,argv[1])==0);
        awe_effective_options effective;assert(awe_plan_effective_options(p,&effective)==0);
        assert(effective.pace==7 && effective.chunks==3);
        assert(effective.pingpong==int(r==AWE_DGEMM || r==AWE_DSYRK));
        size_t size=0;assert(awe_plan_info(p,nullptr,0,&size)==0);std::vector<char> info(size);
        assert(awe_plan_info(p,info.data(),size-1,&size)==AWE_INVALID_ARGUMENT);
        assert(awe_plan_info(p,info.data(),info.size(),&size)==0);
        assert(std::string(info.data()).find("\"pingpong\":"+std::to_string(effective.pingpong))!=std::string::npos);
        assert(std::string(info.data()).find("\"poison\":0")!=std::string::npos);
        double a[]={1,2,3,4},b[]={5,6,7,8},c[]={-99,-99,-99,-99};
        if(r==AWE_DGEMM){assert(awe_dgemm(p,a,b,c)==0);assert(c[0]==23&&c[1]==34&&c[2]==31&&c[3]==46);}
        if(r==AWE_DSYRK){assert(awe_dsyrk(p,a,c)==0);assert(c[0]==10&&c[1]==14&&c[2]==-99&&c[3]==20);}
        if(r==AWE_DTRMM){a[2]=NAN;assert(awe_dtrmm(p,a,b,c)==0);assert(c[0]==5&&c[1]==34&&c[2]==7&&c[3]==46);}
        awe_complex za[]={{1,1},{2,0},{3,-1},{4,0}},zb[]={{5,0},{6,0},{7,0},{8,0}},zc[4]{};
        if(r==AWE_ZTRMM){za[2]={NAN,NAN};assert(awe_ztrmm(p,za,zb,zc)==0);assert(zc[0].real==5&&zc[0].imag==5&&zc[1].real==34);}
        if(r==AWE_ZHERK){zc[2]={-99,13};assert(awe_zherk(p,za,zc)==0);assert(zc[0].real==12&&zc[0].imag==0&&zc[1].real==14&&zc[1].imag==2&&zc[2].real==-99);}
        assert(awe_execute(p,a,b,a)==AWE_INVALID_ARGUMENT);
        fake_device=1;assert(awe_plan_destroy(p)==AWE_INVALID_ARGUMENT);fake_device=0;
        assert(awe_plan_destroy(p)==0);
    }
    // Independently form full symmetric/Hermitian matrices from the used triangle.
    for (auto routine : {AWE_ZGEMM, AWE_ZSYMM, AWE_ZHEMM, AWE_ZSYR2K, AWE_ZHER2K}) {
        d = desc(routine);
        assert(awe_plan_create(&p, &d, argv[1]) == 0);
        awe_complex a[]={{1,2},{2,1},{3,-1},{4,-1}}, b[]={{1,2},{3,1},{2,-1},{1,4}}, c[4]{};
        std::complex<double> expected[4]{};
        for(int j=0;j<2;++j)for(int i=0;i<2;++i)for(int q=0;q<2;++q) {
            auto at=[&](int row,int col) { return std::complex<double>(a[row+col*2].real,a[row+col*2].imag); };
            auto bt=[&](int row,int col) { return std::complex<double>(b[row+col*2].real,b[row+col*2].imag); };
            auto av=at(i,q);
            if(routine==AWE_ZSYMM || routine==AWE_ZHEMM) {
                if(i<q)av=at(q,i);
                if(routine==AWE_ZHEMM && i<q)av=std::conj(av);
                if(routine==AWE_ZHEMM && i==q)av={av.real(),0};
            }
            if(routine==AWE_ZSYR2K)expected[i+j*2]+=at(i,q)*bt(j,q)+bt(i,q)*at(j,q);
            else if(routine==AWE_ZHER2K)expected[i+j*2]+=at(i,q)*std::conj(bt(j,q))+bt(i,q)*std::conj(at(j,q));
            else expected[i+j*2]+=av*bt(q,j);
        }
        if(routine==AWE_ZSYMM || routine==AWE_ZHEMM)a[2]={NAN,NAN};
        if(routine==AWE_ZHER2K || routine==AWE_ZSYR2K)c[2]={-99,13};
        assert(awe_execute(p,a,b,c)==0);
        for(int j=0;j<2;++j)for(int i=0;i<2;++i) {
            if((routine==AWE_ZHER2K || routine==AWE_ZSYR2K) && i<j){assert(c[i+j*2].real==-99 && c[i+j*2].imag==13);continue;}
            assert(c[i+j*2].real==expected[i+j*2].real());
            assert(c[i+j*2].imag==expected[i+j*2].imag());
        }
        assert(awe_plan_destroy(p)==0);
    }
    d=desc(AWE_DGEMM);fake_minor=0;assert(awe_plan_create(&p,&d,argv[1])==AWE_NOT_SUPPORTED&&!p);
    fake_minor=0;
    for(auto r:{AWE_DSYRK,AWE_DSYR2K,AWE_ZSYRK,AWE_ZHERK,AWE_SSYRK,AWE_I8GEMM,AWE_CGEMM}){
        d=desc(r,3);d.backend=AWE_K64;if(r==AWE_DSYR2K)d.ldb=2;
        options=defaults;
        assert(awe_plan_create_with_options(&p,&d,&options,argv[1])==0);
        if(r==AWE_DSYRK){double a[]={1,2,3,4,5,6},c[]={0,0,-99,0};assert(awe_dsyrk(p,a,c)==0);assert(c[0]==35&&c[1]==44&&c[2]==-99&&c[3]==56);}
        if(r==AWE_ZHERK){awe_complex a[]={{1,1},{2,0},{3,-1},{4,0},{1,0},{0,1}},c[4]{};c[2]={-99,13};assert(awe_zherk(p,a,c)==0);assert(c[0].real==13&&c[0].imag==0&&c[1].real==14&&c[1].imag==3&&c[2].real==-99&&c[2].imag==13);}
        assert(awe_plan_destroy(p)==0);
    }
    // Width/name/backend checks precede any CUDA work or module creation.
    d=desc(AWE_CGEMM,3);d.backend=AWE_K64;options=defaults;
    for(int bits:{0,24}) {
        options.quantize_bits=bits;
        assert(awe_plan_create_with_options(&p,&d,&options,argv[1])==0);
        auto selected=awe::select(d,bits);
        assert(selected.engine=="k64_c24" && selected.bits==24 && selected.moduli==11);
        char info[4096];size_t used=0;
        assert(awe_plan_info(p,info,sizeof(info),&used)==0);
        assert(std::strstr(info,"\"engine_name\":\"c24_m11_gemm\""));
        assert(std::strstr(info,"\"quantize_bits\":24"));
        assert(awe_plan_destroy(p)==0);
    }
    for(int bits:{20,22,23}) {
        options.quantize_bits=bits;options.engine_name=nullptr;
        assert(awe_plan_create_with_options(&p,&d,&options,argv[1])==AWE_NOT_SUPPORTED&&!p);
        options.engine_name=bits==20?"c20_m9_gemm":bits==22?"c22_m10_gemm":"c23_m10_gemm";
        assert(awe_plan_create_with_options(&p,&d,&options,argv[1])==0);
        assert(awe_plan_destroy(p)==0);
    }
    d.backend=AWE_K96;assert(awe::select(d).bits==22);
    d.backend=AWE_K64;options=defaults;
    options.engine_name="c24_m11_gemm";
    assert(awe_plan_create_with_options(&p,&d,&options,argv[1])==0);
    assert(awe::select(d,0,options.engine_name).bits==24);
    assert(awe::select(d,24,options.engine_name).moduli==11);
    assert(awe_plan_destroy(p)==0);
    options.quantize_bits=22;
    assert(awe_plan_create_with_options(&p,&d,&options,argv[1])==AWE_NOT_SUPPORTED&&!p);
    options.quantize_bits=24;d.backend=AWE_K96;
    assert(awe_plan_create_with_options(&p,&d,&options,argv[1])==AWE_NOT_SUPPORTED&&!p);
    d.backend=AWE_K64;d.routine=AWE_ZGEMM;
    assert(awe_plan_create_with_options(&p,&d,&options,argv[1])==AWE_NOT_SUPPORTED&&!p);
    d.routine=AWE_CGEMM;d.k=d.ldb=16385;
    assert(awe_plan_create_with_options(&p,&d,&options,argv[1])==AWE_NOT_SUPPORTED&&!p);
    d.k=d.ldb=16384;
    assert(awe::select(d,24,options.engine_name).engine=="k64_c24");
    assert(awe::select(d).engine=="k64_c24");
    d.k=d.ldb=16385;assert(awe_validate(&d)==AWE_NOT_SUPPORTED);
    for (auto r : {AWE_SSYMM, AWE_SSYR2K, AWE_STRMM, AWE_SSYRK}) {
        d=desc(r,8192);d.backend=AWE_K64;d.m=d.n=d.lda=d.ldb=d.ldc=8192;
        options=defaults;options.quantize_bits=r==AWE_SSYRK?23:r==AWE_SSYR2K?20:22;
        assert(awe_plan_create_with_options(&p,&d,&options,argv[1])==AWE_NOT_SUPPORTED&&!p);
        auto selected=awe::select_default(d,options.quantize_bits);
        options.engine_name=awe::public_engine_name(selected.engine);
        assert(awe::select(d,options.quantize_bits,options.engine_name).engine==selected.engine);
    }
    d=desc(AWE_DTRSM);d.backend=AWE_K64;d.m=d.k=3;d.n=2;d.lda=4;d.ldb=5;d.ldc=6;
    double ta[]={2,1,3,99,NAN,2,1,99,NAN,NAN,4,99},tb[]={2,5,17,99,99,4,8,25,99,99},tc[12]{};
    options=defaults;
    assert(awe_plan_create(&p,&d,argv[1])==AWE_NOT_SUPPORTED&&!p);
    for(int bits:{53,57}){options.quantize_bits=bits;options.engine_name=bits==57?"d57_m20_trsm":nullptr;assert(awe_plan_create_with_options(&p,&d,&options,argv[1])==0);assert(awe_dtrsm(p,ta,tb,tc)==0);assert(tc[0]==1&&tc[1]==2&&tc[2]==3&&tc[6]==2&&tc[7]==3&&tc[8]==4);assert(awe_plan_destroy(p)==0);}
    options.engine_name=nullptr;options.quantize_bits=54;assert(awe_plan_create_with_options(&p,&d,&options,argv[1])==AWE_NOT_SUPPORTED&&!p);
    d.k=2;assert(awe_validate(&d)==AWE_NOT_SUPPORTED);
        assert(awe_plan_destroy(nullptr)==0);
}
