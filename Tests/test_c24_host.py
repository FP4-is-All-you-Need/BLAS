"""Check the imported complex-float host policy without loading CUDA."""
from pathlib import Path
import re
import subprocess
import tempfile
import unittest
from test_measured_defaults import function

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT/'Platforms/NVIDIA/CUDA-Inline/GEMM/SM100a_K64/C24'


class ComplexFloatHostTest(unittest.TestCase):
    def test_production_policy_bound_and_configured_entry(self):
        plan = (ENGINE/'l8rt_plan.cuh').read_text()
        start = plan.index('const U128 qm =')
        stop = plan.index('\n        }', start)
        coverage = plan[start:stop]
        assignment = re.search(r'sh.pingpong = 1;.*?if \(sh.pingpong\)', plan, re.S)[0]
        assignment = assignment[:assignment.rfind('if (sh.pingpong)')]
        text = (ENGINE/'l8rt.cu').read_text()
        code = '''#include <cassert>
#include <stdexcept>
#include <string>
#include "Platforms/NVIDIA/CUDA-Common/Common/additional_options.h"
using U128=unsigned __int128;
#define ZL8_NMODC 11
struct Plan {
    struct {int pace_s=1,nchunk=1,item_order=0,pingpong=0,zk16=0;} sh;
    struct {int fast=0;} enc;
    int crt_chunks=1,kplan=8192;
    void init() {
''' + assignment + '''
    }
};
namespace cl8rt {std::string error; struct Ctx {Plan plan;};}
void *c_create_ld(int,int,int,long long,long long,long long) {
    auto *p=new cl8rt::Ctx; p->plan.init(); return p;
}
void coverage(U128 P,int k,int bits) {
''' + coverage + '\n}\n'
        code += function(text, 'c_create_configured') + '\n'
        code += function(text, 'c_effective_options') + '\n'
        code += '''int main() {
    U128 product=1;for(int p:{8,9,11,73,89,97,101,109,137,145,169})product*=p;
    coverage(product,16384,24);
    bool bad=false;try{coverage(product,16385,24);}catch(...){bad=true;}assert(bad);
    const U128 q=(U128(1)<<24)-1;
    bad=false;try{coverage(4*17*q*q,17,24);}catch(...){bad=true;}assert(bad);
    coverage(4*17*q*q+1,17,24);
    awe_engine_configuration config{};config.struct_size=sizeof(config);
    for(const char *value: {"0","1","7"}) {
        setenv("L8_PINGPONG",value,1);setenv("L8C_BITS","bk",1);
        auto *p=c_create_configured(17,17,17,17,17,17,1,&config);assert(p);
        awe_engine_effective effective{};
        assert(c_effective_options(p,&effective)==0);
        assert(effective.pingpong==1&&effective.moduli==11);
        assert(awe_additional_env("L8C_BITS")==nullptr);
        delete static_cast<cl8rt::Ctx *>(p);
    }
    assert(!c_create_configured(17,17,17,17,17,17,3,&config));
    config.pingpong=1;
    assert(!c_create_configured(17,17,17,17,17,17,1,&config));
}
'''
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory); (p/'test.cpp').write_text(code)
            subprocess.run(['g++','-std=c++17','-I'+str(ROOT),str(p/'test.cpp'),'-o',str(p/'test')],check=True)
            subprocess.run([str(p/'test')],check=True)
