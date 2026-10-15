"""Compile production host option wrappers and table conditions without CUDA."""
import json
from pathlib import Path
import re
import subprocess
import tempfile
import unittest
ROOT = Path(__file__).resolve().parents[1]


def function(text, name):
    start = re.search(r'extern "C"[^\n]*\b' + re.escape(name) + r'\(', text).start()
    opening = text.index('{', start)
    depth = 1
    end = opening + 1
    while depth:
        depth += (text[end] == '{') - (text[end] == '}')
        end += 1
    return text[start:end]


class MeasuredDefaultsTest(unittest.TestCase):
    def test_production_create_and_effective_options(self):
        expected = {'k96_d17': (1, 1), 'k96_d18': (1, 1),
                    'k64_d17': (1, 1), 'k64_d18': (1, 1),
                    'k96_d18_syrk': (2, 1), 'k96_d21_syrk': (2, 1),
                    'k96_d18_syrk_trmm': (3, 0), 'k96_d21_syrk_trmm': (3, 0),
                    'k96_z53_syrk_trmm': (3, 0), 'k96_z51_herk': (4, 0), 'k64_d21': (1, 0)}
        for engine in json.loads((ROOT / 'Docs/Provenance/engines.json').read_text()):
            with self.subTest(engine=engine['id']):
                path = ROOT / engine['path']
                z = '_z' in engine['id']
                prefix, namespace = ('z', 'zl8rt') if z else ('d', 'dvrt')
                text = (path / ('zl8_dtab.cuh' if z else 'l8rt_dtab.cuh')).read_text()
                # Use the real plan's environment-to-shape assignment as well as both public wrappers.
                assignment = re.search(r'sh.pingpong = 0;.*?if \(const char \*e = awe_option_env\(options, "L8_PINGPONG"\)\) \{.*?\n        \}',
                                       (path / 'l8rt_plan.cuh').read_text(), re.S).group()
                op, pp = expected[engine['id']]
                code = '''#include <cassert>
#include <string>
#include "Platforms/NVIDIA/CUDA-Common/Common/host_options.h"
#define L8K_NMOD 18
struct Shape { int pingpong=0,zk16=0,pace_s=7,nchunk=3,item_order=1; };
struct Plan { Shape sh; int crt_chunks=2;
void init(const awe_engine_options *options) { int kplan=8192;
''' + assignment + '\n}};\n'
                code += f'namespace {namespace} {{ std::string last_error; struct Ctx {{Plan plan; int poison=1;}}; }}\n'
                code += f'''static void *{prefix}_create_impl(int,int,int,long long,long long,long long,const awe_engine_options *options) {{
    auto *p=new {namespace}::Ctx; p->plan.init(options); return p;
}}
'''
                for name in ('create_configured', 'effective_options', 'poison'):
                    code += function(text, prefix + '_' + name) + '\n'
                code += f'''int main() {{
    awe_engine_configuration config{{}}; config.struct_size=sizeof(config);
    for (const char *env : {{"0","1","17"}}) {{
        setenv("L8_PINGPONG",env,1);
        auto *p={prefix}_create_configured(7,7,97,7,97,7,{op},&config); assert(p);
        awe_engine_effective effective{{}};
        assert({prefix}_effective_options(p,&effective)==0 && effective.pingpong=={pp});
        assert(effective.pace==7 && effective.chunks==3 && effective.crt_chunks==2);
        assert({prefix}_poison(p,0)==1 && {prefix}_poison(p,0)==0);
        delete static_cast<{namespace}::Ctx*>(p);
        awe_engine_options legacy{{{op}}};
        assert(std::string(awe_option_env(&legacy,"L8_PINGPONG"))==env);
    }}
    config.pingpong=1;
    assert(!{prefix}_create_configured(7,7,97,7,97,7,{op},&config));
}}
'''
                with tempfile.TemporaryDirectory() as directory:
                    p = Path(directory)
                    (p / 'check.cpp').write_text(code)
                    subprocess.run(['g++', '-std=c++17', '-I' + str(ROOT), str(p / 'check.cpp'), '-o', str(p / 'check')], check=True)
                    subprocess.run([str(p / 'check')], check=True)
