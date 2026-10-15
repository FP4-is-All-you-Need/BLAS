"""Check K64 table restrictions on the CPU without loading any CUDA library."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]

class K64TableTest(unittest.TestCase):
    def test_pingpong_tables(self):
        engines = json.loads((ROOT/'Docs/Provenance/additional_engines.json').read_text())
        for engine in engines:
            folder = ROOT/engine['path']
            plan = folder/'l8rt_plan.cuh'
            if engine['k'] != 64 or not plan.exists() or 'sh.pingpong = 1;' not in plan.read_text():
                continue
            table = folder/('zl8_tables_rt.h' if (folder/'zl8_tables_rt.h').exists() else 'l8k_tables_rt_f32.h')
            with self.subTest(engine=engine['id']), tempfile.TemporaryDirectory() as directory:
                d = Path(directory)
                (d/'test.cpp').write_text('#include <cassert>\n#include <initializer_list>\n#include "'+str(table)+'''"
int main(){
    for(auto table : {l8k_mods59_k8192,l8k_mods59_k16384})
        for(int i=0;i<L8K_NMOD;++i)
            for(int j=0;j<table[i].nprod;++j)
                assert(table[i].nacc!=1 || ((table[i].prod[j]>>4)&1u)==0);
}
''')
                subprocess.run(['g++','-std=c++17',str(d/'test.cpp'),'-o',str(d/'test')],check=True)
                subprocess.run([str(d/'test')],check=True)
