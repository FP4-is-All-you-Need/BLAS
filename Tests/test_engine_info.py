"""Exercise the production info formatter with CPU plan fields, never CUDA."""
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
ROOT = Path(__file__).resolve().parents[1]
FREE_TEXT = {'source', 'set', 'encoder', 'crt'}


def render(text):
    fragment = text[text.index('    char buf[2048];'):text.index('    c->info = buf;')]
    code = '''#include <cstdio>
struct Shape {int cuts=43,tiles=5,clusters=7,nchunk=1,crt_pf=1,crt_prop=1,crt2=0,
                  herk=1,syrk=1,trmm=1,pace_s=10,pingpong=1;};
struct Plan {Shape sh; int kplan=8192,accs=23,crt_chunks=16;
             struct Encoder {int fast=1;} enc;} p;
struct {const char *name="CPU formatter fixture";} prop;
int runtime=13020,m=17,n=19,k=4097,pbits=124;
long long lda=17,ldb=4097,ldc=17;
int L8K_NMOD=18,ZL8_NMODC=17,kParts=3,L8K_PRODUCTS59=62,L8K_PLANES59=39;
int L8K_FTS=32,kSlots=2,Side=128,Cols=128,L8_FAST_ENC=1,L8_BITS=51,L8_CRT2C=1;
int main(){for(int chunks:{1,3})for(int fast:{0,1}){
    p.sh.nchunk=chunks;p.enc.fast=L8_FAST_ENC=fast;
''' + fragment + '\nputs(buf);}}\n'
    code = '#include <initializer_list>\n' + code
    with tempfile.TemporaryDirectory() as directory:
        p = Path(directory)
        (p / 'info.cpp').write_text(code)
        subprocess.run(['g++', '-std=c++17', '-Werror=format', str(p / 'info.cpp'), '-o', str(p / 'info')], check=True)
        return [json.loads(line) for line in subprocess.check_output([str(p / 'info')], text=True).splitlines()]


def contract(rows):
    # Freeze every key and every value except the four explicitly edited prose values.
    retained = [{k: v for k, v in row.items() if k not in FREE_TEXT} for row in rows]
    return {'keys': sorted(rows[0]), 'nonprose_sha256': hashlib.sha256(json.dumps(retained, sort_keys=True).encode()).hexdigest(),
            'variants': len(rows)}


class EngineInfoTest(unittest.TestCase):
    def test_info_keys_and_machine_readable_values(self):
        baseline = json.loads((ROOT / 'Tests/Fixtures/engine_info_contract.json').read_text())
        for path, expected in baseline['files'].items():
            with self.subTest(path=path):
                rows = render((ROOT / path).read_text())
                self.assertEqual(contract(rows), expected)
                for row in rows:
                    for key in FREE_TEXT & row.keys():
                        self.assertNotRegex(row[key],
                                            r'op' + r'us|RI' + r'KEN|\bs[0-9]+\b|\b[0-9]{4,6}\b')
                    for key in ('quantize_bits', 'moduli', 'mma_products'):
                        self.assertIsInstance(row[key], int)
                    if 'pingpong' in row:
                        self.assertEqual(row['pingpong'], 1)
