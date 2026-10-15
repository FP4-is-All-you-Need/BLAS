"""Check actual encoder/descriptor address formulas without CUDA or a GPU."""
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'Tools'))
from source_tokens import tokens

GEOMETRY_FILES = ('zl8_encode.cuh', 'fp64k_common.h', 'l8_encode.cuh',
                  'l8k_kernel_rt.cuh', 'l8rt.cu', 'l8_probe.cuh', 'l8_encode_fast.cuh')


def definition(values, name):
    """Find one definition, ignoring calls and braces inside string tokens."""
    def end(start, opening, closing):
        depth = 0
        for i in range(start, len(values)):
            if values[i] == opening: depth += 1
            elif values[i] == closing:
                depth -= 1
                if depth == 0: return i+1
        raise AssertionError('unclosed definition: '+name)
    found = []
    for i, word in enumerate(values[:-1]):
        if word != name or values[i+1] != '(': continue
        body = end(i+1, '(', ')')
        if body < len(values) and values[body] == '{':
            found.append((i, body, end(body, '{', '}')))
    if len(found) != 1: raise AssertionError('expected one definition: '+name)
    return found[0]


def geometry_sources(folder):
    return {name: (folder/name).read_text() for name in GEOMETRY_FILES}


def check_reference_geometry(candidate, reference):
    # The K64 double-complex geometry passed B200 checks. Only the explicit
    # float input/storage changes below and CRT reconstruction are excluded.
    for name in GEOMETRY_FILES:
        expected = reference[name]
        if name == 'zl8_encode.cuh':
            expected = (expected.replace('double2', 'float2')
                        .replace('const double *', 'const float *')
                        .replace('AStageZ * 16', 'AStageZ * 8'))
        if name == 'l8k_kernel_rt.cuh':
            expected = re.sub(r'double\s*\*\s*cout\b', 'float *cout', expected)
        a, b = tokens(candidate[name]), tokens(expected)
        if name == 'l8k_kernel_rt.cuh':
            for values in (a, b):
                for excluded in ('crt_cols_rt', 'l8k_crt_after'):
                    _, start, stop = definition(values, excluded)
                    values[start:stop] = ['{', '}']
        if name == 'l8rt.cu':
            for selected in ('tile_desc', 'face_mma'):
                ai, _, az = definition(a, selected)
                bi, _, bz = definition(b, selected)
                while ai and a[ai-1] not in (';', '}'): ai -= 1
                while bi and b[bi-1] not in (';', '}'): bi -= 1
                if a[ai:az] != b[bi:bz]:
                    raise AssertionError(name+': '+selected+' differs from K64 reference')
        elif a != b:
            raise AssertionError(name+': differs from K64 reference')


def function(text, name):
    # Extract the real pure arithmetic body, not a copy of its formula.
    match = re.search(r'(?:constexpr |inline )?(?:int|size_t|unsigned long long) '+name+r'\([^{}]+\)\s*\{[^{}]*\}', text)
    if not match: raise AssertionError(name)
    return match.group()


class GeometryTests(unittest.TestCase):
    def test_geometry_matches_k64_reference(self):
        root = ROOT/'Platforms/NVIDIA/CUDA-Inline/GEMM/SM100a_K64'
        reference = geometry_sources(root/'Z53_General')
        for bits in (20,22,23):
            with self.subTest(bits=bits):
                check_reference_geometry(geometry_sources(root/f'C{bits}'), reference)

    def test_reject_encoder_grid_and_tma_regressions(self):
        root = ROOT/'Platforms/NVIDIA/CUDA-Inline/GEMM/SM100a_K64'
        reference = geometry_sources(root/'Z53_General')
        mutations = {
            'enc12': ('zl8_encode.cuh', [('linear / 8', 'linear / 12'),
                       ('linear % 8', 'linear % 12'), ('cut * 64', 'cut * 96')]),
            'grid6': ('zl8_encode.cuh', [('const dim3 grid(4,', 'const dim3 grid(6,')]),
            'tmarow': ('l8k_kernel_rt.cuh', [('TileBytes / 256', 'TileBytes / 384')]),
        }
        for bits in (20,22,23):
            original = geometry_sources(root/f'C{bits}')
            check_reference_geometry(original, reference)
            for label, (name, replacements) in mutations.items():
                with self.subTest(bits=bits,mutation=label):
                    broken = dict(original)
                    for before, after in replacements:
                        self.assertIn(before, broken[name])
                        broken[name] = broken[name].replace(before, after)
                    with self.assertRaisesRegex(AssertionError, re.escape(name)):
                        check_reference_geometry(broken, reference)

    def test_packed_coordinates(self):
        engines=json.loads((ROOT/'Docs/Provenance/additional_engines.json').read_text())
        for bits in (20,22,23):
            e=next(e for e in engines if e['id']==f'k64_c{bits}')
            folder=ROOT/e['path']; cu=(folder/'l8rt.cu').read_text()
            enc=(folder/'l8_encode.cuh').read_text(); plan=(folder/'l8rt_plan.cuh').read_text()
            constants=re.search(r'constexpr int kSlots =.*?constexpr int kBoxTiles = \d+;',cu,re.S).group()
            box=re.search(r'const cuuint32_t box\[2\] = .*?;',plan).group()
            defines='\n'.join('#define '+a[2:].replace('=',' ',1) for a in e['arguments'] if a.startswith('-D'))
            code='''#include <cassert>
#include <vector>
#include <set>
#include <initializer_list>
#include "fp64k_common.h"
#include "zl8_tables_rt.h"
using namespace fp64k;
using cuuint32_t = unsigned;
#ifndef L8_SLOTS
#define L8_SLOTS 3
#endif
'''+constants+'\n'+'\n'.join(function(enc,n) for n in ('cuts_of','tile_block_bytes','plane_offset'))+'\n'+function(cu,'tile_desc')+'''
int main() {
    assert(Cuts==cuts_of(K)); assert(TileBytes==128*64/2);
    for(int k : {1,63,64,65,127,128,129,4096,8192,16384}) {
        assert(cuts_of(k)%2==0 && cuts_of(k)*64>=k && (cuts_of(k)-2)*64<k);
    }
    std::set<int> seen;
    const auto descriptor=tile_desc(0);
    const int leading=int((descriptor>>16)&0x3fff)*16;
    const int stride=int((descriptor>>32)&0x3fff)*16;
    assert(leading==128 && stride==256);
    for(int row=0;row<128;++row)for(int nibble=0;nibble<64;++nibble) {
        int byte=tile_byte(row,nibble/2);
        int consumed=(row/8)*stride+(nibble/32)*leading+(row%8)*16+(nibble%32)/2;
        assert(byte==consumed && byte>=0 && byte<TileBytes);
        assert(seen.insert(byte*2+nibble%2).second);
    }
    assert(seen.size()==128*64);
    '''+box+'''
    assert(box[0]==256 && box[1]==32);
    assert(box[0]*box[1]==2*TileBytes);
    assert(kOperand==L8K_FTS*TileBytes && kGroup==2*kOperand);
    assert(kShared==kSlots*kGroup && kShared<=227*1024);
    for(int cuts : {2,64,128,256}) {
        std::set<size_t> addresses;
        int total=0;
        for(int planes : {1,2,3}) {
            for(unsigned rank=0;rank<2;++rank)for(int cut=0;cut<cuts;++cut)for(int p=0;p<planes;++p) {
                auto off=plane_offset(cuts,total,planes,rank,cut,p);
                assert(off%TileBytes==0);
                assert(addresses.insert(off).second);
            }
            total+=planes;
        }
        assert(addresses.size()==size_t(2*cuts*total));
        assert(*addresses.begin()==0);
        assert(*addresses.rbegin()+TileBytes==tile_block_bytes(cuts,total));
    }
}
'''
            with self.subTest(bits=bits), tempfile.TemporaryDirectory() as directory:
                p=Path(directory);(p/'test.cpp').write_text(defines+'\n'+code)
                subprocess.run(['g++','-std=c++17','-I'+str(folder),str(p/'test.cpp'),'-o',str(p/'test')],check=True,capture_output=True)
                subprocess.run([str(p/'test')],check=True)
