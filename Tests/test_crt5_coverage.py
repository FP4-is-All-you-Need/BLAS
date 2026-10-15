"""Check the production five-word range gate with independent Python integers."""
import math
from pathlib import Path
import random
import re
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / 'Platforms/NVIDIA/CUDA-Inline/TRSM/SM100a_K64/D57_TRSM'

class Crt5CoverageTest(unittest.TestCase):
    def test_product_words_and_strict_gate(self):
        text = (ENGINE / 'l8k_crt5.h').read_text()
        def array(name):
            body = re.search(r'\b' + name + r'\[.*?=\s*\{(.*?)\};', text, re.S)[1]
            return [int(x.rstrip('uUlL'), 0) for x in re.findall(r'0x[0-9a-fA-F]+[uUlL]*|\b[0-9]+[uUlL]*', body)]
        product = math.prod(array('l8k_crt5_moduli'))
        words = array('l8k_crt5_p')
        self.assertEqual(product, sum(w << (32*i) for i,w in enumerate(words)))
        self.assertEqual(product // 2, sum(w << (32*i) for i,w in enumerate(array('l8k_crt5_ph'))))
        rng = random.Random(714)
        cases = [(product, 57, k) for k in [1, 128, 2048, 8192, 8388, 8389, 16384]]
        for bits in [1, 8, 24, 53, 57, 63]:
            for k in [1, 7, 8192, (1 << 32)-1]:
                bound = 2*k*((1 << bits)-1)**2
                cases.extend((bound + delta, bits, k) for delta in [-1, 0, 1])
        cases += [(rng.randrange(1 << 160), rng.randrange(1,64), rng.randrange(1,1 << 32)) for _ in range(100)]
        code = '#include <cassert>\n#include "' + str(ENGINE / 'crt5_coverage.h') + '"\nint main(){\n'
        for p,b,k in cases:
            expected = p > 2*k*((1 << b)-1)**2
            code += '{unsigned p[5]={' + ','.join(str((p >> (32*i)) & 0xffffffff)+'u' for i in range(5)) + '};'
            code += f'assert(crt5_covers(p,{b}u,{k}u)=={str(expected).lower()});}}\n'
        code += '}\n'
        with tempfile.TemporaryDirectory() as directory:
            d = Path(directory)
            (d / 'check.cpp').write_text(code)
            subprocess.run(['g++','-std=c++17',str(d/'check.cpp'),'-o',str(d/'check')],check=True)
            subprocess.run([str(d/'check')],check=True)
