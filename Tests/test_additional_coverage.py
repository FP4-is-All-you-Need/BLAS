"""Check imported CRT tables independently with Python's unbounded integers."""
import json
import math
from pathlib import Path
import re
import unittest
ROOT = Path(__file__).resolve().parents[1]

class AdditionalCoverageTest(unittest.TestCase):
    def test_complex_tables_and_strict_boundaries(self):
        for engine in json.loads((ROOT/'Docs/Provenance/additional_engines.json').read_text()):
            table = ROOT/engine['path']/'zl8_tables_rt.h'
            if table.exists():
                values = re.search(r'zl8_p_ct\(int q\).*?=\s*\{([^}]+)', table.read_text(), re.S)[1]
                moduli = [int(x) for x in values.split(',')]
                factor = 4
            elif engine['id'].startswith(('k96_d','k96_s','k64_s','k64_d57')):
                table = ROOT/engine['path']/('l8k_tables_rt_f32.h' if engine['id'].startswith(('k96_s','k64_s')) else 'l8k_tables_rt.h')
                rows = re.search(r'l8k_mods59_k(?:8192|16384)[^=]*=\s*\{(.*?)\n\};', table.read_text(), re.S)[1]
                moduli = [int(x) for x in re.findall(r'^    \{(\d+),', rows, re.M)]
                factor = 2
            else:
                continue
            for i, a in enumerate(moduli):
                for b in moduli[:i]:
                    self.assertEqual(math.gcd(a,b), 1)
            product = math.prod(moduli)
            step = factor*((1<<engine['bits'])-1)**2
            bound = (product-1)//step
            self.assertLess(step*bound, product)
            self.assertGreaterEqual(step*(bound+1), product)
            self.assertLessEqual(engine['max_k'], bound)
            self.assertEqual(str(product), engine['coverage']['product'])
            self.assertEqual(moduli, engine['coverage']['moduli_values'])

    def test_assembled_single_sources_require_explicit_names(self):
        records = json.loads((ROOT/'Docs/Provenance/import_k64.json').read_text())
        expected = {'k64_'+x['name'] for x in records['single_source_assembly']
                    if x['status'].startswith('assembled source')}
        header = (ROOT/'Routines/Common/untested_engines.hpp').read_text()
        actual = set(re.findall(r'"(k64_s[^" ]+)"', header))
        self.assertEqual(actual, expected)
        self.assertEqual(len(expected), 7)
