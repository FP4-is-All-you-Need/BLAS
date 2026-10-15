import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('coverage', ROOT / 'Tools/coverage_bounds.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class CoverageTest(unittest.TestCase):
    def test_integer_limits_and_equality(self):
        self.assertEqual(m.real_limit([2, 3], 1), (6, 2))
        rows = m.report(ROOT)
        self.assertEqual(len(rows), 9)
        self.assertEqual({len(x['moduli']): x['strict_coverage_max_k'] for x in rows},
                         {17: 8245, 18: 8254, 21: 18826})
        self.assertTrue(all(x['at_limit_passes'] and x['next_k_fails'] for x in rows))

    def test_production_guards_at_strict_boundary(self):
        paths = ['Platforms/NVIDIA/CUDA-Inline/SYRK/SM103a_K96/D21_SYRK', 'Platforms/NVIDIA/CUDA-Inline/Level3/SM103a_K96/D21_SYRK_TRMM', 'Platforms/NVIDIA/CUDA-Inline/GEMM/SM100a_K64/D53_M21']
        source = '#include <cassert>\n#include <stdexcept>\nusing U128=unsigned __int128;\n'
        for i, path in enumerate(paths):
            text = (ROOT / path / 'l8rt_plan.cuh').read_text()
            block = text.split('        { // Strict real reconstruction coverage,', 1)[1]
            block = '{ // Strict real reconstruction coverage,' + block.split('        enc.set_flips_for_k', 1)[0]
            source += f'bool check{i}(U128 P,int bits,int k) {{try {block} catch(...){{return false;}} return true;}}\n'
        source += 'int main(){ U128 q=(U128(1)<<53)-1;\n'
        for i in range(3):
            source += (f'assert(!check{i}(2*17*q*q,53,17)); assert(check{i}(2*17*q*q+1,53,17));'
                       f'assert(!check{i}(2*17*q*q-1,53,17));\n')
            product = next(r['product'] for r in m.report(ROOT) if len(r['moduli']) == 21)
            source += 'U128 p'+str(i)+'=0; for(char c : "'+product+'") if(c) p'+str(i)+'=p'+str(i)+'*10+(c-\'0\');\n'
            source += f'assert(check{i}(p{i},53,18826)); assert(!check{i}(p{i},53,18827));\n'
        source += '}\n'
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            (p / 'test.cpp').write_text(source)
            subprocess.run(['g++', '-std=c++17', str(p / 'test.cpp'), '-o', str(p / 'test')], check=True)
            subprocess.run([str(p / 'test')], check=True)


if __name__ == '__main__':
    unittest.main()
