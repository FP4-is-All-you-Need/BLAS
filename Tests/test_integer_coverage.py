"""Exhaust all INT8 input pairs using integers independent of the CUDA encoder."""
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]

class IntegerCoverageTest(unittest.TestCase):
    def test_every_input_pair_and_accumulation_bounds(self):
        text = (ROOT/'Platforms/NVIDIA/CUDA-Inline/GEMM/SM103a_K96/INT8/lut1728.h').read_text()
        tables = []
        for side in ('A','B'):
            body = re.search(r'LUT_'+side+r'\[256\]\[6\]\s*=\s*\{(.*?)\};', text, re.S)[1]
            rows = [[int(v) for v in re.findall(r'-?\d+', row)] for row in re.findall(r'\{([^}]+)\}', body)]
            self.assertEqual(len(rows), 256)
            self.assertTrue(all(len(row)==6 for row in rows))
            self.assertTrue(all(abs(v) in (0,1,2,3,4,6,8,12) for row in rows for v in row))
            tables.append(rows)
        coef = [1,14,56,1,4,56]
        maximum = [0,0]
        for x,a in enumerate(tables[0], -128):
            for y,b in enumerate(tables[1], -128):
                sums = [sum(coef[j]*a[j]*b[j] for j in range(g*3,g*3+3)) for g in (0,1)]
                # Block scales restore the table units; the final payout divides by two.
                self.assertEqual(37*sums[0]+7*sums[1], 2*x*y)
                for g in (0,1):
                    maximum[g]=max(maximum[g],sum(coef[j]*abs(a[j])*abs(b[j]) for j in range(g*3,g*3+3)))
        self.assertEqual(maximum,[974,1152])
        for bound in maximum:
            safe=(1<<24)//bound
            self.assertLessEqual(safe*bound,1<<24)
            self.assertGreater((safe+1)*bound,1<<24)
        self.assertLessEqual(16384*128**2,2**31-1)
        self.assertGreater(131072*128**2,2**31-1)
