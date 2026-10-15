from pathlib import Path
import importlib.util
import json
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('awe_validation',ROOT/'Tools/awe_validation.py')
v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
class ReferenceTest(unittest.TestCase):
    def test_exact_dot_and_ties(self):
        self.assertEqual(v.cpu_checks()['cpu_reference'],'PASS')
        self.assertEqual(v.reference([[1.,2.,-3.]],[[1.],[1.],[1.]]),[[0.]])
        self.assertEqual(v.quantize([0.,0.],53), ([(0,0),(0,0)],0))
        self.assertEqual(v.reference([[2.**-500]],[[2.**-500]]),[[2.**-1000]])
    def test_complex_quantization_shared_component_scale(self):
        self.assertEqual(v.quantize([1+0.625j,0.875+0j],2), ([(2,1),(2,0)],1))
        self.assertEqual(v.reference([[1+2j]],[[3+4j]],53,True),[[-5+10j]])
    def test_cross_device_rejects_changed_inputs_or_outputs(self):
        rows=[{'case':str(i),'sha256':'a','input_sha256':'b'} for i in range(9)]
        with tempfile.TemporaryDirectory() as directory:
            paths=[Path(directory)/f'{k}.json' for k in (64,96)]
            for path,k in zip(paths,(64,96)):path.write_text(json.dumps({'backend':k,'records':rows}))
            self.assertEqual(v.compare_records(paths)['compared_cases'],9)
            for field in ('sha256','input_sha256'):
                changed=[dict(r) for r in rows];changed[0][field]='different'
                paths[1].write_text(json.dumps({'backend':96,'records':changed}))
                with self.assertRaises(AssertionError):v.compare_records(paths)

    def test_forced_18_21_comparison_and_mismatch(self):
        cases=v.validation_cases(64)
        for k in (4096,4097,8192):
            for mod in (18,21):
                self.assertIn(('gemm',1,7,9,k,mod),cases)
            self.assertNotIn(('gemm',1,7,9,k,21),v.validation_cases(96))
        outputs={'gemm:7:9:8192:18':'same'}
        self.assertEqual(v.check_forced_moduli(outputs,'gemm',7,9,8192,21,'same'),[18])
        with self.assertRaises(AssertionError):
            v.check_forced_moduli(outputs,'gemm',7,9,8192,21,'different')
