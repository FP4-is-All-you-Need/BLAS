"""Exercise GPU batch arithmetic and failure handling without loading CUDA."""
import copy
import ctypes as ct
from fractions import Fraction
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'Tests/GPU'))
from level3_cases import ROUTINES,cases,exact_cell,solve_reference,ulp_error,rounded_reference
from run_level3 import compare_pair,compare_moduli
from level3_worker import native_operation,make_inputs,device_input

class BatchTests(unittest.TestCase):
    def test_all_routines_and_forced_pairs(self):
        for backend in (64,96):
            rows=cases(backend,'small')
            self.assertEqual({x['routine'] for x in rows},set(ROUTINES)-({'dtrsm'} if backend==96 else set()))
        for group in ('small','4096','8192','boundaries'):
            d=[x for x in cases(64,group) if x['routine']=='dgemm']
            self.assertEqual({x['moduli'] for x in d},{18,21})
        r=subprocess.run([sys.executable,str(ROOT/'Tests/GPU/b300.py')],capture_output=True,text=True,check=True)
        self.assertFalse(json.loads(r.stdout)['gpu_execution'])
        r=subprocess.run([sys.executable,str(ROOT/'Tests/GPU/level3_worker.py'),
            '--case','absent','--library','absent','--output','absent'],capture_output=True,text=True)
        self.assertNotEqual(r.returncode,0);self.assertIn('requires --gpu',r.stderr)

    def test_target_sources_and_separate_history(self):
        for backend in (64,96):
            for group in ('small','4096','8192','boundaries'):
                for c in cases(backend,group):
                    self.assertEqual(c['validation_suite'],'new_input_seed')
                    p=c['target_provenance']
                    self.assertFalse(p['historical_reproduction'])
                    self.assertEqual(p['enforced'],c['enforce_native_target'])
                    self.assertTrue((ROOT/p['measurement_source']).is_file())
        c=next(c for c in cases(96,'4096') if c['routine']=='ssyr2k' and c['bits']==21 and c['distribution']=='normal')
        self.assertTrue(c['enforce_native_target'])
        self.assertIn('8192',c['target_provenance']['historical_scope'])
        old=json.loads((ROOT/'Tests/GPU/historical_cases.json').read_text())['cases']
        self.assertTrue(old)
        for c in old:
            self.assertEqual(c['validation_suite'],'historical_reproduction')
            self.assertEqual(c['status'],'not_run')
            self.assertEqual(len(c['measurement_sha256']),64)
            self.assertNotIn('seed',c)

    def test_reference_all_small_kinds(self):
        import numpy as np
        for name in ROUTINES:
            c={'routine':name,'m':4,'n':4,'k':4,'distribution':'dyadic','seed':193}
            a,b,_=make_inputs(c);kind=name[1:]
            if kind=='trsm':
                got=solve_reference(a,b)
                self.assertTrue(np.array_equal(a@np.array(got,dtype=float),b));continue
            if kind in ('syrk','herk'):want=a@(a.conj().T if kind=='herk' else a.T)
            elif kind in ('syr2k','her2k'):want=a@(b.conj().T if kind=='her2k' else b.T)+b@(a.conj().T if kind=='her2k' else a.T)
            else:want=a.astype(np.complex128)@b.astype(np.complex128)
            for i in range(4):
                for j in range(4):
                    r,z=exact_cell(name,a,b,i,j)
                    self.assertEqual(complex(float(r),float(z)),want[i,j],name)
        a=np.array([[2.**52,1.,-2.**52]]);b=np.ones((3,1))
        self.assertEqual(exact_cell('dgemm',a,b,0,0),(Fraction(1),Fraction(0)))
        self.assertEqual(ulp_error(1.+2.**-52,Fraction(1),53),1)
        self.assertEqual(ulp_error(2.**-1074,Fraction(0),53),1)
        # A value just above a float32 halfway point must not double-round down.
        self.assertEqual(rounded_reference(Fraction(1)+Fraction(1,2**24)+Fraction(1,2**80),24),1.+2.**-23)
        self.assertEqual(rounded_reference(Fraction(1,2**150),24),0.)

    def test_pair_rejects_changes(self):
        base={'case':{},'input_sha256':'input','effective_options':{'pingpong':1},
              'info':{'poison':0},'output_sha256':{'awe':'output'},
              'reference_sha256':'reference','qualified_against_native':True}
        compare_pair(base,copy.deepcopy(base))
        for key,value in [('input_sha256','different'),('effective_options',{'pingpong':0}),
                          ('info',{'poison':1}),('output_sha256',{'awe':'different'}),
                          ('qualified_against_native',False)]:
            changed=copy.deepcopy(base);changed[key]=value
            with self.assertRaises(AssertionError):compare_pair(base,changed)
        rows=[{'case':{'routine':'dgemm','moduli':x,'k':4096},'input_sha256':'i','output_sha256':{'awe':'o'}} for x in (18,21)]
        self.assertEqual(compare_moduli(rows,64),1)
        rows[1]['output_sha256']['awe']='bad'
        with self.assertRaises(AssertionError):compare_moduli(rows,64)

    def test_precision_pairs_and_record_only(self):
        for backend in (64,96):
            rows=cases(backend,'8192')
            narrow=next(c for c in rows if c['routine']=='cgemm' and c['bits']==20 and c['distribution']=='normal')
            self.assertFalse(narrow['enforce_native_target'])
            self.assertTrue(next(c for c in rows if c['routine']=='cgemm' and c['bits']==20 and c['distribution']=='uniform')['enforce_native_target'])
        self.assertTrue(all(not c['enforce_native_target'] for c in cases(64,'small') if c['routine']=='dtrsm' and c['bits']==53))
        base={'case':{'enforce_native_target':False},'input_sha256':'i','effective_options':{},
              'info':{'poison':0},'output_sha256':{'awe':'o'},'reference_sha256':'r','qualified_against_native':False}
        compare_pair(base,copy.deepcopy(base))
        chosen=[c for c in cases(96,'4096') if c['routine']=='ssyrk' and c['bits']==23]
        self.assertEqual(len(chosen),4)
        for c in chosen:
            self.assertEqual(c['enforce_native_target'], c['engine_name'].endswith('_a') == (c['distribution']=='normal'))
        self.assertTrue(any(c['engine_name']=='d53_m18_gemm_crt' for c in cases(96,'small')))
        self.assertTrue(all(not c['enforce_native_target'] for c in cases(96,'4096')
                            if c['routine']=='zherk' and c['distribution']=='uniform'))
        self.assertEqual([c['bits'] for c in cases(96,'4096')
                          if c['routine']=='zgemm' and c['distribution']=='uniform'
                          and c['enforce_native_target']], [53])
        bounds=cases(96,'boundaries')
        for name,bits,k in [('dgemm',53,8254),('zsyrk',52,8958),('zgemm',48,15187)]:
            self.assertTrue(any((c['routine'],c['bits'],c['k'])==(name,bits,k) for c in bounds))
        import numpy as np
        c={'routine':'zhemm','m':4,'n':4,'k':4,'distribution':'dyadic','seed':193}
        a,_,_=make_inputs(c); gpu_a=device_input(a,'zhemm')
        self.assertTrue(np.all(gpu_a.diagonal().imag==7))
        self.assertTrue(np.all(a.diagonal().imag==0))
        self.assertTrue(np.all(np.isnan(gpu_a[np.triu_indices(4,1)])))

    def test_batch_continues_after_failed_worker(self):
        from unittest.mock import patch
        import tempfile
        import run_level3
        rows=cases(96,'small')[:2]
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/'results';calls=[]
            def fake_run(command, **kwargs):
                case=json.loads(Path(command[command.index('--case')+1]).read_text());calls.append(case)
                if case==rows[0]: raise subprocess.CalledProcessError(1,command)
                record={'case':case,'input_sha256':'i','effective_options':{},'info':{'poison':0,'engine_name':case.get('engine_name')},
                        'output_sha256':{'awe':'o'},'reference_sha256':'r','qualified_against_native':True}
                Path(command[command.index('--output')+1]).write_text(json.dumps(record))
            argv=['batch','--gpu','--old-library','old','--new-library','new','--output',str(out)]
            with patch.object(sys,'argv',argv),patch.object(run_level3,'cases',return_value=rows),patch.object(run_level3.subprocess,'run',side_effect=fake_run):
                with self.assertRaises(SystemExit): run_level3.main(96)
            result=json.loads((out/'summary.json').read_text())
            self.assertEqual(result['completed'],[1]);self.assertEqual(len(result['failures']),1)
            self.assertFalse((out/'passed.json').exists());self.assertEqual(len(calls),5)

    def test_native_abi_argument_counts(self):
        class Function:
            def __call__(self,*args):
                if len(args)!=len(self.argtypes):raise AssertionError((len(args),len(self.argtypes)))
                return 0
        class Library:
            def __getattr__(self,key):return Function()
        class Array: data=ct.c_void_p(256)
        for name in ROUTINES:
            with self.subTest(name=name):native_operation(Library(),ct.c_void_p(1),name,(16,16,16),[Array()]*3)()

if __name__=='__main__':unittest.main()
