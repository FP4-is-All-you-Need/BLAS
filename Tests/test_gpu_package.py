"""CPU tests for the bounded single-build GPU package."""
import contextlib
import copy
from fractions import Fraction
import gzip
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parent/'gpu'))
from level3_cases import ROUTINES, exact_cell, solve_reference
from level3_worker import make_inputs, save_verdict, repeat_outputs
from reference_cache import write_cache, read_cache, exact_references, reference_key
from run_single import check_arch, comparison_results, execute, run_bounded, validate_record, verify_package, order_cases, estimated_seconds, display_counts
import run_single


class PackageTests(unittest.TestCase):
    def test_preflight_uses_verified_base_and_records_long_startup(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory)/'base';base.mkdir()
            (base/'manifest.json').write_text('{"backend":64}')
            digest=hashlib.sha256((base/'manifest.json').read_bytes()).hexdigest()
            (base/'SHA256SUMS').write_text(digest+'  manifest.json\n')
            work=Path(directory)/'work';work.mkdir()
            (work/'manifest.json').write_text('{"backend":64}')
            (work/'reference').write_text('{}')
            row=dict(id='one',stage='small',case=self.row()['case'],samples=4,reference='reference')
            (work/'cases.json').write_text(json.dumps([row]))
            clock=[0];worker_calls=[]
            def fake(command,log,timeout,heartbeat=None):
                if '--preflight-output' in command:
                    clock[0]+=min(20,timeout)
                    if timeout<20: raise subprocess.TimeoutExpired(command,timeout)
                    # Run the real CLI path and real checksum check, only replacing CUDA.
                    with patch.object(sys,'argv',command[1:]),patch('run_single.preflight',return_value={'architecture':[10,0]}):
                        run_single.main()
                    return 0
                worker_calls.append(command)
                Path(command[command.index('--output')+1]).write_text(json.dumps(self.row()))
                return 0
            with patch('run_single.run_bounded',side_effect=fake),patch('run_single.time.monotonic',side_effect=lambda:clock[0]),contextlib.redirect_stdout(io.StringIO()):
                wrong=execute(work,work/'wrong-path',500,time.time())
                short=execute(work,work/'short-limit',500,time.time(),verification_root=base,preflight_timeout=15)
                fixed=execute(work,work/'fixed',500,time.time(),verification_root=base)
            self.assertIn('SHA256SUMS',wrong['stop_reason'])
            self.assertIn(str(work),wrong['stop_reason'])
            self.assertEqual(wrong['counts']['not_run'],1)
            self.assertIn('timed out',short['stop_reason'])
            self.assertEqual(short['counts']['not_run'],1)
            self.assertEqual(fixed['counts']['pass'],1)
            self.assertEqual(len(worker_calls),1)
            attempts=[json.loads((work/name/'preflight-attempts.jsonl').read_text()) for name in ('short-limit','fixed')]
            self.assertEqual([r['timeout_seconds'] for r in attempts],[15,300])
            self.assertEqual([r['elapsed_seconds'] for r in attempts],[15,20])
            self.assertEqual([r['status'] for r in attempts],['failed','passed'])
            self.assertTrue((work/'fixed/environment.json').exists())

    def test_correctness_repetitions_do_not_read_timer(self):
        events=[]
        with patch('level3_worker.time.perf_counter',side_effect=AssertionError('timer read')):
            raw,times,hashes=repeat_outputs(lambda:events.append('reset'),
                lambda:events.append('execute'),lambda:events.append('sync'),
                lambda:b'answer',3,0,False)
        self.assertEqual(raw,b'answer');self.assertEqual(times,[])
        self.assertEqual(len(hashes),3);self.assertEqual(len(set(hashes)),1)
        self.assertEqual(events,['reset','sync','execute','sync']*3)
        values=iter([b'answer',b'changed'])
        _,_,hashes=repeat_outputs(lambda:None,lambda:None,lambda:None,
                                 lambda:next(values),2,0,False)
        self.assertEqual(len(set(hashes)),2)

    def test_integer_reference_matches_fraction(self):
        for name in ROUTINES:
            for distribution in ('dyadic','normal','uniform'):
                case=dict(routine=name,m=4,n=4,k=4,distribution=distribution,seed=193)
                a,b,_=make_inputs(case)
                reference=exact_references(case,a,b,8)
                solve=solve_reference(a,b) if name=='dtrsm' else None
                for i,j,value in reference:
                    want=(solve[i][j],Fraction(0)) if solve else exact_cell(name,a,b,i,j)
                    self.assertEqual(tuple(Fraction(*x) for x in value),want,(name,distribution,i,j))
        import numpy as np
        c=dict(routine='dgemm',m=1,n=1,k=3,distribution='dyadic',seed=193)
        a=np.array([[2.**52,1.,-2.**52]]);b=np.ones((3,1))
        self.assertEqual(exact_references(c,a,b,1)[0][2],[[1,1],[0,1]])

    def test_cache_rejects_wrong_case_input_and_content(self):
        case=dict(routine='zher2k',m=4,n=4,k=4,distribution='dyadic',seed=193)
        a,b,_=make_inputs(case)
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'reference.gz';data=write_cache(p,case,a,b,4)
            self.assertEqual(data,read_cache(p,case,a,b,4))
            self.assertEqual(reference_key(case,4),reference_key(dict(case,bits=51,engine_name='x'),4))
            with self.assertRaises(ValueError):read_cache(p,dict(case,seed=194),a,b,4)
            changed=a.copy();changed[0,0]+=1
            with self.assertRaises(ValueError):read_cache(p,case,changed,b,4)
            data['references'][0][2][0][0]+=1
            p.write_bytes(gzip.compress(json.dumps(data).encode()))
            with self.assertRaises(ValueError):read_cache(p,case,a,b,4)

    def test_arch_without_cuda(self):
        self.assertEqual(check_arch(96,10,3),(10,3))
        self.assertEqual(check_arch(64,10,0),(10,0))
        for args in ((96,10,0),(64,10,3),(96,9,0)):
            with self.assertRaises(ValueError):check_arch(*args)

    def test_package_sha_and_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'value').write_bytes(b'abc')
            (root/'SHA256SUMS').write_text(hashlib.sha256(b'abc').hexdigest()+'  value\n')
            self.assertEqual(verify_package(root),1)
            (root/'value').write_bytes(b'bad')
            with self.assertRaises(ValueError):verify_package(root)
            (root/'SHA256SUMS').write_text('x  ../outside\n')
            with self.assertRaises(ValueError):verify_package(root)

    @staticmethod
    def row():
        c=dict(routine='dgemm',m=16,n=16,k=16,bits=53,moduli=18,distribution='dyadic',seed=193,
               engine_name='d53_m18_gemm',implementation='Inline',enforce_native_target=True)
        return dict(case=c,input_sha256='input',reference_sha256='reference',reference_precomputed=True,
                    effective_options={'pingpong':1},output_sha256={'awe':'output'},
                    info={'poison':0,'engine_name':c['engine_name']},canaries=True,inputs_unchanged=True,
                    qualified_against_native=True)

    def test_correctness_only_record_requires_three_hashes_and_no_time(self):
        r=self.row();r.update(timing_measured=False,repeated_output_sha256={'awe':['a']*3,'native':['b']*3})
        self.assertEqual(validate_record(r,r['case'],False),'pass')
        for change in ({'time_ms':{}},{'timing_measured':True},
                       {'repeated_output_sha256':{'awe':['a'],'native':['b']}},
                       {'repeated_output_sha256':{'awe':['a','b','a'],'native':['b']*3}}):
            bad=copy.deepcopy(r);bad.update(change)
            with self.assertRaises(ValueError):validate_record(bad,bad['case'],False)
        r['case']['expected_info']={'create_operation':1,'operation':3,'native_adapter':False}
        with self.assertRaises(ValueError):validate_record(r,r['case'],False)
        r['info'].update(r['case']['expected_info'])
        self.assertEqual(validate_record(r,r['case'],False),'pass')

    def test_targets_and_comparison_scope(self):
        direct=self.row();structured=copy.deepcopy(direct);structured['case']['implementation']='Template'
        comparisons=comparison_results([direct,structured],96)
        self.assertEqual(comparisons['Inline_Template'][0]['status'],'pass')
        self.assertEqual(comparisons['old_new_comparison'],'not_run')
        structured['output_sha256']['awe']='wrong'
        self.assertEqual(comparison_results([direct,structured],96)['Inline_Template'][0]['status'],'fail')
        self.assertEqual(comparison_results([direct],96)['Inline_Template'][0]['status'],'not_run')
        for change in ({'input_sha256':'wrong'},{'effective_options':{'pingpong':0}}):
            bad=copy.deepcopy(direct);bad['case']['implementation']='Template';bad.update(change)
            self.assertEqual(comparison_results([direct,bad],96)['Inline_Template'][0]['status'],'fail')
        pair=[]
        for mod in (18,21):
            r=copy.deepcopy(direct);r['case'].update(moduli=mod,engine_name=None);pair.append(r)
        self.assertEqual(comparison_results(pair,64)['forced_18_21'][0]['status'],'pass')
        pair[1]['output_sha256']['awe']='wrong'
        self.assertEqual(comparison_results(pair,64)['forced_18_21'][0]['status'],'fail')
        self.assertEqual(validate_record(direct,direct['case']),'pass')
        direct['qualified_against_native']=False
        with self.assertRaises(ValueError):validate_record(direct,direct['case'])
        direct['case']['enforce_native_target']=False
        self.assertEqual(validate_record(direct,direct['case']),'record_only')

    def test_failure_saved_before_raise(self):
        record=self.row();record.update(scores={'awe':{'max':123}},first_mismatches=[{'row':0,'column':1,'actual':2}])
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'worker.json'
            with self.assertRaisesRegex(AssertionError,'full reference'):
                save_verdict(output,record,['small full reference mismatch'])
            saved=json.loads(output.read_text())
            self.assertEqual(saved['scores']['awe']['max'],123)
            self.assertEqual(saved['first_mismatches'][0]['actual'],2)
            self.assertEqual(saved['output_sha256']['awe'],'output')
            self.assertEqual(saved['correctness'],'fail')
            with self.assertRaises(ValueError):validate_record(saved,saved['case'])

    def test_nonfinite_evidence_fails_and_counts_do_not_overlap(self):
        record=self.row();record['scores']={'awe':{'max':float('inf')}}
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'worker.json'
            with self.assertRaisesRegex(AssertionError,'non-finite'):
                save_verdict(output,record,[])
            self.assertEqual(json.loads(output.read_text())['scores']['awe']['max'],'inf')
        record.update(scores={},reference_check='all_correctly_rounded',native_target_enforced=True,qualified_against_native=True)
        counts=display_counts([record],[])
        self.assertEqual(counts['full_reference_match'],1)
        self.assertEqual(counts['native_target_pass'],0)

    def test_routine_rounds_and_recent_budget(self):
        rows=[]
        for routine in ('dgemm','ssymm','cgemm'):
            for target in (True,True,False):
                rows.append(dict(stage='4096',case=dict(routine=routine,enforce_native_target=target)))
        ordered=order_cases(rows)
        self.assertEqual([r['case']['routine'] for r in ordered[:3]],['dgemm','ssymm','cgemm'])
        self.assertTrue(all(r['case']['enforce_native_target'] for r in ordered[:6]))
        self.assertEqual(estimated_seconds(rows[0],{('4096','dgemm'):20}),26)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'reference').write_text('{}');(root/'manifest.json').write_text('{"backend":96}')
            c=self.row()['case'];planned=[dict(id=str(i),stage='small',case=c,samples=4,reference='reference') for i in range(2)]
            (root/'cases.json').write_text(json.dumps(planned));calls=[];clock=[0]
            def fake(command,log,timeout,heartbeat=None):
                calls.append(1);clock[0]+=10;Path(command[command.index('--output')+1]).write_text(json.dumps(self.row()));return 0
            # First case takes 10 seconds. Only 6 seconds remain for the next.
            with patch('run_single.time.monotonic',side_effect=lambda:clock[0]),patch('run_single.run_bounded',side_effect=fake),contextlib.redirect_stdout(io.StringIO()):
                result=execute(root,root/'results',76,time.time()+.001,health=lambda:None)
            self.assertEqual(len(calls),1)
            self.assertEqual(result['counts']['not_run'],1)
            self.assertFalse(result['cases'][1]['attempted'])

    def test_process_timeout(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(subprocess.TimeoutExpired):
                run_bounded([sys.executable,'-c','import time;time.sleep(30)'],Path(directory)/'log',.05)

    def test_failed_case_continues_and_deadline_records_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'reference').write_text('{}')
            (root/'manifest.json').write_text('{"backend":96}')
            rows=[dict(id=str(i),stage='small',case=self.row()['case'],samples=4,reference='reference') for i in range(3)]
            (root/'cases.json').write_text(json.dumps(rows))
            calls=[]
            def fake(command,log,timeout,heartbeat=None):
                calls.append(command)
                if len(calls)==1:
                    bad=self.row();bad['validation_failures']=['small full reference mismatch']
                    Path(command[command.index('--output')+1]).write_text(json.dumps(bad));return 3
                Path(command[command.index('--output')+1]).write_text(json.dumps(self.row()))
                return 0
            with patch('run_single.run_bounded',side_effect=fake),contextlib.redirect_stdout(io.StringIO()):
                result=execute(root,root/'results',150,time.time(),health=lambda:None)
            self.assertEqual(result['counts']['fail'],1);self.assertEqual(result['counts']['pass'],2)
            self.assertTrue((root/'results/0/worker.json').exists())
            self.assertTrue(result['cases'][0]['attempted'])
            self.assertEqual(result['old_new_comparison'],'not_run');self.assertFalse(result['passed'])
            self.assertEqual(len((root/'results/progress.jsonl').read_text().splitlines()),8)
            with contextlib.redirect_stdout(io.StringIO()):
                result=execute(root,root/'expired',90,time.time()-90,health=lambda:None)
            self.assertEqual(result['counts']['not_run'],3)
            with self.assertRaises(FileExistsError):execute(root,root/'results',150,time.time(),health=lambda:None)

    def test_timeout_unhealthy_stops_and_preflight_rejects(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'reference').write_text('{}');(root/'manifest.json').write_text('{"backend":96}')
            rows=[dict(id=str(i),stage='small',case=self.row()['case'],samples=4,reference='reference') for i in range(2)]
            (root/'cases.json').write_text(json.dumps(rows));checks=[]
            def health():
                checks.append(1)
                if len(checks)>1:raise RuntimeError('unhealthy')
            with patch('run_single.run_bounded',side_effect=subprocess.TimeoutExpired('worker',20)),contextlib.redirect_stdout(io.StringIO()):
                result=execute(root,root/'results',150,time.time(),health=health)
            self.assertEqual(result['counts']['timeout'],1);self.assertEqual(result['counts']['not_run'],1)
            with contextlib.redirect_stdout(io.StringIO()):
                result=execute(root,root/'wrong-arch',150,time.time(),health=lambda:check_arch(96,10,0))
            self.assertEqual(result['counts']['not_run'],2)
            self.assertIn('requires',result['stop_reason'])

if __name__=='__main__':unittest.main()
