#!/usr/bin/env python3
"""Run a bounded check of one public build with saved CPU references."""
import argparse
import ctypes as ct
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def write_json(path, value):
    temporary = path.with_suffix(path.suffix+'.part')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def verify_package(root):
    count = 0
    for line in (root/'SHA256SUMS').read_text().splitlines():
        expected, name = line.split('  ', 1)
        path = (root/name).resolve()
        if root.resolve() not in path.parents: raise ValueError('file outside package')
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('SHA mismatch: '+name)
        count += 1
    return count


def check_arch(backend, major, minor):
    expected = (10, 3 if backend == 96 else 0)
    if (major, minor) != expected:
        raise ValueError(f'K{backend} requires {expected}, found {(major, minor)}')
    return expected


def preflight(backend):
    """Only the --gpu path calls CUDA, including architecture queries."""
    cuda = ct.CDLL('libcudart.so.13')
    def call(name, params, *args):
        fn = getattr(cuda, name); fn.argtypes = params; fn.restype = ct.c_int
        status = fn(*args)
        if status: raise RuntimeError(f'{name}: CUDA status {status}')
    value = ct.c_int; pointer = ct.POINTER(value)
    call('cudaSetDevice', [value], 0)
    major, minor, runtime, driver = value(), value(), value(), value()
    call('cudaDeviceGetAttribute', [pointer,value,value], ct.byref(major),75,0)
    call('cudaDeviceGetAttribute', [pointer,value,value], ct.byref(minor),76,0)
    check_arch(backend, major.value, minor.value)
    call('cudaRuntimeGetVersion', [pointer], ct.byref(runtime))
    call('cudaDriverGetVersion', [pointer], ct.byref(driver))
    call('cudaDeviceSynchronize', [])
    blas=ct.CDLL('libcublas.so.13');handle=ct.c_void_p();version=value()
    def blas_call(name,params,*args):
        fn=getattr(blas,name);fn.argtypes=params;fn.restype=ct.c_int
        status=fn(*args)
        if status: raise RuntimeError(f'{name}: cuBLAS status {status}')
    blas_call('cublasCreate_v2',[ct.POINTER(ct.c_void_p)],ct.byref(handle))
    try: blas_call('cublasGetVersion_v2',[ct.c_void_p,pointer],handle,ct.byref(version))
    finally: blas_call('cublasDestroy_v2',[ct.c_void_p],handle)
    smi = subprocess.run(['nvidia-smi','--query-gpu=name,uuid,driver_version','--format=csv,noheader'],
                         capture_output=True,text=True,timeout=5,check=True)
    return dict(architecture=[major.value,minor.value],cuda_runtime=runtime.value,
                cuda_driver=driver.value,cublas_version=version.value,gpus=smi.stdout.strip(),python=sys.version)


def run_bounded(command, log_path, timeout, heartbeat=None):
    """Kill the whole child group on timeout; leave the log in place."""
    started = time.monotonic()
    with log_path.open('w') as log:
        child = subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            while True:
                remaining = timeout-(time.monotonic()-started)
                if remaining <= 0: raise subprocess.TimeoutExpired(command,timeout)
                try: return child.wait(timeout=min(10,remaining))
                except subprocess.TimeoutExpired:
                    if heartbeat: heartbeat()
        except BaseException:
            try: os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError: pass
            try: child.wait(timeout=2)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL); child.wait(timeout=2)
            raise


def validate_record(record, case, measure_time=None):
    if record['case'] != case: raise ValueError('worker returned another case')
    if case.get('expected_not_supported'):
        if record.get('contract_check') != 'pass' or record.get('status') != 2:
            raise ValueError('default rejection check failed')
        return 'contract_pass'
    if record.get('validation_failures'): raise ValueError('; '.join(record['validation_failures']))
    if not record.get('reference_precomputed'): raise ValueError('reference was not precomputed')
    if not record['canaries'] or not record['inputs_unchanged']: raise ValueError('memory check failed')
    if record['info']['poison'] != 0: raise ValueError('poison differs')
    if case.get('engine_name') and record['info'].get('engine_name') != case['engine_name']:
        raise ValueError('explicit engine differs')
    for key, value in case.get('expected_info', {}).items():
        if record['info'].get(key) != value: raise ValueError('host route differs: '+key)
    hashes = record.get('repeated_output_sha256', {})
    if measure_time is False:
        if record.get('timing_measured') is not False or 'time_ms' in record or 'median_ms' in record:
            raise ValueError('correctness-only case contains timing data')
        if set(hashes) != {'awe','native'} or any(len(v)!=3 for v in hashes.values()):
            raise ValueError('correctness-only case needs three hashes per implementation')
    if any(len(set(values)) != 1 for values in hashes.values()):
        raise ValueError('repeated output hash differs')
    if case['enforce_native_target'] and not record['qualified_against_native']:
        raise ValueError('native p95/max target failed for the specified width and input')
    return 'pass' if case['enforce_native_target'] or record.get('reference_check')=='all_correctly_rounded' else 'record_only'


def comparison_results(records, backend):
    templates = {}; moduli = {}
    for record in records:
        c = record['case']
        if 'output_sha256' not in record or c['routine'] != 'dgemm': continue
        key = (c['m'],c['n'],c['k'],c['bits'],c['distribution'],c['seed'])
        if c.get('engine_name') in ('d53_m17_gemm','d53_m18_gemm'):
            pair = templates.setdefault(key+(c['engine_name'],),{})
            value = (record['input_sha256'],record['reference_sha256'],record['effective_options'],
                     record['output_sha256']['awe'])
            label = c.get('implementation','Inline')
            if label in pair and pair[label] != value: pair['repeat_mismatch'] = True
            pair[label] = value
        if backend == 64 and c['moduli'] in (18,21) and c.get('implementation','Inline') == 'Inline' and c.get('engine_name') is None:
            pair = moduli.setdefault(key,{})
            value = (record['input_sha256'],record['reference_sha256'],record['output_sha256']['awe'])
            if c['moduli'] in pair and pair[c['moduli']] != value: pair['repeat_mismatch'] = True
            pair[c['moduli']] = value
    def result(pairs, labels):
        values=[]
        for key,pair in pairs.items():
            status = 'not_run'
            if pair.get('repeat_mismatch'): status = 'fail'
            elif all(label in pair for label in labels): status = 'pass' if pair[labels[0]] == pair[labels[1]] else 'fail'
            values.append(dict(key=list(key),status=status))
        return values
    return dict(Inline_Template=result(templates,('Inline','Template')),
                forced_18_21=result(moduli,(18,21)),old_new_comparison='not_run')


def order_cases(rows):
    """Keep smoke/small first, then one target per routine in each round."""
    first=[r for r in rows if r['stage']!='4096']
    targets={}; extra=[]
    for row in rows:
        if row['stage']!='4096': continue
        if row['case']['enforce_native_target']:
            targets.setdefault(row['case']['routine'],[]).append(row)
        else: extra.append(row)
    while any(targets.values()):
        for group in targets.values():
            if group: first.append(group.pop(0))
    return first+extra


def estimated_seconds(row, recent):
    # Do not truncate a case below a recent run of the same routine and size.
    return max(5,recent.get((row['stage'],row['case']['routine']),0)*1.2+2)


def display_counts(records, outcomes):
    successful=[r for r in records if not r.get('validation_failures') and 'scores' in r]
    return dict(execution_completed=sum('scores' in r for r in records),
                full_reference_match=sum(r.get('reference_check')=='all_correctly_rounded' for r in successful),
                native_target_pass=sum(r.get('reference_check')!='all_correctly_rounded' and r.get('native_target_enforced',False) and r.get('qualified_against_native',False) for r in successful),
                record_only=sum(not r.get('native_target_enforced',False) and r.get('reference_check')!='all_correctly_rounded' for r in successful),
                default_rejection_pass=sum(r['status']=='contract_pass' for r in outcomes),
                attempted_failed=sum(r['status'] in ('fail','timeout') for r in outcomes),
                not_run=sum(r['status']=='not_run' for r in outcomes))


def execute(root, output, max_seconds, started_epoch, worker=None, health=None,
            verification_root=None, preflight_timeout=300):
    if preflight_timeout <= 0: raise ValueError('preflight timeout must be positive')
    verification_root = Path(verification_root) if verification_root is not None else root
    rows = order_cases(json.loads((root/'cases.json').read_text()))
    manifest = json.loads((root/'manifest.json').read_text())
    output.mkdir(parents=True,exist_ok=False)
    deadline = time.monotonic()+max(0,max_seconds-(time.time()-started_epoch))
    records=[]; outcomes=[]; stop_reason=None; recent={}
    def event(value):
        value=dict(time=time.time(),remaining_seconds=round(max(0,deadline-time.monotonic()),2),**value)
        with (output/'progress.jsonl').open('a') as stream:
            stream.write(json.dumps(value)+'\n'); stream.flush(); os.fsync(stream.fileno())
        print(json.dumps(value),flush=True)
    write_json(output/'manifest.json',manifest)
    event(dict(status='starting',max_seconds=max_seconds,cases=len(rows)))
    def check_health():
        if health is not None: return health()
        target=output/'environment.part.json'
        command=[sys.executable,str(Path(__file__).resolve()),'--gpu','--package',str(verification_root),
                 '--preflight-output',str(target)]
        limit=min(preflight_timeout,max(1,deadline-time.monotonic()))
        attempt=dict(command=command,timeout_seconds=limit,status='failed')
        started=time.monotonic()
        try:
            code=run_bounded(command,output/'preflight.log',limit)
            attempt['exit_code']=code
            if code: raise RuntimeError('GPU preflight failed; see preflight.log')
            target.replace(output/'environment.json')
            attempt['status']='passed'
        except Exception as exc:
            attempt['error']=str(exc)
            raise
        finally:
            attempt['elapsed_seconds']=time.monotonic()-started
            with (output/'preflight-attempts.jsonl').open('a') as stream:
                stream.write(json.dumps(attempt)+'\n')
    try:
        if deadline-time.monotonic() < 65: stop_reason='time budget used by setup'
        else: check_health()
    except Exception as exc: stop_reason='preflight: '+str(exc)
    for row in rows:
        case=row['case']; remaining=deadline-time.monotonic()-60
        state=dict(id=row['id'],stage=row['stage'],routine=case['routine'],
                   engine_name=case.get('engine_name'),implementation=case.get('implementation','Inline'),attempted=False)
        if stop_reason or remaining < estimated_seconds(row,recent):
            state.update(status='not_run',reason=stop_reason or 'remaining budget is below the recent case duration plus margin')
        elif not (root/row['reference']).is_file():
            state.update(status='not_run',reason='precomputed reference is absent')
        else:
            state['attempted']=True
            folder=output/row['id'];folder.mkdir()
            write_json(folder/'case.json',case);event(dict(state,status='started'))
            partial=folder/'worker.part.json'
            limit=min(60 if row['stage']=='4096' else 45 if case['routine']=='dtrsm' or case.get('implementation')=='Template' else 20,remaining)
            command=[sys.executable,str(worker or Path(__file__).with_name('level3_worker.py')),'--gpu',
                     '--case',str(folder/'case.json'),'--library',str(root/'Lib/libawe.so'),
                     '--engine-directory',str(root/('Lib/Template' if case.get('implementation')=='Template' else 'Lib')),
                     '--output',str(partial),'--reference',str(root/row['reference']),'--require-reference',
                     '--samples',str(row['samples']),'--warmup','1' if row['stage']=='4096' else '0',
                     '--repeat','3' if row['stage']=='4096' else '1']
            if manifest.get('measure_time') is False:
                command[command.index('--warmup')+1] = '0'
                command[command.index('--repeat')+1] = '3'
                command.append('--no-timing')
            def heartbeat():
                phase_path=folder/'phase.json'
                phase=json.loads(phase_path.read_text())['phase'] if phase_path.exists() else 'worker_start'
                event(dict(state,status='running',phase=phase))
            t=time.monotonic()
            try:
                code=run_bounded(command,folder/'worker.log',limit,heartbeat)
                if partial.exists():
                    record=json.loads(partial.read_text());partial.replace(folder/'worker.json')
                    records.append(record)
                    state['execution_completed']='scores' in record
                if code: raise RuntimeError(f'worker exit {code}; see saved evidence and worker.log')
                if not (folder/'worker.json').exists(): raise RuntimeError('worker did not write evidence')
                state['status']=validate_record(record,case,manifest.get('measure_time'))
            except subprocess.TimeoutExpired:
                state.update(status='timeout',reason=f'case limit {limit:.1f} seconds')
                try: check_health()
                except Exception as exc: stop_reason='device not responsive after timeout: '+str(exc)
            except Exception as exc: state.update(status='fail',reason=str(exc))
            state['seconds']=round(time.monotonic()-t,3)
            if state['status'] in ('pass','record_only'):
                recent[(row['stage'],case['routine'])]=state['seconds']
            write_json(folder/'result.json',state)
        outcomes.append(state);event(state)
        comparisons=comparison_results(records,manifest['backend'])
        write_json(output/'comparisons.json',comparisons)
    counts={s:sum(r['status']==s for r in outcomes) for s in ('pass','record_only','contract_pass','fail','timeout','not_run')}
    comparisons=comparison_results(records,manifest['backend'])
    pair_failure=any(r['status']=='fail' for group in ('Inline_Template','forced_18_21') for r in comparisons[group])
    coverage={}
    for library in manifest.get('shared_libraries',[]):
        matches=[r for r in records if r.get('engine_sha256')==library['sha256']]
        coverage[library['path']]=dict(recorded_runs=len(matches),status='recorded' if matches else 'no_numeric_record',note='case attempts are listed separately; a missing record does not mean no attempt')
    result=dict(counts=counts,display_counts=display_counts(records,outcomes),cases=outcomes,comparisons=comparisons,engine_coverage=coverage,
                pass_scope='listed cases only; see engine_coverage for libraries not exercised',
                passed=not pair_failure and not any(counts[s] for s in ('fail','timeout','not_run')),
                gpu_execution=True,old_new_comparison='not_run',stop_reason=stop_reason,
                elapsed_seconds=time.time()-started_epoch)
    write_json(output/'summary.json',result);event(dict(status='finished',counts=counts,passed=result['passed']))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package',type=Path,required=True)
    parser.add_argument('--gpu',action='store_true');parser.add_argument('--list',action='store_true')
    parser.add_argument('--check-package',action='store_true');parser.add_argument('--max-seconds',type=int)
    parser.add_argument('--started-epoch',type=float,default=time.time());parser.add_argument('--output',type=Path)
    parser.add_argument('--preflight-output',type=Path)
    args=parser.parse_args();root=args.package.resolve()
    if args.list:
        print((root/'cases.json').read_text());return
    if args.preflight_output:
        print(json.dumps(dict(phase='package_verification_started')),flush=True)
    check_started=time.monotonic()
    count=verify_package(root)
    if args.preflight_output:
        print(json.dumps(dict(phase='package_verification_complete',files=count,
                              elapsed_seconds=time.monotonic()-check_started)),flush=True)
    if args.check_package:
        print(json.dumps(dict(gpu_execution=False,sha_files=count)));return
    if not args.gpu: parser.error('GPU execution requires --gpu')
    backend=json.loads((root/'manifest.json').read_text())['backend']
    if args.preflight_output:
        print(json.dumps(dict(phase='cuda_preflight_started')),flush=True)
        probe_started=time.monotonic()
        report=preflight(backend)
        report['preflight_seconds']=time.monotonic()-probe_started
        write_json(args.preflight_output,report)
        print(json.dumps(dict(phase='cuda_preflight_complete',elapsed_seconds=report['preflight_seconds'])),flush=True)
        return
    if args.max_seconds is None or args.max_seconds < 90: parser.error('--max-seconds must be at least 90')
    output=args.output or root/'results'
    result=execute(root,output,args.max_seconds,args.started_epoch)
    if not result['passed']: raise SystemExit(1)

if __name__=='__main__': main()
