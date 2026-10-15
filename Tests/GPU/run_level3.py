#!/usr/bin/env python3
"""List or run one shape group. GPU work requires --gpu and two API builds."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from level3_cases import cases, ROUTINES


def compare_pair(old, new):
    assert old['case']==new['case'], 'different descriptors'
    assert old['input_sha256']==new['input_sha256'], 'different inputs'
    assert old['effective_options']==new['effective_options'], 'different engine settings'
    assert old['info']['poison']==new['info']['poison']==0, 'poison differs'
    if new['case'].get('engine_name'):
        assert old['info'].get('engine_name')==new['info'].get('engine_name')==new['case']['engine_name'], 'explicit engine choice differs'
    assert old['output_sha256']['awe']==new['output_sha256']['awe'], 'old/new output hash differs'
    assert old['reference_sha256']==new['reference_sha256'], 'different reference'
    if new['case'].get('enforce_native_target', True):
        assert new['qualified_against_native'], 'new result does not meet native p95/max target'


def compare_moduli(records, backend):
    pairs={}
    for r in records:
        c=r['case']
        if c['routine']!='dgemm' or c['moduli'] not in (18,21): continue
        key=tuple((k,v) for k,v in sorted(c.items()) if k!='moduli')
        values=pairs.setdefault(key,{})
        value=(r['input_sha256'],r['output_sha256']['awe'])
        if c['moduli'] in values: assert values[c['moduli']]==value, 'repeated output differs'
        values[c['moduli']]=value
    if backend==64:
        assert pairs, 'missing forced K64 modulus comparisons'
        for p in pairs.values():
            assert set(p)=={18,21}, 'one forced modulus set is absent'
            assert p[18]==p[21], 'forced 18/21 inputs or output bits differ'
    return len(pairs)


def main(backend=None):
    p=argparse.ArgumentParser(description=__doc__)
    if backend is None: p.add_argument('--backend',type=int,choices=[64,96],required=True)
    p.add_argument('--group',choices=['small','4096','8192','boundaries'],default='small')
    p.add_argument('--routine',choices=list(ROUTINES))
    p.add_argument('--gpu',action='store_true');p.add_argument('--old-library');p.add_argument('--new-library')
    p.add_argument('--output',type=Path);p.add_argument('--cudart',default='libcudart.so.13')
    p.add_argument('--cublas',default='libcublas.so.13');p.add_argument('--device',type=int,default=0)
    p.add_argument('--samples',type=int,default=1024);p.add_argument('--repeat',type=int,default=20)
    p.add_argument('--warmup',type=int,default=3);p.add_argument('--timeout',type=int,default=1800)
    args=p.parse_args(); backend=backend or args.backend
    selected=cases(backend,args.group)
    if args.routine: selected=[c for c in selected if c['routine']==args.routine]
    if not selected: p.error('no cases for this routine, backend and group')
    if not args.gpu:
        print(json.dumps({'gpu_execution':False,'backend':backend,'group':args.group,'cases':selected},indent=2));return
    if not args.old_library or not args.new_library or not args.output:
        p.error('--gpu needs --old-library, --new-library and --output')
    if args.output.exists(): p.error('choose a new output directory; existing evidence is never overwritten')
    if min(args.samples,args.repeat,args.timeout)<1 or args.warmup<0: p.error('invalid count')
    args.output.mkdir(parents=True)
    worker=Path(__file__).with_name('level3_worker.py');records=[]
    manifest={'backend':backend,'group':args.group,'gpu_execution':True,'order':['old','new','new','old'],
              'worker_sha256':hashlib.sha256(worker.read_bytes()).hexdigest(),'cases':selected}
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    failures=[]; completed=[]
    for index,case in enumerate(selected):
        try:
            folder=args.output/f'{index:04d}';folder.mkdir()
            case_path=folder/'case.json';case_path.write_text(json.dumps(case)+'\n');runs={}
            for round_id,label in enumerate(manifest['order']):
                output=folder/f'{round_id}-{label}.json'
                command=[sys.executable,str(worker),'--gpu','--case',str(case_path),
                         '--library',getattr(args,label+'_library'),'--output',str(output),
                         '--cudart',args.cudart,'--cublas',args.cublas,'--device',str(args.device),
                         '--samples',str(args.samples),'--repeat',str(args.repeat),'--warmup',str(args.warmup)]
                with (folder/f'{round_id}-{label}.log').open('w') as log:
                    subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=args.timeout)
                record=json.loads(output.read_text());runs.setdefault(label,[]).append(record)
                if label=='new': records.append(record)
            for old,new in zip(runs['old'],runs['new']): compare_pair(old,new)
            assert runs['new'][0]['output_sha256']==runs['new'][1]['output_sha256'], 'repeat hash changed'
            completed.append(index)
        except (AssertionError, OSError, ValueError, subprocess.SubprocessError) as exc:
            failures.append({'index':index,'case':case,'error':str(exc)})
    count=0
    try:
        if any(c['routine']=='dgemm' for c in selected): count=compare_moduli(records,backend)
    except AssertionError as exc:
        failures.append({'check':'forced_18_21','error':str(exc)})
    summary={'passed':not failures,'cases':len(selected),'completed':completed,
             'failures':failures,'forced_18_21_pairs':count,'gpu_execution':True}
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    if not failures: (args.output/'passed.json').write_text(json.dumps(summary,indent=2)+'\n')
    else: raise SystemExit(1)

if __name__=='__main__': main()
