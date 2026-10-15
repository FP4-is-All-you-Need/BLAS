#!/usr/bin/env python3
"""Integer/rational reference, explicit GPU checks, and cross-device record comparison."""
import argparse
import ctypes as ct
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import struct
import time


def quantize(values, bits):
    """One shared binary exponent; nearest-even integers for real/complex components."""
    parts = [(float(complex(v).real), float(complex(v).imag)) for v in values]
    if any(not math.isfinite(x) for p in parts for x in p):
        raise ValueError('non-finite value in used input')
    maximum = max((abs(x) for p in parts for x in p), default=0.0)
    shift = bits - math.frexp(maximum)[1] if maximum else 0
    scale = Fraction(2) ** shift
    return [(round(Fraction(a) * scale), round(Fraction(b) * scale)) for a, b in parts], shift


def reference(a, b, bits=53, complex_output=False):
    """Row-list matrices; products/sums are unbounded integers, final scaling rational."""
    m, k, n = len(a), len(a[0]), len(b[0])
    assert len(b) == k and all(len(row) == k for row in a) and all(len(row) == n for row in b)
    qa = [quantize(row, bits) for row in a]
    qb = [quantize([b[t][j] for t in range(k)], bits) for j in range(n)]
    out = []
    for i in range(m):
        row = []
        for j in range(n):
            ar, sa = qa[i]; br, sb = qb[j]
            real = sum(x*u-y*v for (x,y),(u,v) in zip(ar,br))
            imag = sum(x*v+y*u for (x,y),(u,v) in zip(ar,br))
            scale = Fraction(2) ** (-sa-sb)
            r, z = float(real*scale), float(imag*scale)
            row.append(complex(r,z) if complex_output else r)
        out.append(row)
    return out


def cpu_checks():
    assert reference([[1.,2.]],[[3.],[4.]]) == [[11.]]
    # An exact dot product retains the low term lost by sequential FP64 summation.
    assert reference([[2.**52,1.,-2.**52]],[[1.],[1.],[1.]]) == [[1.]]
    assert quantize([1.25,1.75],3)[0] == [(5,0),(7,0)]
    assert quantize([1.,0.625,0.875],2)[0] == [(2,0),(1,0),(2,0)]
    assert reference([[1+2j,3-1j]],[[1-2j],[3+1j]],51,True) == [[15+0j]]
    try: quantize([float('nan')],53)
    except ValueError: pass
    else: raise AssertionError('NaN accepted')
    return {'cpu_reference':'PASS','gpu_execution':False}


class Complex(ct.Structure):
    _fields_=[('real',ct.c_double),('imag',ct.c_double)]
class Descriptor(ct.Structure):
    _fields_=[('struct_size',ct.c_size_t),('routine',ct.c_int),('backend',ct.c_int),('moduli',ct.c_int)] + [
        (n,ct.c_int64) for n in ('m','n','k','lda','ldb','ldc')] + [
        (n,ct.c_int) for n in ('trans_a','trans_b','triangle','side','diagonal')] + [('alpha',Complex),('beta',Complex)]


def validation_cases(backend):
    cases=[('gemm',1,7,9,k,mod) for k,mod in [(4096,0),(4097,0),(8192,0),(8193,0),(8254,0),(8255,0),(4096,17),(4096,18),(4097,17),(4097,18),(8192,17),(8192,18)]]
    if backend==64: cases += [('gemm',1,7,9,k,21) for k in (4096,4097,8192,8255,16384)]
    else: cases += [('syrk',3,7,7,k,0) for k in (97,8255)] + [('trmm',4,17,17,17,0),('ztrmm',5,17,17,17,0),('zherk',6,7,7,97,0)]
    return cases


def check_forced_moduli(outputs, label, m, n, k, mod, digest):
    matched=[]
    if mod in (17,18,21):
        for peer in (17,18,21):
            other=f'{label}:{m}:{n}:{k}:{peer}'
            if peer != mod and other in outputs:
                assert outputs[other]==digest, f'{mod}/{peer} mismatch at K={k}'
                matched.append(peer)
    return matched


class EffectiveOptions(ct.Structure):
    _fields_=[(name,ct.c_int) for name in (
        'pace','chunks','crt_chunks','item_order','encoder_minb','pingpong',
        'skip_b_padding','pdl','shift_form','extra_fx','encoder_fast','moduli')]


def gpu_checks(args):
    # This branch is reachable only with explicit --gpu; CPU tests do not import/load CUDA.
    import numpy as np
    cuda=ct.CDLL(args.cudart)
    for name, params in {
        'cudaMalloc':[ct.POINTER(ct.c_void_p),ct.c_size_t], 'cudaFree':[ct.c_void_p],
        'cudaMemcpy':[ct.c_void_p,ct.c_void_p,ct.c_size_t,ct.c_int], 'cudaSetDevice':[ct.c_int],
    }.items():
        getattr(cuda,name).argtypes=params; getattr(cuda,name).restype=ct.c_int
    def ck(status):
        if status: raise RuntimeError(f'CUDA status {status}')
    ck(cuda.cudaSetDevice(args.device))
    api=ct.CDLL(str(Path(args.library).resolve()))
    api.awe_descriptor_init.argtypes=[ct.POINTER(Descriptor),ct.c_int]
    api.awe_validate.argtypes=[ct.POINTER(Descriptor)];api.awe_validate.restype=ct.c_int
    api.awe_plan_create.argtypes=[ct.POINTER(ct.c_void_p),ct.POINTER(Descriptor),ct.c_char_p]
    api.awe_plan_create.restype=ct.c_int
    api.awe_execute.argtypes=[ct.c_void_p]*4;api.awe_execute.restype=ct.c_int
    api.awe_plan_destroy.argtypes=[ct.c_void_p];api.awe_plan_destroy.restype=ct.c_int
    api.awe_plan_effective_options.argtypes=[ct.c_void_p,ct.POINTER(EffectiveOptions)]
    api.awe_plan_effective_options.restype=ct.c_int
    api.awe_last_error.restype=ct.c_char_p
    api.awe_plan_info.argtypes=[ct.c_void_p,ct.c_void_p,ct.c_size_t,ct.POINTER(ct.c_size_t)]
    def call(status):
        if status: raise RuntimeError(api.awe_last_error().decode())
    class Allocation:
        def __init__(self, data):
            self.raw=bytes([0xD3])*128+data+bytes([0xA7])*128
            self.pointer=ct.c_void_p();ck(cuda.cudaMalloc(ct.byref(self.pointer),len(self.raw)))
            self.data=self.pointer.value+128;self.n=len(data)
            host=ct.create_string_buffer(self.raw);ck(cuda.cudaMemcpy(self.pointer,host,len(self.raw),1))
        def read(self):
            out=ct.create_string_buffer(len(self.raw));ck(cuda.cudaMemcpy(out,self.pointer,len(self.raw),2))
            data=out.raw
            assert data[:128]==self.raw[:128] and data[-128:]==self.raw[-128:], 'canary modified'
            return data[128:-128]
        def close(self): ck(cuda.cudaFree(self.pointer))
    cases=validation_cases(args.backend)
    records=[];plans={}; outputs={}
    started=time.monotonic()
    try:
        # Keep plans alive and reverse order to exercise the per-module CRT owner path.
        for pass_id, sequence in enumerate((cases,list(reversed(cases)))):
            for label,r,m,n,k,mod in sequence:
                key=f'{label}:{m}:{n}:{k}:{mod}'
                d=Descriptor();api.awe_descriptor_init(ct.byref(d),r)
                d.backend=args.backend;d.moduli=mod;d.m=m;d.n=n;d.k=k;d.lda=m;d.ldb=k;d.ldc=m
                if r==1: d.lda+=3;d.ldb+=5;d.ldc+=7
                validation=api.awe_validate(ct.byref(d))
                if mod==0 and r==1 and (k==8255 or (args.backend==64 and k>8192)):
                    assert validation==2, 'AUTO accepted beyond its supported table/coverage range'
                    if pass_id==0: records.append({'case':key,'rejected':True})
                    continue
                call(validation)
                if key not in plans:
                    plan=ct.c_void_p();call(api.awe_plan_create(ct.byref(plan),ct.byref(d),None));plans[key]=plan
                plan=plans[key]
                rng=np.random.default_rng(193+m+n+k)
                z=r in (5,6);dtype=np.complex128 if z else np.float64
                a=np.asfortranarray(rng.standard_normal((m,k)).astype(dtype));b=np.asfortranarray(rng.standard_normal((k,n)).astype(dtype))
                if z: a+=1j*rng.standard_normal(a.shape);b+=1j*rng.standard_normal(b.shape)
                rank=r in (3,6)
                if r in (4,5):
                    a[np.triu_indices(m,1)]=complex(float('nan'),float('nan')) if z else float('nan')
                    refa=np.tril(a)
                else: refa=a
                refb=a.conj().T if r==6 else a.T if r==3 else b
                expected=np.asarray(reference(refa.tolist(),refb.tolist(),51 if r==6 else 53,z),dtype=dtype)
                aa=np.full((d.lda,k),-701,dtype=dtype,order='F');aa[:m]=a
                bb=np.full((d.ldb,n),-702,dtype=dtype,order='F');bb[:k]=b
                cc=np.full((d.ldc,n),-703,dtype=dtype,order='F')
                memories=[]
                try:
                    A=Allocation(aa.tobytes(order='F'));memories.append(A)
                    B=Allocation(bb.tobytes(order='F'));memories.append(B)
                    C=Allocation(cc.tobytes(order='F'));memories.append(C)
                    begin=time.monotonic();call(api.awe_execute(plan,A.data,None if rank else B.data,C.data));elapsed=time.monotonic()-begin
                    got=C.read();assert A.read()==aa.tobytes(order='F') and B.read()==bb.tobytes(order='F'),'input changed'
                    want=cc.copy(order='F')
                    if rank:
                        lo=np.tril_indices(n);want[lo]=expected[lo]
                    else: want[:m]=expected
                    assert got==want.tobytes(order='F'),f'bit mismatch: {key}'
                    valid=np.frombuffer(got,dtype=dtype).reshape((d.ldc,n),order='F')[:m]
                    digest=hashlib.sha256(np.asfortranarray(valid).tobytes(order='F')).hexdigest()
                    assert key not in outputs or outputs[key]==digest,'alternating plan changed output'
                    outputs[key]=digest
                    matched=check_forced_moduli(outputs,label,m,n,k,mod,digest)
                    size=ct.c_size_t();call(api.awe_plan_info(plan,None,0,ct.byref(size)))
                    info=ct.create_string_buffer(size.value);call(api.awe_plan_info(plan,info,size.value,ct.byref(size)))
                    info_values=json.loads(info.value)
                    effective=EffectiveOptions();call(api.awe_plan_effective_options(plan,ct.byref(effective)))
                    assert all(info_values[name]==getattr(effective,name) for name,_ in EffectiveOptions._fields_), 'typed/info settings differ'
                    expected_pingpong=int(r==3 or (r==1 and mod!=21))
                    assert info_values['pingpong']==expected_pingpong and info_values['poison']==0, 'measured defaults differ'
                    records.append({'case':key,'pass':pass_id,'sha256':digest,'cpu_reference':True,'canary':True,
                        'inputs_unchanged':True,'execute_seconds':elapsed,'info':info_values,'matched_forced_moduli':matched,
                        'input_sha256':hashlib.sha256(aa.tobytes(order='F')+bb.tobytes(order='F')).hexdigest()})
                finally:
                    for memory in reversed(memories): memory.close()
    finally:
        for plan in plans.values(): call(api.awe_plan_destroy(plan))
    result={'backend':args.backend,'device':args.device,'numpy':np.__version__,'gpu_execution':True,
            'wall_seconds':time.monotonic()-started,'records':records}
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
    return result


def compare_records(paths):
    reports=[json.loads(Path(p).read_text()) for p in paths]
    assert {r['backend'] for r in reports}=={64,96}, 'provide one K64 and one K96 report'
    maps=[{x['case']:x for x in r['records'] if 'sha256' in x} for r in reports]
    common=maps[0].keys() & maps[1].keys();assert len(common)>=9,'incomplete cross-geometry records'
    for key in sorted(common):
        assert maps[0][key]['input_sha256']==maps[1][key]['input_sha256'], f'inputs differ: {key}'
        assert maps[0][key]['sha256']==maps[1][key]['sha256'], f'outputs differ: {key}'
    return {'cross_geometry':'PASS','compared_cases':len(common),'gpu_execution':False}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu',action='store_true',help='explicitly load CUDA and execute GPU Tests')
    p.add_argument('--compare',nargs=2,metavar=('K64_JSON','K96_JSON'))
    p.add_argument('--backend',type=int,choices=(64,96),default=96)
    p.add_argument('--library',default='Build/Lib/libawe.so');p.add_argument('--cudart',default='libcudart.so.13')
    p.add_argument('--device',type=int,default=0);p.add_argument('--output',default='validation.json')
    args=p.parse_args()
    if args.gpu and args.compare: p.error('--gpu and --compare are mutually exclusive')
    result=compare_records(args.compare) if args.compare else gpu_checks(args) if args.gpu else cpu_checks()
    print(json.dumps(result,indent=2))
if __name__=='__main__': main()
