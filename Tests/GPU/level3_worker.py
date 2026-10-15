"""One GPU case in an isolated process. Requires an explicit --gpu switch."""
import argparse
import ctypes as ct
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'Tools'))
from awe_validation import Descriptor, EffectiveOptions
from level3_cases import ROUTINES, exact_cell, solve_reference, ulp_error, percentile95, rounded_reference

V = ct.c_void_p
I = ct.c_int
class Options(ct.Structure):
    _fields_ = [('struct_size', ct.c_size_t)] + [(n, I) for n in
        ('pacing', 'chunks', 'order', 'encoder_minb', 'pingpong', 'encoder', 'quantize_bits')] + [('engine_name', ct.c_char_p)]


def bind(library, name, arguments):
    fn = getattr(library, name); fn.argtypes = arguments; fn.restype = I
    return fn


def check(status):
    if status: raise RuntimeError(f'CUDA/cuBLAS status {status}')


class DeviceArray:
    def __init__(self, cuda, array):
        self.cuda = cuda; self.pointer = V(); self.initial = array.tobytes(order='F')
        self.raw = b'\xd3'*128+self.initial+b'\xa7'*128
        check(cuda.cudaMalloc(ct.byref(self.pointer), len(self.raw)))
        self.data = V(self.pointer.value+128)
        try: self.reset()
        except BaseException: self.close(); raise

    def reset(self):
        host = ct.create_string_buffer(self.raw)
        check(self.cuda.cudaMemcpy(self.pointer, host, len(self.raw), 1))

    def read(self):
        host = ct.create_string_buffer(len(self.raw))
        check(self.cuda.cudaMemcpy(host, self.pointer, len(self.raw), 2))
        assert host.raw[:128] == self.raw[:128] and host.raw[-128:] == self.raw[-128:], 'canary changed'
        return host.raw[128:-128]

    def close(self):
        if self.pointer.value:
            check(self.cuda.cudaFree(self.pointer)); self.pointer = V()


def repeat_outputs(reset, execute, synchronize, read, repeat, warmup, measure_time):
    """Reset each call, check its output, and optionally collect timings."""
    for _ in range(warmup):
        reset(); execute(); synchronize()
    elapsed, hashes = [], []
    for _ in range(repeat):
        reset(); synchronize()
        start = time.perf_counter() if measure_time else None
        execute(); synchronize()
        if measure_time: elapsed.append((time.perf_counter()-start)*1000)
        raw = read()
        hashes.append(hashlib.sha256(raw).hexdigest())
    return raw, elapsed, hashes


def native_operation(lib, handle, name, shape, arrays):
    """Call the native typed BLAS routine, with the same lower/left conventions."""
    m, n, k = shape
    A, B, C = (x.data for x in arrays)
    kind = name[1:]; scalar = ct.c_float if name[0] in 'sc' else ct.c_double
    complex_type = name[0] in 'zc'
    one = (scalar*2)(1, 0) if complex_type else scalar(1)
    zero = (scalar*2)(0, 0) if complex_type else scalar(0)
    alpha, beta = ct.byref(one), ct.byref(zero)
    prefix = 'cublas'+name[0].upper()
    lda = m; ldb = m if kind in ('syr2k', 'her2k') else k; ldc = m
    if name == 'i8gemm':
        one_i, zero_i = I(1), I(0)
        # CUDA_R_8I=3, CUDA_R_32I=10, CUBLAS_COMPUTE_32I=72.
        fn = bind(lib, 'cublasGemmEx', [V,I,I,I,I,I,V,V,I,I,V,I,I,V,V,I,I,I,I])
        return lambda: check(fn(handle,0,0,m,n,k,ct.byref(one_i),A,3,lda,B,3,ldb,
                                ct.byref(zero_i),C,10,ldc,72,-1))
    if kind == 'gemm':
        fn = bind(lib, prefix+'gemm_v2', [V,I,I,I,I,I,V,V,I,V,I,V,V,I])
        args = [handle,0,0,m,n,k,alpha,A,lda,B,ldb,beta,C,ldc]
    elif kind in ('symm', 'hemm'):
        fn = bind(lib, prefix+kind+'_v2', [V,I,I,I,I,V,V,I,V,I,V,V,I])
        args = [handle,0,0,m,n,alpha,A,lda,B,ldb,beta,C,ldc]
    elif kind in ('syrk', 'herk'):
        if kind == 'herk':
            one_r, zero_r = scalar(1), scalar(0)
            alpha, beta = ct.byref(one_r), ct.byref(zero_r)
        fn = bind(lib, prefix+kind+'_v2', [V,I,I,I,I,V,V,I,V,V,I])
        args = [handle,0,0,n,k,alpha,A,lda,beta,C,ldc]
    elif kind in ('syr2k', 'her2k'):
        if kind == 'her2k':
            zero_r = scalar(0); beta = ct.byref(zero_r)
        fn = bind(lib, prefix+kind+'_v2', [V,I,I,I,I,V,V,I,V,I,V,V,I])
        args = [handle,0,0,n,k,alpha,A,lda,B,ldb,beta,C,ldc]
    elif kind == 'trmm':
        fn = bind(lib, prefix+'trmm_v2', [V,I,I,I,I,I,I,V,V,I,V,I,V,I])
        args = [handle,0,0,0,0,m,n,alpha,A,lda,B,ldb,C,ldc]
    elif kind == 'trsm':
        fn = bind(lib, prefix+'trsm_v2', [V,I,I,I,I,I,I,V,V,I,V,I])
        # cuBLAS TRSM is in-place; reset C to B outside the timed region.
        args = [handle,0,0,0,0,m,n,alpha,A,lda,C,ldc]
    else: raise ValueError(name)
    return lambda: check(fn(*args))


def make_inputs(case):
    import numpy as np
    name = case['routine']; m,n,k = (case[x] for x in ('m','n','k'))
    dtype = {'d': np.float64, 'z': np.complex128, 's': np.float32,
             'c': np.complex64, 'i': np.int8}[name[0]]
    rng = np.random.default_rng(case['seed']); kind = name[1:]
    def draw(shape):
        if name == 'i8gemm': x = rng.integers(-128, 128, size=shape)
        elif case['distribution'] == 'dyadic': x = rng.integers(-3, 4, size=shape)/8
        elif case['distribution'] == 'normal': x = rng.standard_normal(shape)
        else: x = rng.uniform(-1, 1, shape)
        if name[0] in 'zc':
            y = rng.integers(-3, 4, size=shape)/8 if case['distribution']=='dyadic' else rng.standard_normal(shape) if case['distribution']=='normal' else rng.uniform(-1,1,shape)
            x = x+1j*y
        return np.asfortranarray(x, dtype=dtype)
    a = draw((m,k)); b = draw((m,k) if kind in ('syr2k','her2k') else (k,n))
    if kind in ('symm','hemm'):
        a = np.tril(a)+np.tril(a,-1).conj().T if kind=='hemm' else np.tril(a)+np.tril(a,-1).T
        if kind=='hemm': a[np.diag_indices(m)] = a.diagonal().real
    elif kind=='trmm': a = np.tril(a)
    elif kind=='trsm':
        # Sparse dyadic solve crosses both 128-row and 2048-row block boundaries.
        a.fill(0); a[np.diag_indices(m)] = 1
        a[np.arange(1,m),np.arange(m-1)] = .125
        if m>2048: a[2048,0] = .0625
    c = np.full((m,n), -703, dtype=np.int32 if name=='i8gemm' else dtype, order='F')
    return np.asfortranarray(a), b, c


def device_input(a, name):
    import numpy as np
    gpu_a = a.copy(order='F')
    if name[1:] in ('symm','hemm','trmm','trsm'):
        gpu_a[np.triu_indices(a.shape[0],1)] = np.nan
    if name[1:] == 'hemm': gpu_a[np.diag_indices(a.shape[0])] += 7j
    return gpu_a


def save_verdict(path, record, failures):
    """Save evidence even when the numeric checks reject the output."""
    failures = list(failures)
    def finite(value):
        if isinstance(value, float) and not math.isfinite(value):
            if "non-finite numeric evidence" not in failures: failures.append("non-finite numeric evidence")
            return str(value)
        if isinstance(value, dict): return {k:finite(v) for k,v in value.items()}
        if isinstance(value, list): return [finite(v) for v in value]
        return value
    record = finite(record)
    record['validation_failures'] = failures
    if failures: record['correctness'] = 'fail'
    path = Path(path); temporary = path.with_suffix(path.suffix+'.writing')
    temporary.write_text(json.dumps(finite(record),indent=2,allow_nan=False)+'\n')
    temporary.replace(path)
    if failures: raise AssertionError('; '.join(failures))


def phase(args, name):
    path=Path(args.output).parent/'phase.json'
    temporary=path.with_suffix('.part.json')
    temporary.write_text(json.dumps(dict(phase=name,time=time.time()))+'\n')
    temporary.replace(path)


def run(args):
    import numpy as np
    case = json.loads(Path(args.case).read_text()); name = case['routine']
    if case.get('validation_suite', 'new_input_seed') != 'new_input_seed':
        raise ValueError('historical reproduction requires its saved inputs, reference points, metric and native library; do not generate new inputs')
    phase(args,'inputs_and_reference')
    a,b,c = make_inputs(case); m,n,k = (case[x] for x in ('m','n','k'))
    cache = None
    if getattr(args, 'reference', None):
        from reference_cache import read_cache
        cache = read_cache(args.reference, case, a, b, args.samples)
    if getattr(args, 'require_reference', False) and cache is None:
        raise ValueError('precomputed reference required')
    rank = name[1:] in ('syrk','herk','syr2k','her2k'); single = name[0] in 'sc'
    phase(args,'cuda_setup')
    cuda = ct.CDLL(args.cudart)
    for symbol, params in {'cudaMalloc':[ct.POINTER(V),ct.c_size_t], 'cudaFree':[V],
                           'cudaMemcpy':[V,V,ct.c_size_t,I], 'cudaSetDevice':[I],
                           'cudaDeviceSynchronize':[], 'cudaDeviceGetAttribute':[ct.POINTER(I),I,I],
                           'cudaRuntimeGetVersion':[ct.POINTER(I)]}.items(): bind(cuda,symbol,params)
    check(cuda.cudaSetDevice(args.device))
    major,minor,runtime=I(),I(),I()
    check(cuda.cudaDeviceGetAttribute(ct.byref(major),75,args.device))
    check(cuda.cudaDeviceGetAttribute(ct.byref(minor),76,args.device))
    check(cuda.cudaRuntimeGetVersion(ct.byref(runtime)))
    assert (major.value,minor.value)==(10,3 if case['backend']==96 else 0), 'wrong device architecture'
    api = ct.CDLL(str(Path(args.library).resolve()))
    api.awe_last_error.restype = ct.c_char_p
    def call(status):
        if status: raise RuntimeError(api.awe_last_error().decode())
    bind(api,'awe_descriptor_init',[ct.POINTER(Descriptor),I])
    bind(api,'awe_options_init',[ct.POINTER(Options)])
    bind(api,'awe_plan_create_with_options',[ct.POINTER(V),ct.POINTER(Descriptor),ct.POINTER(Options),ct.c_char_p])
    bind(api,'awe_plan_destroy',[V])
    bind(api,'awe_plan_effective_options',[V,ct.POINTER(EffectiveOptions)])
    bind(api,'awe_plan_info',[V,V,ct.c_size_t,ct.POINTER(ct.c_size_t)])
    rank_one = name[1:] in ('syrk','herk')
    typed = bind(api,'awe_'+name,[V,V,V] if rank_one else [V,V,V,V])
    d = Descriptor(); api.awe_descriptor_init(ct.byref(d),ROUTINES[name])
    for key in ('backend','m','n','k','moduli'): setattr(d,key,case[key])
    d.lda=m; d.ldb=m if name[1:] in ('syr2k','her2k') else k; d.ldc=m
    opts = Options(); api.awe_options_init(ct.byref(opts)); opts.quantize_bits=case['bits']
    opts.engine_name = case.get('engine_name').encode() if case.get('engine_name') else None
    plan=V(); handle=V(); allocations=[]
    blas = ct.CDLL(args.cublas)
    bind(blas,'cublasCreate_v2',[ct.POINTER(V)]); bind(blas,'cublasDestroy_v2',[V])
    bind(blas,'cublasSetMathMode',[V,I])
    try:
        engine_directory = Path(args.engine_directory).resolve() if getattr(args,'engine_directory',None) else Path(args.library).resolve().parent
        phase(args,'plan_create')
        status = api.awe_plan_create_with_options(ct.byref(plan),ct.byref(d),ct.byref(opts),str(engine_directory).encode())
        if case.get('expected_not_supported'):
            if status != 2 or plan.value: raise AssertionError('expected NOT_SUPPORTED without a plan')
            message = api.awe_last_error().decode()
            if 'not run on a GPU' not in message: raise AssertionError('wrong rejection reason: '+message)
            Path(args.output).write_text(json.dumps({'case':case,'gpu_execution':True,
                'contract_check':'pass','correctness':'not_run','old_new_comparison':'not_run',
                'status':status,'reason':message})+'\n')
            return
        call(status)
        effective=EffectiveOptions(); call(api.awe_plan_effective_options(plan,ct.byref(effective)))
        size=ct.c_size_t(); call(api.awe_plan_info(plan,None,0,ct.byref(size)))
        info_buffer=ct.create_string_buffer(size.value); call(api.awe_plan_info(plan,info_buffer,size.value,ct.byref(size)))
        info=json.loads(info_buffer.value)
        settings={key:getattr(effective,key) for key,_ in EffectiveOptions._fields_}
        assert all(info[key]==value for key,value in settings.items()), 'info/effective mismatch'
        assert info['poison']==0, 'poison must be disabled'
        input_digest=hashlib.sha256(a.tobytes(order='F')+b.tobytes(order='F')).hexdigest()
        # Unused upper input data must not affect the answer.
        gpu_a=device_input(a,name)
        for array in (gpu_a,b,c): allocations.append(DeviceArray(cuda,array))
        A,B,C=allocations
        def ours(): call(typed(plan,A.data,C.data) if rank_one else typed(plan,A.data,B.data,C.data))
        check(blas.cublasCreate_v2(ct.byref(handle))); check(blas.cublasSetMathMode(handle,2))
        version=I();check(bind(blas,'cublasGetVersion_v2',[V,ct.POINTER(I)])(handle,ct.byref(version)))
        native=native_operation(blas,handle,name,(m,n,k),allocations)
        outputs={}; times={}; repeated_hashes={}
        for label, execute in [('awe',ours),('native',native)]:
            phase(args,label+'_execute')
            def reset():
                C.reset()
                if name=='dtrsm' and label=='native': check(cuda.cudaMemcpy(C.data,B.data,len(B.initial),3))
            raw, elapsed, hashes = repeat_outputs(reset, execute,
                lambda: check(cuda.cudaDeviceSynchronize()), C.read, args.repeat,
                args.warmup, not getattr(args, 'no_timing', False))
            repeated_hashes[label] = hashes
            got=np.frombuffer(raw,dtype=c.dtype).reshape((m,n),order='F').copy()
            if rank: assert np.array_equal(np.triu(got,1),np.triu(c,1)), 'upper output changed'
            outputs[label]=got;times[label]=elapsed
            assert A.read()==A.initial and B.read()==B.initial, 'input changed'
        if cache is not None:
            points = [(r[0],r[1]) for r in cache['references']]
        else:
            from reference_cache import points_for
            points = points_for(case,args.samples)
        reference_solve=solve_reference(a,b) if name=='dtrsm' and cache is None else None
        phase(args,'compare_reference')
        errors={'awe':[],'native':[]}; exact_matches={'awe':0,'native':0}
        references=[]; mismatches=[]
        for point_index,(i,j) in enumerate(points):
            exact=(Fraction(0),Fraction(0)) if cache is not None else (reference_solve[i][j],0) if reference_solve is not None else exact_cell(name,a,b,i,j)
            if cache is not None:
                exact=tuple(Fraction(*part) for part in cache['references'][point_index][2])
            references.append([i,j,[[v.numerator,v.denominator] if hasattr(v,'numerator') else [v,1] for v in exact]])
            for label in errors:
                value=complex(outputs[label][i,j]); parts=[value.real,value.imag] if name[0] in 'zc' else [value.real]
                for index,part in enumerate(parts):
                    want=rounded_reference(exact[index],24 if single else 53)
                    exact_matches[label]+=int(part==want)
                    if label=='awe' and part!=want and len(mismatches)<16:
                        mismatches.append(dict(row=i,column=j,component=index,actual=part,
                            rounded_reference=want,exact=[exact[index].numerator,exact[index].denominator]))
                    errors[label].append(ulp_error(part,exact[index],24 if single else 53))
        scores={key:{'p95':percentile95(value),'max':max(value)} for key,value in errors.items()}
        qualified=all(scores['awe'][key]<=scores['native'][key] for key in ('p95','max'))
        exact_required=case['distribution']=='dyadic' and name!='dtrsm'
        failures=[]
        if any(len(set(values)) != 1 for values in repeated_hashes.values()):
            failures.append("repeated output hash differs")
        if exact_required and exact_matches['awe']!=len(errors['awe']): failures.append('small full reference mismatch')
        if name=='i8gemm' and not np.array_equal(outputs['awe'],outputs['native']): failures.append('INT8 exact mismatch')
        if case['enforce_native_target'] and not qualified: failures.append('native p95/max target failed')
        engine_path=engine_directory/('libawe_'+info['engine']+'.so')
        record={'implementation':case.get('implementation','Inline'),'engine_directory':str(engine_directory),
                'old_new_comparison':'not_run','reference_precomputed':cache is not None,
                'cuda_runtime':runtime.value,'cublas_version':version.value,'device':args.device,
                'device_architecture':[major.value,minor.value],
                'engine_sha256':hashlib.sha256(engine_path.read_bytes()).hexdigest(),'case':case,'gpu_execution':True,'library_sha256':hashlib.sha256(Path(args.library).read_bytes()).hexdigest(),
                'input_sha256':input_digest,'info':info,'effective_options':settings,
                'output_sha256':{label:hashlib.sha256(value.tobytes(order='F')).hexdigest() for label,value in outputs.items()},
                'reference_sha256':hashlib.sha256(json.dumps(references).encode()).hexdigest(),
                'correctness':('pass' if qualified else 'fail') if case['enforce_native_target'] else 'record_only',
                'reference_check':('all_correctly_rounded' if exact_matches['awe']==len(errors['awe']) else 'full_reference_mismatch') if exact_required else 'errors_recorded',
                'first_mismatches':mismatches,
                'reference_points':len(points),'reference_all_outputs':case['distribution']=='dyadic',
                'scores':scores,'qualified_against_native':qualified,
                'native_target_enforced':case['enforce_native_target'],
                'target_result':('pass' if qualified else 'fail') if case['enforce_native_target'] else 'record_only','correctly_rounded_components':exact_matches,
                'canaries':True,'inputs_unchanged':True,'timing_scope':'CPU wall time, execute plus device synchronization; excludes reset, creation and transfers',
                'repeated_output_sha256':repeated_hashes,'timing_measured':not getattr(args,'no_timing',False)}
        if getattr(args,'no_timing',False):
            record['timing_scope']='not measured'
        else:
            record['time_ms']=times
            record['median_ms']={key:statistics.median(value) for key,value in times.items()}
        phase(args,'save_verdict')
        save_verdict(args.output,record,failures)
    finally:
        if handle.value: check(blas.cublasDestroy_v2(handle))
        for allocation in reversed(allocations): allocation.close()
        if plan.value: call(api.awe_plan_destroy(plan))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu',action='store_true');p.add_argument('--case',required=True)
    p.add_argument('--library',required=True);p.add_argument('--output',required=True)
    p.add_argument('--cudart',default='libcudart.so.13');p.add_argument('--cublas',default='libcublas.so.13')
    p.add_argument('--engine-directory');p.add_argument('--reference');p.add_argument('--require-reference',action='store_true')
    p.add_argument('--device',type=int,default=0);p.add_argument('--samples',type=int,default=1024)
    p.add_argument('--no-timing',action='store_true')
    p.add_argument('--warmup',type=int,default=3);p.add_argument('--repeat',type=int,default=20)
    args=p.parse_args()
    if not args.gpu: p.error('GPU execution requires --gpu')
    if min(args.samples,args.repeat)<1 or args.warmup<0: p.error('invalid repeat/sample count')
    run(args)

if __name__=='__main__': main()
