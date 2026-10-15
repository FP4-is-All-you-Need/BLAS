"""Case definitions and exact arithmetic; importing this module never loads CUDA."""
from fractions import Fraction
import math

ROUTINES = {'dgemm': 1, 'zgemm': 2, 'dsyrk': 3, 'dtrmm': 4,
            'ztrmm': 5, 'zherk': 6, 'dsymm': 7, 'dsyr2k': 8,
            'zhemm': 9, 'zher2k': 10, 'zsymm': 11, 'zsyrk': 12,
            'zsyr2k': 13, 'sgemm': 14, 'ssymm': 15, 'ssyrk': 16,
            'ssyr2k': 17, 'strmm': 18, 'cgemm': 19, 'i8gemm': 20,
            'dtrsm': 21}


def widths(name, backend, k):
    if name == 'i8gemm': return [8]
    if name == 'dtrsm': return [57, 53]
    if backend == 64:
        if name[0] in 'dz': return [53]
        if name == 'cgemm': return [22, 20, 23] if k <= 4096 else [22, 20]
        if name == 'sgemm': return [23, 21]
        return ({'ssymm': [22, 21], 'ssyrk': [23], 'ssyr2k': [23, 22], 'strmm': [23, 21]}
                if k <= 4096 else
                {'ssymm': [22, 20], 'ssyrk': [24, 23], 'ssyr2k': [21, 20], 'strmm': [22]})[name]
    if k <= 4096 and name in ('cgemm','zsyrk','zherk'):
        return {'cgemm':[22,23,20],'zsyrk':[53,52,50],'zherk':[51,49,53]}[name]
    if k <= 4096 and name[0] == 's':
        return {'sgemm':[23,22,21], 'ssymm':[22,21,20], 'ssyrk':[23,24],
                'ssyr2k':[21,22,20], 'strmm':[23,21,22]}[name]
    return {'dgemm': [53, 52, 51], 'zgemm': [53, 48], 'zsyrk': [52, 50],
            'zherk': [51, 49], 'sgemm': [23, 21], 'ssymm': [22, 20],
            'ssyrk': [24, 23], 'ssyr2k': [21, 20], 'strmm': [22],
            'cgemm': [22, 20]}.get(name, [53])


def explicit_engine(name, backend, bits, k):
    # These sources need explicit selection until GPU validation succeeds.
    if backend != 64: return None
    if name == 'sgemm' and bits == 23: return 's23_m9_gemm'
    if name == 'ssyrk' and bits == 24: return 's24_m12_level3'
    if name == 'cgemm': return f'c{bits}_m{10 if bits >= 22 else 9}_gemm'
    if name == 'dtrsm' and bits == 57: return 'd57_m20_trsm'
    if name == 'ssymm' and bits == 20 or name == 'ssyr2k' and bits == 20:
        return 's20_m10_level3'
    if name[0] == 's' and name != 'sgemm':
        if bits == 22: return f's22_m11_k{4096 if k <= 4096 else 8192}'
        if k > 4096 and bits in (21, 23): return f's{bits}_m{10 if bits == 21 else 12}_k8192'
    return None


def precision_target(name, backend, bits, k, distribution):
    if name == 'dtrsm': return bits == 57
    if distribution == 'dyadic' or name == 'i8gemm': return True
    choices = widths(name, backend, k)
    # First width is the normal choice; the last is the uniform choice.
    normal, uniform = choices[0], choices[-1]
    if backend == 96 and k <= 4096 and name[0] == 's':
        normal, uniform = {'sgemm':(23,22),'ssymm':(22,21),'ssyrk':(23,23),
                           'ssyr2k':(21,22),'strmm':(23,21)}[name]
    if backend == 96 and k <= 4096:
        if name == 'zgemm': normal,uniform=53,53
        if name == 'cgemm': normal,uniform=22,23
        if name == 'zsyrk': normal,uniform=53,53
        if name == 'zherk': normal,uniform=51,None
    return bits == (normal if distribution == 'normal' else uniform)


def target_provenance(case):
    """State why a goal is applied; new inputs do not reproduce an earlier run."""
    name, backend = case['routine'], case['backend']
    small = case['distribution'] == 'dyadic' and name != 'dtrsm'
    source = ('k64_measurements.json' if backend == 64 else
              ('import_small_complex.json' if case['bits']==23 else 'import_cgemm.json') if name == 'cgemm' else
              'import_small_single.json' if name[0] == 's' and case['k'] <= 4096 else
              'import_single.json' if name[0] == 's' else 'README.md')
    result = dict(policy_source='Tests/GPU/level3_cases.py:precision_target',
                  width_source='Docs/api.md', measurement_source='Docs/Provenance/'+source,
                  enforced=case['enforce_native_target'],
                  goal='all output components correctly rounded' if small else
                       'AWE p95 and max ULP <= native on the same inputs' if case['enforce_native_target'] else
                       'record only; no native accuracy goal',
                  metric='absolute error from exact unrounded reference / reference-binade ULP; linear p95',
                  historical_reproduction=False,
                  historical_scope='See the linked width/measurement table; this case uses new inputs and its installed cuBLAS.')
    if backend == 96 and name == 'ssyr2k' and case['distribution'] == 'normal' and case['bits'] == 21:
        result['historical_scope']='Width selected from 8192 normal measurements; no matching 4096 normal measurement. The 4096 goal remains enforced as a new-input test.'
    if backend == 64 and name == 'sgemm':
        result['historical_scope']='Earlier standalone SGEMM checks covered accuracy, not this 4096 native p95/max goal.'
    if backend == 64 and name == 'cgemm':
        result['historical_scope']='Earlier timing compared correctly rounded counts and an INT8 implementation, not this native p95/max goal.'
    return result


def boundary_limit(name, backend, bits):
    if name == 'i8gemm': return 16384
    if backend == 64:
        if name == 'cgemm': return 4096 if bits == 23 else 16384
        if name[0] == 'z': return 16384
        if name[0] == 'd': return 8192
        return 8192
    if name == 'sgemm' and bits == 22 or name == 'ssyr2k' and bits == 22: return 4096
    if name == 'dgemm': return {53:8254, 52:8560, 51:8720}[bits]
    if name == 'zsyrk': return {53:16384,52:8958,50:15925}[bits]
    if name == 'cgemm' and bits == 23: return 4096
    if name == 'zgemm': return 15187 if bits == 48 else 16384
    if name[0] in 'dzc': return 16384
    return 8192


def cases(backend, group):
    if group not in ('small', '4096', '8192', 'boundaries'):
        raise ValueError('unknown shape group')
    result = []
    for name in ROUTINES:
        if name == 'dtrsm' and backend != 64: continue
        if group == 'boundaries':
            # Square-only routines already reach their limits in the 8192 group.
            if name[1:] in ('symm', 'hemm', 'trmm', 'trsm'): continue
            shapes = [(16, 16, k) for k in sorted({boundary_limit(name, backend, b)
                       for b in widths(name, backend, 4096) + widths(name, backend, 8192)})]
        elif name == 'dtrsm':
            if group != 'small': continue
            shapes = [(257, 4, 257), (2049, 2, 2049)]
        elif group == 'small':
            shapes = [(16, 16, 16), (20, 20, 20)] if name == 'i8gemm' else [(16, 16, 16), (17, 17, 17)]
        else:
            n = int(group); shapes = [(n, n, n)]
        if group == 'boundaries' and backend == 96 and name == 'ssyrk': shapes.append((16,16,4096))
        for m, n, k in shapes:
            candidate_widths = widths(name, backend, k)
            if group == 'boundaries' and backend == 96 and name in ('zsyrk','zherk') and k == 16384:
                candidate_widths = sorted(set(candidate_widths + [53]))
            if group == 'boundaries' and name == 'cgemm' and backend == 64 and k == 4096:
                candidate_widths = [23]
            for bits in candidate_widths:
                if group == 'boundaries' and k != boundary_limit(name, backend, bits) and not (backend==96 and name=='ssyrk' and bits==23 and k==4096): continue
                mods = [18, 21] if backend == 64 and name == 'dgemm' else [0]
                for mod in mods:
                    for distribution in (['dyadic'] if group == 'small' else ['normal', 'uniform']):
                        result.append({'routine': name, 'm': m, 'n': n, 'k': k,
                                       'bits': bits, 'moduli': mod, 'backend': backend,
                                       'distribution': distribution, 'seed': 193,
                                       'engine_name': explicit_engine(name, backend, bits, k),
                                       'enforce_native_target': precision_target(name, backend, bits, k, distribution)})
    expanded=[]
    for c in result:
        if backend==96 and c['routine']=='ssyrk' and c['bits']==23 and c['k']<=4096:
            for suffix,distribution in [('a','normal'),('b','uniform')]:
                row=dict(c,engine_name='s23_m11_syrk_k4096_'+suffix)
                row['enforce_native_target']=c['distribution'] in ('dyadic',distribution)
                expanded.append(row)
        elif backend==96 and c['routine']=='dgemm' and c['bits']==53:
            expanded.append(c)
            expanded.append(dict(c,engine_name='d53_m18_gemm_crt'))
        else: expanded.append(c)
    for row in expanded:
        row["validation_suite"] = "new_input_seed"
        row["target_provenance"] = target_provenance(row)
    return expanded


def pair(x):
    z = complex(x)
    return Fraction(z.real), Fraction(z.imag)


def mul(a, b):
    return a[0]*b[0]-a[1]*b[1], a[0]*b[1]+a[1]*b[0]


def exact_dot(a, b):
    real = Fraction(0); imag = Fraction(0)
    for x, y in zip(a, b):
        r, i = mul(pair(x), pair(y)); real += r; imag += i
    return real, imag


def exact_cell(name, a, b, i, j):
    """Reference to the original inputs, before any AWE quantization."""
    kind = name[1:]
    if kind in ('syrk', 'herk'):
        row = a[j].conjugate() if kind == 'herk' else a[j]
        r, z = exact_dot(a[i], row)
    elif kind in ('syr2k', 'her2k'):
        bj = b[j].conjugate() if kind == 'her2k' else b[j]
        aj = a[j].conjugate() if kind == 'her2k' else a[j]
        r, z = exact_dot(a[i], bj); rr, zz = exact_dot(b[i], aj)
        r += rr; z += zz
    else:
        r, z = exact_dot(a[i], b[:, j])
    if kind in ('herk', 'her2k') and i == j: z = Fraction(0)
    return r, z


def solve_reference(a, b):
    """Exact forward substitution, including division, for small solve cases."""
    out = [[Fraction(0) for _ in range(b.shape[1])] for _ in range(a.shape[0])]
    for i in range(a.shape[0]):
        used = [(k, Fraction(float(a[i, k]))) for k in range(i) if a[i, k] != 0]
        diagonal = Fraction(float(a[i, i]))
        for j in range(b.shape[1]):
            out[i][j] = (Fraction(float(b[i, j]))-sum(v*out[k][j] for k, v in used))/diagonal
    return out


def reference_unit(exact, mantissa):
    exact = abs(Fraction(exact))
    minimum = -149 if mantissa == 24 else -1074
    if not exact: return Fraction(2) ** minimum
    exponent = exact.numerator.bit_length()-exact.denominator.bit_length()
    if exact < Fraction(2)**exponent: exponent -= 1
    return Fraction(2) ** max(exponent+1-mantissa, minimum)


def rounded_reference(exact, mantissa):
    unit = reference_unit(exact, mantissa)
    return float(round(Fraction(exact)/unit)*unit)


def ulp_error(value, exact, mantissa):
    if not math.isfinite(float(value)): return float('inf')
    return float(abs(Fraction(float(value))-exact)/reference_unit(exact, mantissa))


def percentile95(values):
    values = sorted(values); x = (len(values)-1)*.95
    lo = int(x); hi = min(lo+1, len(values)-1)
    return values[lo]+(values[hi]-values[lo])*(x-lo)
