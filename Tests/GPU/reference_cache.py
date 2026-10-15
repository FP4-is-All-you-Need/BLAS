"""Exact CPU references saved before a GPU allocation is rented."""
from fractions import Fraction
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
from level3_cases import solve_reference


def reference_key(case, samples):
    fields = {k: case[k] for k in ('routine', 'm', 'n', 'k', 'distribution', 'seed')}
    fields.update(samples=samples, algorithm='integer-dot-v1', numpy=np.__version__)
    return hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()


def input_sha(a, b):
    h = hashlib.sha256()
    h.update(a.tobytes(order='F')); h.update(b.tobytes(order='F'))
    return h.hexdigest()


def points_for(case, samples):
    m, n = case['m'], case['n']
    rank = case['routine'][1:] in ('syrk', 'herk', 'syr2k', 'her2k')
    if case['distribution'] == 'dyadic':
        return [(i, j) for j in range(n) for i in range(m) if not rank or i >= j]
    rng = np.random.default_rng(case['seed']+1); points = []
    while len(points) < min(samples, m*n):
        i, j = int(rng.integers(m)), int(rng.integers(n))
        if not rank or i >= j: points.append((i, j))
    return points


def integer_vector(values):
    pairs = [float(x).as_integer_ratio() for x in values]
    denominator = max(d for _, d in pairs)
    return [n*(denominator//d) for n, d in pairs], denominator


def exact_references(case, a, b, samples):
    points = points_for(case, samples); kind = case['routine'][1:]
    if kind == 'trsm':
        out = solve_reference(a, b)
        return [[i, j, [[out[i][j].numerator, out[i][j].denominator], [0, 1]]] for i, j in points]
    vectors = {}
    def vector(which, index, column, imag):
        key = which, index, column, imag
        if key not in vectors:
            matrix = a if which == 'a' else b
            v = matrix[:, index] if column else matrix[index]
            vectors[key] = integer_vector(v.imag if imag else v.real)
        return vectors[key]
    def dot(x, y):
        xv, xd = vector(*x); yv, yd = vector(*y)
        return Fraction(sum(v*w for v, w in zip(xv, yv)), xd*yd)
    def product(x, y, conjugate=False):
        real = dot((*x, False), (*y, False)); imag = Fraction(0)
        if case['routine'][0] in 'zc':
            ii = dot((*x, True), (*y, True))
            real += ii if conjugate else -ii
            imag = dot((*x, True), (*y, False))
            ri = dot((*x, False), (*y, True))
            imag += -ri if conjugate else ri
        return real, imag
    result = []
    for i, j in points:
        if kind in ('syrk', 'herk'):
            r, z = product(('a', i, False), ('a', j, False), kind == 'herk')
        elif kind in ('syr2k', 'her2k'):
            r, z = product(('a', i, False), ('b', j, False), kind == 'her2k')
            rr, zz = product(('b', i, False), ('a', j, False), kind == 'her2k')
            r += rr; z += zz
        else:
            r, z = product(('a', i, False), ('b', j, True))
        if kind in ('herk', 'her2k') and i == j: z = Fraction(0)
        result.append([i, j, [[r.numerator, r.denominator], [z.numerator, z.denominator]]])
    return result


def write_cache(path, case, a, b, samples):
    refs = exact_references(case, a, b, samples)
    data = dict(key=reference_key(case, samples), input_sha256=input_sha(a, b), references=refs,
                reference_sha256=hashlib.sha256(json.dumps(refs).encode()).hexdigest())
    Path(path).write_bytes(gzip.compress(json.dumps(data).encode(), mtime=0))
    return data


def read_cache(path, case, a, b, samples):
    data = json.loads(gzip.decompress(Path(path).read_bytes()))
    if data['key'] != reference_key(case, samples): raise ValueError('reference case/version differs')
    if data['input_sha256'] != input_sha(a, b): raise ValueError('reference input SHA differs')
    if data['reference_sha256'] != hashlib.sha256(json.dumps(data['references']).encode()).hexdigest():
        raise ValueError('reference SHA differs')
    expected = points_for(case, samples)
    if [(r[0], r[1]) for r in data['references']] != expected: raise ValueError('reference points differ')
    return data
