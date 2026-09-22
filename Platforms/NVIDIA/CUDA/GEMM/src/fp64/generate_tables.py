#!/usr/bin/env python3
import argparse
import hashlib
import json
import math
from pathlib import Path
parser = argparse.ArgumentParser()
parser.add_argument('source', type=Path)
parser.add_argument('output', type=Path)
args = parser.parse_args()
data = json.loads(args.source.read_text())
rows = {r['modulus']: r for r in data['rows'] if r['coefficient_denominator'] == 1}
plans = {}
for p in data['plans']:
    mods = tuple(p['moduli'])
    if p['method'] != 3 or not all((m in rows for m in mods)):
        continue
    product = int(p['modulus_product'])
    assert product == math.prod(mods)
    assert all((math.gcd(m, n) == 1 for (i, m) in enumerate(mods) for n in mods[i + 1:]))
    assert p['real_faces'] == sum((rows[m]['products'] for m in mods))
    if product >= 2 ** 127:
        continue
    plans[mods] = (p['real_faces'], product)
assert max((p[1] for p in plans.values())) > 2 * (2 ** 31 - 1) * 128 ** 2
used = sorted({m for mods in plans for m in mods})
fp4 = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 6: 5, 8: 6, 12: 7}

def pack(v):
    return fp4[abs(v)] | (8 if v < 0 else 0)
out = ['// Generated from certified residue tables; do not edit by hand.', '// Source SHA256: ' + hashlib.sha256(args.source.read_bytes()).hexdigest(), 'static const Witness witnesses[] = {']
for m in used:
    r = rows[m]
    assert r.get('verification', {'passed': True})['passed']
    lut = r['face_lut']
    for a in range(m):
        for b in range(m):
            value = sum((c * lut[a][i] * lut[b][j] for (c, (i, j)) in zip(r['coefficients_numerator'], r['face_pairs'])))
            assert (value - a * b) % m == 0
    faces = []
    for (c, (i, j)) in zip(r['coefficients_numerator'], r['face_pairs']):
        aa = ','.join((str(pack(lut[v][i])) for v in range(m)))
        bb = ','.join((str(pack(lut[v][j])) for v in range(m)))
        faces.append('{%d,{%s},{%s}}' % (c, aa, bb))
    out.append('{%d,%d,%d,{%s}},' % (m, r['products'], r['K_safe'], ','.join(faces)))
out += ['};', 'static const Choice choices[] = {']
for (mods, (cost, product)) in sorted(plans.items(), key=lambda p: (p[1][0], p[1][1])):
    out.append('{%d,%d,((U128(%dULL)<<64)|%dULL),{%s}},' % (cost, len(mods), product >> 64, product % 2 ** 64, ','.join((str(used.index(m)) for m in mods))))
out += ['};', '']
args.output.write_text('\n'.join(out))
print(json.dumps(dict(witnesses=len(used), choices=len(plans), max_product=str(max((p[1] for p in plans.values()))))))
