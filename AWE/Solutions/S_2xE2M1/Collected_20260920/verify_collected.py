#!/usr/bin/env python3
"""Independently check integer descriptors; no search or hardware execution."""
import json
from pathlib import Path


def check(e):
    S = set(e['storage']['values'])
    assert len(S) == 15
    reps = e['storage']['representatives']
    assert len(reps) == 13 and {x % 13 for x in reps} == set(range(13))
    c, forms, weights = e['coefficients'], e['faces'], e['limb_weights']
    assert len(c) == e['r'] == len(forms['A']) == len(forms['B'])
    for i, a in enumerate(weights['A']):
        for j, b in enumerate(weights['B']):
            assert sum(cp * u[i] * v[j] for cp, u, v in zip(c, forms['A'], forms['B'])) == a*b
    expanded = {}
    for side in ('A', 'B'):
        lut = e['encoding_lut'][side]
        lo, hi = e['input_domain'][side]
        rejected = set(e['correction']['rejected_values'][side])
        retained = {int(v) for v in lut}
        assert not retained & rejected
        assert retained | rejected == set(range(lo, hi+1))
        gains = e['face_gains'][side]
        assert len(gains) == len(c)
        expanded[side] = []
        for value, limbs in lut.items():
            value = int(value)
            assert len(limbs) == len(weights[side]) and set(limbs) <= S
            assert sum(w*t for w,t in zip(weights[side],limbs)) == value
            face = [sum(u*t for u,t in zip(form,limbs)) for form in forms[side]]
            assert all(f*g in S for f,g in zip(face,gains))
            expanded[side].append((value, face))
    tight = 0
    for a, u in expanded['A']:
        for b, v in expanded['B']:
            terms = [cp*x*y for cp,x,y in zip(c,u,v)]
            assert sum(terms) == a*b
            tight = max(tight, sum(abs(x) for x in terms))
    n = len(expanded['A'])*len(expanded['B'])
    na, nb = (e['input_domain'][s][1]-e['input_domain'][s][0]+1 for s in ('A','B'))
    return {'id':e['id'], 'retained_pairs_checked':n, 'full_domain_pairs':na*nb,
            'pairs_requiring_exact_correction':na*nb-n, 'maximum_absolute_term_sum':tight,
            'mismatches':0, 'integer_identity':True, 'storage_and_encoding':True,
            'correction_partition':True, 'scale_format_verified':False,
            'method':'integer matrix identity, every retained input pair, storage membership and correction partition'}


if __name__ == '__main__':
    here = Path(__file__).resolve().parent
    for name in ('catalog_representatives.jsonl','catalog_supplementary.jsonl','recent_six_product.jsonl'):
        for line in (here/name).read_text().splitlines():
            e = json.loads(line)
            result = check(e)
            assert result['retained_pairs_checked'] == e['verified']['n_pairs']
            print(json.dumps(result,sort_keys=True))
