#!/usr/bin/env python3
"""Report strict real CRT coverage using Python integers, without loading CUDA."""
import argparse
import json
import math
from pathlib import Path
import re


def real_limit(moduli, bits):
    if not moduli or any(p < 2 for p in moduli) or bits < 1:
        raise ValueError('Positive bit width and moduli greater than one are required')
    product = math.prod(moduli)
    return product, (product - 1) // (2 * ((1 << bits) - 1) ** 2)


def report(root):
    rows = []
    for engine in json.loads((root / 'Docs/Provenance/engines.json').read_text()):
        if '_z' in engine['id']:
            continue
        text = (root / engine['path'] / 'l8_tables.h').read_text()
        match = re.search(r'l8_mod\[L8_NMOD\]\s*=\s*\{([^}]+)', text)
        if not match:
            raise ValueError('Missing modulus array: ' + engine['id'])
        moduli = [int(v) for v in re.findall(r'\d+', match[1])]
        product, maximum = real_limit(moduli, 53)
        rows.append({'id': engine['id'], 'bits': 53, 'moduli': moduli,
                     'product': str(product), 'strict_coverage_max_k': maximum,
                     'table_max_k': 8192 if engine['id'].endswith('_d17') else 16384,
                     'at_limit_passes': product > 2 * maximum * ((1 << 53) - 1) ** 2,
                     'next_k_fails': product <= 2 * (maximum + 1) * ((1 << 53) - 1) ** 2})
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    print(json.dumps(report(args.root), indent=2))
