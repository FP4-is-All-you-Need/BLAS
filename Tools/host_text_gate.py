#!/usr/bin/env python3
"""Compare host .text bytes and disassembly without loading either shared library."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import struct
import subprocess

TEMPORARY_INITIALIZER = re.compile(r'(?<=<_GLOBAL__sub_I_)tmpxft_[0-9a-f]{8}_[0-9a-f]{8}_[0-9]+_')


def text_bytes(path):
    data = path.read_bytes()
    if len(data) < 64 or data[:6] != b'\x7fELF\x02\x01':
        raise ValueError('Expected a little-endian ELF64 shared library')
    if struct.unpack_from('<H', data, 18)[0] != 62:
        raise ValueError('Expected x86-64 host code')
    offset = struct.unpack_from('<Q', data, 40)[0]
    size, count, names_index = struct.unpack_from('<HHH', data, 58)
    if size < 64 or not count or names_index >= count or offset + count * size > len(data):
        raise ValueError('Unsupported or truncated section table')
    sections = [struct.unpack_from('<IIQQQQIIQQ', data, offset + i * size) for i in range(count)]
    names_section = sections[names_index]
    names = data[names_section[4]:names_section[4] + names_section[5]]
    for section in sections:
        name = names[section[0]:].split(b'\0', 1)[0]
        if name == b'.text':
            start, length = section[4:6]
            if not length or start + length > len(data):
                raise ValueError('Empty or truncated .text section')
            return data[start:start + length]
    raise ValueError('Missing .text section')


def disassembly(path, objdump):
    listing = subprocess.check_output([objdump, '-d', '-w', '-j', '.text', str(path)], text=True)
    marker = 'Disassembly of section .text:\n'
    if marker not in listing:
        raise ValueError('Unrecognized host disassembly')
    body = listing.split(marker, 1)[1].strip()
    if not re.search(r'^\s*[0-9a-f]+:\s+[0-9a-f]{2}', body, re.M):
        raise ValueError('No host instructions')
    # Only the input-filename banner is removed here.
    return body


def compare(reference, candidate, objdump='objdump'):
    a, b = text_bytes(reference), text_bytes(candidate)
    da, db = disassembly(reference, objdump), disassembly(candidate, objdump)
    # Independent builds of the same CUDA snapshot differ in this initializer
    # label's temporary filename only. Raw instruction bytes still match exactly.
    na, nb = TEMPORARY_INITIALIZER.sub('', da), TEMPORARY_INITIALIZER.sub('', db)
    digest = lambda value: hashlib.sha256(value).hexdigest()
    return {'reference_text_bytes': len(a), 'candidate_text_bytes': len(b),
            'reference_text_sha256': digest(a), 'candidate_text_sha256': digest(b),
            'raw_text_equal': a == b, 'raw_disassembly_equal': da == db,
            'disassembly_equal': na == nb,
            'reference_disassembly_sha256': digest(da.encode()),
            'candidate_disassembly_sha256': digest(db.encode()),
            'normalized_reference_disassembly_sha256': digest(na.encode()),
            'normalized_candidate_disassembly_sha256': digest(nb.encode()),
            'temporary_name_normalization': TEMPORARY_INITIALIZER.pattern,
            'reference_normalized_labels': len(TEMPORARY_INITIALIZER.findall(da)),
            'candidate_normalized_labels': len(TEMPORARY_INITIALIZER.findall(db)),
            'scope': 'Host .text only; no host data, dynamic linking, or runtime behavior claim.',
            'passed': a == b and na == nb}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference', type=Path)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('--objdump', default='objdump')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    result = compare(args.reference, args.candidate, args.objdump)
    output = json.dumps(result, indent=2) + '\n'
    if args.report:
        args.report.write_text(output)
    print(output, end='')
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
