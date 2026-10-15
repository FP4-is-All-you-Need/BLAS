#!/usr/bin/env python3
"""Compare device instructions without loading a CUDA module or using a GPU."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess


def parse_sass(text):
    functions = []
    arch = None
    current = None
    for line in text.splitlines():
        match = re.search(r'arch\s*=\s*(sm_\w+)', line)
        if match:
            arch = match.group(1)
        match = re.match(r'\s*Function\s*:\s*(\S+)', line)
        if match:
            if arch is None:
                raise ValueError('Function without architecture')
            current = {'arch': arch, 'name': match.group(1), 'instructions': []}
            functions.append(current)
            continue
        match = re.match(r'\s*/\*([0-9a-fA-F]+)\*/\s*(.*?)\s*/\*\s*(0x[0-9a-fA-F]+)\s*\*/', line)
        if match:
            if current is None:
                raise ValueError('Instruction outside a function')
            instruction = ' '.join(match.group(2).split())
            if not instruction.endswith(';'):
                raise ValueError('Unrecognized instruction: ' + line)
            current['instructions'].append([instruction, match.group(3).lower()])
            continue
        match = re.fullmatch(r'\s*/\*\s*(0x[0-9a-fA-F]+)\s*\*/\s*', line)
        if match:
            if current is None or not current['instructions']:
                raise ValueError('Encoding without instruction')
            current['instructions'][-1].append(match.group(1).lower())
        elif re.match(r'\s*/\*[0-9a-fA-F]+\*/', line):
            raise ValueError('Unrecognized instruction line: ' + line)
    if not functions or any(not f['instructions'] for f in functions):
        raise ValueError('Empty disassembly or function')
    return functions


def fingerprint(functions, ignore_names=False):
    # Preserve duplicate functions and architecture; keep operands and control bits.
    return Counter((f['arch'], '' if ignore_names else f['name'],
                    tuple(tuple(i) for i in f['instructions'])) for f in functions)


def digest(functions, ignore_names=False):
    counts = fingerprint(functions, ignore_names)
    return hashlib.sha256(repr(sorted(counts.items())).encode()).hexdigest()


def disassemble(path, cuobjdump):
    result = subprocess.run([cuobjdump, '-sass', str(path)], check=True,
                            capture_output=True, text=True)
    return parse_sass(result.stdout)


def parse_inventory(text):
    entries = []
    for line in text.splitlines():
        match = re.match(r'\s*(ELF|PTX) file\s+\d+:\s*(.+)', line)
        if match:
            arch = re.search(r'\.(sm_[^.]+|compute_[^.]+)\.(?:cubin|ptx)$', match[2])
            if not arch:
                raise ValueError('Unknown embedded image name: ' + line)
            entries.append((match[1], arch[1]))
    if not entries:
        raise ValueError('No embedded ELF images')
    if any(kind == 'PTX' for kind, _ in entries):
        raise ValueError('PTX is not permitted in a frozen cubin-only build')
    return sorted(entries)


def parse_resources(text):
    records = []
    arch = scope = None
    for line in text.splitlines():
        match = re.search(r'arch\s*=\s*(sm_\w+)', line)
        if match:
            arch, scope = match[1], 'common'
        match = re.match(r'\s*Function (\S+):', line)
        if match:
            scope = match[1]
        if re.search(r'\b(?:REG|GLOBAL|STACK|SHARED|LOCAL|CONSTANT\[\d+\]):', line):
            if arch is None:
                raise ValueError('Resources without architecture')
            records.append((arch, scope, ' '.join(line.split())))
    if not any('REG:' in r[2] for r in records):
        raise ValueError('No kernel resource records')
    return sorted(records)


def parse_elf_metadata(text):
    records, data = [], []
    # Offsets in section headers and symbol-table indices are not addresses in the
    # runtime contract. Resolve references to symbols, retaining parameter offsets.
    for block in text.split('Fatbin elf code:')[1:]:
        arch_match = re.search(r'arch\s*=\s*(sm_\w+)', block)
        if not arch_match:
            raise ValueError('ELF without architecture')
        arch = arch_match[1]
        symbols = {}
        for line in block.splitlines():
            match = re.match(r'\s*(0x[0-9a-f]+)\s+\S+\s+\S+\s+\S+\s+\S+\s+\S+\s+(\S+)\s*$', line)
            if match:
                symbols[int(match[1], 16)] = match[2]
        sections = re.split(r'(?m)^(\.nv\.(?:info(?:\.[^\n]+)?|constant\d+(?:\.[^\n]+)?|global\.init))\s*\n', block)
        for i in range(1, len(sections), 2):
            name, body = sections[i], sections[i + 1]
            body = re.split(r'(?m)^\.(?:section|nv\.|text)', body)[0]
            if name.startswith('.nv.info'):
                for attr in re.split(r'(?:^|\n)\s*<0x[0-9a-f]+>\s*\n', body):
                    if 'Attribute:' not in attr:
                        continue
                    kind = re.search(r'Attribute:\s*(EIATTR_\w+)', attr)
                    if not kind:
                        raise ValueError('Unrecognized ELF attribute')
                    attr = re.sub(r'(function:\s*\S+?)\(0x[0-9a-f]+\)', r'\1', attr)
                    if kind[1] == 'EIATTR_PARAM_CBANK':
                        def resolve(match):
                            index = int(match[1], 16)
                            if index not in symbols:
                                raise ValueError('Unresolved constant-bank symbol')
                            return 'Value: ' + symbols[index]
                        attr = re.sub(r'Value:\s*(0x[0-9a-f]+)', resolve, attr, count=1)
                    records.append((arch, name, ' '.join(attr.split())))
            else:
                # The dump prints words without address prefixes. Keep all bytes,
                # including initialized CUDA runtime constants and zero padding.
                lines = [line.strip() for line in body.splitlines() if line.strip()]
                if lines and all(re.fullmatch(r'(?:0x[0-9a-fA-F]+\s*)+', line) for line in lines):
                    data.append((arch, name, tuple(re.findall(r'0x[0-9a-fA-F]+', ' '.join(lines)))))
                elif lines and lines != ['No data to be dumped.']:
                    raise ValueError('Unrecognized device data section ' + name)
    if not records:
        raise ValueError('No ELF attributes')
    return {'attributes': sorted(records), 'initialized_data': sorted(data)}


def device_metadata(path, cuobjdump):
    def run(*options):
        result = subprocess.run([cuobjdump, *options, str(path)], check=True,
                                capture_output=True, text=True)
        return result.stdout
    elf = parse_elf_metadata(run('-elf'))
    return {'inventory': parse_inventory(run('-lelf', '-lptx')),
            'resources': parse_resources(run('-res-usage')), **elf}


def metadata_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference', type=Path)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('--cuobjdump', default='cuobjdump')
    parser.add_argument('--ignore-kernel-names', action='store_true')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    ref = disassemble(args.reference, args.cuobjdump)
    new = disassemble(args.candidate, args.cuobjdump)
    sass_equal = fingerprint(ref, args.ignore_kernel_names) == fingerprint(new, args.ignore_kernel_names)
    ref_meta = device_metadata(args.reference, args.cuobjdump)
    new_meta = device_metadata(args.candidate, args.cuobjdump)
    if args.ignore_kernel_names:
        def rename_metadata(meta, functions):
            encoded = json.dumps(meta)
            for f in sorted(functions, key=lambda item: len(item['name']), reverse=True):
                name = 'body_' + hashlib.sha256(repr(f['instructions']).encode()).hexdigest()
                encoded = encoded.replace(f['name'], name)
            return {k: sorted(v, key=repr) for k, v in json.loads(encoded).items()}
        ref_meta = rename_metadata(ref_meta, ref)
        new_meta = rename_metadata(new_meta, new)
    metadata_checks = {key: ref_meta[key] == new_meta[key] for key in ref_meta}
    equal = sass_equal and all(metadata_checks.values())
    report = {'passed': equal, 'sass_equal': sass_equal, 'device_metadata_checks': metadata_checks,
              'reference_device_metadata_sha256': {k: metadata_digest(v) for k, v in ref_meta.items()},
              'candidate_device_metadata_sha256': {k: metadata_digest(v) for k, v in new_meta.items()}, 'ignore_kernel_names': args.ignore_kernel_names,
              'reference_sha256': hashlib.sha256(args.reference.read_bytes()).hexdigest(),
              'candidate_sha256': hashlib.sha256(args.candidate.read_bytes()).hexdigest(),
              'reference_sass_sha256': digest(ref, args.ignore_kernel_names),
              'candidate_sass_sha256': digest(new, args.ignore_kernel_names),
              'reference_functions': len(ref), 'candidate_functions': len(new),
              'reference_instructions': sum(len(f['instructions']) for f in ref),
              'candidate_instructions': sum(len(f['instructions']) for f in new)}
    output = json.dumps(report, indent=2) + '\n'
    if args.report:
        args.report.write_text(output)
    print(output, end='')
    return 0 if equal else 1


if __name__ == '__main__':
    raise SystemExit(main())
