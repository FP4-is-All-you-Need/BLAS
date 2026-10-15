#!/usr/bin/env python3
"""Conservative second-stage x86-64 host gate for explicit rodata string edits.

Never loads a library. Keeps the raw gate result, and fails closed on layouts it
cannot prove: text addresses/function extents/instruction lengths must be fixed.
Unwind and exception-table decoding is outside this gate's scope.
"""
import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re
import struct

import host_text_gate

INTERNAL = re.compile(rb'_INTERNAL_[0-9a-f]{8}_')
TEMP = re.compile(r'(?<=_GLOBAL__sub_I_)tmpxft_[0-9a-f]{8}_[0-9a-f]{8}_[0-9]+_')
RIP = re.compile(r'(-?0x[0-9a-f]+)\(%rip\)')


def require(condition, message):
    if not condition:
        raise ValueError(message)


@dataclass
class Section:
    name: str
    address: int
    size: int
    data: bytes
    alignment: int
    kind: int


class Elf:
    def __init__(self, path):
        # Reuse the raw gate's format/architecture validation.
        host_text_gate.text_bytes(path)
        data = path.read_bytes()
        offset = struct.unpack_from('<Q', data, 40)[0]
        stride, count, names_index = struct.unpack_from('<HHH', data, 58)
        headers = [struct.unpack_from('<IIQQQQIIQQ', data, offset + i * stride) for i in range(count)]
        def contents(h):
            if h[1] == 8:
                return b''
            require(h[4] + h[5] <= len(data), 'Truncated ELF section')
            return data[h[4]:h[4] + h[5]]
        names = contents(headers[names_index])
        self.sections = [Section(names[h[0]:].split(b'\0', 1)[0].decode(), h[3], h[5],
                                 contents(h), h[8], h[1]) for h in headers]
        self.by_name = {s.name: s for s in self.sections}
        self.symbols = []
        self.relocations = {}
        self.relocation_sections = {}
        for h in headers:
            if h[1] != 2:  # Full symtab is required; stripped inputs fail closed.
                continue
            strings = contents(headers[h[6]])
            require(h[9] == 24, 'Unsupported symbol format')
            for pos in range(0, h[5], 24):
                name, info, _, index, value, size = struct.unpack_from('<IBBHQQ', contents(h), pos)
                if index < count and name:
                    self.symbols.append((TEMP.sub('', strings[name:].split(b'\0', 1)[0].decode()),
                                         info & 15, self.sections[index].name, value, size))
        require(self.symbols, 'Full host symbol table required')
        # Dynamic relocation identity, including symbol and addend, is retained.
        for h in headers:
            if h[1] != 4:
                continue
            require(h[9] == 24, 'Unsupported relocation format')
            symtab = headers[h[6]]
            strings = contents(headers[symtab[6]])
            for pos in range(0, h[5], 24):
                address, info, addend = struct.unpack_from('<QQq', contents(h), pos)
                sym_index, kind = info >> 32, info & 0xffffffff
                name_index = struct.unpack_from('<I', contents(symtab), sym_index * 24)[0]
                name = strings[name_index:].split(b'\0', 1)[0].decode()
                require(address not in self.relocations, 'Duplicate relocation destination')
                self.relocations[address] = (kind, name, addend)
                self.relocation_sections[address] = names[h[0]:].split(b'\0', 1)[0].decode()

    def locate(self, address):
        matches = [s for s in self.sections if s.address and s.address <= address < s.address + s.size]
        require(len(matches) == 1, f'Ambiguous or absent target section at {address:x}')
        return matches[0]

    def identities(self, address):
        section = self.locate(address).name
        return {(name, address - value, size) for name, _, sec, value, size in self.symbols
                if sec == section and size and value <= address < value + size}


def instructions(path, objdump):
    rows = []
    for line in host_text_gate.disassembly(path, objdump).splitlines():
        match = re.match(r'\s*([0-9a-f]+):\s+((?:[0-9a-f]{2} )+)\s*(.*)', line)
        if match:
            rows.append((int(match[1], 16), bytes.fromhex(match[2]), match[3].split('#')[0].strip()))
    return rows


def rodata_map(a, b, changes, report):
    sa, sb = a.by_name['.rodata'], b.by_name['.rodata']
    da, db = sa.data, sb.data
    fa, fb = a.by_name.get('.nv_fatbin'), b.by_name.get('.nv_fatbin')
    require((fa is None) == (fb is None), 'Fatbin section added or removed')
    if fa is not None:
        require(INTERNAL.sub(b'_INTERNAL_00000000_', fa.data) ==
                INTERNAL.sub(b'_INTERNAL_00000000_', fb.data),
                'Changed fatbin bytes beyond compiler symbol names')
        report['normalized_fatbin_equal'] = True
    # NVCC registration strings name compiler-local device symbols. Accept their
    # hash component only with a complete byte proof of the containing fatbin.
    ha, hb = INTERNAL.findall(da), INTERNAL.findall(db)
    if fa is not None and ha == hb:
        require(fa.data == fb.data, 'Fatbin changed without a corresponding registration-name change')
    if ha != hb:
        require(fa is not None and fb is not None and
                INTERNAL.sub(b'_INTERNAL_00000000_', fa.data) == INTERNAL.sub(b'_INTERNAL_00000000_', fb.data),
                'Compiler device-symbol names changed without identical normalized fatbin')
        require(len(set(ha)) == len(set(hb)) == 1 and len(ha) == len(hb), 'Ambiguous compiler symbol mapping')
        registrations = []
        for data, fatbin in ((da, fa.data), (db, fb.data)):
            names = [s for s in data.split(b'\0') if INTERNAL.search(s)]
            require(all(s.startswith(b'_ZN') and s + b'\0' in fatbin for s in names),
                    'Registration string does not name a symbol in its own fatbin')
            registrations.append(names)
        require([INTERNAL.sub(b'_INTERNAL_00000000_', s) for s in registrations[0]] ==
                [INTERNAL.sub(b'_INTERNAL_00000000_', s) for s in registrations[1]],
                'Compiler registration symbol identity changed')
        report['compiler_symbol_identity'] = {'before': ha[0].decode(), 'after': hb[0].decode(),
            'registration_names': len(ha), 'entire_normalized_fatbin_equal': True,
            'names_before': [s.decode() for s in registrations[0]],
            'names_after': [s.decode() for s in registrations[1]]}
        da = INTERNAL.sub(b'_INTERNAL_00000000_', da)
        db = INTERNAL.sub(b'_INTERNAL_00000000_', db)
    spans = [[], []]
    report['strings'] = []
    for index, pair in enumerate(changes):
        old, new = pair['before'].encode(), pair['after'].encode()
        require(old and new and b'\0' not in old + new and old != new, 'Invalid string allowance')
        pa, pb = da.find(old), db.find(new)
        if pa == pb == -1:
            continue
        require(pa >= 0 and pb >= 0 and da.count(old) == db.count(new) == 1,
                'String allowance missing or not unique')
        starts = [da.rfind(b'\0', 0, pa) + 1, db.rfind(b'\0', 0, pb) + 1]
        ends = [da.find(b'\0', pa), db.find(b'\0', pb)]
        require(min(ends) >= 0, 'Unterminated allowed string')
        ca, cb = da[starts[0]:ends[0]], db[starts[1]:ends[1]]
        require(ca.replace(old, new) == cb, 'Other changes inside allowed string')
        report['strings'].append({'before': ca.decode(), 'after': cb.decode(),
                                  'reference_offset': starts[0], 'candidate_offset': starts[1]})
        for side in (0, 1):
            spans[side].append((starts[side], ends[side], index))

    def tokenize(data, exceptions):
        result, cursor = [], 0
        def plain(start, end):
            for match in re.finditer(rb'\0+|[^\0]+', data[start:end]):
                value = match[0]
                result.append(('zero' if value[0] == 0 else 'bytes', value,
                               start + match.start(), start + match.end()))
        for start, end, index in sorted(exceptions):
            require(start >= cursor, 'Overlapping string allowances')
            plain(cursor, start)
            result.append(('allowed', index, start, end))
            cursor = end
        plain(cursor, len(data))
        return result

    ta, tb = tokenize(da, spans[0]), tokenize(db, spans[1])
    require(len(ta) == len(tb), 'Unapproved rodata insertion/deletion')
    mapping, approved, padding = [], [], []
    for i, (x, y) in enumerate(zip(ta, tb)):
        kind, value, start, end = x
        other, other_value, dest, stop = y
        require(kind == other, 'Changed rodata object kind')
        if kind == 'allowed':
            require(value == other_value, 'Allowed strings reordered')
            approved.append((start, dest))  # Only the complete string's start may be referenced.
        elif kind == 'zero' and value != other_value:
            alignment = min(max(sa.alignment, sb.alignment), 32)
            prev = ta[i - 1] if i else None
            printable = prev and (prev[0] == 'allowed' or
                (prev[0] == 'bytes' and len(prev[1]) >= 4 and all(32 <= c < 127 or c in (9, 10, 13) for c in prev[1])))
            require(printable and max(len(value), len(other_value)) <= alignment and
                    any((sa.address + end) % n == (sb.address + stop) % n == 0 for n in (8, 16, 32)),
                    'Unproved rodata padding change')
            for elf, section, lo, hi in ((a, sa, start + 1, end), (b, sb, dest + 1, stop)):
                require(not any(kind == 1 and size and sec == '.rodata' and
                                max(address, section.address + lo) < min(address + size, section.address + hi)
                                for _, kind, sec, address, size in elf.symbols),
                        'Alignment padding overlaps a sized STT_OBJECT after the first NUL')
            # Keep the terminating NUL; padding itself must never be a target.
            mapping.append((start, start + 1, dest))
            padding.append({'reference_offset': start, 'candidate_offset': dest,
                            'before_zero_bytes': len(value), 'after_zero_bytes': len(other_value)})
        else:
            require(value == other_value, f'Unapproved rodata content change at {start:x}')
            mapping.append((start, end, dest))
    report['alignment_padding'] = padding
    def corresponds(old, new):
        x, y = old - sa.address, new - sb.address
        return (x, y) in approved or any(lo <= x < hi and y == target + x - lo for lo, hi, target in mapping)
    return corresponds


def data_relocations(a, b, maps, report):
    """Pair every RELA entry by destination section/offset, including PLT entries."""
    def indexed(elf):
        rows = {}
        for address, (kind, symbol, addend) in elf.relocations.items():
            # These x86-64 relocations all occupy one 64-bit word. Unknown kinds
            # must not get a guessed mask width or implicit addend normalization.
            require(kind in (1, 6, 7, 8), f'Unsupported relocation type {kind}')
            section = elf.locate(address)
            require(address + 8 <= section.address + section.size, 'Relocation crosses a section')
            key = (section.name, address - section.address)
            rows[key] = (elf.relocation_sections[address], kind, symbol, addend)
        return rows
    left, right = indexed(a), indexed(b)
    require(left.keys() == right.keys(), 'Relocation destinations added, removed or moved within a section')
    pairs = []
    for key, x in sorted(left.items()):
        y = right[key]
        require(x[:3] == y[:3], f'Relocation type or symbol changed at {key}')
        if x[1] == 8:  # R_X86_64_RELATIVE: addend is a load-base-relative address.
            require(not x[2], 'RELATIVE relocation has a symbol')
            sa, sb = a.locate(x[3]), b.locate(y[3])
            require(sa.name == sb.name, f'Relative target section changed at {key}')
            if sa.name == '.rodata':
                require(maps(x[3], y[3]), f'Wrong relocated rodata pointer at {key}')
            elif sa.name == '.text':
                require(x[3] == y[3], f'Relocated code pointer changed at {key}')
            else:
                require(x[3] - sa.address == y[3] - sb.address and sa.size == sb.size,
                        f'Relative target offset or section size changed at {key}')
                require(sa.name == '.nv_fatbin' or a.identities(x[3]) & b.identities(y[3]) or
                        sa.data == sb.data, f'Unproved relative target at {key}')
            target = {'section': sa.name, 'reference': hex(x[3]), 'candidate': hex(y[3])}
        else:
            require(x[3] == y[3], f'Symbol relocation addend changed at {key}')
            target = {'symbol': x[2], 'addend': x[3]}
        pairs.append({'relocation_section': x[0], 'section': key[0], 'offset': key[1],
                      'type': x[1], 'target': target})
    checked = {}
    for name in ('.data.rel.ro', '.init_array', '.fini_array', '.data', '.nvFatBinSegment'):
        sa, sb = a.by_name.get(name), b.by_name.get(name)
        require((sa is None) == (sb is None), f'{name} added or removed')
        if sa is None:
            continue
        require(sa.size == sb.size, f'{name} size changed')
        da, db = bytearray(sa.data), bytearray(sb.data)
        offsets = sorted(offset for section, offset in left if section == name)
        require(all(x + 8 <= y for x, y in zip(offsets, offsets[1:])), 'Overlapping relocation words')
        for offset in offsets:
            da[offset:offset + 8] = db[offset:offset + 8] = b'\0' * 8
        require(da == db, f'Non-relocated bytes changed in {name}')
        checked[name] = {'bytes': sa.size, 'relocation_words': len(offsets), 'remaining_bytes_equal': True}
    report['relocation_pairs'] = pairs
    report['relocations_checked'] = len(pairs)
    report['data_sections'] = checked


def compare(reference, candidate, changes=(), objdump='objdump'):
    raw = host_text_gate.compare(reference, candidate, objdump)
    result = {'raw': raw, 'passed': False, 'stage': 'relocation', 'gpu_execution': False,
              'scope': 'Fixed-layout host .text, RIP references, all RELA entries and selected host data sections; device gate required separately.',
              'outside_scope': ['Unwind and exception-table decoding (.eh_frame, .eh_frame_hdr, .gcc_except_table).']}
    try:
        a, b = Elf(reference), Elf(candidate)
        sa, sb = a.by_name['.text'], b.by_name['.text']
        require((sa.address, sa.size) == (sb.address, sb.size), 'Host text layout changed')
        fa = sorted(s for s in a.symbols if s[1] == 2 and s[2] == '.text')
        fb = sorted(s for s in b.symbols if s[1] == 2 and s[2] == '.text')
        require(fa == fb, 'Function identities or extents changed')
        maps = rodata_map(a, b, changes, result)
        data_relocations(a, b, maps, result)
        ia, ib = instructions(reference, objdump), instructions(candidate, objdump)
        require(len(ia) == len(ib), 'Instruction count changed')
        require(sum(len(x[1]) for x in ia) == sa.size, 'Incomplete text disassembly')
        checked, moved = 0, []
        for x, y in zip(ia, ib):
            addr, code, asm = x
            address, other_code, other_asm = y
            require(addr == address and len(code) == len(other_code), 'Instruction boundary changed')
            ma, mb = RIP.search(asm), RIP.search(other_asm)
            if not ma and not mb:
                # Direct control flow, immediate values, and registers stay exact.
                require(code == other_code, f'Non-RIP instruction changed at {addr:x}')
                continue
            require(ma and mb and RIP.sub('DISP(%rip)', asm) == RIP.sub('DISP(%rip)', other_asm),
                    f'Opcode/register/non-address operand changed at {addr:x}')
            disp, other_disp = int(ma[1], 16), int(mb[1], 16)
            da, db = struct.pack('<i', disp), struct.pack('<i', other_disp)
            positions = [i for i in range(len(code) - 3) if code[i:i+4] == da and other_code[i:i+4] == db
                         and code[:i] == other_code[:i] and code[i+4:] == other_code[i+4:]]
            require(len(positions) == 1, f'Unproved displacement encoding at {addr:x}')
            target, other_target = addr + len(code) + disp, address + len(other_code) + other_disp
            left, right = a.locate(target), b.locate(other_target)
            require(left.name == right.name, f'Target section changed at {addr:x}')
            if left.name == '.rodata':
                require(maps(target, other_target), f'Wrong rodata reference at {addr:x}')
            else:
                # Same symbol plus offset, or identical whole section plus offset.
                identity = a.identities(target) & b.identities(other_target)
                require(identity or (target - left.address == other_target - right.address and
                                     left.size == right.size and left.data == right.data),
                        f'Unproved target identity at {addr:x}')
                # All dynamic relocation words were already paired above, with
                # address addends resolved through the same rodata mapping.
            checked += 1
            if code != other_code:
                moved.append({'instruction': hex(addr), 'section': left.name,
                              'reference_target': hex(target), 'candidate_target': hex(other_target)})
        result.update(passed=True, rip_references_checked=checked, relocated_instructions=moved,
                      relocated_instruction_count=len(moved), control_flow='Identical bytes and fixed function layout')
        result['stage'] = 'raw' if raw['passed'] else 'relocation'
    except (ValueError, KeyError, UnicodeError, struct.error) as error:
        result['error'] = str(error)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference', type=Path)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('--strings', type=Path, help='JSON list of exact before/after description edits')
    parser.add_argument('--objdump', default='objdump')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    result = compare(args.reference, args.candidate,
                     json.loads(args.strings.read_text()) if args.strings else (), args.objdump)
    output = json.dumps(result, indent=2) + '\n'
    if args.report:
        args.report.write_text(output)
    print(output, end='')
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
