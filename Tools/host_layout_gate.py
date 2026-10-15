"""Prove equal instructions and data when compiler names reorder GOT slots."""
from pathlib import Path

if not __debug__:
    raise RuntimeError("Run this validation without Python optimization")
import sys, re, struct, collections, json, subprocess
sys.path.insert(0, str(Path(__file__).resolve().parent))
from host_relocation_gate import Elf, instructions, RIP
NAME = re.compile(b'(_GLOBAL__N__|_INTERNAL_)[0-9a-f]{8}_')
TMP = re.compile(b'tmpxft_[0-9a-f]{8}_[0-9a-f]{8}')

def norm(x):
    return TMP.sub(b'tmpxft_00000000_00000000', NAME.sub(b'\\g<1>00000000_', x))

def compare(ref, new):
    (a, b) = (Elf(ref), Elf(new))
    out = {'passed': False, 'scope': 'Same ELF layout, compiler-name mapping, all dynamic relocations, GOT slot identities, raw data and instructions except proven RIP displacements.'}
    try:
        assert list(a.by_name) == list(b.by_name)
        for s in a.sections:
            t = b.by_name[s.name]
            assert (s.address, s.size, s.kind, s.alignment) == (t.address, t.size, t.kind, t.alignment), (s.name, 'layout')
        syms = lambda e: sorted(((norm(n.encode()).decode(), typ, sec, val, size) for (n, typ, sec, val, size) in e.symbols))
        assert syms(a) == syms(b), 'symbols'

        def names(e):
            return {m.group() for s in e.sections for m in NAME.finditer(s.data)}
        translations = {}
        for sec in ('.rodata', '.nv_fatbin'):
            if sec not in a.by_name:
                continue
            aa = a.by_name[sec].data
            bb = b.by_name[sec].data
            assert norm(aa) == norm(bb), (sec, 'non-name bytes')
            for (x, y) in zip(NAME.finditer(aa), NAME.finditer(bb)):
                assert x.span() == y.span()
                (before, after) = (x.group(), y.group())
                assert before not in translations or translations[before] == after
                translations[before] = after
        left_symbols = sorted(a.symbols, key=lambda s: (norm(s[0].encode()), *s[1:]))
        right_symbols = sorted(b.symbols, key=lambda s: (norm(s[0].encode()), *s[1:]))
        for left, right in zip(left_symbols, right_symbols):
            for x, y in zip(NAME.finditer(left[0].encode()), NAME.finditer(right[0].encode())):
                before, after = x.group(), y.group()
                assert before not in translations or translations[before] == after, 'Inconsistent compiler symbol name'
                translations[before] = after
        assert set(translations) == names(a) and set(translations.values()) == names(b), 'Unaccounted compiler name'
        assert len(translations) == len(set(translations.values())), 'name map not bijective'

        def got(e):
            s = e.by_name['.got']
            result = {}
            occupied = set()
            for (addr, (typ, sym, addend)) in e.relocations.items():
                if not s.address <= addr < s.address + s.size:
                    continue
                assert typ in (6, 8, 16), 'unexpected GOT relocation'
                offset = addr - s.address
                size = 16 if typ == 16 else 8
                assert offset + size <= s.size
                payload = s.data[offset:offset + size]
                key = (typ, sym, addend, payload)
                assert key not in result, 'ambiguous GOT slot'
                assert not occupied.intersection(range(offset, offset + size))
                occupied.update(range(offset, offset + size))
                result[key] = addr
            remainder = bytes((v for (i, v) in enumerate(s.data) if i not in occupied))
            return (result, remainder, occupied)
        (ga, ra, oa) = got(a)
        (gb, rb, ob) = got(b)
        assert ga.keys() == gb.keys() and ra == rb and oa == ob, 'GOT entries'
        remap = {ga[k]: gb[k] for k in ga}
        byte_targets = {ga[key] + offset: gb[key] + offset for key in ga for offset in range(len(key[3]))}
        aa = collections.Counter(((a.relocation_sections[addr], remap.get(addr, addr), *value) for (addr, value) in a.relocations.items()))
        bb = collections.Counter(((b.relocation_sections[addr], addr, *value) for (addr, value) in b.relocations.items()))
        assert aa == bb, 'relocations'
        if '.note.gnu.build-id' in a.by_name:
            assert a.by_name['.note.gnu.build-id'].data[:16] == b.by_name['.note.gnu.build-id'].data[:16], 'build-id note header'
        code_sections = {'.text', '.init', '.fini', '.plt', '.plt.got', '.plt.sec'} & set(a.by_name)
        except_sections = code_sections | {'.note.gnu.build-id', '.text', '.rodata', '.nv_fatbin', '.strtab', '.symtab', '.rela.dyn', '.got'}
        for s in a.sections:
            if s.name not in except_sections:
                assert s.data == b.by_name[s.name].data, (s.name, 'bytes')
        assert collections.Counter(norm(a.by_name['.strtab'].data).split(b'\x00')) == collections.Counter(norm(b.by_name['.strtab'].data).split(b'\x00')), 'string table'

        def full_symbols(e):
            data = e.by_name['.symtab'].data
            strings = e.by_name['.strtab'].data
            result = []
            for offset in range(0, len(data), 24):
                (name, info, other, index, value, size) = struct.unpack_from('<IBBHQQ', data, offset)
                result.append((norm(strings[name:].split(b'\x00', 1)[0]), info, other, index, value, size))
            return sorted(result)
        assert full_symbols(a) == full_symbols(b), 'full symbol table'
        def all_instructions(path):
            args = ['objdump', '-d', '--disassemble-zeroes', '--insn-width=16']
            for sec in sorted(code_sections):
                args += ['-j', sec]
            text = subprocess.check_output(args + [str(path)], text=True)
            rows = []
            for line in text.splitlines():
                match = re.match(r'\s*([0-9a-f]+):\s+((?:[0-9a-f]{2} )+)\s*(.*)', line)
                if match:
                    rows.append((int(match[1],16),bytes.fromhex(match[2]),match[3].split('#')[0].strip()))
            return rows
        ia, ib = all_instructions(ref), all_instructions(new)
        assert len(ia) == len(ib)
        assert sum((len(x[1]) for x in ia)) == sum(a.by_name[sec].size for sec in code_sections)
        moves = []
        for (x, y) in zip(ia, ib):
            (addr, code, asm) = x
            (addr2, code2, asm2) = y
            assert addr == addr2 and len(code) == len(code2)
            if code == code2:
                same_rip = RIP.search(asm)
                if same_rip:
                    target = addr + len(code) + int(same_rip[1], 16)
                    assert byte_targets.get(target, target) == target, 'Unchanged instruction now points at a different GOT entry'
                continue
            (ma, mb) = (RIP.search(asm), RIP.search(asm2))
            assert ma and mb, 'non-RIP change'
            assert RIP.sub('RIP', asm.split('#')[0]) == RIP.sub('RIP', asm2.split('#')[0]), 'opcode/register/immediate'
            (d1, d2) = (int(ma[1], 16), int(mb[1], 16))
            (v1, v2) = (struct.pack('<i', d1), struct.pack('<i', d2))
            slots = [i for i in range(len(code) - 3) if code[i:i + 4] == v1 and code2[i:i + 4] == v2 and (code[:i] == code2[:i]) and (code[i + 4:] == code2[i + 4:])]
            assert len(slots) == 1, 'displacement encoding'
            t1 = addr + len(code) + d1
            t2 = addr + len(code2) + d2
            assert t1 in remap and remap[t1] == t2, 'RIP target identity'
            moves.append({'instruction': hex(addr), 'before_target': hex(t1), 'after_target': hex(t2)})
        out.update(passed=True, relocations=len(a.relocations), got_slots=len(ga), rip_changes=moves, compiler_names={x.decode(): y.decode() for (x, y) in translations.items()}, unwind_and_exception_sections='raw equal', data='All sections exact except recorded names, build ID and paired GOT/relocation words')
    except (AssertionError, ValueError) as e:
        out['error'] = str(e)
    return out
if __name__ == '__main__':
    r = compare(Path(sys.argv[1]), Path(sys.argv[2]))
    print(json.dumps(r, indent=2))
    sys.exit(0 if r['passed'] else 1)
