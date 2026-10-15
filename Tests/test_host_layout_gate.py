"""Reject changed instructions, referenced data and relocation words."""
from pathlib import Path
import importlib.util
import struct
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('host_layout_gate', ROOT/'Tools/host_layout_gate.py')
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)
section_spec = importlib.util.spec_from_file_location('elf_sections_gate', ROOT/'Tools/elf_sections_gate.py')
sections = importlib.util.module_from_spec(section_spec)
section_spec.loader.exec_module(sections)


class HostLayoutTest(unittest.TestCase):
    def test_same_layout_and_reject_mutations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/'fixture.cpp'
            source.write_text('''
extern "C" int puts(const char*);
const char *const messages[] = {"first message", "second message"};
__attribute__((used,section(".nv_fatbin"))) const char device_data[] = "device bytes";
extern "C" int call(int i) { return puts(messages[i & 1]); }
''')
            reference = root/'reference.so'
            subprocess.run(['g++', '-shared', '-fPIC', '-O2', str(source), '-o', str(reference)], check=True)
            self.assertTrue(gate.compare(reference, reference)['passed'])
            self.assertTrue(sections.compare(reference, reference)['passed'])
            data = reference.read_bytes()
            headers = struct.unpack_from('<Q', data, 40)[0]
            stride, count = struct.unpack_from('<HH', data, 58)
            elf = gate.Elf(reference)
            for section in ('.text', '.init', '.rodata', '.data.rel.ro', '.rela.dyn', '.got', '.nv_fatbin'):
                with self.subTest(section=section):
                    index = next(i for i, s in enumerate(elf.sections) if s.name == section)
                    offset = struct.unpack_from('<Q', data, headers + index*stride + 24)[0]
                    changed = bytearray(data)
                    changed[offset + (16 if section == '.rela.dyn' else 0)] ^= 1
                    candidate = root/'changed.so'
                    candidate.write_bytes(changed)
                    self.assertFalse(gate.compare(reference, candidate)['passed'])
                    self.assertFalse(sections.compare(reference, candidate)['passed'])

            # Redirect a relocated pointer to another valid string, preserving ELF layout.
            section = elf.by_name['.data.rel.ro']
            before = elf.relocations[section.address]
            after = elf.relocations[section.address + 8]
            self.assertEqual(before[0], 8)
            changed = bytearray(data)
            for index, sec in enumerate(elf.sections):
                offset = struct.unpack_from('<Q', data, headers + index*stride + 24)[0]
                if sec.name == '.data.rel.ro':
                    struct.pack_into('<Q', changed, offset, after[2])
                if sec.name == '.rela.dyn':
                    for pos in range(offset, offset + sec.size, 24):
                        if struct.unpack_from('<Q', data, pos)[0] == section.address:
                            struct.pack_into('<q', changed, pos + 16, after[2])
            candidate.write_bytes(changed)
            self.assertFalse(gate.compare(reference, candidate)['passed'])
