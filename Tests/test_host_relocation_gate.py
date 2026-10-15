"""Adversarial CPU-only ELF fixtures for the second-stage host gate."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'Tools'))
import host_relocation_gate as gate


class HostRelocationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.changes = [{'before': 'Description before', 'after': 'Description after with more detail'}]

    def build(self, name, description, target='payload', opcode='lea', immediate=7, extra=''):
        path = self.root / f'{name}.so'
        source = path.with_suffix('.S')
        source.write_text(f'''
.section .rodata
.balign 8
message: .asciz "{description}"
.balign 8
payload: .asciz "unchanged payload"
other: .asciz "different payload"
.text
.globl read_data
.type read_data,@function
read_data:
 {opcode} {target}(%rip), %rax
 mov ${immediate}, %edx
 test %edx, %edx
 je .Ldone
 lea message(%rip), %rcx
.Ldone:
 ret
.size read_data, .-read_data
{extra}
.section .note.GNU-stack,"",@progbits
''')
        subprocess.run(['gcc', '-shared', '-nostdlib', str(source), '-o', str(path)], check=True)
        return path

    def pair(self, **kwargs):
        return (self.build('before', self.changes[0]['before']),
                self.build('after', self.changes[0]['after'], **kwargs))

    def test_allowed_length_change_passes_only_second_stage(self):
        result = gate.compare(*self.pair(), self.changes)
        self.assertTrue(result['passed'], result)
        self.assertFalse(result['raw']['passed'])
        self.assertEqual(result['stage'], 'relocation')
        self.assertGreater(result['relocated_instruction_count'], 0)
        self.assertEqual(result['strings'][0]['before'], self.changes[0]['before'])

    def test_reference_to_different_string_rejected(self):
        result = gate.compare(*self.pair(target='other'), self.changes)
        self.assertFalse(result['passed'])
        self.assertIn('Wrong rodata reference', result['error'])

    def test_opcode_change_rejected(self):
        result = gate.compare(*self.pair(opcode='mov'), self.changes)
        self.assertFalse(result['passed'])
        self.assertIn('Opcode/register', result['error'])

    def test_non_address_immediate_change_rejected(self):
        self.assertFalse(gate.compare(*self.pair(immediate=9), self.changes)['passed'])

    def test_unlisted_string_change_rejected(self):
        self.assertFalse(gate.compare(*self.pair())['passed'])

    def test_control_target_change_rejected(self):
        before, after = self.pair()
        data = after.read_bytes()
        # Change the short conditional jump's relative target without relinking.
        elf = gate.Elf(after)
        code = elf.by_name['.text'].data
        pos = code.index(b'\x74\x07')
        offset = data.index(code)
        data = bytearray(data)
        data[offset + pos + 1] = 6
        after.write_bytes(data)
        self.assertFalse(gate.compare(before, after, self.changes)['passed'])

    def test_equal_raw_code_does_not_hide_unapproved_data_change(self):
        a = self.build('before', 'Description before')
        b = self.build('after', 'Description BEfore')
        result = gate.compare(a, b)
        self.assertTrue(result['raw']['passed'])
        self.assertFalse(result['passed'])

    def test_changed_table_data_rejected(self):
        a, b = self.pair()
        b.write_bytes(b.read_bytes().replace(b'unchanged payload', b'unchanged payloae'))
        self.assertFalse(gate.compare(a, b, self.changes)['passed'])

    def test_data_pointer_relocation_is_resolved_not_just_masked(self):
        def table(target):
            return f'''.section .data.rel.ro,"aw"
.type pointers,@object
pointers: .quad message, {target}
.size pointers, .-pointers
'''
        a = self.build('before', self.changes[0]['before'], extra=table('payload'))
        good = self.build('good', self.changes[0]['after'], extra=table('payload'))
        bad = self.build('bad', self.changes[0]['after'], extra=table('other'))
        result = gate.compare(a, good, self.changes)
        self.assertTrue(result['passed'], result)
        self.assertEqual(result['relocations_checked'], 2)
        result = gate.compare(a, bad, self.changes)
        self.assertFalse(result['passed'])
        self.assertIn('Wrong relocated rodata pointer', result['error'])

    def test_non_relocated_data_word_change_rejected(self):
        a = self.build('before', 'Description before', extra='.data\n.quad 7')
        b = self.build('after', 'Description before', extra='.data\n.quad 9')
        result = gate.compare(a, b)
        self.assertFalse(result['passed'])
        self.assertIn('Non-relocated bytes changed', result['error'])

    def test_numeric_zero_elements_are_not_alignment_padding(self):
        def table(zeros):
            return f'''.section .rodata
.balign 16
.type numeric_table,@object
numeric_table: .ascii "ABCD"
.zero {zeros}
.size numeric_table, .-numeric_table
.ascii "NEXT"
'''
        a = self.build('before', 'Description before', extra=table(4))
        b = self.build('after', 'Description before', extra=table(12))
        result = gate.compare(a, b)
        self.assertFalse(result['passed'])
        self.assertIn('STT_OBJECT', result['error'])

    def test_fatbin_byte_corruption_rejected_with_or_without_name_changes(self):
        def fatbin(identity, byte=7):
            symbol = f'_ZN37_INTERNAL_{identity}_fixtureE'
            return f'''.section .rodata
.asciz "{symbol}"
.section .nv_fatbin,"a"
.asciz "{symbol}"
.byte {byte}
'''
        a = self.build('before', 'Description before', extra=fatbin('01234567'))
        good = self.build('good', 'Description before', extra=fatbin('abcdef01'))
        result = gate.compare(a, good)
        self.assertTrue(result['passed'], result)
        for identity in ('01234567', 'abcdef01'):
            b = self.build('bad', 'Description before', extra=fatbin(identity, 8))
            result = gate.compare(a, b)
            self.assertFalse(result['passed'])
            self.assertIn('fatbin', result['error'].lower())
