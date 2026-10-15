import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('sass_gate', Path(__file__).resolve().parents[1] / 'Tools/sass_gate.py')
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)
SASS = '''Fatbin elf code:
arch = sm_103a
        Function : kernel_a
/*0000*/ MOV R1, 0x1 ; /* 0x0000000100017802 */
                       /* 0x000fe20000000f00 */
/*0010*/ EXIT ;        /* 0x000000000000794d */
                       /* 0x000fea0003800000 */
'''


class ComparisonTest(unittest.TestCase):
    def test_ignores_metadata(self):
        decorated = SASS.replace('Function :', '// line 12\n        Function :')
        self.assertEqual(gate.fingerprint(gate.parse_sass(SASS)), gate.fingerprint(gate.parse_sass(decorated)))

    def test_renaming_requires_explicit_mode(self):
        a, b = map(gate.parse_sass, [SASS, SASS.replace('kernel_a', 'kernel_b')])
        self.assertNotEqual(gate.fingerprint(a), gate.fingerprint(b))
        self.assertEqual(gate.fingerprint(a, True), gate.fingerprint(b, True))

    def test_detects_operand_control_arch_and_multiplicity(self):
        for changed in [SASS.replace('0x1 ;', '0x2 ;'),
                        SASS.replace('0x000fe20000000f00', '0x000fe40000000f00'),
                        SASS.replace('sm_103a', 'sm_100a'), SASS + SASS]:
            self.assertNotEqual(gate.fingerprint(gate.parse_sass(SASS), True),
                                gate.fingerprint(gate.parse_sass(changed), True))

    def test_rejects_missing_or_unparsed_code(self):
        for bad in ['', 'arch = sm_103a\nFunction : empty', SASS.replace('EXIT ;', 'unknown')]:
            with self.assertRaises(ValueError):
                gate.parse_sass(bad)

class MetadataTest(unittest.TestCase):
    def test_inventory_rejects_ptx_and_preserves_count(self):
        x = 'ELF file 1: a.1.sm_103a.cubin\nELF file 2: a.2.sm_103a.cubin'
        self.assertEqual(len(gate.parse_inventory(x)), 2)
        self.assertEqual(gate.parse_inventory(x), gate.parse_inventory(x.replace(': a.', ': b.')))
        with self.assertRaises(ValueError):
            gate.parse_inventory(x + '\nPTX file 1: a.1.sm_103a.ptx')

    def test_resources(self):
        x = 'arch = sm_103a\nFunction k:\nREG:16 STACK:0 SHARED:1024 LOCAL:0'
        self.assertNotEqual(gate.parse_resources(x), gate.parse_resources(x.replace('REG:16', 'REG:32')))

    def test_attributes_and_initialized_data(self):
        x = '''Fatbin elf code:
arch = sm_103a
.nv.info.k
<0x1>
Attribute: EIATTR_MAX_THREADS
Format: EIFMT_SVAL
Value: 0x100 0x1 0x1
.nv.constant3
0x00000001 0x00000002
'''
        self.assertNotEqual(gate.parse_elf_metadata(x), gate.parse_elf_metadata(x.replace('0x100', '0x200')))
        self.assertNotEqual(gate.parse_elf_metadata(x), gate.parse_elf_metadata(x.replace('0x00000002', '0x00000003')))
        self.assertEqual(gate.parse_elf_metadata(x), gate.parse_elf_metadata(x.replace('<0x1>', '<0x4>')))


if __name__ == '__main__':
    unittest.main()
