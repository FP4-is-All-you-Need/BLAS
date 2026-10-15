import sys
from pathlib import Path
import subprocess
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'Tools'))
import host_text_gate


class HostTextTest(unittest.TestCase):
    def test_only_observed_initializer_temporary_prefix_is_normalized(self):
        pattern = host_text_gate.TEMPORARY_INITIALIZER
        label = '<_GLOBAL__sub_I_tmpxft_000413f0_00000000_6_l8rt.cudafe1.cpp>'
        self.assertEqual(pattern.sub('', label), '<_GLOBAL__sub_I_l8rt.cudafe1.cpp>')
        operand = '<user_tmpxft_000413f0_00000000_6_l8rt>'
        self.assertEqual(pattern.sub('', operand), operand)

    def test_independent_builds_and_changed_host_instruction(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            binaries = []
            for i, value in enumerate((7, 7, 9)):
                source = root / f'input{i}.c'
                binary = root / f'output{i}.so'
                source.write_text(f'int answer(void) {{ return {value}; }}\n')
                subprocess.run(['gcc', '-shared', '-fPIC', '-O2', str(source), '-o', str(binary)], check=True)
                binaries.append(binary)
            self.assertTrue(host_text_gate.compare(binaries[0], binaries[1])['passed'])
            self.assertFalse(host_text_gate.compare(binaries[0], binaries[2])['passed'])

    def test_non_elf_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / 'not-elf'
            path.write_text('not a library')
            with self.assertRaises(ValueError):
                host_text_gate.text_bytes(path)


if __name__ == '__main__':
    unittest.main()
