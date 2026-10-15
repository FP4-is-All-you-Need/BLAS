import importlib.util
from pathlib import Path
import unittest
spec = importlib.util.spec_from_file_location('source_tokens', Path(__file__).resolve().parents[1] / 'Tools/source_tokens.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class TokensTest(unittest.TestCase):
    def test_comments_and_strings(self):
        a = 'int x = 2; // why\nconst char* p = "// keep /* this */";'
        b = 'int /* comment */ x=2;\nconst char *p="// keep /* this */";'
        self.assertEqual(m.tokens(a), m.tokens(b))
        self.assertNotEqual(m.tokens(a), m.tokens(b.replace('this', 'that')))

    def test_raw_strings_and_splicing(self):
        a = 'asm(R"ptx(// keep\n/* keep */)ptx"); // remove\\\n also remove\n'
        b = 'asm(R"ptx(// keep\n/* keep */)ptx");'
        self.assertEqual(m.tokens(a), m.tokens(b))
        self.assertNotEqual(m.tokens(a), m.tokens(b.replace('keep', 'change')))

    def test_token_boundaries(self):
        self.assertNotEqual(m.tokens('a + +b'), m.tokens('a++b'))
        self.assertNotEqual(m.tokens('1.5'), m.tokens('1 .5'))

class FormattingTest(unittest.TestCase):
    def test_added_braces_only(self):
        self.assertTrue(m.formatting_tokens_equal('if (x) f();', 'if (x) { f(); }'))
        self.assertFalse(m.formatting_tokens_equal('if (x) { f(); }', 'if (x) f();'))
        self.assertFalse(m.formatting_tokens_equal('int a[]={1,2};', 'int a[]={1,3};'))
        self.assertFalse(m.formatting_tokens_equal('f<<<1,256>>>();', 'f<<<1,128>>>();'))
        self.assertFalse(m.formatting_tokens_equal('if (x) f();', 'if (x) { f();'))

    def test_rejects_expanded_or_unrelated_scope(self):
        self.assertFalse(m.formatting_tokens_equal('if (c) a(); b();', 'if (c) { a(); b(); }'))
        self.assertFalse(m.formatting_tokens_equal('a();', '{ a(); }'))
        self.assertFalse(m.formatting_tokens_equal('if (c) a();', 'if (c) {{ a(); }}'))

    def test_nested_control_flow_and_dangling_else(self):
        self.assertTrue(m.formatting_tokens_equal(
            'if (a) if (b) f(); else g();', 'if (a) { if (b) { f(); } else { g(); } }'))
        self.assertFalse(m.formatting_tokens_equal(
            'if (a) if (b) f(); else g();', 'if (a) { if (b) f(); } else g();'))
        self.assertTrue(m.formatting_tokens_equal('do f(); while (c);', 'do { f(); } while (c);'))
        self.assertFalse(m.formatting_tokens_equal('do f(); while (c);', 'do f(); while (c) {;}'))

    def test_preprocessor_boundaries(self):
        a = 'for (;;)\n#pragma unroll\nf();'
        self.assertFalse(m.formatting_tokens_equal(a, 'for (;;) {\n#pragma unroll\nf();}'))
        self.assertTrue(m.formatting_tokens_equal(
            '#if X\nelse if (a) f();\n#endif', '#if X\nelse if (a) {f();}\n#endif'))
        self.assertFalse(m.formatting_tokens_equal(
            'if(a) if(b) f();\n#if X\nelse g();\n#endif',
            'if(a) {if(b) f();}\n#if X\nelse g();\n#endif'))
        self.assertFalse(m.source_tokens_equal('#define X 1\n+2\nf();', '#define X 1+2\nf();'))
        self.assertFalse(m.source_tokens_equal('#define F(x) x\n', '#define F (x) x\n'))

    def test_macros_are_not_assumed_to_be_single_statements(self):
        prefix = '#define M f(); g()\n'
        self.assertFalse(m.formatting_tokens_equal(prefix + 'if (c) M;', prefix + 'if (c) {M;}'))
        self.assertFalse(m.formatting_tokens_equal(prefix + 'if (c) for (;;) M;',
                                                  prefix + 'if (c) {for (;;) M;}'))
        prefix = '#define M(x) asm volatile(x)\n'
        self.assertTrue(m.formatting_tokens_equal(prefix + 'if (c) M("nop;");',
                                                 prefix + 'if (c) {M("nop;");}'))

    def test_other_header_macros_are_not_single_statements(self):
        import tempfile
        from pathlib import Path
        for definition, invocation in (('#define M a(); b()', 'M;'),
                                       ('#define lower(x) a(); b()', 'lower(1);')):
            with tempfile.TemporaryDirectory() as directory:
                left, right = Path(directory) / 'before', Path(directory) / 'after'
                left.mkdir(); right.mkdir()
                for root in (left, right):
                    (root / 'helpers.h').write_text(definition + '\n')
                (left / 'kernel.cu').write_text('if(c) ' + invocation)
                (right / 'kernel.cu').write_text('if(c) {' + invocation + '}')
                with self.assertRaises(ValueError):
                    m.compare(left, right, formatting=True)
        self.assertFalse(m.formatting_tokens_equal('if(c) UNKNOWN;', 'if(c) {UNKNOWN;}'))

    def test_initializer_lambda_and_asm_are_one_statement(self):
        for body in ('int a[]={1,2};', 'f([](){ return 1; });', 'asm volatile("{ nop; }");'):
            self.assertTrue(m.formatting_tokens_equal('if(c) '+body, 'if(c) {'+body+'}'))


if __name__ == '__main__':
    unittest.main()
