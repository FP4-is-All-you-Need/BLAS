import sys
from pathlib import Path
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'Tools'))
import comment_gate


class CommentAuditTest(unittest.TestCase):
    def check_source(self, source):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'test.cu').write_text(source)
            return comment_gate.audit(root)

    def test_artifacts_are_comments_only(self):
        result = self.check_source('const char *p = "// ;: . r2";\nint a; //\n// bad;: text .\n')
        self.assertEqual(len(result['failures']), 3)
        self.assertEqual({x['reason'] for x in result['failures']},
                         {'empty line comment', 'semicolon-colon artifact', 'space before period'})

    def test_repeated_text_lists_distinct_code(self):
        result = self.check_source('// Operation.\nf();\n// Operation.\ng();\n// Operation.\nh();\n')
        self.assertEqual(len(result['repeated_comments_require_reading']), 1)
        self.assertEqual(result['repeated_comments_require_reading'][0]['distinct_code_contexts'], 3)

    def test_revision_labels_are_not_hardware_or_expression_identifiers(self):
        result = self.check_source('// r2: old\n// s14: old\n// 6285\n'
                                   '// RTX PRO 6000\nf();\n// (c0 + lane, r0 + j)\nf(r0);\n')
        self.assertEqual([x['label'] for x in result['failures']], ['r2', 's14', '6285'])
        self.assertEqual(len(result['non_label_matches']), 2)

    def test_repeated_sentence_inside_different_comments(self):
        result = self.check_source('// Shared explanation. Specific alpha.\nf();\n'
                                   '// Shared explanation. Specific beta.\ng();\n'
                                   '// Shared explanation.: Specific gamma.\nh();\n')
        self.assertIn('Shared explanation.',
                      [x['comment'] for x in result['repeated_comments_require_reading']])


if __name__ == '__main__':
    unittest.main()
