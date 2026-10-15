"""Compare every ELF section with explicit compiler-name exceptions."""
from pathlib import Path
if not __debug__:
    raise RuntimeError("Run this validation without Python optimization")
import sys, hashlib
sys.path.insert(0, str(Path(__file__).resolve().parent))
from host_relocation_gate import Elf, rodata_map, INTERNAL, TEMP

def compare(reference, candidate):
    (a, b) = (Elf(reference), Elf(candidate))
    result = {'passed': False, 'scope': 'All ELF section layouts and bytes; GNU build ID and established compiler-generated names are recorded separately.'}
    try:
        assert list(a.by_name) == list(b.by_name), 'Section inventory differs'
        detail = {}
        rodata_map(a, b, (), detail)
        assert not detail.get('strings') and (not detail.get('alignment_padding')), 'Non-compiler data change'
        rows = []
        for s in a.sections:
            t = b.by_name[s.name]
            assert (s.address, s.size, s.kind, s.alignment) == (t.address, t.size, t.kind, t.alignment), (s.name, 'layout')
            equal = s.data == t.data
            kind = 'raw'
            if not equal:
                if s.name == '.note.gnu.build-id':
                    assert s.data[:16] == t.data[:16], 'Build-ID note header changed'
                    kind = 'build-id'
                elif s.name in ('.rodata', '.nv_fatbin'):
                    assert detail.get('normalized_fatbin_equal'), s.name
                    kind = 'existing compiler-name check'
                elif s.name == '.strtab':

                    def norm(x):
                        return __import__('re').sub(b'tmpxft_[0-9a-f]{8}_[0-9a-f]{8}', b'tmpxft_00000000_00000000', INTERNAL.sub(b'_INTERNAL_00000000_', x))
                    assert norm(s.data) == norm(t.data), 'Other symbol strings changed'
                    kind = 'compiler names'
                else:
                    raise AssertionError((s.name, 'bytes'))
            rows.append({'section': s.name, 'size': s.size, 'raw_equal': equal, 'comparison': kind, 'reference_sha256': hashlib.sha256(s.data).hexdigest(), 'candidate_sha256': hashlib.sha256(t.data).hexdigest()})
        result.update(passed=True, sections=rows, compiler_names=detail)
    except (AssertionError, ValueError) as e:
        result['error'] = str(e)
    return result
if __name__ == '__main__':
    import argparse, json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference', type=Path)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    result = compare(args.reference, args.candidate)
    output = json.dumps(result, indent=2) + '\n'
    if args.report:
        args.report.write_text(output)
    print(output, end='')
    raise SystemExit(0 if result['passed'] else 1)
