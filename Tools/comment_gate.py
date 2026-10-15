#!/usr/bin/env python3
"""Audit CUDA comments and list repeated text attached to different code."""
import argparse
from collections import defaultdict
from bisect import bisect_left
import json
from pathlib import Path
import re

from source_tokens import LEX, tokens

SOURCE_SUFFIXES = {'.cu', '.cuh', '.h', '.hpp', '.inc'}
LABEL = re.compile(r'\b(?:r[0-9]+[a-z]*|s(?:[1-9]|1[0-4])[a-z]*|[67][0-9]{3})\b')


def audit(root):
    failures = []
    repeated = defaultdict(list)
    non_labels = []
    count = 0
    for path in sorted(root.rglob('*')):
        if not path.is_file() or path.suffix not in SOURCE_SUFFIXES:
            continue
        count += 1
        text = path.read_text()
        lexed = list(LEX.finditer(text))
        code = [(m.start(), m.group()) for m in lexed if m.lastgroup not in ('comment', 'space')]
        starts = [position for position, value in code]
        for match in lexed:
            if match.lastgroup != 'comment':
                continue
            comment = match.group()
            line = text.count('\n', 0, match.start()) + 1
            location = {'path': str(path.relative_to(root)), 'line': line}
            code_before = text[text.rfind('\n', 0, match.start()) + 1:match.start()]
            index = bisect_left(starts, match.end())
            following = [value for position, value in code[index:index + 32]]
            attached = tokens(code_before) or following
            context = ' '.join(attached)
            for reason, bad in (
                ('empty line comment', comment.startswith('//') and not comment[2:].strip()),
                ('semicolon-colon artifact', ';:' in comment),
                ('space before period', re.search(r'\s+\.(?!\.)', comment)),
                ('punctuation-only comment', not re.search(r'[A-Za-z0-9]', comment) and bool(re.search(r'[;:]', comment))),
            ):
                if bad:
                    failures.append({**location, 'reason': reason, 'comment': comment})
            for label in LABEL.finditer(comment):
                value = label.group()
                # Source identifiers and explicit device/shape values are not revision labels.
                if value in attached + following and re.match(r'r\d+$', value):
                    why = 'identifier used by the attached code'
                elif value == '6000' and re.search(r'RTX PRO\s+6000\b', comment):
                    why = 'GPU model number'
                elif value == '6000' and re.search(r'4000\s+x\s+6000\s+x\s+5003', comment):
                    why = 'documented matrix dimension'
                else:
                    failures.append({**location, 'reason': 'internal revision label', 'label': value, 'comment': comment})
                    continue
                non_labels.append({**location, 'value': value, 'reason': why})
            normalized = re.sub(r'\s+', ' ', re.sub(r'^//\s*|^/\*\s*|\s*\*/$', '', comment)).strip()
            phrases = {normalized}
            phrases.update(part.strip(' :;') for part in re.split(r'(?<=[.!?])(?:\s+|(?=[:;]))', normalized)
                           if len(re.findall(r'[A-Za-z]+', part)) >= 2)
            for phrase in phrases:
                if re.search(r'[A-Za-z]', phrase):
                    repeated[phrase].append({**location, 'code': context})
    review = []
    for comment, locations in sorted(repeated.items()):
        contexts = {x['code'] for x in locations}
        if len(contexts) >= 3:
            review.append({'comment': comment, 'distinct_code_contexts': len(contexts), 'locations': locations})
    return {'files': count, 'failures': failures, 'non_label_matches': non_labels,
            'repeated_comments_require_reading': review,
            'repeat_definition': 'Same normalized comment or sentence at three or more distinct attached code token sequences; mirrored identical code is deduplicated.',
            'passed': not failures}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.root)
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(f"{report['files']} files; {len(report['failures'])} failures; "
          f"{len(report['repeated_comments_require_reading'])} repeated comments to read")
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
