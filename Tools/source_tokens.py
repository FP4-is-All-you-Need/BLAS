#!/usr/bin/env python3
"""Compare C++ preprocessing tokens while excluding comments and whitespace."""
import argparse
from pathlib import Path
import re

# String alternatives precede comments: comment markers inside PTX strings are data.
LEX = re.compile(r'''(?P<raw>(?:u8|u|U|L)?R"(?P<delim>[^ ()\\\t\r\n]{0,16})\(.*?\)(?P=delim)")|(?P<string>(?:u8|u|U|L)?"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*')|(?P<comment>//[^\n]*|/\*.*?\*/)|(?P<space>\s+)|(?P<token>[A-Za-z_]\w*|(?:\d|\.\d)(?:[eEpP][+-]|[\w.])*|::|->\*|->|\.\*|\.\.\.|<<=|>>=|<<|>>|<=|>=|==|!=|&&|\|\||\+\+|--|[+*/%&|^!~=-]=|##|.)''', re.S)


def tokens(text):
    text = re.sub(r'\\\r?\n', '', text)
    return [m.group() for m in LEX.finditer(text) if m.lastgroup not in ('comment', 'space')]


def _source(text):
    """Keep logical preprocessor lines and their positions among ordinary tokens."""
    from bisect import bisect_right
    text = re.sub(r'\\\r?\n', '', text)
    newlines = [m.start() for m in re.finditer('\n', text)]
    lexed = [m for m in LEX.finditer(text) if m.lastgroup not in ('comment', 'space')]
    items = [(m.group(), bisect_right(newlines, m.start())) for m in lexed]
    values = [value for value, line in items]
    directives, directive_indices = [], set()
    ordinary = 0
    i = 0
    while i < len(items):
        value, line = items[i]
        if value == '#' and (i == 0 or items[i - 1][1] != line):
            j = i + 1
            while j < len(items) and items[j][1] == line:
                j += 1
            function_macro = (values[i:i + 2] == ['#', 'define'] and i + 3 < j
                              and values[i + 3] == '('
                              and lexed[i + 2].end() == lexed[i + 3].start())
            directives.append((ordinary, tuple(values[i:j]), function_macro))
            directive_indices.update(range(i, j))
            i = j
        else:
            ordinary += value not in ('{', '}')
            i += 1
    return values, directives, directive_indices


def preprocessor_lines_equal(before, after):
    return _source(before)[1] == _source(after)[1]


def source_tokens_equal(before, after):
    a, app, _ = _source(before)
    b, bpp, _ = _source(after)
    return a == b and app == bpp


class _Statements:
    """Conservative statement boundaries, without interpreting preprocessor branches.

    A candidate insertion must wrap one complete original statement immediately
    after a control head. Directives inside that span prevent approval. Unknown
    or incomplete syntax fails closed when an insertion depends on it.
    """
    def __init__(self, values, directives, directive_lines=None):
        self.t = values
        self.pp = directives
        self.directive_lines = directive_lines
        self.cache = {}

    def balanced(self, start):
        pairs = {'(': ')', '[': ']', '{': '}'}
        stack = []
        for i in range(start, len(self.t)):
            if i in self.pp:
                return None
            value = self.t[i]
            if value in pairs:
                stack.append(pairs[value])
            elif value in (')', ']', '}'):
                if not stack or value != stack.pop():
                    return None
                if not stack:
                    return i + 1
        return None

    def body_start(self, i):
        if i in self.pp:
            return None
        word = self.t[i]
        if word in ('else', 'do'):
            return i + 1
        if word not in ('if', 'for', 'while', 'switch'):
            return None
        j = i + 1
        if word == 'if' and j < len(self.t) and self.t[j] == 'constexpr':
            j += 1
        if j >= len(self.t) or self.t[j] != '(':
            return None
        return self.balanced(j)

    def end(self, i):
        if i in self.cache:
            return self.cache[i]
        self.cache[i] = None
        result = self._end(i)
        self.cache[i] = result
        return result

    def _end(self, i):
        if i >= len(self.t) or i in self.pp:
            return None
        if self.directive_lines is not None and not _safe_macro_statement(
                self.t, i, len(self.t), self.directive_lines):
            return None
        word = self.t[i]
        if word == '{':
            return self.balanced(i)
        if word in ('if', 'for', 'while', 'switch', 'do'):
            body = self.body_start(i)
            if body is None:
                return None
            end = self.end(body)
            if end is None:
                return None
            if word == 'if' and end < len(self.t) and self.t[end] == 'else':
                return self.end(end + 1)
            if word == 'if' and end in self.pp:
                following = end
                while following in self.pp:
                    following += 1
                if following < len(self.t) and self.t[following] == 'else':
                    return None
            if word == 'do':
                if end >= len(self.t) or self.t[end] != 'while':
                    return None
                tail = self.body_start(end)
                return tail + 1 if tail is not None and tail < len(self.t) and self.t[tail] == ';' else None
            return end
        # Labels still label the entire following statement.
        if i + 1 < len(self.t) and re.fullmatch(r'[A-Za-z_]\w*', word) and self.t[i + 1] == ':':
            return self.end(i + 2)
        if word in ('else', 'try', 'catch', 'case', 'default'):
            return None
        # Expression/declaration/return/asm statements end at a top-level semicolon.
        # Balanced initializers and lambda bodies are part of that same statement.
        j = i
        while j < len(self.t):
            value = self.t[j]
            if j in self.pp or value in ('}', 'else'):
                return None
            if value == ';':
                return j + 1
            if value in ('(', '[', '{'):
                j = self.balanced(j)
                if j is None:
                    return None
            else:
                j += 1
        return None

    def insertion_pairs(self):
        pairs = {}
        do_tails = set()
        for i, word in enumerate(self.t):
            if word == 'do' and i not in self.pp:
                tail = self.end(i + 1)
                if tail is not None:
                    do_tails.add(tail)
        for i, word in enumerate(self.t):
            if word not in ('if', 'for', 'while', 'else', 'do') or i in self.pp or i in do_tails:
                continue
            start = self.body_start(i)
            if start is None or start >= len(self.t) or self.t[start] == '{' or start in self.pp:
                continue
            end = self.end(start)
            if end is not None and not any(j in self.pp for j in range(start, end)):
                pairs[start] = end
        return pairs


def _safe_macro_statement(values, start, end, directives):
    """Only accept a macro body proved to be one asm statement.

    Other macro invocations at statement start require preprocessing or review;
    treating an arbitrary macro invocation as one statement would be unsound.
    """
    definitions = {}
    for _, directive, function_macro in directives:
        if directive[:2] == ('#', 'define') and len(directive) >= 3:
            definitions.setdefault(directive[2], []).append((list(directive[3:]), function_macro))
    word = values[start]
    looks_like_macro = re.fullmatch(r'[A-Z_][A-Z0-9_]*', word)
    if word not in definitions:
        return not (looks_like_macro and start + 1 < end
                    and values[start + 1] in ('(', ';'))
    for body, function_macro in definitions[word]:
        if function_macro:
            parameter_end = _Statements(body, set()).balanced(0)
            if parameter_end is None:
                return False
            body = body[parameter_end:]
        if not body or body[0] not in ('asm', '__asm__'):
            return False
        j = 2 if len(body) > 1 and body[1] in ('volatile', '__volatile__') else 1
        if j >= len(body) or body[j] != '(' or _Statements(body, set()).balanced(j) != len(body):
            return False
    return True


def formatting_tokens_equal(before, after, macro_directives=()):
    """Allow one brace pair around an original single control-body statement only."""
    a, app, pp = _source(before)
    b, bpp, _ = _source(after)
    if app != bpp:
        return False
    context = list(app) + list(macro_directives)
    allowed = _Statements(a, pp, context).insertion_pairs()
    stack, used = [], set()
    i = 0
    for value in b:
        # Close an inserted block before consuming an original adjacent closing brace.
        if stack and stack[-1] == i:
            if value != '}':
                return False
            stack.pop()
        elif value == '{' and i in allowed and i not in used:
            end = allowed[i]
            if (stack and end > stack[-1]) or not _safe_macro_statement(a, i, end, context):
                return False
            stack.append(end)
            used.add(i)
        elif i < len(a) and value == a[i]:
            i += 1
        else:
            return False
    return i == len(a) and not stack


def macro_context(root):
    """Collect all local definitions, including lowercase and object-like macros.

    Include resolution and conditional configuration are deliberately conservative:
    every definition in the engine must prove the same single-statement property.
    """
    definitions = set()
    for path in root.rglob('*'):
        if path.is_file() and path.suffix in ('.cu', '.cuh', '.h', '.hpp', '.inc'):
            for _, directive, function_macro in _source(path.read_text())[1]:
                if directive[:2] == ('#', 'define'):
                    definitions.add((directive, function_macro))
    return [(0, directive, function_macro) for directive, function_macro in sorted(definitions)]


def compare(left, right, formatting=False):
    def files(root):
        return {p.relative_to(root): p for p in root.rglob('*')
                if p.is_file() and p.suffix in ('.cu', '.cuh', '.h', '.hpp', '.inc')}
    a, b = files(left), files(right)
    if a.keys() != b.keys():
        raise ValueError('Source file sets differ')
    check = formatting_tokens_equal if formatting else source_tokens_equal
    contexts = {}
    bad = []
    for p in a:
        extra = ()
        if formatting:
            # When a whole backend tree is compared, keep engine definitions separate.
            engine = next((parent for parent in a[p].parents
                           if parent.is_relative_to(left) and (parent / 'l8rt.cu').exists()), left)
            if engine not in contexts:
                contexts[engine] = macro_context(engine)
            extra = (contexts[engine],)
        if not check(a[p].read_text(), b[p].read_text(), *extra):
            bad.append(str(p))
    if bad:
        raise ValueError('Non-comment tokens differ: ' + ', '.join(bad))
    return len(a)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('left', type=Path)
    parser.add_argument('right', type=Path)
    parser.add_argument('--formatting', action='store_true')
    args = parser.parse_args()
    print(f'PASS: {compare(args.left, args.right, args.formatting)} source files')
