#!/usr/bin/env python3
"""Compile a copied source tree with recorded arguments, then compare SASS."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('build', type=Path)
    parser.add_argument('--arguments', required=True, type=Path,
                        help='JSON array of nvcc arguments, excluding executable, -o, --keep and --keep-dir')
    parser.add_argument('--nvcc', default='nvcc')
    parser.add_argument('--cuobjdump', default='cuobjdump')
    parser.add_argument('--reference', type=Path)
    args = parser.parse_args()
    source, build = args.source.resolve(), args.build.resolve()
    if build == source or build.is_relative_to(source) or source.is_relative_to(build):
        parser.error('source and build must be separate trees')
    if build.exists():
        parser.error('build directory already exists; choose a fresh directory')
    flags = json.loads(args.arguments.read_text())
    if not isinstance(flags, list) or not all(isinstance(x, str) for x in flags):
        parser.error('arguments must be a JSON array of strings')
    if any(x in ('-o', '--output-file', '--keep', '--keep-dir') or x.startswith(('--output-file=', '--keep-dir=')) for x in flags):
        parser.error('output and intermediate paths are managed by this tool')
    env = os.environ.copy()
    for key in ('NVCC_PREPEND_FLAGS', 'NVCC_APPEND_FLAGS', 'NVCC_CCBIN'):
        env.pop(key, None)
    version = subprocess.check_output([args.nvcc, '--version'], text=True, env=env)
    host_version = subprocess.check_output(['g++', '--version'], text=True, env=env)
    if 'V13.2.86' not in version:
        parser.error('the baseline requires nvcc V13.2.86')
    build.mkdir(parents=True)
    work = build / 'source'
    work.mkdir()
    hashes = {}
    for path in sorted(source.rglob('*')):
        if path.is_symlink():
            raise ValueError('Source symlink requires an explicit snapshot: ' + str(path))
        if path.is_file() and path.suffix in ('.cu', '.cuh', '.h', '.hpp', '.inc'):
            rel = path.relative_to(source)
            dest = work / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, dest)
            hashes[str(rel)] = hashlib.sha256(path.read_bytes()).hexdigest()
    if not hashes:
        raise ValueError('No source files')
    keep = build / 'keep'
    keep.mkdir()
    output = build / 'engine.so'
    command = [args.nvcc, *flags, '-o', str(output), '--keep', '--keep-dir', str(keep)]
    with (build / 'build.log').open('w') as log:
        log.write(version)
        log.flush()
        result = subprocess.run(command, cwd=work, stdout=log, stderr=subprocess.STDOUT, env=env)
    record = {'nvcc_version': version, 'host_gxx_version': host_version,
              'nvcc_environment': {k: env.get(k) for k in ('NVCC_PREPEND_FLAGS', 'NVCC_APPEND_FLAGS', 'NVCC_CCBIN')}, 'arguments': flags, 'source_sha256': hashes,
              'compile_exit_code': result.returncode}
    (build / 'build.json').write_text(json.dumps(record, indent=2) + '\n')
    if result.returncode:
        print('Compile failed; inspect ' + str(build / 'build.log'), file=sys.stderr)
        return result.returncode
    if args.reference:
        return subprocess.run([sys.executable, str(Path(__file__).with_name('sass_gate.py')),
                               str(args.reference.resolve()), str(output), '--cuobjdump', args.cuobjdump,
                               '--report', str(build / 'sass.json')]).returncode
    print('Compiled a baseline only; no measured-binary comparison was performed.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
