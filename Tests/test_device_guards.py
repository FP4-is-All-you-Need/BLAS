"""Compile the production create guards with a CPU CUDA-runtime stub."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class DeviceGuardTest(unittest.TestCase):
    def test_all_create_guards_use_active_device_and_exact_architectures(self):
        source = '''#include <stdexcept>
#include <cassert>
struct cudaDeviceProp { int major, minor; };
static int active = 2, queried = -1, major_value, minor_value;
int cudaGetDevice(int* p) { *p=active; return 0; }
int cudaGetDeviceProperties(cudaDeviceProp* p, int d) {
    queried=d; p->major=major_value; p->minor=minor_value; return 0;
}
void ck(int status) { if(status) throw std::runtime_error("CUDA failure"); }
'''
        calls = []
        for engine in json.loads((ROOT / 'Docs/Provenance/engines.json').read_text()):
            for path in sorted((ROOT / engine['path']).glob('*dtab.cuh')):
                text = path.read_text()
                block = text.split('    cudaDeviceProp prop{};', 1)[1].split('    int runtime = 0;', 1)[0]
                name = f'guard_{len(calls)}'
                source += f'void {name}() {{ cudaDeviceProp prop{{}};\n{block}\n}}\n'
                calls.append((name, engine['k']))
        source += 'int main() {\n'
        for name, k in calls:
            for major, minor in [(10, 0), (10, 3), (10, 1), (9, 0), (12, 0)]:
                accepts = major == 10 and (minor == 3 or (k == 64 and minor == 0))
                source += (f'{{ major_value={major}; minor_value={minor}; queried=-1; bool ok=true;'
                           f'try{{{name}();}}catch(const std::exception&){{ok=false;}}'
                           f'assert(ok=={str(accepts).lower()}); assert(queried==active); }}\n')
        source += '}\n'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / 'test.cpp').write_text(source)
            subprocess.run(['g++', '-std=c++17', str(path / 'test.cpp'), '-o', str(path / 'test')], check=True)
            subprocess.run([str(path / 'test')], check=True)


if __name__ == '__main__':
    unittest.main()
