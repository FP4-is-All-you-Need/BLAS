"""Exercise symbol isolation with CPU-only stand-ins, never CUDA modules."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class EngineLoaderTest(unittest.TestCase):
    def test_isolation_shared_ownership_and_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            for name, number in [('a', 41), ('b', 73), ('interposer', 666)]:
                source = ('static int state=0;\n'
                          f'extern "C" int d_value() {{return {number};}}\n'
                          'extern "C" int d_wrapped() {return d_value();}\n'
                          'extern "C" void d_set(int value) {state=value;}\n'
                          'extern "C" int d_get() {return state;}\n')
                (p / (name + '.cpp')).write_text(source)
                subprocess.run(['g++', '-shared', '-fPIC', str(p / (name + '.cpp')),
                                '-Wl,-Bsymbolic', '-Wl,--version-script=' + str(ROOT / 'CMake/engine_exports.map'),
                                '-o', str(p / (name + '.so'))], check=True)
            (p / 'test.cpp').write_text(r'''
#include "engine_loader.hpp"
#include <cassert>
#include <dlfcn.h>
#include <stdexcept>
#include <string>
int main(int argc,char** argv) {
    assert(argc==2); std::string root=argv[1];
    void* global=dlopen((root+"/interposer.so").c_str(),RTLD_NOW|RTLD_GLOBAL);
    assert(global);
    auto a=awe::EngineModule::open(root+"/a.so");
    auto b=awe::EngineModule::open(root+"/b.so");
    auto same=awe::EngineModule::open(root+"/./a.so");
    assert(a==same && &a->mutex()==&same->mutex() && a!=b);
    using Get=int(*)(); using Set=void(*)(int);
    { std::lock_guard<std::mutex> lock(a->mutex());
      assert(reinterpret_cast<Get>(a->symbol("d_wrapped"))()==41);
      reinterpret_cast<Set>(a->symbol("d_set"))(19); }
    { std::lock_guard<std::mutex> lock(b->mutex());
      assert(reinterpret_cast<Get>(b->symbol("d_wrapped"))()==73);
      assert(reinterpret_cast<Get>(b->symbol("d_get"))()==0); }
    assert(reinterpret_cast<Get>(same->symbol("d_get"))()==19);
    bool failed=false; try { a->symbol("missing"); } catch(const std::exception&) { failed=true; }
    assert(failed); failed=false;
    try { awe::EngineModule::open(root+"/absent.so"); } catch(const std::exception&) { failed=true; }
    assert(failed); dlclose(global);
}
''')
            subprocess.run(['g++', '-std=c++17', '-pthread', '-I' + str(ROOT / 'Routines/Common'),
                            str(p / 'test.cpp'), str(ROOT / 'Routines/Common/engine_loader.cpp'),
                            '-ldl', '-o', str(p / 'test')], check=True)
            subprocess.run([str(p / 'test'), str(p)], check=True)


if __name__ == '__main__':
    unittest.main()
