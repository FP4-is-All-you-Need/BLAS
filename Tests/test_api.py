"""Compile the real host API against CPU stubs; never load CUDA binaries."""
from pathlib import Path
import json
import shutil
import subprocess
import tempfile
import unittest
ROOT = Path(__file__).resolve().parents[1]
class ApiTest(unittest.TestCase):
    def test_dispatch_and_level3_copy_contract(self):
        source = ROOT / 'Tests/CPU_CUDA'
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            for name in ('engine', 'support'):
                subprocess.run(['g++','-std=c++17','-shared','-fPIC',str(source / (name+'.cpp')),
                                '-I'+str(source),'-I'+str(ROOT/'Include'),'-o',str(p/(name+'.so'))],check=True)
            for engine in (json.loads((ROOT/'Docs/Provenance/engines.json').read_text()) + json.loads((ROOT/'Docs/Provenance/additional_engines.json').read_text())):
                shutil.copyfile(p/'engine.so',p/('libawe_'+engine['id']+'.so'))
            for kind in ('real','complex','float'):
                shutil.copyfile(p/'support.so',p/('libawe_level3_'+kind+'.so'))
            subprocess.run(['g++','-std=c++17','-pthread','-I'+str(source),'-I'+str(ROOT/'Include'),
                '-I'+str(ROOT/'Routines/Common'),str(source/'api_test.cpp'),str(ROOT/'API/api.cpp'),
                str(ROOT/'Routines/Common/engine_loader.cpp'),'-ldl','-o',str(p/'test')],check=True)
            subprocess.run([str(p/'test'),str(p)],check=True)
            subprocess.run(['cc','-x','c','-fsyntax-only','-I'+str(ROOT/'Include'),'-'],input='#include <AWE/awe.h>\n',text=True,check=True)
