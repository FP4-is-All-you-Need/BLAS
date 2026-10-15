#!/usr/bin/env python3
"""Run the frozen real host table checks on the CPU, without CUDA headers/runtime."""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile
ROOT=Path(__file__).resolve().parents[1]
def inspect():
    results=[]
    for e in json.loads((ROOT/'Docs/Provenance/engines.json').read_text()):
        if '_z' in e['id']: continue
        source=ROOT/e['path']
        text=(source/'l8_encode.cuh').read_text()
        face=''
        if 'inline int l8_face_value(' in text:
            a=text.index('inline int l8_face_value(');b=text.index('\n}',a)+2
            face=text[a:b]
        has4096=(source/'l8k_tables_k4096.h').exists() and '-DL8_KPLAN4096=1' in e['arguments']
        includes='#include "l8_tables.h"\n#include "l8k_tables_rt.h"\n'
        if has4096: includes+='#include "l8k_tables_k4096.h"\n'
        code='#include <algorithm>\n#include <cstring>\n#include <iostream>\n'+includes+face+'\n#include "l8k_host.cuh"\n'
        code+='''std::string check(int k) {
        L8kMod mods[L8K_NMOD];
        const L8kMod *base = k<=8192 ? l8k_mods59_k8192 : l8k_mods59_k16384;
'''
        if has4096: code+='if(k<=4096)base=l8k_mods59_k4096;\n'
        code+=f'const int cuts=((k+{e["k"]-1})/{e["k"]}+1)/2*2;\n'
        code+='''std::memcpy(mods,base,sizeof(mods));int gt=0,gbase=0,pbase=0;
        for(auto &m:mods)gt+=(cuts+L8K_FTS/m.planes-1)/(L8K_FTS/m.planes);
        for(auto &m:mods){m.cpg=L8K_FTS/m.planes;m.groups=(cuts+m.cpg-1)/m.cpg;
            m.gbase=gbase;m.pbase=pbase;m.gt=gt;gbase+=m.groups;pbase+=m.planes;}
        try {l8k_check_tables(mods,L8K_NMOD,k,cuts,L8K_FTS,2,false);return "";}
        catch(const std::exception &e){return e.what();}
        }
        int main(){int last=0;for(int k=1;k<=16384;++k){auto err=check(k);if(!err.empty()){
            std::cout<<last<<"\\n"<<err<<"\\n";return 0;}last=k;}
            std::cout<<last<<"\\n";}
'''
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'check.cpp').write_text(code)
            subprocess.run(['g++','-O2','-std=c++17','-I'+str(source),str(p/'check.cpp'),'-o',str(p/'check')],check=True)
            lines=subprocess.check_output([str(p/'check')],text=True).splitlines()
        results.append({'id':e['id'],'last_contiguous_k_passing_table_checks':int(lines[0]),'first_failure':lines[1] if len(lines)>1 else None})
    return results
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path);args=parser.parse_args()
    result=inspect();out=json.dumps(result,indent=2)+'\n'
    if args.output:args.output.write_text(out)
    print(out)
