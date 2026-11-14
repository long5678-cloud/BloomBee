"""Run the bounded CPU review suite into a new output directory."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();output=args.output.absolute();output.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[1]
    env=os.environ.copy()
    env['PYTHONPATH']=os.pathsep.join(map(str,[root/'src',root,root/'tests/kv_network',root/'tests/kv_network/cache']))+os.pathsep+env.get('PYTHONPATH','')
    env['HF_HUB_OFFLINE']='1';env['PYTHONDONTWRITEBYTECODE']='1'
    # Keep the platform socket directory: long project paths exceed AF_UNIX
    # limits in torch_shm_manager. Persistent test outputs remain under output.
    commands=[
        ['-m','pytest','--noconftest','tests/kv_network','tests/test_kv_cache_placement.py',
         'tests/test_kv_cache_telemetry_api.py','tests/test_glm_mla_chunks.py',
         'tests/test_recovery_history_source.py','-q','-p','no:cacheprovider','--basetemp',str(output/'pytest-tmp')],
        ['tests/kv_network/check_page_retention.py',str(output/'page-retention')],
        ['-m','bloombee.kv_cache.network','check','--local','dynamo','--prefetch','transition','--telemetry'],
    ]
    results=[]
    for i,command in enumerate(commands,1):
        process=subprocess.run([sys.executable,*command],cwd=root,env=env,capture_output=True,text=True)
        (output/(str(i)+'.log')).write_text(process.stdout+process.stderr)
        results.append(dict(command=command,exit_code=process.returncode,log=str(i)+'.log'))
    receipt=dict(scope='CPU production components, local DHT/HTTP, contracts and fixtures; no GPU/native SDK acceptance',
                 passed=all(r['exit_code']==0 for r in results),results=results)
    with (output/'receipt.json').open('x') as stream:json.dump(receipt,stream,indent=2)
    print(json.dumps(receipt,indent=2))
    return 0 if receipt['passed'] else 1

if __name__=='__main__':raise SystemExit(main())
