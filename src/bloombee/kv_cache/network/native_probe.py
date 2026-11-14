"""Real official coordinator + CPU MP servers + filesystem L2 + SDK tensors.

No mocks, no model substitution, no L1 deletion to manufacture a cold target.
"""
from pathlib import Path
import json
import os
import signal
import socket
import subprocess
import sys
import time
import uuid

import httpx
import torch
import zmq
from lmcache.sdk.kvcache import connect
from lmcache.utils import EngineType
from lmcache.v1.multiprocess.group_view import EngineGroupInfo
from lmcache.v1.multiprocess.transfer_context import create_transfer_context
from lmcache.v1.multiprocess.transport.factory import RequestClientFactory

BASE=Path(__file__).resolve().parents[1]
BOOT=None  # Services use the normally installed LMCache interpreter.
CHUNK=16
CLIENT=httpx.Client(trust_env=False)


def free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0))
        return sock.getsockname()[1]


def poll(fn, timeout=90):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        result=fn()
        if result:return result
        time.sleep(.05)
    raise TimeoutError('native service readiness/result timeout')


class Services:
    def __init__(self, root):
        self.root=root
        root.mkdir(parents=True,exist_ok=False)
        self.processes=[]
        self.coordinator=f'http://127.0.0.1:{free_port()}'
        self.l2=root/'l2'
        self.l2.mkdir()
        try:
            self.start('coordinator','lmcache.v1.mp_coordinator',[
                '--host','127.0.0.1','--port',self.coordinator.rsplit(':',1)[1],
                '--chunk-size',str(CHUNK),'--health-check-interval','0',
                '--eviction-check-interval','0','--disable-metrics'])
            poll(lambda:self.answers(self.coordinator+'/healthz'))
        except Exception:
            self.close()
            raise

    def start(self,name,module,args):
        log=(self.root/(name+'.log')).open('xb')
        proc=subprocess.Popen([sys.executable,'-m',module,*args],
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        self.processes.append((name,proc,log))
        self.snapshot()
        return proc

    def snapshot(self):
        # Append launch identity; never overwrite earlier records.
        with (self.root/'processes.jsonl').open('a') as f:
            f.write(json.dumps(dict(time_ns=time.monotonic_ns(),processes=[
                dict(name=n,pid=p.pid,returncode=p.poll()) for n,p,_ in self.processes]))+'\n')

    def answers(self,url):
        for name,proc,_ in self.processes:
            if proc.poll() is not None:
                raise RuntimeError(f'{name} exited {proc.returncode}; inspect its retained log')
        try:return CLIENT.get(url,timeout=1).status_code==200
        except httpx.HTTPError:return False

    def server(self,name):
        rpc,http=free_port(),free_port()
        result=dict(instance_id=name,url=f'tcp://127.0.0.1:{rpc}',http_url=f'http://127.0.0.1:{http}')
        self.start(name,'lmcache.v1.multiprocess.http_server',[
            '--instance-id',name,'--host','127.0.0.1','--http-host','127.0.0.1',
            '--port',str(rpc),'--http-port',str(http),'--chunk-size',str(CHUNK),
            '--l1-size-gb','0.0625','--eviction-policy','LRU',
            '--supported-transfer-mode','auto','--no-l1-use-lazy',
            '--l2-adapter',json.dumps(dict(type='fs',shared=True,base_path=str(self.l2))),
            '--coordinator-url',self.coordinator,'--coordinator-advertise-ip','127.0.0.1',
            '--coordinator-event-reporting','--coordinator-event-flush-interval','0.2',
            '--coordinator-heartbeat-interval','1','--disable-metrics',
            '--trace-level','events','--trace-output',str(self.root/(name+'.lct'))])
        poll(lambda:self.answers(result['http_url']+'/healthcheck'))
        def registered():
            r=CLIENT.get(self.coordinator+'/instances',timeout=2);r.raise_for_status()
            return any(x['instance_id']==name for x in r.json()['instances'])
        poll(registered)
        return result

    def directory(self):
        response=CLIENT.get(self.coordinator+'/directory/keys',params=dict(limit=10000),timeout=5)
        response.raise_for_status()
        return response.json()

    def warm(self,server,model_name,tokens,cache_salt=""):
        started=time.monotonic_ns()
        response=CLIENT.post(self.coordinator+'/cache/prefetches',json=dict(
            instance_id=server['instance_id'],model_name=model_name,world_size=1,token_ids=list(tokens),cache_salt=cache_salt),timeout=10)
        response.raise_for_status();submitted=response.json()
        if not submitted.get('request_id'):
            return dict(started_ns=started,submitted=submitted,completed_ns=time.monotonic_ns())
        def done():
            r=CLIENT.get(self.coordinator+f"/cache/prefetches/{server['instance_id']}/{submitted['request_id']}",timeout=5)
            r.raise_for_status();data=r.json()
            return data if data.get('status')=='completed' else None
        completed=poll(done)
        return dict(started_ns=started,submitted=submitted,completed=completed,completed_ns=time.monotonic_ns())

    def close(self):
        records=[]
        for name,proc,log in reversed(self.processes):
            if proc.poll() is None:
                proc.send_signal(signal.SIGINT)
                try:proc.wait(15)
                except subprocess.TimeoutExpired:
                    proc.terminate()
                    try:proc.wait(5)
                    except subprocess.TimeoutExpired:proc.kill();proc.wait()
            records.append(dict(name=name,pid=proc.pid,returncode=proc.poll()))
            log.close()
        with (self.root/'closed.json').open('x') as f:json.dump(records,f,indent=2)


class Geometry:
    """Register actual tensor geometry with official CPU shared-memory layout registration."""
    def __init__(self,server,model_name,layers,heads,dim,dtype):
        self.buffers={f'layer.{i}':torch.zeros((2,2,heads,CHUNK,dim),dtype=dtype) for i in range(layers)}
        self.zmq=zmq.Context()
        self.zmq.setsockopt(zmq.LINGER,0)
        self.client=RequestClientFactory.create(server['url'],context=self.zmq)
        self.transfer=create_transfer_context(self.buffers,mode='lmcache_driven',instance_id=uuid.uuid4().int%(2**30),req_client=self.client)
        self.transfer.register(self.buffers,model_name,1,1,30,
            layout_hints=dict(kv_layout='HND',num_kv_heads=heads,tokens_per_block=CHUNK,head_dim=dim),
            engine_group_infos=[EngineGroupInfo(engine_group_id=0,layer_indices=tuple(range(layers)),tokens_per_block=CHUNK)],
            engine_type=EngineType.VLLM)
        # VLLM is the declared tensor-layout enum only; no vLLM library/process executes.

    def close(self):
        self.transfer.close()
        if hasattr(self.client,'close'):self.client.close()
        self.zmq.destroy(linger=0)


def placements(data,instance,model_name):
    return [row for row in data['keys'] if row['key']['model_name']==model_name
            and any(p['instance_id']==instance and p['tier']=='l1' for p in row['placements'])]


def main():
    root=BASE/'results'/('native_'+uuid.uuid4().hex[:8])
    services=Services(root)
    resources=[]
    result=dict(completed=False,scope='official CPU LMCache, real filesystem L2, local interprocess; no model yet')
    try:
        source=services.server('source_a1');target=services.server('target_a1')
        model_name='model_a1'
        for server in (source,target):
            geometry=Geometry(server,model_name,2,2,8,torch.float32);resources.append(geometry)
        src=connect(source['url'],source['http_url'],model_name,timeout=30);resources.append(src)
        dst=connect(target['url'],target['http_url'],model_name,timeout=30);resources.append(dst)
        tokens=list(range(32))
        kv=torch.arange(2*2*32*16,dtype=torch.float32).reshape(2,2,32,16)
        result['store_ok']=src.store(kv,tokens)
        poll(lambda:len(list(services.l2.rglob('*.data')))>=2)
        before=services.directory()
        assert not placements(before,target['instance_id'],model_name)
        result['directory_before']=before
        result['l2_files_before']=[dict(name=p.name,nbytes=p.stat().st_size) for p in services.l2.rglob('*.data')]
        result['warm']=services.warm(target,model_name,tokens)
        assert result['warm']['completed']['found_keys']==2
        after=poll(lambda:services.directory() if placements(services.directory(),target['instance_id'],model_name) else None)
        result['directory_after']=after
        retrieved=dst.retrieve(tokens)
        result['retrieved_equal']=bool(torch.equal(kv,retrieved))
        assert result['retrieved_equal']
        result['subchunk_noop']=services.warm(target,model_name,tokens[:5])
        result['unknown_retrieve_is_none']=dst.retrieve(list(range(100,132))) is None
        result['completed']=True
    except Exception as error:
        result['error_type']=type(error).__name__;result['error']=str(error)
        raise
    finally:
        for resource in reversed(resources):
            try:resource.close()
            except Exception as error:result.setdefault('close_errors',[]).append(type(error).__name__+': '+str(error))
        services.close()
        (root/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(dict(result_path=str(root/'result.json'),completed=result['completed']),indent=2),flush=True)


if __name__=='__main__':main()
