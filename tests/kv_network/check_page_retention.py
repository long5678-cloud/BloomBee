"""Offline CPU tensors through the actual page-pool allocator and original byte Placement."""
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import struct
import sys
from threading import RLock
from types import ModuleType, SimpleNamespace

import torch

OUT = Path(sys.argv[1])
OUT.mkdir(exist_ok=False)
from bloombee.kv_cache import retention_completion as ledger, page_retention as ret, dynamo as cpu, placement as old
from bloombee.server import dynamo_pages as gpu
now = [1000.]
# Isolate test clocks to modules that consume the ledger.
ret.time = ledger.time = SimpleNamespace(monotonic=lambda: now[0])
def group(token):
    layers, data = [], {}
    for i in range(32):
        ident = {"p":token, "layer_start":i, "layer_end":i+1}
        head = json.dumps({"identity":ident}).encode()
        blob = b"KVSLAB01"+struct.pack("!I",len(head))+head
        sha = hashlib.sha256(blob).hexdigest()
        layers.append(old.Layer(json.dumps(ident),sha,len(blob),1e12,"loop"))
        data[sha] = blob
    return old.Group(tuple(layers)),data

groups = {k:group(k) for k in (1,2,3,4,5)}
assert len({g.nbytes for g,_ in groups.values()}) == 1
runs=[]
def run(name, tier, seq, cap=2):
    now[0]=1000.
    file=OUT/f"{name}_{tier}_{len(runs)}.jsonl"
    file.touch()
    os.environ.update(BLOOMBEE_RETENTION_POLICY=name,BLOOMBEE_HISTORY_TIER=tier,
                      BLOOMBEE_RETENTION_TERMINALS=str(file),BLOOMBEE_RETAINED_GPU_BLOCKS=str(cap))
    state=SimpleNamespace(binding="binding",lock=RLock(),remote_cpu_sources={},remote_gpu_sources={},
        cache=cpu.DynamoCache(cap,16,context="binding",expected_layers=2,host_offload_min_priority=50),
        pool=gpu.EnginePagePool(cap+2,16,1,2,dtype=torch.float32,device="cpu"))
    ref=old.Placement(name if name!="dynamo" else "lru",cap*groups[1][0].nbytes,clock=lambda:now[0])
    receipts=[]
    success_seq=0
    for i,(k,ok,dt) in enumerate(seq):
        now[0]+=dt
        ids=[k]*16; attempt=f"req{i}"
        req=ref.begin(groups[k][0],attempt)
        fetch=req.bind_fetcher(lambda endpoint,sha,**kwargs:groups[k][1][sha])
        for obj in groups[k][0].layers:
            fetch("loop",obj.sha256,allowed_endpoints=["loop"])
        cfg=dict(attempt_id=attempt,token_ids=ids)
        plan=ret.begin_prefill(state,cfg)
        key=("binding",tuple(ids))
        session=state.pool.begin([key])
        if not session.length:
            tensor=torch.full((16,1,2),float(k))
            session.append(tensor,tensor+1)
        def reader(index):
            view=session.block(index)
            return {0:{"key":view["key"],"value":view["value"]},1:{"key":view["value"],"value":view["key"]}}
        ret.stage(state,session,cfg,reader)
        flags=plan["cpu"]["action"], plan["gpu"]["action"]
        session.release()
        if flags[0]=="stage": assert key not in state.cache.entries
        if flags[1]=="stage":
            assert key not in state.pool.lookup and any(p.refs==1 for p in state.pool.pages)
        if ok:
            success_seq+=1
            row=dict(request_id=attempt,attempt_id=attempt,status="complete_success",kv_reuse_success=(ok != "cold"),completion_event_seq=success_seq,
                     actual_output_tokens=1,requested_output_tokens=1,terminal_ref=attempt,
                     token_ids=ids,prefix_key=ret.prefix_key(ids),completed_monotonic=now[0])
            with file.open("a") as f:f.write(json.dumps(row)+"\n")
        terminal=dict(attempt_id=attempt,status="complete_success" if ok else "failed",token_ids=ids)
        result=ret.terminal(state,terminal)
        assert ret.terminal(state,terminal)==result
        assert not any(p.refs for p in state.pool.pages)
        req.finish(reuse_success=bool(ok and (ok != "cold" or name == "dynamo")))
        expect={t for t in groups if groups[t][0].key in ref.entries}
        cpu_keys={ids[0] for _,ids in state.cache.entries}
        gpu_keys={ids[0] for _,ids in state.pool.lookup}
        assert cpu_keys==(expect if tier in ("cpu","both") else set()), (name,k,cpu_keys,expect)
        assert gpu_keys==(expect if tier in ("gpu","both") else set()), (name,k,gpu_keys,expect)
        for ck,block in state.cache.entries.items():
            assert block.layers[0]["key"][0,0,0].item()==ck[1][0]
        for gk,slot in state.pool.lookup.items():
            assert state.pool.k[slot,0,0,0].item()==gk[1][0]
        receipts.append(dict(input=k,ok=ok,actions=flags,residents=sorted(expect),
                             receipt=result))
    runs.append(dict(policy=name,tier=tier,comparisons=len(seq),rows=receipts))

seq=[(k,True,1) for k in [1,1,1,2,3,3,3,3,2,4,2,4,2,5,5,1,4,5]]
seq += [(2,False,61),(2,True,1),(4,False,1),(5,True,1)]
for name in ("lru","hotness","decay"):run(name,"both",seq)
for tier in ("cpu","gpu"):run("dynamo",tier,seq)
assert len({json.dumps([r["residents"] for r in x["rows"]]) for x in runs[:3]})==3
(OUT/"result.json").write_text(json.dumps(runs,indent=2))
print(f"{sum(r['comparisons'] for r in runs)} terminal comparisons passed; three distinct trajectories.")
# Four history pages with fifth demand; distinguish active-protection from no space.
for name in ("lru","hotness","decay"):
    run(name,"both",[(k,True,1) for k in [1,2,3,4,1,5,5,5,5,2,1,5,4,3,5]],cap=4)
os.environ.update(BLOOMBEE_RETENTION_POLICY="lru",BLOOMBEE_HISTORY_TIER="both",
                  BLOOMBEE_RETAINED_GPU_BLOCKS="1")
file=OUT/"control_race.jsonl"; file.touch();os.environ["BLOOMBEE_RETENTION_TERMINALS"]=str(file)
state=SimpleNamespace(binding="x",lock=RLock(),remote_cpu_sources={},remote_gpu_sources={},
    cache=cpu.DynamoCache(1,16,context="x",expected_layers=2),
    pool=gpu.EnginePagePool(3,16,1,2,dtype=torch.float32,device="cpu"))
cfg=dict(token_ids=[1]*17,attempt_id="one")
ret.begin_prefill(state,cfg)
s=state.pool.begin()
x=torch.ones((16,1,2));s.append(x,x)
read=lambda idx:{i:{"key":x,"value":x} for i in range(2)}
ret.stage(state,s,cfg,read)
# Concurrent next admission sees the in-flight reservation but cannot access those tensors.
cfg2=dict(token_ids=[2]*17,attempt_id="two")
plan=ret.begin_prefill(state,cfg2)
assert all(plan[t]["action"]=="no_idle_capacity" for t in ("cpu","gpu"))
assert not state.pool.lookup and not state.cache.entries
ret.finish_attempt(state,"two",False)
try:ret.finish_attempt(state,"one",True)
except ValueError:pass
else:raise AssertionError("missing successful ledger must not publish a page")
with file.open("a") as f:f.write(json.dumps(dict(request_id="one",attempt_id="one",status="complete_success",kv_reuse_success=True,
    completion_event_seq=1,actual_output_tokens=1,requested_output_tokens=1,terminal_ref="one",
    token_ids=cfg["token_ids"],prefix_key=ret.prefix_key(cfg["token_ids"]),completed_monotonic=now[0]))+"\n")
ret.finish_attempt(state,"one",True)
assert state.pool.pages[0].refs==1
active=state.cache.get([1]*16);assert active[1]==16
cfg3=dict(token_ids=[3]*17,attempt_id="three")
plan=ret.begin_prefill(state,cfg3)
assert all(plan[t]["action"]=="no_idle_capacity" for t in ("cpu","gpu"))
ret.finish_attempt(state,"three",False)
s.release();state.cache.release()
assert state.pool.pages[0].refs==0
# The work allocator refuses to evict registered history even if it would
# otherwise make room for this oversized working request.
s=state.pool.begin()
try:s.append(torch.ones((48,1,2)),torch.ones((48,1,2)))
except gpu.PageCapacityError:pass
else:raise AssertionError("active request borrowed the retained-history budget")
assert state.pool.matching_tokens([("x",tuple([1]*16))])==16
s.release()
(OUT/"all_checks.json").write_text(json.dumps({"terminal_comparisons":sum(r["comparisons"] for r in runs),
    "cap4":runs[-3:],"pending_hidden":True,"pending_reservations_counted":True,
    "terminal_before_release_keeps_model_ref":True,"active_cpu_gpu_leases_protected":True,
    "work_does_not_displace_history":True},indent=2))
print("Additional 45 four-slot comparisons and active lease / terminal ordering checks passed.")
mix = [(1,True,1),(2,True,1), (3,"cold",1), (4,"cold",1), (3,"cold",1), (4,"cold",1),
       (3,True,1),(2,False,1),(4,"cold",1),(5,True,60)]
for name in ("lru","hotness","decay"):
    run(name,"both",mix)
    for req in runs[-1]["rows"]:
        if req["ok"]=="cold":
            assert req["receipt"]["status"]=="cold_discard"
run("dynamo","cpu",mix)
run("dynamo","gpu",mix)
for req in runs[-1]["rows"]:
    if req["ok"]=="cold":assert req["receipt"]["status"]=="committed"
os.environ["BLOOMBEE_RETENTION_POLICY"]="decay"
def event(k,t,used):
    return dict(prefix_key=str(k),completed_monotonic=t,kv_reuse_success=used)
view=dict(events=[event(1,100,True)]+[event(2,101,False)]*4+[event(2,102,True)]*4)
scores,last=ret._scores(view,103)
assert scores["1"]==0.5 and last=={"1":1,"2":5}
assert "1" not in ret._scores(view,160)[0]
(OUT/"cold_gate.json").write_text(json.dumps({"cold_matrix":runs[-5:],"cold_steps_do_not_decay":True,
    "four_actual_reuses_halve":True,"age_at_60_excluded":True,"terminal_comparisons":sum(r["comparisons"] for r in runs)},indent=2))
print("50 mixed cold/reuse/failed comparisons passed; decay ignores four cold completes, halves on four reuses.")
