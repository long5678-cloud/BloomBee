"""Audited optional GPU retention hints; separate from CPU missing-priority=0."""
import torch
from bloombee.server.dynamo_pages import EnginePagePool


def pool():return EnginePagePool(2,2,2,4,dtype=torch.float32,device="cpu")


def seed(store,name,priority=None):
    s=store.begin();s.append(torch.ones(2,2,4),torch.zeros(2,2,4))
    s.publish([name],None if priority is None else [priority])
    slot=s.block(0)["slot"];s.release();return slot


def test_missing_gpu_hint_preserves_p90_so_pressure_reclaims_q50():
    store=pool();p=seed(store,"P",90);q=seed(store,"Q",50)
    hit=store.begin(["P"])
    assert store.pages[p].priority==90
    hit.publish(["P"],[None]);hit.release()
    assert store.pages[p].priority==90
    r=seed(store,"R")
    assert r==q and "P" in store.lookup and "Q" not in store.lookup
    assert store.pages[r].priority==35  # newly allocated optional fallback only


def test_explicit_claim_updates_low_to_high_and_zero_can_lower_before_publish():
    store=pool();p=seed(store,"P",10);q=seed(store,"Q",50)
    hit=store.begin(["P"],priorities=[100])
    assert store.pages[p].priority==100  # not deferred to publish
    hit.release()
    assert seed(store,"R",35)==q and "P" in store.lookup
    hit=store.begin(["P"],priorities=[0])
    assert store.pages[p].priority==0
    hit.release()
    assert seed(store,"S",100)==p and "P" not in store.lookup


def test_new_page_missing_and_none_use_default_but_repeated_none_does_not_reset():
    store=pool();slot=seed(store,"P")
    assert store.pages[slot].priority==35
    hit=store.begin(["P"],priorities=[80]);hit.publish(["P"],[80]);hit.release()
    hit=store.begin(["P"],priorities=[None]);hit.publish(["P"]);hit.release()
    assert store.pages[slot].priority==80
