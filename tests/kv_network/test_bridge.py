"""Authentic recorded identities + finite measurement tensors; no original model KV."""
import importlib.util,json,hashlib,os,sys,time,multiprocessing as mp
from pathlib import Path
ROOT=Path(__file__).resolve().parent
import torch
from bloombee.kv_cache import page_bundle as codec
from bloombee.kv_cache.identity import CacheIdentity
from bloombee.kv_cache.transport import pack_snapshot
from bloombee.kv_cache.network.bundle_contract import *
from bloombee.kv_cache.network.ready_bank import ReadyBank,TicketHTTPServer,LocalTicketClient
from bloombee.kv_cache.network.native_page_bridge import NativePageBundleBridge

def wire_identity(identity,block_size):
    return dict(cache_identity=identity.to_dict(),schema='dynamo_page_block_v1',
                block_start=identity.prefix_length-block_size,block_end=identity.prefix_length)

def reject(fn):
    try:fn()
    except (ValueError,KeyError):return
    raise AssertionError('expected rejection')

def serve(pipe,manifest,blobs):
    bank=ReadyBank();ticket=bank.offer(manifest,blobs,dict(network_bytes=None,network_attempts=None,network_receipt=None),{})
    server=TicketHTTPServer(('127.0.0.1',0),bank,lambda *a:dict(status='physical_release_not_tested'))
    import threading
    thread=threading.Thread(target=server.serve_forever);thread.start();pipe.send((server.server_port,os.getpid()))
    pipe.recv();server.shutdown();thread.join();server.server_close();pipe.send(bank.accounting())

def test_cpu_bridge():
    torch.set_num_threads(1)
    source=ROOT/'fixtures'
    tokens=json.loads((source/'inputs.json').read_text())['rows']['P_source']['ids'][:16]
    events=json.loads((source/'published_identity.json').read_text())['events'];events.sort(key=lambda e:e['remote_publication']['records'][0]['cache_identity']['layer_start'])
    ids=[[CacheIdentity.from_dict(e['remote_publication']['records'][0]['cache_identity']) for e in events]]
    blobs=[[]]
    for layer,identity in enumerate(ids[0]):
        values=(torch.arange(4*65536,dtype=torch.float32).reshape(4,65536)%97/128+layer/64).half()
        tensors=dict(key=values[0].reshape(1,32,16,128),value=values[1].reshape(1,32,16,128),aux_input=values[2].reshape(1,1,16,4096),aux_output=values[3].reshape(1,1,16,4096))
        blobs[0].append(pack_snapshot(wire_identity(identity,16),tensors))
    hashes=[[sha(b) for b in blobs[0]]];layout=codec.PageBundleLayout(ids,tokens);native=layout.encode(blobs,hashes)
    assert layout.decode(native,hashes)==blobs
    manifest=make_manifest(layout,blobs,hashes,native,source_attempt_id='cpu_measurement',run_id='cpu_contract',source_instance='cpu_source')
    transfer=dict(network_bytes=None,network_attempts=None,network_receipt=None)
    h=ticket_header(manifest,'cpu_ticket',time.monotonic()+300,transfer,{})
    frame=b''.join(frame_parts(h,blobs));got,decoded=unpack_ticket_frame(frame);assert decoded==blobs
    bad=dict(h,cache_salt='different');reject(lambda:unpack_ticket_frame(b''.join(frame_parts(bad,blobs))))
    reject(lambda:unpack_ticket_frame(frame+b'x'));reject(lambda:unpack_ticket_frame(frame[:-1]))
    reject(lambda:ReadyBank(max_bytes=1).offer(manifest,blobs,transfer,{}))
    reject(lambda:ReadyBank().offer(manifest,[],transfer,{}))
    # Metadata-only defer path: no native endpoint or retirement network mocked as success.
    bridge=object.__new__(NativePageBundleBridge);bridge.pending_retirements={};bridge.retirement_lock=__import__('threading').Lock()
    bank=ReadyBank(ticket_ttl_s=.01);bank.retire_notify=bridge.defer_retirement
    t=bank.offer(manifest,blobs,transfer,{},native_warm_receipt={'not_physical':True},expected_key={'not_physical':True});epoch=bank.epoch;inv=bank.inventory();assert inv['bank_epoch']==epoch and inv['objects'][0]['validated_complete_32layer_sha'] and inv['objects'][0]['manifest_sha256']==digest(manifest)
    assert bank.epoch==epoch and t['state']=='ready'
    t['expires_at']=time.monotonic()-1
    assert bank.inventory()['objects'][0]['expired'] and not bank.inventory()['objects'][0]['claimable'] and bank.epoch==epoch
    assert not bank.has_object(manifest['object_id']) and bank.accounting()['payload_bytes']==0
    assert len(bridge.pending_retirements)==1 and 'blobs' not in next(iter(bridge.pending_retirements.values()))['ticket']
    # Negative control only: corrupt mocked SDK read must preserve unknown
    # staging and transfer evidence. No mocked success is a physical receipt.
    from unittest.mock import Mock,patch
    source_manifest=dict(manifest,source_readback_exact=True,native_registration=dict(storage_layers=64))
    fake=object.__new__(NativePageBundleBridge);fake.config=dict(run_id='cpu_contract',source_instance='cpu_source',native_endpoint=dict(instance_id='cpu_target'),coordinator_url='http://127.0.0.1:1');fake.bank=ReadyBank();fake.emit=lambda *a,**k:None;fake.probe=None;fake.native=Mock();fake.native.sdk.return_value.retrieve.return_value=torch.zeros_like(native);fake.locks={};fake.lock=__import__('threading').Lock();fake.retirement_lock=__import__('threading').Lock();fake.pending_retirements={};fake.release_fn=lambda *a,**k:dict(status='physical_release_not_tested')
    key=dict(model_name=layout.model_name,cache_salt=layout.cache_salt,kv_rank=16777472,object_group_id=0)
    warm=dict(request_id='negative_control',status={},after=dict(registration={},chunks=[dict(incarnation='negative_control')]))
    import types
    negative_native_warm=types.ModuleType('native_warm');negative_native_warm.warm_registered=Mock(return_value=warm)
    with patch.dict(sys.modules,{'bloombee.kv_cache.network.native_warm':negative_native_warm}),patch('httpx.Client') as client_class:
        client_class.return_value.__enter__.return_value.post.return_value.json.return_value=dict(chunks=1,results=[dict(key=key)])
        failed=fake.warm_to_bank(layout,source_manifest,{})
    assert failed['status']=='warm_failed_not_ready' and failed['transfer']['network_bytes'] is None and len(fake.pending_retirements)==1
    assert fake.bank.accounting()['payload_bytes']==0
    # Filter negative controls with fixed simulated metadata responses. No
    # physical E/C claim: independent audit is supplied by its separate owner.
    from bloombee.kv_cache.network.joint_strategy import PageCandidateFilter
    config=dict(run_id='cpu_contract',source_instance='cpu_source',native_endpoint=dict(instance_id='cpu_target'),coordinator_url='http://127.0.0.1:1')
    event=dict(request_id='past_cpu',token_ids=tokens)
    placement=dict(instance_id='cpu_source',tier='l2',backend='valkey',size_bytes=layout.tensor_bytes,incarnation='cpu_control')
    response=dict(chunks=1,results=[dict(key=key,placements=[placement])])
    negative_history=types.ModuleType('history_candidates');negative_history.stream_state=lambda *a,**k:('fresh',{})
    with patch.dict(sys.modules,{'bloombee.kv_cache.network.history_candidates':negative_history}),patch('bloombee.kv_cache.network.joint_strategy.layout_for_tokens',return_value=layout),patch('httpx.Client') as client_class:
        client=client_class.return_value.__enter__.return_value;client.get.return_value.json.return_value={};client.post.return_value.json.side_effect=lambda:response
        filtering=PageCandidateFilter(config,ReadyBank(),lambda *args:dict(known=True,complete=False,demand_inflight=False))
        missing=filtering.filter([event]);assert missing['unknown'] and missing['decisions'][0]['reason']=='remote_bundle_missing_disclosed_manifest'
        response['results'][0]['placements']=[]
        absent=filtering.filter([event]);assert not absent['unknown'] and not absent['candidates']
        response['results'][0]['placements']=[placement]
        full=dict(event,source_bundle_manifest=manifest);qualified=filtering.filter([full]);assert len(qualified['candidates'])==1 and qualified['bank_inventory_before']['objects']==qualified['bank_inventory_after']['objects']
        filtering.local_page_probe=lambda *args:dict(known=True,complete=False,demand_inflight=True)
        assert len(filtering.filter([full])['candidates'])==1
        inflight=filtering.filter([full],exclude_inflight=True);assert not inflight['candidates'] and inflight['decisions'][0]['reason']=='target_demand_inflight'
        filtering.bank.offer(manifest,blobs,transfer,{})
        held=filtering.bank.claim(attempt_id='arrived_cpu',page_identity_digest=digest(ids[0][0].to_dict()))
        assert filtering.filter([full])['unknown']
    # Actual serialization over A-local HTTP between separate CPU processes.
    ctx=mp.get_context('fork');parent,child=ctx.Pipe();process=ctx.Process(target=serve,args=(child,manifest,blobs),daemon=True);process.start();port,pid=parent.recv()
    assert pid!=os.getpid();client=LocalTicketClient('http://127.0.0.1:'+str(port))
    packet=client.claim_for_layout('arrived_cpu_attempt',layout);assert packet and packet[1]==blobs
    assert client.claim_for_layout('another_cpu_attempt',layout) is None
    ack=client.ack(packet[0]['ticket_id'],'arrived_cpu_attempt',False);assert ack['status']=='not_ready'
    client.terminal('arrived_cpu_attempt','failed');client.close();parent.send('close');accounting=parent.recv();process.join(10);assert process.exitcode==0 and accounting['payload_bytes']==0
    from bloombee.server.page_bundle_reader import RequestBundleRegistry
    registry=RequestBundleRegistry(max_bytes=64*1024**2,clock_domain='cpu_measurement')
    prep=registry.accept_ready_payload(page_ready_payload(packet[0],packet[1],attempt_id='cpu_prepare',runtime_pid=os.getpid(),clock_domain='cpu_measurement'))
    assert prep['prepared'] and not prep['page_ready'] and 'cpu_prepare' not in registry.active
    reject(lambda:registry.accept_ready_payload(dict(version=1,phase='activate',attempt_id='cpu_prepare',runtime_pid=os.getpid(),clock_domain='cpu_measurement',selection_id=packet[0]['selection_id'],object_id=manifest['object_id'],release_result=dict(status='unverified'))))
    assert registry.terminal(None,dict(attempt_id='cpu_prepare',status='failed'))['closed']
    result=dict(status='passed',source='authentic recorded identities with finite deterministic measurement tensors; not original model KV',original_wire_objects=32,native_shape=list(native.shape),wire_bytes=sum(map(len,blobs[0])),local_http_body_bytes=packet[0]['local_handoff_http_body_bytes'],sender_pid=pid,receiver_pid=os.getpid(),negative_checks=['namespace','trailing_bytes','truncated_body','bank_budget','missing_entire_wire_grid','pure_read_inventory_full_manifest_binding','expired_pending_metadata_only','corrupt_mock_SDK_preserves_pending_unknown_cost','duplicate_claim','unverified_release_cannot_activate','missing_manifest_queries_real_key_control','same_bank_snapshot_control','inflight_action_only_control','claimed_bank_unknown_control'],reader_prepare_not_visible=True,post_terminal_bank=accounting,native_sdk_tested=False,physical_release_tested=False,B_to_A_tested=False,model_execution_tested=False)
    assert result['status']=='passed'
