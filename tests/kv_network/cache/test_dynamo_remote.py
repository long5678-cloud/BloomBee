"""Real loopback two-peer DHT/HTTP + ordinary attention component entry.
CPU component evidence only, not two physical nodes or ordinary RPC acceptance.
"""
from dataclasses import replace
import socket

import pytest
import torch
from hivemind import DHT
from test_dynamo_page_service import actual_attention_components
from bloombee.server.dynamo_remote import Settings, RemotePlane, RAMStore, block_identity, wire_identity
from bloombee.kv_cache.transport import unpack_snapshot, pack_snapshot


@pytest.fixture(autouse=True)
def enable_explicit_demand_remote_profile(monkeypatch):
    # These are the older explicit demand-remote tests, not the joint ready-bank
    # profile which intentionally disables on-demand remote recovery.
    from bloombee.server import page_control
    monkeypatch.setattr(page_control, '_remote_allowed', True)


def settings(mode, endpoint, holders=()):
    return Settings(mode=mode,context='immutable-test',model_id='test-model',revision='a'*40,
        execution_digest='b'*64,tokenizer_digest='c'*64,allowed_endpoints=(endpoint,),
        allowed_holders=holders,endpoint=endpoint if mode=='publish' else '',bind_host='127.0.0.1')


@pytest.fixture
def peers():
    nodes=[]
    try:
        nodes.append(DHT(start=True,host_maddrs=['/ip4/127.0.0.1/tcp/0'],use_relay=False,client_mode=False))
        nodes.append(DHT(start=True,initial_peers=nodes[0].get_visible_maddrs(),
            host_maddrs=['/ip4/127.0.0.1/tcp/0'],use_relay=False,client_mode=False))
        assert nodes[0].peer_id != nodes[1].peer_id
        yield nodes
    finally:
        for node in reversed(nodes):node.shutdown()


def test_ram_budget_and_settings():
    store=RAMStore(4,4)
    assert store.put(b'ab')==store.put(b'ab') and store.nbytes==2
    with pytest.raises(ValueError):store.put(b'123')
    with pytest.raises(ValueError):store.put(b'12345')
    with pytest.raises(ValueError):settings('consume','http://127.0.0.1:9999')
    with pytest.raises(ValueError):settings('publish','http://example.com:9999')


def test_two_peers_remote_gate_gpu_and_cpu_provenance(monkeypatch,peers):
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    endpoint=f'http://127.0.0.1:{port}'
    publisher=RemotePlane(peers[0],settings('publish',endpoint))
    holder=peers[0].peer_id.to_base58()
    consumer=RemotePlane(peers[1],settings('consume',endpoint,(holder,)))
    try:
        with actual_attention_components(monkeypatch) as (b,bm,brequest,_,_):
            bm._dynamo_remote_plane=publisher
            ids=list(range(1,9))
            expected,be,_=brequest(ids)
            assert be['remote_publication']['published_blocks']==4
            with actual_attention_components(monkeypatch) as (a,am,request,_,_):
                am._dynamo_remote_plane=consumer
                a.name='different-node.0'
                output,event,_=request(ids,priorities=[50,0,100,100])
                torch.testing.assert_close(output,expected,atol=2e-3,rtol=2e-3)
                assert not event['remote_errors'], event['remote_errors']
                assert event['remote_hit_tokens']==6 and event['gpu_hit_tokens']==event['cpu_hit_tokens']==0
                assert event['remote_records'][0]['cache_identity']==be['remote_publication']['records'][0]['cache_identity']
                assert event['remote_records'][0]['cache_identity']['prefix_length']==2
                assert event['remote_cpu_admitted_tokens']==2
                assert event['remote_cpu_cutoff']==1
                assert len(a._dynamo_service_state.cache.entries)==1
                assert all(r['origin']=='remote_http' for r in event['gpu_block_sources'][:3])
                assert event['gpu_block_sources'][3]['origin']=='local_compute'
                cpu=event['cpu_block_sources'][0]
                assert cpu['cpu_block_digest']==cpu['source_block_digest']
                assert cpu['sha256']==event['remote_records'][0]['sha256']
                _,hot,_=request(ids,priorities=[50,0,100,100])
                assert hot['gpu_hit_tokens']==6 and hot['remote_http_bytes']==hot['remote_lookup_count']==0
                # Actual finite capacity pressure, no direct entry deletion/preseed.
                for start in (100,200,300):request(list(range(start,start+8)),priorities=[100]*4)
                before=publisher.http.sent_bytes
                restored,ce,_=request(ids[:3],priorities=[50,0])
                torch.testing.assert_close(restored,expected[:,:3],atol=2e-3,rtol=2e-3)
                assert ce['cpu_hit_tokens']==2 and ce['remote_http_bytes']==0
                assert ce['gpu_block_sources'][0]['sha256']==cpu['sha256']
                assert ce['gpu_block_sources'][0]['last_transfer']=='cpu_to_gpu'
                assert publisher.http.sent_bytes==before
                # Local GPU range remains pinned; fetch only the missing suffix.
                _,mixed,_=request(ids,priorities=[50,0,100,100])
                assert mixed['gpu_hit_tokens']==2 and mixed['remote_hit_tokens']==4
                assert mixed['remote_read_range']==[2,6]
                assert mixed['remote_cpu_cutoff']==1
                assert len(mixed['remote_records'])==2
                state=a._dynamo_service_state
                identity=block_identity(consumer.settings,a,state,ids,2)
                original_settings=consumer.settings
                record=consumer.io.submit(consumer.directory.lookup,identity).result()[0]
                blob=publisher.store.objects[record.payload_sha256]
                tensors=unpack_snapshot(blob,wire_identity(identity,2),max_bytes=consumer.settings.max_object)
                args=(identity,2,state.hidden,state.input_dtype,tensors['aux_input'][:,0],state.cache._digest)
                bad=list(args);bad[4]=bad[4]+1
                value,rejected=consumer.io.submit(consumer._fetch,*bad).result()
                assert value is None and any('actual layer input differs' in r for r in rejected['rejections'])
                consumer.settings=replace(original_settings,allowed_holders=('unapproved',))
                value,rejected=consumer.io.submit(consumer._fetch,*args).result()
                assert value is None and rejected['http_attempts']==0
                consumer.settings=original_settings
                # Unit fault injection in immutable RAM bytes: real HTTP SHA rejects it.
                publisher.store.objects[record.payload_sha256]=blob[:-1]+bytes([blob[-1]^1])
                try:
                    value,rejected=consumer.io.submit(consumer._fetch,*args).result()
                    assert value is None and rejected['http_attempts']==1
                    assert any('校验和不匹配' in r for r in rejected['rejections'])
                    assert rejected['http_bytes']==len(blob)
                finally:publisher.store.objects[record.payload_sha256]=blob
                # Immutable deployment namespace separates otherwise matching input.
                consumer.settings=replace(consumer.settings,context='other-context')
                _,miss,_=request(list(range(400,408)))
                assert miss['remote_hit_tokens']==0
                assert miss['remote_http_bytes']==0
        events=list(publisher.http.events)
        assert len(events)==7 and all(e['status']=='sent' for e in events)
        assert sum(e['bytes'] for e in events)==publisher.http.sent_bytes
        assert all(e['peer_ip']=='127.0.0.1' for e in events)
    finally:
        consumer.close();publisher.close()


def test_real_transformer_backend_tp_chunks_over_dht_http(monkeypatch,peers,tmp_path):
    from test_dynamo_pages_chunk import build_backend, full_reference
    from bloombee.server.backend import TransformerBackend
    for key,value in dict(SERVICE='1',GPU_ENGINE='pages',CONTEXT='actual-prod',BLOCK_SIZE='2',
        CAPACITY_BLOCKS='32',HOST_MIN_PRIORITY='50',GPU_CAPACITY_BLOCKS='8',GPU_BUDGET_BYTES='16384').items():
        monkeypatch.setenv('BLOOMBEE_DYNAMO_'+key,value)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    endpoint=f'http://127.0.0.1:{port}'
    publisher=RemotePlane(peers[0],settings('publish',endpoint))
    consumer=RemotePlane(peers[1],settings('consume',endpoint,(peers[0].peer_id.to_base58(),)))
    resources=[]
    try:
        dirs=[tmp_path/'b',tmp_path/'a']
        for directory in dirs:directory.mkdir()
        b,bm,module,_,bi=build_backend(dirs[0]);resources.append((bm,bi))
        a,am,_,_,ai=build_backend(dirs[1]);resources.append((am,ai))
        bm._dynamo_remote_plane=publisher;am._dynamo_remote_plane=consumer
        inputs=torch.randn(1,8,16,dtype=torch.float16)*.2
        cfg=dict(context='actual-prod',token_ids=list(range(8)),priorities=[100]*4)
        bi=replace(bi,dynamo_cache=cfg);ai=replace(ai,dynamo_cache=dict(cfg,priorities=[50,0,100,100]))
        for backend in (b,a):
            backend.max_chunk_size_bytes=2*backend._max_shard_num_heads*backend.dtype_bytes*8
        expected=full_reference(module,inputs)
        original=TransformerBackend.inference_step(b,inputs.clone(),torch.tensor([0]),bi)[0]
        output=TransformerBackend.inference_step(a,inputs.clone(),torch.tensor([0]),ai)[0]
        torch.testing.assert_close(output,original,atol=2e-3,rtol=2e-3)
        torch.testing.assert_close(output,expected,atol=2e-3,rtol=2e-3)
        event=a._dynamo_service_events[-1]
        assert not event['remote_errors'], event['remote_errors']
        assert event['remote_hit_tokens']==6 and event['computed_tokens']==2
        assert event['remote_cpu_cutoff']==1 and event['cpu_store_error'] is None
        assert event['cpu_progress_finished'] and event['cpu_put_chunks']==1
        assert all(row['source_block_digest']==row['cpu_block_digest'] for row in event['cpu_block_sources'])
    finally:
        for manager,info in resources:
            manager.cache.force_free(*info.cache_handles)
            with manager.cache.use_cache():pass
            manager._reorder_executor.shutdown(wait=True)
        consumer.close();publisher.close()
