"""Offline source/config/gate checks using explicitly synthetic metadata fixtures."""
import copy,hashlib,json
from pathlib import Path
import pytest
from bloombee.kv_cache.network import independent_audit as audit
from bloombee.kv_cache.network.configuration import generate_bundle,validate_bundle,validate_deployment


def deployment_fixture(tmp_path):
    identity=json.loads((Path(__file__).parent/'fixtures/published_identity.json').read_text())['events'][0]['remote_publication']['records'][0]['cache_identity']
    declared = audit.digest('fixture-declared-execution')
    identity['execution_digest'] = audit.digest(dict(declared=declared, context='fixture_context',
        schema='dynamo_page_block_v1', block_size=16, auxiliary_hidden=4096, auxiliary_dtype='torch.float16'))
    templates=[dict(identity,layer_start=i,layer_end=i+1) for i in range(32)]
    source='source';cfg=dict(page_identity_template_by_layer=templates,source_instance=source,source_run_id='source_a1')
    _,layout=audit.reconstruct(cfg,list(range(16)))
    manifest=dict(format='bb_joint_page_bundle_v1',layout=layout,sha_by_block=[[audit.digest(i) for i in range(32)]],
        wire_bytes_by_block=[[512]*32],native_value_sha256=audit.digest('fixture'),native_tensor_bytes=audit.NATIVE_BYTES,
        source_instance=source,run_id='source_a1',source_readback_exact=True,native_registration=dict(
            model_layers=32,storage_layers=64,endpoint=dict(instance_id=source,http_url='http://127.0.0.1:9011')))
    manifest['object_id']=audit.digest({k:manifest[k] for k in ('format','layout','sha_by_block','native_value_sha256','wire_bytes_by_block')})
    def save(name,value):
        p=tmp_path/name;p.write_text(json.dumps(value));return str(p)
    return dict(declared_execution_digest=declared,run_id='run_a1',source_run_id='source_a1',consumer_id='consumer',context='fixture_context',source_incarnation=7,
        source_native_endpoint=dict(instance_id='source',url='tcp://127.0.0.1:9010',http_url='http://127.0.0.1:9011'),
        target_native_endpoint=dict(instance_id='target',url='tcp://127.0.0.1:9012',http_url='http://127.0.0.1:9013'),
        coordinator_url='http://127.0.0.1:9000',redis_endpoint='127.0.0.1:9001',source_page_endpoint='http://127.0.0.1:9020',
        source_holder_peers=['fixture_peer'],page_control_url='http://127.0.0.1:9021',model_path=str(tmp_path/'model'),
        model_dht_prefix='fixture',source_model_dht_prefix='fixture_source',peer='fixture_target_peer',request_count=3,
        identity_templates_file=save('templates.json',templates),source_manifests_file=save('manifests.json',[manifest]),
        initial_history_file=save('history.json',[]),history_boundary=dict(truncated=False,consumer_id='consumer',
            window_id='fixture',initial_completion_seq=0,clock_domain_id='fixture',
            completion_sequence_semantics='unique_successful_full_request_disclosure'),
        clock_domain_file=str(tmp_path/'clock.json'),ingress_clock_domain='target_fixture_domain',
        source_state_dir=str(tmp_path/'source'),target_state_dir=str(tmp_path/'target'),
        source_config_dir=str(tmp_path/'source_bundle'),target_config_dir=str(tmp_path/'target_bundle'))


def test_generate_complete_bundle_and_static_validate_without_services(tmp_path):
    cfg=deployment_fixture(tmp_path);folder=generate_bundle(cfg,tmp_path/'bundle')
    result=validate_bundle(folder)
    assert result['valid'] and result['audit']=='builtin'
    files={p.name for p in folder.iterdir()}
    assert {'worker.json','client.json','source-native.json','source-server.json','target-server.json',
            'launch.json','job.json','initial-history.json','source-manifests.json'}<=files
    client=json.loads((folder/'client.json').read_text());worker=json.loads((folder/'worker.json').read_text())
    assert client['distributed_telemetry'] is False and 'telemetry_namespace' not in client
    assert client['prefetch_controller']['independent_audit']=='builtin'
    commands=json.loads((folder/'launch.json').read_text())
    assert {'redis','coordinator','source_mp','target_mp','source_model','target_model','worker'}<=commands.keys()
    assert worker['audit_source_incarnation']==7
    assert 'bootstrap' not in json.dumps(commands)
    with pytest.raises(ValueError,match='unavailable'):validate_bundle(folder,runtime_files=True)
    worker['release_helper']['sha256']='0'*64;(folder/'worker.json').write_text(json.dumps(worker))
    with pytest.raises(ValueError,match='pin'):validate_bundle(folder)

@pytest.mark.parametrize('mutation',[lambda c:c.pop('source_incarnation'),
    lambda c:c['history_boundary'].update(truncated=True),
    lambda c:c['history_boundary'].update(initial_completion_seq=2),
    lambda c:c['target_native_endpoint'].update(instance_id='source'),
    lambda c:c.update(coordinator_url='http://user:secret@127.0.0.1:1'),
    lambda c:c.update(distributed_telemetry='false')])
def test_deployment_rejects_incomplete_or_untrusted_fields(tmp_path,mutation):
    cfg=deployment_fixture(tmp_path);mutation(cfg)
    with pytest.raises((ValueError,TypeError)):validate_deployment(cfg)


def test_terminal_receipt_sha_is_checked_before_generation(tmp_path):
    cfg=deployment_fixture(tmp_path)
    receipt=tmp_path/'terminal.json';receipt.write_text(json.dumps(dict(request_id='r',attempt_id='a',status='complete_success',actual_output_tokens=3,requested_output_tokens=3)))
    history=[dict(request_id='r',attempt_id='a',token_ids=list(range(16)),outcome='processed_success',
        consumer_instance_id='consumer',clock_domain_id='fixture',completion_event_seq=1,disclosed_ns=1,
        terminal_receipt_ref=dict(path=str(receipt),sha256='0'*64))]
    Path(cfg['initial_history_file']).write_text(json.dumps(history))
    with pytest.raises(ValueError,match='SHA'):generate_bundle(cfg,tmp_path/'rejected')
    assert not (tmp_path/'rejected').exists()

@pytest.mark.parametrize('selection',['default',None,'bad-sha'])
def test_controller_constructs_builtin_or_fails_closed(tmp_path,selection):
    from bloombee.kv_cache.network.controller import Controller
    cfg=deployment_fixture(tmp_path);cfg['target_config_dir']=str(tmp_path/'bundle')
    folder=generate_bundle(cfg,tmp_path/'bundle');client=json.loads((folder/'client.json').read_text())
    root=tmp_path/'client';root.mkdir()
    if selection=='default':client['prefetch_controller'].pop('independent_audit')
    elif selection is None:client['prefetch_controller']['independent_audit']=None
    else:client['prefetch_controller']['independent_audit']=dict(path=str(Path(audit.__file__)),sha256='0'*64)
    if selection=='bad-sha':
        with pytest.raises(ValueError,match='SHA'):Controller({},client,root)
    else:
        controller=Controller({},client,root)
        try:assert controller.audit is (audit.audit if selection=='default' else None)
        finally:controller.http.close()


@pytest.mark.parametrize('threshold',[0,50,100])
def test_native_configuration_preserves_public_cpu_priority_threshold(tmp_path,threshold):
    source=tmp_path/'metadata';source.mkdir();cfg=deployment_fixture(source)
    cfg['host_min_priority']=threshold
    bundle=generate_bundle(cfg,tmp_path/'bundle')
    for name in ['source','target']:
        env=json.loads((bundle/(name+'-server.json')).read_text())['server_env']
        assert env['BLOOMBEE_DYNAMO_HOST_MIN_PRIORITY']==str(threshold)

@pytest.mark.parametrize('threshold',[-1,101,True,50.0,'50'])
def test_native_configuration_rejects_invalid_cpu_priority_threshold(tmp_path,threshold):
    source=tmp_path/'metadata';source.mkdir();cfg=deployment_fixture(source)
    cfg['host_min_priority']=threshold
    with pytest.raises(ValueError,match='host_min_priority'):generate_bundle(cfg,tmp_path/'bundle')


def test_long_session_and_published_identity(tmp_path):
    cfg = deployment_fixture(tmp_path)
    cfg['max_session_tokens'] = 1179
    folder = generate_bundle(cfg, tmp_path/'bundle')
    template = json.loads(Path(cfg['identity_templates_file']).read_text())[0]
    for role in ('source', 'target'):
        env = json.loads((folder/(role+'-server.json')).read_text())['server_env']
        assert env['BLOOMBEE_DYNAMO_REMOTE_EXECUTION_DIGEST'] == cfg['declared_execution_digest']
        assert env['BLOOMBEE_DYNAMO_REMOTE_EXECUTION_DIGEST'] != template['execution_digest']
        assert env['BLOOMBEE_DYNAMO_GPU_CAPACITY_BLOCKS'] == '74'
        assert int(env['BLOOMBEE_DYNAMO_GPU_BUDGET_BYTES']) == 74 * 524288
    path = folder/'target-server.json'
    target = json.loads(path.read_text())
    target['server_env']['BLOOMBEE_DYNAMO_REMOTE_EXECUTION_DIGEST'] = template['execution_digest']
    path.write_text(json.dumps(target))
    with pytest.raises(ValueError, match='target server configuration mismatch'):
        validate_bundle(folder)


@pytest.mark.parametrize('change,reason', [
    (dict(declared_execution_digest=None), 'declared_execution_digest'),
    (dict(declared_execution_digest='f'*64), 'published identity'),
    (dict(context='other'), 'published identity'),
    (dict(max_session_tokens=1179, gpu_capacity_blocks=73), 'gpu_capacity_blocks'),
    (dict(gpu_capacity_blocks=74, gpu_budget_bytes=74*524288-1), 'gpu_budget_bytes')])
def test_reject_incompatible_deployment(tmp_path, change, reason):
    cfg = deployment_fixture(tmp_path)
    cfg.update(change)
    with pytest.raises(ValueError, match=reason):
        generate_bundle(cfg, tmp_path/'bundle')
    assert not (tmp_path/'bundle').exists()


@pytest.mark.parametrize('tokens',[1179,16384,32768])
def test_complete_request_capacity_reaches_normal_model_cli(tmp_path,tokens):
    cfg=deployment_fixture(tmp_path);cfg.update(max_session_tokens=tokens,max_chunk_size_bytes=8388608)
    bundle=generate_bundle(cfg,tmp_path/'bundle')
    launch=json.loads((bundle/'launch.json').read_text())
    for role in ['source','target']:
        argv=launch[role+'_model']['argv']
        for key in ['inference_max_length','attn_cache_tokens','max_batch_size']:
            assert int(argv[argv.index('--'+key)+1])>=tokens
        assert argv[argv.index('--max_chunk_size_bytes')+1]=='8388608'

@pytest.mark.parametrize('key',['inference_max_length','attn_cache_tokens','max_batch_size'])
def test_request_window_is_not_silently_larger_than_service_limits(tmp_path,key):
    cfg=deployment_fixture(tmp_path);cfg.update(max_session_tokens=1179);cfg[key]=1178
    with pytest.raises(ValueError,match=key):generate_bundle(cfg,tmp_path/'bundle')
