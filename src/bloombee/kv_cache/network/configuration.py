"""Offline generation/validation for the fixed 32-layer native deployment.

Input identity, publication, incarnation and terminal files must come from real
services. Generation performs no model call, native transfer or service launch.
"""
import copy
import hashlib
import ipaddress
import json
import re
from pathlib import Path
import sys
from urllib.parse import urlsplit
from .api import plugin_spec
from .lifecycle import boolean
from . import independent_audit as gate


def read(path):
    return json.loads(Path(path).read_text())


def pin(path):
    path=Path(path)
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def endpoint(value, scheme='http'):
    parsed=urlsplit(value)
    if (parsed.scheme!=scheme or not parsed.hostname or not parsed.port or
        parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path not in ('','/')):
        raise ValueError('explicit credential-free '+scheme+' host:port required')
    ipaddress.ip_address(parsed.hostname)
    return parsed


def native_endpoint(value):
    if not isinstance(value.get('instance_id'),str) or not value['instance_id']:
        raise ValueError('native instance_id required')
    endpoint(value['url'],'tcp');endpoint(value['http_url'])
    return value


def page_settings(cfg, templates):
    """Recreate the published execution namespace for this fixed FP16 profile."""
    declared = cfg.get('declared_execution_digest')
    if not isinstance(declared, str) or not re.fullmatch('[0-9a-f]{64}', declared):
        raise ValueError('declared_execution_digest from the source model launch is required')
    expected = gate.digest(dict(declared=declared, context=cfg['context'],
        schema='dynamo_page_block_v1', block_size=16, auxiliary_hidden=4096,
        auxiliary_dtype='torch.float16'))
    if len(templates) != 32 or any(t['execution_digest'] != expected for t in templates):
        raise ValueError('declared_execution_digest and context must reproduce the published identity')

    # Each layer allocates K, V, auxiliary input and auxiliary output.
    bytes_per_page = 4 * 16 * 4096 * 2
    tokens = cfg.get('max_session_tokens', 1)
    if type(tokens) is not int or tokens < 1:
        raise ValueError('max_session_tokens must be a positive integer')
    for key in ('inference_max_length', 'attn_cache_tokens', 'max_batch_size'):
        if key in cfg and (type(cfg[key]) is not int or cfg[key] < tokens):
            raise ValueError(key + ' must cover max_session_tokens')
    if 'max_chunk_size_bytes' in cfg and (type(cfg['max_chunk_size_bytes']) is not int or cfg['max_chunk_size_bytes'] <= 0):
        raise ValueError('max_chunk_size_bytes must be a positive integer')
    pages_needed = (tokens + 15) // 16
    pages = cfg.get('gpu_capacity_blocks', max(17, pages_needed))
    if type(pages) is not int or pages < pages_needed:
        raise ValueError('gpu_capacity_blocks cannot hold the declared maximum session')
    minimum = pages * bytes_per_page
    budget = cfg.get('gpu_budget_bytes', minimum)
    if type(budget) is not int or budget < minimum:
        raise ValueError('gpu_budget_bytes cannot hold the configured pages per layer')
    return {'BLOOMBEE_DYNAMO_GPU_CAPACITY_BLOCKS': str(pages),
            'BLOOMBEE_DYNAMO_GPU_BUDGET_BYTES': str(budget),
            'BLOOMBEE_DYNAMO_REMOTE_EXECUTION_DIGEST': declared}


def validate_deployment(raw):
    cfg=copy.deepcopy(raw)
    required=('run_id','source_run_id','consumer_id','context','source_incarnation',
        'source_native_endpoint','target_native_endpoint','coordinator_url','redis_endpoint',
        'source_page_endpoint','source_holder_peers','page_control_url','model_path','model_dht_prefix',
        'peer','source_model_dht_prefix','request_count','identity_templates_file','source_manifests_file','initial_history_file',
        'history_boundary','clock_domain_file','ingress_clock_domain','source_state_dir','target_state_dir',
        'source_config_dir','target_config_dir')
    for key in required:
        if key not in cfg:raise ValueError('missing deployment field: '+key)
    for key in ('run_id','source_run_id','consumer_id','context','peer','model_dht_prefix','source_model_dht_prefix','ingress_clock_domain'):
        if not isinstance(cfg[key],str) or not cfg[key].strip() or 'REPLACE' in cfg[key].upper():
            raise ValueError('real nonempty '+key+' required')
    if type(cfg['request_count'])is not int or cfg['request_count']<=0:raise ValueError('positive request_count required')
    if type(cfg['source_incarnation'])is not int or cfg['source_incarnation']<0:raise ValueError('actual source incarnation required')
    if cfg['source_model_dht_prefix']==cfg['model_dht_prefix']:
        raise ValueError('source and target ordinary model swarms must be distinct')
    source=native_endpoint(cfg['source_native_endpoint']);target=native_endpoint(cfg['target_native_endpoint'])
    if source['instance_id']==target['instance_id']:raise ValueError('distinct source and target registrations required')
    for key in ('coordinator_url','source_page_endpoint','page_control_url'):endpoint(cfg[key])
    endpoint('redis://'+cfg['redis_endpoint'],'redis')
    if not cfg['source_holder_peers'] or not all(isinstance(x,str) and x for x in cfg['source_holder_peers']):
        raise ValueError('actual source publisher peer identities required')
    if cfg.get('prefetch_strategy','lrfu') not in ('lrfu','transition'):raise ValueError('invalid predictor')
    boolean(cfg,'distributed_telemetry')
    threshold=cfg.get('host_min_priority',0)
    if type(threshold)is not int or not 0<=threshold<=100:raise ValueError('host_min_priority must be an integer in 0..100')
    if cfg.get('retention_policy','dynamo') not in ('dynamo','lru','hotness','decay'):
        raise ValueError('invalid retention policy')
    if cfg.get('history_tier','both') not in ('cpu','gpu','both'):raise ValueError('invalid history tier')
    for key in ('source_state_dir','target_state_dir','source_config_dir','target_config_dir','clock_domain_file','model_path'):
        if not Path(cfg[key]).is_absolute():raise ValueError('deployment path must be absolute: '+key)
    if 'provider_log' in cfg and not Path(cfg['provider_log']).is_absolute():
        raise ValueError('provider_log must be absolute')
    templates=read(cfg['identity_templates_file'])
    manifests=read(cfg['source_manifests_file'])
    if isinstance(manifests,dict):manifests=manifests['manifests']
    if not isinstance(manifests,list) or not manifests:raise ValueError('actual immutable source manifests required')
    history=read(cfg['initial_history_file'])
    if not isinstance(history,list):raise ValueError('full initial history JSON list required')
    boundary=cfg['history_boundary']
    if boundary.get('initial_completion_seq')!=0 or boundary.get('truncated') is not False:
        raise ValueError('native audit requires untruncated history starting at sequence zero')
    if boundary.get('consumer_id')!=cfg['consumer_id']:raise ValueError('history consumer mismatch')
    from .strategy_runner import validate_demand_ledger
    import time
    validate_demand_ledger(history,boundary,time.monotonic_ns())
    for event in history:
        ref=event['terminal_receipt_ref'];raw=Path(ref['path']).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=ref['sha256']:raise ValueError('terminal receipt SHA mismatch')
        proof=json.loads(raw)
        if proof.get('status')!='complete_success' or proof.get('request_id')!=event['request_id'] or proof.get('attempt_id')!=event.get('attempt_id'):
            raise ValueError('terminal receipt owner/outcome mismatch')
        if proof.get('actual_output_tokens')!=proof.get('requested_output_tokens') or type(proof.get('actual_output_tokens'))is not int:
            raise ValueError('full terminal output proof required')
    audit_cfg=dict(page_identity_template_by_layer=templates,source_instance=source['instance_id'],
        source_run_id=cfg['source_run_id'])
    for manifest in manifests:
        _,layout=gate.reconstruct(audit_cfg,manifest['layout']['tokens'])
        gate.manifest_check(manifest,layout,audit_cfg)
    cfg['page_settings'] = page_settings(cfg, templates)
    cfg.update(templates=templates,manifests=manifests,history=history,
               page_model_name=manifests[0]['layout']['model_name'])
    return cfg


def generate_bundle(deployment, output):
    cfg=validate_deployment(deployment)
    output=Path(output).absolute();output.mkdir(parents=True,exist_ok=False)
    target_dir=Path(cfg['target_state_dir']);source_dir=Path(cfg['source_state_dir'])
    target_config=Path(cfg['target_config_dir']);source_config=Path(cfg['source_config_dir'])
    folder=Path(__file__).parent
    worker=dict(run_id=cfg['run_id'],source_run_id=cfg['source_run_id'],
        source_instance=cfg['source_native_endpoint']['instance_id'],native_endpoint=cfg['target_native_endpoint'],
        lmcache_target=cfg['target_native_endpoint'],candidate_consumer_id=cfg['consumer_id'],
        coordinator_url=cfg['coordinator_url'],page_control_url=cfg['page_control_url'],
        page_identity_template_by_layer=cfg['templates'],page_model_name=cfg['page_model_name'],
        prefix_length=16,prefetch_strategy=cfg.get('prefetch_strategy','lrfu'),strategy_transition_plugin=plugin_spec(),
        strategy_attempt_dir=str(target_dir/'attempts_a1'),clock_domain_file=cfg['clock_domain_file'],
        release_helper=pin(folder/'release_l1.py'),audit_source_incarnation=cfg['source_incarnation'],
        native_timeout_s=120,audit_metadata_timeout_s=120,ready_bank_max_bytes=64*1024**2,
        ready_bank_max_prefixes=4,ready_ticket_ttl_s=300,ticket_port=0)
    if cfg.get('ticket_control_token'):
        worker['ticket_control_token']=cfg['ticket_control_token']
    controller=dict(independent_audit='builtin',inbox=str(target_dir/'inbox_a1'),
        worker_config_path=str(target_config/'worker.json'),request_count=cfg['request_count'],
        initial_history=cfg['history'],history_boundary=cfg['history_boundary'],
        source_manifests=str(target_config/'source-manifests.json'),score_metadata=dict(
            segment_id=cfg['run_id'],run_id=cfg['run_id'],active_strategy=worker['prefetch_strategy'],
            source_id=cfg['source_run_id'],registry_sha256=gate.digest(cfg['manifests']),
            clock_domain_id=cfg['history_boundary']['clock_domain_id'],freeze_sha256=gate.digest(cfg['history']),
            raw_qualification_ref='assigned_per_actual_decision'))
    client=dict(model_path=cfg['model_path'],peer=cfg['peer'],consumer_node_id=cfg['consumer_id'],
        distributed_telemetry=boolean(cfg,'distributed_telemetry'),cache_evidence=True,
        prefetch_adapter='builtin',prefetch_controller=controller,provider_log=cfg.get('provider_log',str(target_dir/'provider.log')),
        terminal_ledger=str(target_dir/'history.jsonl'),control_endpoint=cfg['page_control_url'],
        ingress_clock_domain=cfg['ingress_clock_domain'])
    if client['distributed_telemetry']:
        if not cfg.get('telemetry_namespace'):raise ValueError('telemetry_namespace required only when enabled')
        client['telemetry_namespace']=cfg['telemetry_namespace']
    native_source=dict(run_id=cfg['source_run_id'],native_endpoint=cfg['source_native_endpoint'],
        published_manifest_path=str(source_dir/'published.jsonl'),source_pending_max_bytes=64*1024**2)
    template=cfg['templates'][0]
    def server_env(source):
        state=source_dir if source else target_dir
        common=dict(BLOOMBEE_DYNAMO_SERVICE='1',BLOOMBEE_DYNAMO_GPU_ENGINE='pages',
            BLOOMBEE_DYNAMO_CONTEXT=cfg['context'],BLOOMBEE_DYNAMO_BLOCK_SIZE='16',
            BLOOMBEE_RETENTION_POLICY=cfg.get('retention_policy','dynamo'),BLOOMBEE_HISTORY_TIER=cfg.get('history_tier','both'),
            BLOOMBEE_DYNAMO_HOST_MIN_PRIORITY=str(cfg.get('host_min_priority',0)),
            BLOOMBEE_DYNAMO_CAPACITY_BLOCKS=str(cfg.get('cpu_capacity_blocks',16)),
            BLOOMBEE_RETENTION_TERMINALS=str(state/'history.jsonl'),
            BLOOMBEE_DYNAMO_REMOTE_MODE='publish' if source else 'consume',
            BLOOMBEE_DYNAMO_REMOTE_CONTEXT=cfg['context'],BLOOMBEE_DYNAMO_REMOTE_MODEL_ID=template['model_id'],
            BLOOMBEE_DYNAMO_REMOTE_MODEL_REVISION=template['model_revision'],
            BLOOMBEE_DYNAMO_REMOTE_TOKENIZER_DIGEST=template['tokenizer_digest'],
            BLOOMBEE_DYNAMO_REMOTE_ALLOWED_ENDPOINTS=json.dumps([cfg['source_page_endpoint']]),
            BLOOMBEE_DYNAMO_REMOTE_ENDPOINT=cfg['source_page_endpoint'],
            BLOOMBEE_DYNAMO_REMOTE_ALLOWED_HOLDERS=json.dumps(cfg['source_holder_peers']))
        common.update(cfg['page_settings'])
        if source:common['BLOOMBEE_NATIVE_PAGE_CONFIG']=str(source_config/'source-native.json')
        else:
            common.update(BLOOMBEE_PAGE_ALLOW_REMOTE='1',
                BLOOMBEE_PAGE_CONTROL_PORT=str(endpoint(cfg['page_control_url']).port))
        return dict(server_env=common)
    # Official CLI flags copied from the pinned integration's infra_a1.py; no
    # private bootstrap, system changes, asset download or GPU selection here.
    coordinator=endpoint(cfg['coordinator_url']);redis=endpoint('redis://'+cfg['redis_endpoint'],'redis')
    native_python=cfg.get('native_python',sys.executable);model_python=cfg.get('model_python',sys.executable)
    launch={'redis':dict(role='source',argv=[cfg.get('redis_binary','redis-server'),'--bind',redis.hostname,
        '--protected-mode','no','--port',str(redis.port),'--dir',str(source_dir),'--save','','--appendonly','no','--maxmemory','256mb','--maxmemory-policy','noeviction']),
        'coordinator':dict(role='source',argv=[native_python,'-m','lmcache.v1.mp_coordinator','--host',coordinator.hostname,
        '--port',str(coordinator.port),'--chunk-size','16','--health-check-interval','0','--eviction-check-interval','0','--disable-metrics'])}
    for name,registration,state,config_dir in [('source',cfg['source_native_endpoint'],source_dir,source_config),
                                            ('target',cfg['target_native_endpoint'],target_dir,target_config)]:
        tcp=endpoint(registration['url'],'tcp');http=endpoint(registration['http_url'])
        adapter=dict(type='valkey',startup_nodes=cfg['redis_endpoint'],cluster_mode=False,key_prefix=cfg['source_run_id'],
            num_workers=1,max_capacity_gb=0,request_timeout=120,connection_timeout=10)
        launch[name+'_mp']=dict(role=name,argv=[native_python,'-m','lmcache.v1.multiprocess.http_server',
            '--instance-id',registration['instance_id'],'--host',tcp.hostname,'--port',str(tcp.port),
            '--http-host',http.hostname,'--http-port',str(http.port),'--chunk-size','16','--l1-size-gb','0.125',
            '--eviction-policy','LRU','--supported-transfer-mode','auto','--no-l1-use-lazy',
            '--l2-adapter',json.dumps(adapter),'--coordinator-url',cfg['coordinator_url'],
            '--coordinator-advertise-ip',http.hostname,'--coordinator-event-reporting',
            '--coordinator-event-flush-interval','0.2','--coordinator-heartbeat-interval','1','--disable-metrics'])
        launch[name+'_model']=dict(role=name,stdout=str(state/'provider.log'),argv=[model_python,'-m',
            'bloombee.kv_cache.network','server','--config',str(config_dir/(name+'-server.json')),'--',
            cfg['model_path'],'--initial_peers',cfg['peer'],'--dht_prefix',cfg['source_model_dht_prefix'] if name=='source' else cfg['model_dht_prefix'],
            '--block_indices','0:32','--device','cuda:0','--torch_dtype','float16'])
        for option in ('inference_max_length', 'attn_cache_tokens', 'max_batch_size'):
            value = cfg.get(option, cfg.get('max_session_tokens'))
            if value is not None:
                launch[name+'_model']['argv'].extend(['--'+option, str(value)])
        if 'max_chunk_size_bytes' in cfg:
            launch[name+'_model']['argv'].extend(['--max_chunk_size_bytes', str(cfg['max_chunk_size_bytes'])])
    launch['worker']=dict(role='target',env={'CUDA_VISIBLE_DEVICES':''},argv=[native_python,'-m',
        'bloombee.kv_cache.network','worker','--config',str(target_config/'worker.json'),
        '--inbox',str(target_dir/'inbox_a1')])
    files={'worker.json':worker,'client.json':client,'job.json':dict(context=cfg['context'],dht_prefix=cfg['model_dht_prefix']),
        'source-native.json':native_source,'source-server.json':server_env(True),'target-server.json':server_env(False),
        'source-manifests.json':cfg['manifests'],'initial-history.json':cfg['history'],'launch.json':launch,
        'deployment.json':deployment}
    for name,value in files.items():
        with (output/name).open('x') as stream:json.dump(value,stream,indent=2)
    validate_bundle(output)
    return output


def validate_bundle(folder, *, runtime_files=False):
    folder=Path(folder);worker=read(folder/'worker.json');client=read(folder/'client.json')
    native_endpoint(worker['native_endpoint'])
    for key in ('coordinator_url','page_control_url'):endpoint(worker[key])
    if worker['prefetch_strategy'] not in ('lrfu','transition'):raise ValueError('invalid native predictor')
    if worker['source_instance']==worker['native_endpoint']['instance_id']:raise ValueError('source equals target')
    for key in ('release_helper','strategy_transition_plugin'):
        spec=worker[key]
        if hashlib.sha256(Path(spec['path']).read_bytes()).hexdigest()!=spec['sha256']:raise ValueError('package source pin mismatch: '+key)
    if client['prefetch_controller']['independent_audit']!='builtin':raise ValueError('builtin independent audit required by generated profile')
    if client['prefetch_controller']['request_count']<=0:raise ValueError('invalid request count')
    boolean(client,'distributed_telemetry')
    manifests=read(folder/'source-manifests.json')
    for manifest in manifests:
        _,layout=gate.reconstruct(worker,manifest['layout']['tokens']);gate.manifest_check(manifest,layout,worker)
    deployment = read(folder/'deployment.json')
    settings = page_settings(deployment, worker['page_identity_template_by_layer'])
    settings['BLOOMBEE_DYNAMO_REMOTE_CONTEXT'] = deployment['context']
    for role in ('source', 'target'):
        env = read(folder/(role+'-server.json'))['server_env']
        for key, value in settings.items():
            if env.get(key) != value:
                raise ValueError(f'{role} server configuration mismatch: {key}')
    if runtime_files:
        for path in (client['model_path'],client['provider_log'],worker['clock_domain_file']):
            if not Path(path).exists():raise ValueError('execution-node reference unavailable: '+path)
        from .clock_domain import current
        domain=current(worker)
        if domain is None or domain['domain_id']!=client['prefetch_controller']['history_boundary']['clock_domain_id']:
            raise ValueError('execution-node history clock domain mismatch')
        from .strategy_runner import validate_demand_ledger
        import time
        history=client['prefetch_controller']['initial_history']
        validate_demand_ledger(history,client['prefetch_controller']['history_boundary'],time.monotonic_ns())
        for event in history:
            ref=event['terminal_receipt_ref']
            if hashlib.sha256(Path(ref['path']).read_bytes()).hexdigest()!=ref['sha256']:
                raise ValueError('execution-node terminal receipt unavailable or changed')
    return dict(valid=True,scope='offline structural/source-pin validation; no service availability or native/GPU proof',
        audit='builtin',manifests=len(manifests))


def launch_component(folder, component):
    """Explicit local CLI launch; no SSH, device selection or process discovery."""
    import os,subprocess
    folder=Path(folder)
    validate_bundle(folder)
    commands=read(folder/'launch.json')
    if component not in commands:raise ValueError('unknown generated component')
    item=commands[component];deployment=read(folder/'deployment.json')
    state=Path(deployment['source_state_dir'] if item['role']=='source' else deployment['target_state_dir'])
    state.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy();env.update(item.get('env',{}))
    log=Path(item.get('stdout',state/(component+'.log')))
    with log.open('xb') as stream:
        return subprocess.run(item['argv'],env=env,stdout=stream,stderr=subprocess.STDOUT).returncode
