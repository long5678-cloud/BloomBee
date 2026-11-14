"""Persistent metadata-only ranker and separate serial official-warm executor.

Inbox supplies only disclosed terminal history/config/provenance. No trace,
queue, ingress or source-private-log reader exists in this worker.
"""
import argparse,concurrent.futures,fcntl,hashlib,importlib.util,json,os,re,sys,time,uuid
from pathlib import Path

PLUGIN_SHA='09c5f813cbfe2d12a3aca9ebc969727cae21c24971d6d8446bfad756b7a32cf5'
SHADOW_SHA='262f6fa7a008537994747ed02038b716c010051c954c275acfda54d77bdbaba3'
COMMON_FILES={'strategy_runner.py': '4718341319bcdccd5841569fcbb51ca7e96a143d495a2fc906a7279d51c02a4a', 'history_candidates.py': '94f26617f8bb90718f5b6b5c201f22a9d0dc71c17962978aa05a368746009880', 'snapshot_bridge.py': '20dccb437cc6b6628b92a6088da22255764668146cf34622e8323575a959c242', 'make_provider_config.py': 'acb5ab2a39f0fdbb803e3e8be396f51be507662a82002dfebd5a56822a85a980', 'strategy_contract.py': 'f7f50525e9f09612ccc8907f9256e556296b44b9d03356c03769b4b42c75938f', 'provider_runtime.py': '0357b63e3914abadd30ae7475721d169b3ff4b4dc25d4f7e271652d02417d3b6', 'native_probe.py': '4e9ec71122799bf4c3e5a84dec14f89c808b74ed331b18bf1027f1f03f4ad10e', 'warm_selected.py': 'bbcf610900594bd967a79f4ea4d7c66740d04c88c9955bd46307e0815b9ffa9d', 'native_warm.py': '210e60223af8a90ff49dbcdc30eef4c7d5ae03472d57fae899cb970bb5f296a6', 'lrfu_policy.py': 'c76430a525f92b125164d47400e28f5a7813bc3404d2b8407126f40e7f9e434b', 'clock_domain.py': '7f7597c45a06379c348a4c30efbb563f96894f6afc556daf32d83a7a90ad914e', 'joint_strategy.py': '083292ad2a813c79ef840c56b8427c29f9f45f925f176a9fb9d6728dd2633545', 'native_page_bridge.py': '631e0c1f10334944a0e4890f88881f39ae75177805d07cc0dd48a1443ac9005f', 'bundle_contract.py': 'da84008639c6214e7bdb16c46f83968159df6ce4e953c345421ffd6fd709b3f2', 'release_l1.py': 'ea39d14685e5354cf25dd22779f9948c1ae41af0940f65b1b70540753823da5d', 'ready_bank.py': '2988b06560321488fa0dc512cb518efd9d36c621c34bbb4b8520da299ddbbfb2', 'joint_runtime.py': '18e5614eaf1871b0444b88b5044146eae55bbcdbba36101f5a33c4df47ee1a50', 'independent_audit.py': '2ebfa9747c0a8970100c3fca68c38417149e0f2a8c1516a1109675e22f705d0b', 'lifecycle.py': '26913d914cff0ea6e539ce6009825ebcf1cc4b84356c13e632fc1e984180539f', 'configuration.py': '637cfb1b89687a1fc2dcdf32472e9628c08ef1243d54cf10c592bbaacae8c3fe'}
COMMON_FILES['controller.py'] = 'a6c9c53e9a278ec7ab7c9f72b3389de2eb71191d420354a622f3b11eb9336cef'
COMMON_FILES['release_ack.py'] = '931e38a5e6540df61523b5f6cf970e6aaf1528aba664e84abaa7f18b92003925'
META_ALLOWED={'segment_id','run_id','active_strategy','source_id','registry_sha256','completed_trace_rows','clock_domain_id','freeze_sha256','raw_qualification_ref'}
META_REQUIRED=META_ALLOWED-{'completed_trace_rows'}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,sort_keys=True,ensure_ascii=False,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    return sha(path)
def sync(path):
    with Path(path).open('a') as stream:stream.flush();os.fsync(stream.fileno())
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


class Worker:
    def __init__(self,args):
        self.args=args;self.inbox=Path(args.inbox).resolve();self.inbox.mkdir(parents=True,exist_ok=True)
        self.config=read(args.config);self.original_config=json.loads(json.dumps(self.config));self.config_sha=sha(args.config);self.jobs={};self.execution_closed=False
        common=Path(args.common_code).resolve()
        for name,expected in COMMON_FILES.items():
            if sha(common/name)!=expected:raise ValueError('common a3 file pin mismatch: '+name)
        sys.path.insert(0,str(common));sys.dont_write_bytecode=True
        from bloombee.kv_cache.network.strategy_contract import digest
        from bloombee.kv_cache.network.strategy_runner import StrategyRunner,build_context,validate_demand_ledger
        self.digest=digest;self.build_context=build_context;self.validate_ledger=validate_demand_ledger
        self.plugin=read(args.plugin)
        if self.plugin.get('sha256')!=PLUGIN_SHA or sha(self.plugin['path'])!=PLUGIN_SHA:raise ValueError('transition plugin pin mismatch')
        if self.config.get('prefetch_strategy') not in ('lrfu','transition'):raise ValueError('invalid active arm')
        configured=self.config.get('strategy_transition_plugin')
        if configured is not None and configured!=self.plugin:raise ValueError('config/CLI plugin mismatch')
        self.config['strategy_transition_plugin']=self.plugin
        capture=Path(args.shadow_common).resolve()
        if capture.is_dir():capture=capture/'capture_shadow.py'
        if sha(capture)!=SHADOW_SHA:raise ValueError('33 capture_shadow SHA mismatch')
        self.capture=load('ranker_shadow_capture',capture).capture
        self.runner=StrategyRunner(self.config,self.emit)
        self.pool=concurrent.futures.ThreadPoolExecutor(max_workers=1,thread_name_prefix='official-warm')
        self.lock_file=(self.inbox/'worker_lock.jsonl').open('a')
        fcntl.flock(self.lock_file.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        self.lock_file.write(json.dumps(dict(pid=os.getpid(),started_ns=time.monotonic_ns(),config_sha256=self.config_sha))+'\n');self.lock_file.flush()
        ready_path=self.inbox/'worker_ready.json'
        if ready_path.exists():ready_path=self.inbox/('worker_ready_'+uuid.uuid4().hex[:8]+'.json')
        write(ready_path,dict(schema_version=1,status='ready',pid=os.getpid(),config_sha256=self.config_sha,plugin_sha256=PLUGIN_SHA,shadow_sha256=SHADOW_SHA,common_source_pins=COMMON_FILES,ready_ns=time.monotonic_ns(),scope='CPU protocol/code readiness; not backend or model health proof'))
        self.emit('worker_started',ready_ref=ready_path.name,ready_sha256=sha(ready_path),pid=os.getpid(),config_sha256=self.config_sha,plugin_sha256=PLUGIN_SHA,shadow_sha256=SHADOW_SHA,common_code=str(common))
    def emit(self,event,**data):
        line=json.dumps(dict(event=event,worker_ns=time.monotonic_ns(),**data),sort_keys=True,ensure_ascii=False)+'\n'
        # One append write per event; the scoring thread and warm thread keep
        # immutable per-decision output files separate.
        with (self.inbox/'worker_events.jsonl').open('a') as stream:stream.write(line);stream.flush()
    def score(self,directory):
        request_path=directory/'score_request.json';ready=read(directory/'score_ready.json')
        if ready.get('score_request_sha256')!=sha(request_path):raise ValueError('ready/request bytes SHA mismatch')
        request=read(request_path)
        allowed={'history','boundary','decision_id','causal_cutoff_ns','metadata','config_ref','config'}
        if set(request)-allowed:raise ValueError('score request contains non-history input')
        if request.get('config_ref') is not None and Path(request['config_ref']).resolve()!=Path(self.args.config).resolve():raise ValueError('dynamic config_ref is forbidden')
        if request.get('config') is not None and self.digest(request['config'])!=self.digest(self.original_config):raise ValueError('inline config differs from frozen config')
        history=request['history'];boundary=request['boundary'];metadata=dict(request['metadata'])
        if not isinstance(history,list):raise ValueError('history must be the complete terminal JSON list')
        self.validate_ledger(history,boundary,request['causal_cutoff_ns'])
        if boundary['consumer_id']!=self.config.get('candidate_consumer_id',self.config['lmcache_target']['instance_id']):raise ValueError('consumer/config mismatch')
        if any(len(event.get('token_ids',[]))<int(self.config.get('prefix_length',16)) for event in history):raise ValueError('unrepresentable_short_terminal_demand')
        if not META_REQUIRED<=metadata.keys() or set(metadata)-META_ALLOWED:raise ValueError('shadow provenance missing or future metadata supplied')
        if metadata['active_strategy']!=self.config['prefetch_strategy'] or metadata['clock_domain_id']!=boundary['clock_domain_id']:raise ValueError('shadow arm/clock mismatch')
        if request['causal_cutoff_ns']>time.monotonic_ns():raise ValueError('future cutoff on decision host clock')
        if any(job.get('decision_id')==request['decision_id'] for job in self.jobs.values()):raise ValueError('duplicate decision id')
        # Exactly ONE initial qualification/context. Capture itself calls both
        # score_context options on deep copies of this same semantic context.
        report=self.runner.qualify(history)
        write(directory/'qualification.json',report)
        context=self.build_context(self.config,report,history,boundary,request['decision_id'],request['causal_cutoff_ns'])
        write(directory/'context.json',context)
        metadata['raw_qualification_ref']=str(directory/'qualification.json')
        with (directory/'shadow.json').open('x'):pass
        shadow=self.capture(context,self.plugin,metadata,directory/'shadow.json',self.args.common_code)
        sync(directory/'shadow.json')
        selection=shadow['shadow_scores'][self.config['prefetch_strategy']]
        result=dict(schema_version=1,status='scored',decision_id=request['decision_id'],active_strategy=self.config['prefetch_strategy'],selection=selection,context_sha256=self.digest(context),selection_sha256=self.digest(selection),qualification_sha256=sha(directory/'qualification.json'),shadow_sha256=sha(directory/'shadow.json'),warm_submitted=False)
        write(directory/'score_result.json',result)
        files={name:sha(directory/name) for name in ('qualification.json','context.json','shadow.json','score_result.json')}
        write(directory/'score_done.json',dict(schema_version=1,status='scored',config_sha256=self.config_sha,decision_id=request['decision_id'],score_request_sha256=ready['score_request_sha256'],files=files,context_sha256=result['context_sha256'],selection_sha256=result['selection_sha256'],done_ns=time.monotonic_ns()))
        self.emit('score_done',directory=directory.name,decision_id=request['decision_id'],context_sha256=result['context_sha256'],selection_sha256=result['selection_sha256'])
        return dict(decision_id=request['decision_id'],request=request,context=context,result=result,execute_handled=False,future=None)
    def recover(self,directory):
        marker=read(directory/'score_done.json')
        if marker['status']!='scored':return dict(decision_id=marker.get('decision_id'),execute_handled=True,future=None)
        if marker.get('config_sha256')!=self.config_sha:raise ValueError('recovery frozen config mismatch')
        if set(marker['files'])!={'qualification.json','context.json','shadow.json','score_result.json'}:raise ValueError('recovery output names mismatch')
        if sha(directory/'score_request.json')!=marker['score_request_sha256']:raise ValueError('recovered request SHA mismatch')
        for name,value in marker['files'].items():
            if sha(directory/name)!=value:raise ValueError('recovered output SHA mismatch')
        if (directory/'execute_done.json').exists() and read(directory/'execute_done.json').get('status')=='accepted_async' and not (directory/'warm_done.json').exists():
            status='recovered_result' if (directory/'warm_result.json').exists() else 'interrupted_unknown_no_retry'
            write(directory/'warm_done.json',dict(status=status,native_retry=False,recovered_ns=time.monotonic_ns()))
        return dict(decision_id=marker['decision_id'],request=read(directory/'score_request.json'),context=read(directory/'context.json'),result=read(directory/'score_result.json'),execute_handled=(directory/'execute_done.json').exists(),future=None)
    def score_error(self,directory,error):
        try:decision_id=read(directory/'score_request.json').get('decision_id')
        except (OSError,ValueError):decision_id=None
        value=dict(schema_version=1,decision_id=decision_id,status='not_scored',error_type=type(error).__name__,reason=str(error),C_status='not_inferred_empty',warm_attempts=0)
        for name in ('qualification.json','context.json','shadow.json','score_result.json'):
            if not (directory/name).exists():write(directory/name,value)
        write(directory/'score_done.json',dict(value,done_ns=time.monotonic_ns()))
        self.emit('score_rejected',directory=directory.name,error=str(error))
        return dict(decision_id=decision_id,execute_handled=True,future=None)
    def warm(self,directory,job):
        try:
            result=self.runner.warm_selected(job['context'],job['result']['selection'],job['request']['history'])
            write(directory/'warm_result.json',result)
            write(directory/'warm_done.json',dict(status='returned',warm_result_sha256=sha(directory/'warm_result.json'),warm_attempts=result.get('warm_attempts',0),done_ns=time.monotonic_ns()))
        except Exception as error:
            if not (directory/'warm_result.json').exists():write(directory/'warm_result.json',dict(status='failed',error_type=type(error).__name__,error=str(error)))
            if not (directory/'warm_done.json').exists():write(directory/'warm_done.json',dict(status='failed',warm_result_sha256=sha(directory/'warm_result.json'),done_ns=time.monotonic_ns()))
            self.emit('warm_worker_failed',decision_id=job['decision_id'],error=str(error))
    def execute(self,directory,job):
        if job['execute_handled'] or not (directory/'execute.json').exists():return
        try:
            command=read(directory/'execute.json');result=job['result']
            if set(command)-{'decision_id','context_sha256','selection_sha256','ec_match','ec_audit_sha256','audit_ref'}:raise ValueError('execute contains non-gate input')
            if command.get('decision_id')!=job['decision_id'] or command.get('context_sha256')!=result['context_sha256'] or command.get('selection_sha256')!=result['selection_sha256']:raise ValueError('execute context/selection SHA or decision mismatch')
            if command.get('ec_match') is not True:raise ValueError('independent E/C was not known equal')
            audit_path=Path(command.get('audit_ref',''))
            if audit_path.resolve()!=(directory/'controller_independent_audit.json').resolve():raise ValueError('execute audit reference is outside this decision')
            if sha(audit_path)!=command.get('ec_audit_sha256'):raise ValueError('execute independent audit SHA mismatch')
            audited=read(audit_path)
            if audited.get('ec_match') is not True or audited.get('unknown')!=[] or audited.get('context_sha256')!=result['context_sha256']:
                raise ValueError('execute independent audit does not authorize this context')
            # Gate marker is committed before submitting one serial executor job.
            # warm_selected also has a persistent one-attempt gate at native POST.
            if (directory/'warm_done.json').exists():status='already_completed'
            elif result['selection']['selected_key'] is None:status='abstained_no_warm'
            else:status='accepted_async'
            write(directory/'execute_done.json',dict(status=status,decision_id=job['decision_id'],execute_sha256=sha(directory/'execute.json'),accepted_ns=time.monotonic_ns()))
            job['execute_handled']=True
            if status=='accepted_async':job['future']=self.pool.submit(self.warm,directory,job)
            self.emit('execute_gate',decision_id=job['decision_id'],status=status)
        except Exception as error:
            job['execute_handled']=True
            if not (directory/'execute_done.json').exists():write(directory/'execute_done.json',dict(status='rejected',error=str(error)))
            self.emit('execute_rejected',directory=directory.name,error=str(error))
    def run(self):
        if (self.inbox/'worker_done.json').exists():
            self.pool.shutdown(wait=True);self.runner.close();self.lock_file.close();return
        try:
            while True:
                directories=sorted((p for p in self.inbox.iterdir() if p.is_dir() and re.fullmatch(r'run_\d+',p.name)),key=lambda p:int(p.name.split('_')[1]))
                for directory in directories:
                    if directory.name not in self.jobs and (directory/'score_ready.json').exists():
                        try:self.jobs[directory.name]=self.recover(directory) if (directory/'score_done.json').exists() else self.score(directory)
                        except Exception as error:
                            if (directory/'score_done.json').exists():raise
                            self.jobs[directory.name]=self.score_error(directory,error)
                # Execute messages are handled independently of model dispatch;
                # score_done never waits for this executor or warm completion.
                for name,job in self.jobs.items():self.execute(self.inbox/name,job)
                if (self.inbox/'finished.json').exists():
                    closed=read(self.inbox/'finished.json');expected=closed.get('decision_ids')
                    if expected is not None and set(expected)!={job['decision_id'] for job in self.jobs.values()}:
                        raise ValueError('finished decision list differs from scored inbox')
                    for name,job in self.jobs.items():
                        if not job['execute_handled']:
                            reason=closed.get('no_execute_reasons',{}).get(job['decision_id'],'input_closed_without_execute')
                            write(self.inbox/name/'execute_done.json',dict(status='not_executed',reason=reason,closed_ns=time.monotonic_ns()));job['execute_handled']=True
                    self.execution_closed=True
                if self.execution_closed and all(job['future'] is None or job['future'].done() for job in self.jobs.values()):break
                time.sleep(.05)
            self.pool.shutdown(wait=True);self.runner.close()
            write(self.inbox/'worker_done.json',dict(schema_version=1,status='drained',decisions=len(self.jobs),warm_jobs=sum(job['future'] is not None for job in self.jobs.values()),interrupted_unknown=sum((self.inbox/name/'warm_done.json').exists() and read(self.inbox/name/'warm_done.json').get('status')=='interrupted_unknown_no_retry' for name in self.jobs),done_ns=time.monotonic_ns()))
        finally:
            self.pool.shutdown(wait=True);self.runner.close();self.lock_file.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('config','inbox','plugin','shadow-common','common-code'):p.add_argument('--'+key,required=True)
    args=p.parse_args();worker=None
    try:worker=Worker(args);worker.run()
    except Exception as error:
        inbox=Path(args.inbox)
        if inbox.exists():
            with (inbox/'worker_errors.jsonl').open('a') as stream:stream.write(json.dumps(dict(error_type=type(error).__name__,error=str(error),pid=os.getpid(),at_ns=time.monotonic_ns()))+'\n')
        raise
if __name__=='__main__':main()
