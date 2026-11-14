"""Record BOTH frozen strategy scores for ONE already qualified causal context.

No directory query, warm, future trace read, request or model call. 32 invokes this
on the actual service decision host before the single active strategy warm.
"""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import sys
import time


def runtime(common_code):
    common=Path(common_code).resolve()
    if str(common) not in sys.path:sys.path.insert(0,str(common))
    sys.dont_write_bytecode=True
    spec=importlib.util.spec_from_file_location('shared_strategy_contract',common/'strategy_contract.py')
    contract=importlib.util.module_from_spec(spec);spec.loader.exec_module(contract)
    spec=importlib.util.spec_from_file_location('shared_strategy_runner',common/'strategy_runner.py')
    runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
    return contract,runner


def capture(context,plugin,metadata,output,common_code):
    contract,runner=runtime(common_code)
    contract.validate_context(context)
    allowed={'segment_id','run_id','active_strategy','source_id','registry_sha256',
             'completed_trace_rows','clock_domain_id','freeze_sha256','raw_qualification_ref'}
    if set(metadata)-allowed:raise ValueError('metadata cannot carry future request/next tokens')
    required={'segment_id','run_id','active_strategy','source_id','registry_sha256','clock_domain_id','freeze_sha256','raw_qualification_ref'}
    if not required<=metadata.keys():raise ValueError('missing pre-registered provenance')
    if metadata['clock_domain_id']!=context['history_boundary']['clock_domain_id']:
        raise ValueError('decision/disclosure clock domains differ')
    if metadata['active_strategy'] not in ('lrfu','transition'):raise ValueError('unknown active strategy')
    started=time.monotonic_ns()
    if context['causal_cutoff_ns']>started:raise ValueError('future disclosure cutoff on current decision clock')
    before=contract.digest(context)
    scores={name:runner.score_context(copy.deepcopy(context),name,plugin if name=='transition' else None)
            for name in ('lrfu','transition')}
    if contract.digest(context)!=before:raise ValueError('context mutated')
    record=dict(event='same_context_shadow',schema_version=1,**metadata,
        decision_id=context['decision_id'],decision_seq=context['decision_seq'],context=context,
        context_sha256=before,decision_started_ns=started,decision_completed_ns=time.monotonic_ns(),
        shadow_scores=scores,shadow_warm_calls=0)
    with Path(output).open('a') as stream:stream.write(json.dumps(record,sort_keys=True,ensure_ascii=False)+'\n')
    return record


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for field in ('context','plugin','metadata','out','common-code'):parser.add_argument('--'+field,required=True)
    args=parser.parse_args()
    capture(json.loads(Path(args.context).read_text()),json.loads(Path(args.plugin).read_text()),
            json.loads(Path(args.metadata).read_text()),args.out,args.common_code)
