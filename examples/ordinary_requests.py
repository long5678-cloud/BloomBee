"""Finite ordinary model requests with one client/history and optional raw scores.

Input is JSONL: {"request_id":"r1","token_ids":[...],"new_tokens":3,
"priorities":null,"purpose":"warmup"}. Omit purpose to inherit the client default
(measured unless configured otherwise). Allowed values are warmup, measured and
prefetch. Priorities are optional; null preserves backend defaults.
"""
import argparse,json,time,hashlib
from pathlib import Path
import torch
from bloombee import AutoDistributedModelForCausalLM
from bloombee.kv_cache.network.ordinary import OrdinaryClient


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('model','peer','dht-prefix','requests-jsonl','output-root'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--config');p.add_argument('--context');p.add_argument('--scores',action='store_true')
    p.add_argument('--continue-on-error',action='store_true')
    args=p.parse_args();rows=[json.loads(line) for line in Path(args.requests_jsonl).read_text().splitlines() if line.strip()]
    if not rows:raise ValueError('nonempty request stream required')
    seen=set()
    for row in rows:
        if not isinstance(row.get('request_id'),str) or not row['request_id'] or row['request_id'] in seen:
            raise ValueError('unique nonempty request_id required')
        seen.add(row['request_id'])
        if not isinstance(row.get('token_ids'),list) or not row['token_ids'] or not all(type(t)is int and t>=0 for t in row['token_ids']):
            raise ValueError('actual nonempty token_ids required')
        if type(row.get('new_tokens',3))is not int or row.get('new_tokens',3)<=0:raise ValueError('positive new_tokens required')
    config=json.loads(Path(args.config).read_text()) if args.config else {}
    torch.manual_seed(0)
    model=AutoDistributedModelForCausalLM.from_pretrained(args.model,initial_peers=[args.peer],
        dht_prefix=args.dht_prefix,local_files_only=True,torch_dtype=torch.float16,device_map='cpu',
        max_retries=1,request_timeout=600,connect_timeout=20,update_period=5).eval()
    try:
        with OrdinaryClient(model,output_root=args.output_root,context=args.context,config=config) as client:
            root=Path(args.output_root)
            with (root/'request-inputs.jsonl').open('xb') as f:f.write(Path(args.requests_jsonl).read_bytes())
            with (root/'effective-config.json').open('x') as f:json.dump(config,f,indent=2)
            for index,row in enumerate(rows):
                ids=torch.tensor([row['token_ids']],dtype=torch.long);captures=[];capture_errors=[]
                def capture(module,inputs,kwargs,output):
                    if len(captures)<3:
                        try:
                            logits=output.get('logits') if isinstance(output,dict) else output.logits
                            captures.append(logits[:,-1:,:].detach().cpu().clone())
                        except Exception as error:capture_errors.append(str(error))
                hook=model.register_forward_hook(capture,with_kwargs=True) if args.scores else None
                started=time.monotonic_ns();error=None;terminal=None;output=None
                try:
                    output,terminal=client.generate(ids,max_new_tokens=row.get('new_tokens',3),request_id=row['request_id'],
                        purpose=row.get('purpose'),priorities=row.get('priorities'),min_new_tokens=row.get('new_tokens',3),do_sample=False,
                        attention_mask=torch.ones_like(ids),eos_token_id=None,pad_token_id=0,logits_to_keep=1)
                except Exception as caught:
                    error=caught;terminal=getattr(caught,'terminal',getattr(caught,'kv_terminal',None))
                finally:
                    ended=time.monotonic_ns()
                    if hook is not None:hook.remove()
                scores=None
                if args.scores:
                    scores='scores_'+str(index)+'.pt';torch.save(captures,root/scores)
                result=dict(request_id=row['request_id'],index=index,model_terminal=terminal,
                    output_ids=output.reshape(-1).tolist() if output is not None else None,
                    error=dict(type=type(error).__name__,message=str(error)) if error else None,
                    request_started_ns=started,request_ended_ns=ended,request_seconds=(ended-started)/1e9,
                    timing_scope='consumer before_request + model + lifecycle; excludes model load and source preparation',
                    scores=scores,capture_errors=capture_errors,history_count=len(client.history))
                with (root/('request_'+str(index)+'.json')).open('x') as f:json.dump(result,f,indent=2)
                print(json.dumps(result),flush=True)
                if error is not None and not args.continue_on_error:raise error
    finally:model.transformer.h.sequence_manager.shutdown()

if __name__=='__main__':main()
