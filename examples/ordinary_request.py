"""One ordinary token-input request to an already running BloomBee provider."""
import argparse
import json
import torch
from bloombee import AutoDistributedModelForCausalLM
from bloombee.kv_cache.network.ordinary import OrdinaryClient


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',required=True)
    parser.add_argument('--peer',required=True)
    parser.add_argument('--dht-prefix',required=True)
    parser.add_argument('--tokens-json',required=True,help='JSON file containing actual input token IDs')
    parser.add_argument('--output-root',required=True,help='New local evidence directory')
    parser.add_argument('--context',help='Immutable execution namespace; omit for dense cold mode')
    parser.add_argument('--config',help='Optional complete adapter JSON; see docs/ordinary-runtime.md')
    parser.add_argument('--new-tokens',type=int,default=3)
    args=parser.parse_args()
    from pathlib import Path
    config=json.loads(Path(args.config).read_text()) if args.config else {}
    ids=torch.tensor([json.loads(Path(args.tokens_json).read_text())],dtype=torch.long)
    model=AutoDistributedModelForCausalLM.from_pretrained(args.model,initial_peers=[args.peer],
        dht_prefix=args.dht_prefix,local_files_only=True,torch_dtype=torch.float16,device_map='cpu').eval()
    try:
        with OrdinaryClient(model,output_root=args.output_root,context=args.context,config=config) as client:
            output,terminal=client.generate(ids,max_new_tokens=args.new_tokens,min_new_tokens=args.new_tokens,
                do_sample=False,attention_mask=torch.ones_like(ids),eos_token_id=None,pad_token_id=0)
            print(json.dumps(dict(output_ids=output.reshape(-1).tolist(),terminal=terminal)))
    finally:
        model.transformer.h.sequence_manager.shutdown()

if __name__=='__main__':main()
