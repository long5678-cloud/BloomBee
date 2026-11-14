"""Legacy ordinary BloomBee client; request JSONL contains actual token_ids/blocks/output_length."""
import argparse
import json
from pathlib import Path
import time

from bloombee import AutoDistributedModelForCausalLM
from bloombee.client.prefix_generation import DistributedPrefixGenerator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--requests', required=True)
    parser.add_argument('--results', required=True)
    parser.add_argument('--mode', choices=['prefetch', 'demand', 'cold'], default='prefetch')
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    model_options = dict(cfg.get('model_options', {}))
    model_options.setdefault('max_retries', 1)
    model = AutoDistributedModelForCausalLM.from_pretrained(cfg['model_path'],
        initial_peers=cfg['initial_peers'], dht_prefix=cfg['dht_prefix'],
        local_files_only=True, torch_dtype='float32',
        **model_options)
    generator = DistributedPrefixGenerator(model, max_context=cfg['max_context'])
    # x mode protects all earlier artifacts; driver controls actual ingress pacing.
    with open(args.requests) as requests, open(args.results, 'x') as results:
        for line_index, line in enumerate(requests):
            row = json.loads(line)
            started = time.perf_counter()
            try:
                output, receipt = generator.generate(row['token_ids'], row['blocks'], mode=args.mode,
                    request_id=str(line_index), max_new_tokens=row['output_length'],
                    eos_token_id=None, pad_token_id=cfg.get('pad_token_id', 0))
                receipt.update(output_ids=output[0].tolist(), completed=True)
            except Exception as error:
                receipt = dict(request_id=str(line_index), completed=False,
                               error_type=type(error).__name__)
            receipt['request_seconds'] = time.perf_counter() - started
            results.write(json.dumps(receipt) + '\n')
            results.flush()


if __name__ == '__main__':
    main()
