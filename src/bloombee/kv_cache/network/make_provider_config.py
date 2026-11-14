"""Reuse 32's verified identity template; exclusively create independent config."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--base-config',required=True);p.add_argument('--out',required=True);p.add_argument('--rpc-url',required=True);p.add_argument('--http-url',required=True);p.add_argument('--instance-id',required=True);p.add_argument('--coordinator-url',required=True);p.add_argument('--events-path',required=True);p.add_argument('--known-prefix-json');p.add_argument('--initial-warm-receipt');p.add_argument('--native-layers',type=int,default=32);p.add_argument('--clock-domain-file');p.add_argument('--storage-contract',choices=['local_fs_interop','network_tcp'],required=True);a=p.parse_args()
cfg=json.loads(Path(a.base_config).read_text());assert isinstance(cfg['identity_template'],dict)
cfg.update(lmcache_target=dict(url=a.rpc_url,http_url=a.http_url,instance_id=a.instance_id),coordinator_url=a.coordinator_url,events_path=a.events_path,storage_contract=a.storage_contract)
cfg['native_layers']=a.native_layers
if a.clock_domain_file:cfg['clock_domain_file']=a.clock_domain_file
if a.initial_warm_receipt:cfg['initial_warm_receipt']=a.initial_warm_receipt
if a.known_prefix_json:cfg['known_prefix_tokens']=json.loads(Path(a.known_prefix_json).read_text())
with Path(a.out).open('x') as f:json.dump(cfg,f,indent=2);f.write('\n')
print('Created independent provider config; original preserved.')
