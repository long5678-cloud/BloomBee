"""Target-local clock-domain receipt. No /proc or external-file inspection.

Operator creates one node-local, non-shared file for helper + provider boot.
Hostname and kernel clock/wall offset checks reject accidental cross-host reuse
or stale/rebooted state. Offset uncertainty fails closed; never asserts a raw
cross-host monotonic ordering. This is trusted local orchestration, not remote
host attestation.
"""
import argparse,json,socket,time,uuid
from pathlib import Path

def observation():
    return dict(host=socket.gethostname(),clock='CLOCK_MONOTONIC',wall_minus_monotonic_ns=time.time_ns()-time.monotonic_ns())

def create(path):
    value=dict(schema=1,domain_id=uuid.uuid4().hex,**observation())
    with Path(path).open('x') as f:json.dump(value,f,indent=2);f.write('\n')
    return value

def current(config):
    path=config.get('clock_domain_file')
    if not path:return None
    stored=json.loads(Path(path).read_text());now=observation()
    if stored.get('host')!=now['host'] or stored.get('clock')!=now['clock'] or abs(stored.get('wall_minus_monotonic_ns',0)-now['wall_minus_monotonic_ns'])>2_000_000_000:return None
    return stored

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();v=create(a.out);print(json.dumps(dict(clock_domain_id=v['domain_id'],path=a.out)))
