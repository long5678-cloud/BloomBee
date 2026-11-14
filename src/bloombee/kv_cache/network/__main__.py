"""Portable launch commands. No server access or GPU selection is implicit."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    s = sub.add_parser('launch-native', help='Explicitly launch one generated component on this host')
    s.add_argument('--bundle', type=Path, required=True)
    s.add_argument('--component', choices=['redis','coordinator','source_mp','target_mp','source_model','target_model','worker'], required=True)
    s = sub.add_parser('configure-native', help='Generate native bundle from real deployment inputs; no launch')
    s.add_argument('--deployment', type=Path, required=True)
    s.add_argument('--output', type=Path, required=True)
    s = sub.add_parser('validate-native', help='Validate bundle/source pins, optionally execution-node files')
    s.add_argument('--bundle', type=Path, required=True)
    s.add_argument('--runtime-files', action='store_true')
    s = sub.add_parser('score', help='Score one qualified context without SDK/model')
    s.add_argument('--context', type=Path, required=True)
    s.add_argument('--strategy', choices=['lrfu','transition'], required=True)
    s = sub.add_parser('worker', help='Start one explicitly configured native CPU worker')
    s.add_argument('--config', type=Path, required=True)
    s.add_argument('--inbox', type=Path, required=True)
    s = sub.add_parser('front', help='Start the finite trace-derived ordinary ingress harness')
    s.add_argument('--run-dir', type=Path, required=True)
    s = sub.add_parser('server', help='Run normal BloomBee CLI with explicit environment config')
    s.add_argument('--config', type=Path, required=True)
    s.add_argument('server_args', nargs=argparse.REMAINDER)
    s = sub.add_parser('check', help='Validate options; does not start components')
    s.add_argument('--local', choices=['lru','hotness','decay','dynamo'])
    s.add_argument('--prefetch', choices=['lrfu','transition'])
    s.add_argument('--telemetry', action='store_true')
    args = parser.parse_args()
    if args.command == 'launch-native':
        from .configuration import launch_component
        raise SystemExit(launch_component(args.bundle,args.component))
    elif args.command == 'configure-native':
        from .configuration import generate_bundle
        folder = generate_bundle(json.loads(args.deployment.read_text()),args.output)
        print(json.dumps(dict(bundle=str(folder),services_started=False)))
    elif args.command == 'validate-native':
        from .configuration import validate_bundle
        print(json.dumps(validate_bundle(args.bundle,runtime_files=args.runtime_files)))
    elif args.command == 'check':
        from dataclasses import asdict
        from .api import Options
        print(json.dumps(asdict(Options(args.local,args.telemetry,args.prefetch))))
    elif args.command == 'score':
        from .api import score
        print(json.dumps(score(json.loads(args.context.read_text()),args.strategy)))
    elif args.command == 'worker':
        if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
            raise ValueError('native sidecar requires CUDA_VISIBLE_DEVICES=""')
        from .api import plugin_spec
        from . import worker
        folder = Path(__file__).parent
        plugin = args.inbox/'packaged_plugin.json'
        args.inbox.mkdir(parents=True,exist_ok=True)
        with plugin.open('x') as stream:
            json.dump(plugin_spec(), stream)
        sys.argv = ['worker','--config',str(args.config),'--inbox',str(args.inbox),
                    '--plugin',str(plugin),'--shadow-common',str(folder/'capture_shadow.py'),
                    '--common-code',str(folder),'--native-code',str(folder),
                    '--framework-root',str(folder.parents[3])]
        worker.main()
    elif args.command == 'front':
        from . import front
        sys.argv = ['front','--run-dir',str(args.run_dir)]
        front.main()
    elif args.command == 'server':
        config = json.loads(args.config.read_text())
        env = config['server_env']
        if any(not key.startswith('BLOOMBEE_') for key in env):
            raise ValueError('server_env only accepts BLOOMBEE_ settings')
        os.environ.update({k:str(v) for k,v in env.items()})
        # Preserve the tested numerical execution settings explicitly.
        os.environ['BLOOMBEE_DISABLE_SDPA_ATTENTION']='1'
        import torch
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False
        from bloombee.cli.run_server import main as run_server
        argv = args.server_args
        if argv and argv[0]=='--':
            argv=argv[1:]
        sys.argv=['run_server',*argv]
        run_server()

if __name__ == '__main__':
    main()
