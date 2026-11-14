"""External warm worker: consume frozen context + score; never rescore.

Caller launches asynchronously, without waiting before next ordinary request.
"""
import argparse,json
from pathlib import Path
from bloombee.kv_cache.network.strategy_runner import StrategyRunner

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--context-json',required=True);p.add_argument('--selection-json',required=True);p.add_argument('--history-json',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    config=json.loads(Path(a.config).read_text())
    def emit(event,**data):
        with Path(config['strategy_events_path']).open('a') as f:f.write(json.dumps(dict(event=event,**data),sort_keys=True)+'\n')
    runner=StrategyRunner(config,emit)
    try:
        result=runner.warm_selected(json.loads(Path(a.context_json).read_text()),json.loads(Path(a.selection_json).read_text()),json.loads(Path(a.history_json).read_text()))
        with Path(a.out).open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    finally:runner.close()
