"""Explicit, independently usable factories for the real KV implementations."""
from dataclasses import dataclass
from pathlib import Path
import hashlib

@dataclass(frozen=True)
class Options:
    local: str | None = None
    telemetry: bool = False
    prefetch: str | None = None

    def __post_init__(self):
        if self.local not in (None, 'lru', 'hotness', 'decay', 'dynamo'):
            raise ValueError('local must be disabled or lru/hotness/decay/dynamo')
        if type(self.telemetry) is not bool:
            raise TypeError('telemetry must be a boolean')
        if self.prefetch not in (None, 'lrfu', 'transition'):
            raise ValueError('prefetch must be disabled or lrfu/transition')


def plugin_spec():
    path = Path(__file__).with_name('transition_policy.py')
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def score(context, strategy):
    """Validate and score disclosed history; never load SDK/model or transfer KV."""
    from .strategy_runner import score_context
    return score_context(context, strategy, plugin_spec() if strategy == 'transition' else None)


class Components:
    """Construct enabled components only; no implicit model/worker launch.

    local_kwargs use the selected storage contract (bytes or Dynamo blocks).
    telemetry_kwargs must include the caller's real DHT and publisher identity.
    prefetch_kwargs are passed to the native JointRuntime factory. Its explicit
    start() starts the loopback ticket service; score alone needs no runtime.
    """
    def __init__(self, options=Options(), *, local_kwargs=None,
                 telemetry_kwargs=None, prefetch_kwargs=None):
        if not isinstance(options, Options):
            raise TypeError('options must be Options')
        self.options = options
        self.local = self.telemetry = self.prefetch = None
        if options.local is not None:
            from bloombee.kv_cache.api import create_cache
            self.local = create_cache(options.local, **(local_kwargs or {}))
        if options.telemetry:
            from bloombee.kv_cache.telemetry import Telemetry
            self.telemetry = Telemetry(**(telemetry_kwargs or {}))
        if options.prefetch is not None:
            from .joint_runtime import create_joint_runtime
            kwargs = dict(prefetch_kwargs or {})
            config = dict(kwargs.pop('config'))
            configured = config.get('prefetch_strategy', options.prefetch)
            if configured != options.prefetch:
                raise ValueError('prefetch option differs from native runtime config')
            config['prefetch_strategy'] = configured
            config['strategy_transition_plugin'] = plugin_spec()
            self.prefetch = create_joint_runtime(config, **kwargs)

    def close(self):
        if self.prefetch is not None:
            return self.prefetch.close()
