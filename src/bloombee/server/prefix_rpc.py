"""Opt-in standard rpc_inference extension, executing in native runtime process.

Requires a full-model Llama provider, ordinary batch1, default native GPU slabs.
Runtime configuration is local trusted loader input, never supplied by RPC caller.
"""
import atexit
from collections import Counter, deque, OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
import os
from pathlib import Path
import threading
import time
from urllib.parse import urlsplit
import http.client
import ipaddress

import torch

from bloombee.kv_cache.identity import CacheIdentity, canonical_digest, token_ids_digest
from bloombee.kv_cache.directory import KVCacheDirectory, CacheRecord
from bloombee.kv_cache.transport import fetch_snapshot
from bloombee.kv_cache.placement import Layer
from bloombee.kv_cache.prefetch import Prefetcher, RemotePrefix
from bloombee.server.remote_prefix_session import (
    RemotePrefixSession, export_prefix, install_runtime_probe, _VARIABLE_FIELDS,
)
from bloombee.utils.misc import DUMMY_INT64

_SERVICES = {}


def execution_binding():
    """Fingerprint the actual runtime code and numeric flags, never RPC claims."""
    import hashlib
    root = Path(__file__).resolve().parents[1]
    return dict(baseline='bb99922071de23f2bf4ccf204c4a30ea64ac1c32',
        torch=torch.__version__, dtype='float16',
        backend_sha256=hashlib.sha256((root/'server/backend.py').read_bytes()).hexdigest(),
        numeric_backend_sha256=hashlib.sha256((root/'flexgen_utils/pytorch_backend.py').read_bytes()).hexdigest(),
        attention_implementation='eager_bmm',
        allow_fp16_reduced_precision_reduction=bool(torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction))


def binding_diagnostic(backend, template, cfg, info=None):
    """Read actual trusted backend fields and active native slab metadata; never change binding."""
    from bloombee.server.remote_prefix_session import _raw
    actual_config = backend.config.to_dict()
    actual = dict(config_digest=canonical_digest(actual_config),
        num_attention_heads=getattr(backend.config, 'num_attention_heads', None),
        num_key_value_heads=getattr(backend.config, 'num_key_value_heads', None),
        dtype=str(backend.dtype).removeprefix('torch.'))
    expected = {k: getattr(template, k) for k in actual}
    sensitive = {'token', 'hf_token', 'access_token', 'use_auth_token', 'api_key', 'password', 'secret'}
    def sanitize(value):
        if isinstance(value, dict):
            return {key: '[redacted]' if str(key).lower() in sensitive else sanitize(item)
                    for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [sanitize(item) for item in value]
        return value
    report = dict(event='binding_diagnostic', monotonic_ns=time.monotonic_ns(), pid=os.getpid(),
        layer=backend.block_index, expected=expected, actual=actual,
        differences={k: dict(expected=expected[k], actual=actual[k]) for k in actual if expected[k] != actual[k]},
        actual_config=sanitize(actual_config), backend_module=type(backend).__module__,
        cache_manager_module=type(backend.cache_manager).__module__,
        source_file=str(Path(__file__).resolve()))
    if info is not None:
        try:
            with backend.cache_manager.use_cache(*info.cache_handles) as cache:
                report['native_slabs'] = [dict(key_dtype=str(_raw(key).dtype), value_dtype=str(_raw(value).dtype),
                    key_shape=list(_raw(key).shape), value_shape=list(_raw(value).shape)) for key,value in cache]
        except Exception as error:
            report['slab_metadata_error_type'] = type(error).__name__
    with open(cfg['events_path'], 'a') as stream:
        stream.write(json.dumps(report, ensure_ascii=False, default=str) + '\n')
    return report


class PrefixRuntime:
    def __init__(self, runner, config_path, runtime_infos=()):
        self._closed = False
        self.dht = self.pool = self.discovery = None
        self._new_probes = []
        try:
            self._initialize(runner, config_path, runtime_infos)
        except Exception:
            self.close()
            # Roll back only probes installed by THIS unsuccessful constructor.
            for backend, finalize, inference in reversed(self._new_probes):
                backend._finalize_cache_update = finalize
                backend.inference_step = inference
                for name in ('_remote_prefix_binding', '_remote_prefix_writes',
                             '_remote_prefix_step_events', '_remote_prefix_event_counter'):
                    if hasattr(backend, name):
                        delattr(backend, name)
            raise

    def _ensure_probe(self, backend):
        expected = {k: v for k, v in self.template.to_dict().items() if k not in _VARIABLE_FIELDS}
        actual = dict(config_digest=canonical_digest(backend.config.to_dict()),
            num_attention_heads=backend.config.num_attention_heads,
            num_key_value_heads=backend.config.num_key_value_heads,
            dtype=str(backend.dtype).removeprefix('torch.'))
        if any(expected.get(k) != v for k,v in actual.items()):
            raise ValueError('trusted loader binding does not match actual backend')
        if hasattr(backend, '_remote_prefix_binding'):
            if (backend._remote_prefix_binding != expected
                    or not all(hasattr(backend, k) for k in ('_remote_prefix_writes',
                        '_remote_prefix_step_events', '_remote_prefix_event_counter'))):
                raise ValueError('existing probe has a different or incomplete trusted binding')
            return  # Same verified binding: never wrap twice or reset the audit counters.
        self._new_probes.append((backend, backend._finalize_cache_update, backend.inference_step))
        install_runtime_probe(backend, self.template.to_dict())

    def _initialize(self, runner, config_path, runtime_infos):
        from hivemind import DHT
        self.config = json.loads(Path(config_path).read_text())
        cfg = self.config
        self.template = CacheIdentity.from_dict(cfg['identity_template'])
        if self.template.execution_digest != canonical_digest(execution_binding()):
            raise ValueError('identity execution digest does not match actual runtime code/numeric flags')
        self.namespace = canonical_digest({k: v for k, v in self.template.to_dict().items()
                                           if k not in _VARIABLE_FIELDS})
        self.backends = tuple(sorted(runner.backends.values(), key=lambda b: b.block_index))
        if (len(self.backends) != self.backends[0].config.num_hidden_layers
                or [b.block_index for b in self.backends] != list(range(len(self.backends)))):
            raise ValueError('prefix RPC currently requires one complete model provider')
        info_by_uid = {info.uid: info for info in runtime_infos}
        for backend in self.backends:
            info = next((info_by_uid[uid] for uid, b in runner.backends.items()
                         if b is backend and uid in info_by_uid), None)
            try:
                self._ensure_probe(backend)
            except Exception:
                binding_diagnostic(backend, self.template, cfg, info)
                raise
        self.dht = DHT(initial_peers=cfg['initial_peers'], host_maddrs=cfg.get('host_maddrs', ['/ip4/0.0.0.0/tcp/0']),
                       start=True, client_mode=False)
        self.directory = KVCacheDirectory(self.dht, allowed_endpoints=cfg['allowed_endpoints'])
        self.pool = Prefetcher(self._fetch, cfg['capacity_bytes'],
                              allowed_endpoints=cfg['allowed_endpoints'], clock=self.directory.clock,
                              history_size=cfg.get('history_size', 32), max_inflight=1)
        self.history = deque(maxlen=cfg.get('history_size', 32))
        self.discovery = ThreadPoolExecutor(max_workers=1, thread_name_prefix='prefix-discovery')
        self.discovery_job = None
        self.event_lock = threading.Lock()
        self.prefix_length = int(cfg.get('prefix_length', 16))
        self._pool_event_count = 0
        self._published = OrderedDict()
        self._holder_health = None
        self.emit('runtime_ready', layers=len(self.backends),
                  config_digest=self.template.config_digest,
                  backend_types=sorted({type(b).__module__+'.'+type(b).__name__ for b in self.backends}),
                  cache_manager_types=sorted({type(b.cache_manager).__module__+'.'+type(b.cache_manager).__name__ for b in self.backends}))
        atexit.register(self.close)
        from multiprocessing.util import Finalize
        self._finalizer = Finalize(self, self.close, exitpriority=10)

    def emit(self, event, **fields):
        row = dict(event=event, monotonic_ns=time.monotonic_ns(), pid=os.getpid(), **fields)
        with self.event_lock:
            with open(self.config['events_path'], 'a') as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')

    def flush_pool(self):
        with self.event_lock:
            cursor, events = self.pool.events_since(self._pool_event_count)
            with open(self.config['events_path'], 'a') as stream:
                for event in events:
                    stream.write(json.dumps(dict(component='prefetch', **event)) + '\n')
            self._pool_event_count = cursor

    def identities(self, tokens, length):
        return tuple(replace(self.template, layer_start=i, layer_end=i + 1,
                             prefix_length=length, prefix_tokens_digest=token_ids_digest(tokens[:length]),
                             position_digest=canonical_digest(list(range(length)))).to_dict()
                     for i in range(len(self.backends)))

    def lookup_prefix(self, tokens, blocks):
        length = self.prefix_length
        if len(tokens) <= length or length % 16 or len(blocks) < length // 16:
            return None, None
        identities = self.identities(tokens, length)
        records, layers = {}, []
        for identity in identities:
            key = CacheIdentity.from_dict(identity)
            found = self.directory.lookup(key, timeout_s=self.config.get('lookup_timeout_s', 5))
            if not found:
                return None, None
            record = found[0]
            records[key.digest()] = record
            layers.append(Layer.from_record(identity, record))
        return RemotePrefix(self.namespace, tuple(blocks[:length // 16]), tuple(layers)), records

    def _fetch(self, endpoint, sha256, *, allowed_endpoints):
        blob = fetch_snapshot(endpoint, sha256, allowed_endpoints=allowed_endpoints,
                              timeout=self.config.get('fetch_timeout_s', 10),
                              max_bytes=self.config.get('max_object_bytes', 256 * 1024**2))
        self.emit('prefetch_http_return', sha256=sha256, object_id=id(blob), nbytes=len(blob))
        return blob

    def _discover(self):
        # Frozen, already served history only. No future rows or client forecasts.
        history = tuple(self.history)
        counter = Counter(tuple(blocks[:self.prefix_length // 16]) for _, blocks in history)
        ordered = sorted(enumerate(history), key=lambda row: (
            -counter[tuple(row[1][1][:self.prefix_length // 16])], -row[0]))
        prefixes, seen = [], set()
        for _, (tokens, blocks) in ordered:
            chain = tuple(blocks[:self.prefix_length // 16])
            if chain in seen:
                continue
            seen.add(chain)
            prefix, _ = self.lookup_prefix(tokens, blocks)
            if prefix:
                prefixes.append(prefix)
            if len(seen) >= self.config.get('discovery_candidates', 4):
                break
        self.pool.schedule(prefixes)
        self.flush_pool()

    def _holder_request(self, endpoint, path, *, blob=None):
        if endpoint not in self.config['allowed_endpoints']:
            raise ValueError('untrusted holder endpoint')
        parsed = urlsplit(endpoint)
        if parsed.scheme != 'http' or parsed.path not in ('', '/') or parsed.query or parsed.fragment:
            raise ValueError('holder requires exact HTTP IP endpoint')
        ipaddress.ip_address(parsed.hostname)
        connection = http.client.HTTPConnection(parsed.hostname, parsed.port or 80,
            timeout=self.config.get('fetch_timeout_s', 10))
        try:
            connection.request('POST' if blob is not None else 'GET', path, body=blob,
                headers={'Content-Type': 'application/octet-stream'} if blob is not None else {})
            response = connection.getresponse()
            if response.status != 200:
                raise RuntimeError('remote holder failed')
            data = response.read(65537)
            if len(data) > 65536:
                raise ValueError('oversized holder acknowledgment')
            return json.loads(data)
        finally:
            connection.close()

    def _publication_key(self, tokens):
        return canonical_digest(self.identities(tokens, self.prefix_length))

    def publish(self, infos, tokens):
        length = self.prefix_length
        if len(tokens) <= length:
            return
        endpoint = self.config.get('publish_endpoint')
        if not endpoint:
            return
        # A bounded complete-group acknowledgment cache, keyed by ACTUAL token identity.
        # Same holder process + immutable nondeleting store implies acknowledged objects remain.
        now = self.directory.clock()
        health = self._holder_health
        if health is None or time.monotonic() - health[0] > self.config.get('holder_health_s', 30):
            peer_id = self._holder_request(endpoint, '/status')['peer_id']
            self._holder_health = (time.monotonic(), peer_id)
        peer_id = self._holder_health[1]
        group_key = self._publication_key(tokens)
        acknowledged = self._published.get(group_key, {})
        identities = self.identities(tokens, length)
        def valid(record):
            return (record is not None and record.holder_id == peer_id and record.endpoint == endpoint
                    and record.expires_at > now + self.config.get('publication_expiry_margin_s', 5))
        if all(valid(acknowledged.get(CacheIdentity.from_dict(i).digest())) for i in identities):
            self._published.move_to_end(group_key)
            self.emit('publication_reused', key=group_key, layers=len(identities), post_bytes=0,
                      holder_health_age_s=time.monotonic() - self._holder_health[0])
            return
        acknowledged = dict(acknowledged)
        self._published[group_key] = acknowledged
        self._published.move_to_end(group_key)
        while len(self._published) > self.config.get('publication_groups', 256):
            self._published.popitem(last=False)
        total = 0
        for backend, info, identity in zip(self.backends, infos, identities):
            identity_key = CacheIdentity.from_dict(identity).digest()
            if valid(acknowledged.get(identity_key)):
                continue
            blob = export_prefix(backend, info, identity, length)
            record = CacheRecord.from_dict(self._holder_request(endpoint, '/publish', blob=blob))
            import hashlib
            if (record.identity_digest != identity_key or record.holder_id != peer_id
                    or record.endpoint != endpoint or record.payload_nbytes != len(blob)
                    or record.payload_sha256 != hashlib.sha256(blob).hexdigest()):
                raise ValueError('publication acknowledgment mismatch')
            acknowledged[identity_key] = record
            total += len(blob)
            self.emit('publication_bytes', key=group_key, layer=identity['layer_start'], nbytes=len(blob))
        self.emit('published', key=group_key, prefix_length=length, layers=len(identities), post_bytes=total)

    def run(self, runner, hidden, hypo, infos, prompts):
        envelope = infos[0].remote_prefix
        if any(info.remote_prefix != envelope for info in infos):
            raise ValueError('mixed prefix envelopes')
        if tuple(runner.backends[info.uid] for info in infos) != self.backends:
            raise ValueError('full ordered provider required')
        tokens, blocks = envelope['token_ids'], envelope['blocks']
        request_id, mode = envelope['request_id'], envelope.get('mode', 'prefetch')
        if (not isinstance(tokens, list) or any(type(x) is not int or x < 0 for x in tokens)
                or len(tokens) != hidden.shape[1] or hidden.shape[0] != 1
                or any(info.prefix_length or info.active_adapter for info in infos)
                or mode not in ('prefetch', 'demand', 'cold')):
            raise ValueError('invalid initial prefix RPC envelope')
        # Block IDs are trace hints only: actual tokens independently determine every KV identity.
        if not isinstance(blocks, list) or any(type(x) not in (int, str) for x in blocks):
            raise ValueError('invalid trace block chain')
        clean = tuple(replace(info, remote_prefix=None) for info in infos)
        epoch = os.urandom(16).hex()
        for backend in self.backends:
            backend._remote_prefix_epoch = epoch
        origin, demand = 'cold', None
        self.emit('rpc_native_enter', request_id=request_id, mode=mode, token_count=len(tokens),
                  rpc_session_id=envelope.get('rpc_session_id'), handler_queue_end_us=envelope.get('handler_queue_end_us'))
        try:
            if mode == 'prefetch':
                demand = self.pool.arrive(request_id, self.namespace, blocks,
                                          timeout=self.config.get('fetch_timeout_s', 10))
            prefix, records = None, None
            if demand is not None and demand.payloads:
                # A matching opaque chain alone is insufficient; verify actual token identities.
                expected = self.identities(tokens, self.prefix_length)
                if tuple(x.identity for x in demand.prefix.layers) == expected:
                    prefix = demand.prefix
                    records = {CacheIdentity.from_dict(layer.identity).digest(): CacheRecord(
                        CacheIdentity.from_dict(layer.identity).digest(), 'prefetch-memory',
                        layer.endpoint, layer.sha256, layer.sha256, layer.nbytes, layer.expires_at)
                        for layer in prefix.layers}
                    origin = 'prefetch'
                else:
                    self.emit('prefetch_token_mismatch', request_id=request_id)
            if mode != 'cold' and prefix is None:
                # Crucial wrong-guess B -> look up ACTUAL A, never use predicted B.
                prefix, records = self.lookup_prefix(tokens, blocks)
                if prefix:
                    origin = 'demand'
            if prefix:
                identities = self.identities(tokens, self.prefix_length)
                def lookup(identity, **kwargs):
                    return records.get(identity.digest())
                lookup.clock = self.directory.clock
                fetcher = fetch_snapshot
                if origin == 'prefetch':
                    blobs = {layer.sha256: blob for layer, blob in zip(prefix.layers, demand.payloads)}
                    def memory_fetch(endpoint, sha, **kwargs):
                        blob = blobs[sha]
                        self.emit('same_object_import', request_id=request_id, sha256=sha,
                                  object_id=id(blob), nbytes=len(blob))
                        return blob
                    memory_fetch.source_kind = 'prefetch_memory'
                    fetcher = memory_fetch
                starting_counts = [getattr(b, '_remote_prefix_event_counter', 0) for b in self.backends]
                session = RemotePrefixSession(self.backends, clean, identities, self.prefix_length,
                    lookup, token_ids=tokens, allowed_endpoints=self.config['allowed_endpoints'],
                    timeout_s=self.config.get('import_timeout_s', 60), fetcher=fetcher)
                if not session.prepare():
                    self._published.pop(self._publication_key(tokens), None)
                    self._holder_health = None
                    self.emit('import_failure_requires_fresh_session', request_id=request_id,
                              metrics=session.metrics)
                    raise RuntimeError('native prefix import failed; restart entire client session cold')
                state = session.run(hidden, hypo)
                receipt = dict(schema=1, rpc_session_id=envelope.get('rpc_session_id'),
                    request_id=request_id, native_session_id=session.receipt.session_id,
                    input_tokens=len(tokens), skipped_tokens=self.prefix_length,
                    token_ids_digest=token_ids_digest(tokens), layer_start=0,
                    layer_end=len(self.backends), origin=origin)
                output = (state, DUMMY_INT64, dict(pool_inner_ms=0.0, remote_prefix_receipt=receipt))
                self.emit('native_reuse', request_id=request_id, origin=origin, metrics=session.metrics,
                          computed_tokens=int(state.shape[1]), skipped_tokens=self.prefix_length,
                          native_backend_steps=[event for b, count in zip(self.backends, starting_counts)
                              for event in getattr(b, '_remote_prefix_step_events', ())
                              if event['sequence'] > count])
            else:
                output = runner(hidden, hypo, clean, *prompts)
                self.emit('native_cold', request_id=request_id, computed_tokens=len(tokens))
            if demand is not None:
                self.pool.finish(demand, reuse_success=origin == 'prefetch')
            # Export only after actual successful native forward; upload real bytes to remote holder.
            try:
                self.publish(clean, tokens)
            except Exception as error:
                self.emit('publication_failure', request_id=request_id, error_type=type(error).__name__)
            self.pool.observe(self.namespace, blocks)
            self.history.append((tuple(tokens), tuple(blocks)))
            if mode == 'prefetch' and (self.discovery_job is None or self.discovery_job.done()):
                if self.discovery_job is not None and self.discovery_job.exception() is not None:
                    self.emit('discovery_failure', error_type=type(self.discovery_job.exception()).__name__)
                self.discovery_job = self.discovery.submit(self._discover)
            self.flush_pool()
            self.emit('rpc_native_exit', request_id=request_id, origin=origin)
            return output
        finally:
            for backend in self.backends:
                backend._remote_prefix_epoch = None

    def close(self):
        if self._closed:
            return
        self._closed = True
        # Each cleanup is attempted even if another one fails; never mask init's cause.
        self.cleanup_errors = []
        actions = []
        if self.discovery is not None:
            actions.append(lambda: self.discovery.shutdown(wait=True))
        if self.pool is not None:
            actions.extend([self.pool.close, self.flush_pool])
        if self.dht is not None:
            actions.append(self.dht.shutdown)
        for action in actions:
            try:
                action()
            except Exception as error:
                self.cleanup_errors.append(type(error).__name__)


def run_prefix_rpc(runner, hidden, hypo, infos, prompts):
    path = os.environ.get('BLOOMBEE_PREFIX_SERVICE_CONFIG')
    if not path:
        raise RuntimeError('prefix RPC not enabled in trusted service configuration')
    key = (os.getpid(), id(runner))
    if key not in _SERVICES:
        _SERVICES[key] = PrefixRuntime(runner, path, infos)
    return _SERVICES[key].run(runner, hidden, hypo, infos, prompts)
