"""Ordinary terminal evidence, one telemetry event, explicit predictor IPC."""
from pathlib import Path
import copy
import dataclasses
import hashlib
from types import SimpleNamespace
from .lifecycle import LifecycleError, boolean, failure
import json
import re
import time

from bloombee.kv_cache.telemetry import Telemetry
from bloombee.kv_cache.telemetry_paged import PagedTelemetry


def save(path, value):
    with Path(path).open('x') as output:
        json.dump(value, output, indent=2)


class _EvidenceRecorder:
    """Validation-only sink. No DHT, namespace, publication or event ledger."""
    def __init__(self, node_id):
        self.ref = SimpleNamespace(node_id=node_id)
        self.clock = time.time
        self.retention_s = 86400
    def record(self, event):
        return False


class Adapter:
    def __init__(self, *, model, job, config, output_root):
        self.job, self.config, self.root = job, dict(config), Path(output_root)
        self.enabled = boolean(self.config, 'distributed_telemetry')
        self.config['distributed_telemetry'] = self.enabled
        self.log = Path(config.get('provider_log', self.root.parent/'server.log'))
        self.offsets, self.observations = {}, {}
        self.history_start = time.time()
        self.publisher = None
        if self.enabled:
            self.publisher = Telemetry(model.transformer.h.sequence_manager.dht,
                namespace=config['telemetry_namespace'], node_id=config['consumer_node_id'],
                retention_s=86400, history_start=self.history_start)
        consumer = config.get('consumer_node_id',
            config.get('prefetch_controller', {}).get('history_boundary', {}).get('consumer_id', 'local-evidence'))
        self.telemetry = PagedTelemetry(self.publisher or _EvidenceRecorder(consumer),
            layer_uids=model.transformer.h.sequence_manager.block_uids)
        self._pending_completions = []
        self.prefetch = None
        self.evidence = PagedTelemetry(_EvidenceRecorder(consumer),
            layer_uids=model.transformer.h.sequence_manager.block_uids)
        self.prefetch_factory = None
        self._prefetch_closed = False
        self._closed = False
        self._close_attempts = 0
        if config.get('prefetch_adapter'):
            import importlib.util
            if config['prefetch_adapter'] == 'builtin':
                from .controller import Controller
                self.prefetch_factory = Controller
            else:
                spec = importlib.util.spec_from_file_location('group_prefetch_ipc', config['prefetch_adapter'])
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                self.prefetch_factory = module.Controller
            if not config.get('restore_initial_history'):
                self.prefetch = self.prefetch_factory(job=job, config=config, output_root=self.root)

    def before_request(self, *, attempt_id, token_ids, actual_ingress_ns, clock_domain):
        if self._closed:
            raise RuntimeError('adapter is closed')
        self._drain_prefetch()
        self.offsets[attempt_id] = self.log.stat().st_size
        if self.prefetch is None and self.prefetch_factory is not None:
            self.prefetch = self.prefetch_factory(job=self.job, config=self.config, output_root=self.root)
        if self.prefetch is not None:
            return self.prefetch.claim_current(attempt_id=attempt_id, token_ids=token_ids,
                actual_ingress_ns=actual_ingress_ns, clock_domain=clock_domain)
        return dict(status='disabled')

    def observe_terminal(self, *, terminal, input_ids, output_ids):
        attempt = terminal['attempt_id']
        if attempt in self.observations:
            return self.observations[attempt]
        start = self.offsets[attempt]
        with self.log.open('rb') as file:
            file.seek(start)
            raw = file.read()
        rows = []
        for line in raw.splitlines():
            _, marker, body = line.partition(b'DYNAMO_SERVICE ')
            if marker:
                row = json.loads(body)
                if row.get('telemetry_attempt_id') == attempt:
                    rows.append(row)
        name = hashlib.sha256(attempt.encode()).hexdigest()
        save(self.root/('layers_'+name+'.json'),dict(attempt_id=attempt, log_path=str(self.log),
            log_start=start, log_end=start+len(raw), rows=rows))
        state = 'completed' if terminal['status'] == 'complete_success' else ('cancelled' if terminal['status']=='cancelled' else 'failed')
        output = output_ids.reshape(-1).tolist() if output_ids is not None else None
        # The same strict evidence validation applies with B disabled. The local
        # recorder validates facts but never constructs a distributed publisher.
        purpose = terminal.get('purpose', self.config.get('purpose', 'measured'))
        if purpose not in ('measured', 'warmup', 'prefetch'):
            raise ValueError('unknown invocation purpose')
        facts = dict(request_id=terminal['request_id'], attempt_id=attempt,
            terminal=state, completed_at=time.time(), input_ids=input_ids, output_ids=output,
            max_new_tokens=terminal['requested_output_tokens'], rows=rows)
        validated = self.evidence.finish(**facts, purpose='measured')
        if self.enabled:
            value = self.telemetry.finish(**facts, purpose=purpose)
        else:
            value = dict(recorded=False,kind='disabled',event=None,
                reason='distributed telemetry disabled; local evidence validated')
        value['evidence_kind'] = validated['kind']
        value['purpose'] = purpose
        if value.get('event') is not None:
            value['event'] = dataclasses.asdict(value['event'])
        value['kv_reuse_success'] = validated['kind']=='reuse_success'
        value['row_count'] = len(rows)
        self.observations[attempt] = value
        save(self.root/('telemetry_'+name+'.json'),value)
        return value

    def finish(self, *, terminal, input_ids, output_ids, history, cache_status=None, disclose=True, observe=True):
        errors = []
        receipt = self.observations.get(terminal['attempt_id'])
        if receipt is None and observe:
            try:
                receipt = self.observe_terminal(terminal=terminal,input_ids=input_ids,output_ids=output_ids)
            except Exception as error:
                errors.append(failure('observe_terminal', error))
                cache_status = 'cancelled'
                disclose = False
        control_terminal = dict(terminal, status=cache_status or terminal['status'])
        if self.prefetch is not None:
            # Preserve the exact successful transaction and its causal history.
            # Retry in order before another claim or close, without model replay.
            attempt = terminal['attempt_id']
            if not any(p['terminal']['attempt_id'] == attempt for p in self._pending_completions):
                self._pending_completions.append(dict(terminal=copy.deepcopy(control_terminal),
                    history=copy.deepcopy(tuple(history)),
                    disclose=disclose and terminal['status']=='complete_success',terminal_done=False))
            try:
                self._drain_prefetch()
            except LifecycleError as error:
                errors.extend(error.errors)
        publication = {'disabled':True}
        if self.enabled:
            try:
                publication = self.publisher.publish(ttl_s=3600)
            except Exception as error:
                errors.append(failure('telemetry.publish',error))
        result = dict(observed=receipt,publication=publication,cache_status=control_terminal['status'],errors=errors)
        if errors:
            raise LifecycleError(errors,terminal=terminal)
        return result

    def resume_pending_completion(self):
        if self._pending_completions:
            self._drain_prefetch()

    def _drain_prefetch(self):
        while self._pending_completions:
            pending = self._pending_completions[0]
            if not pending['terminal_done']:
                try:
                    self.prefetch.terminal(terminal=pending['terminal'])
                    pending['terminal_done'] = True
                except Exception as error:
                    raise LifecycleError([failure('prefetch.terminal', error)]) from error
            if pending['disclose']:
                try:
                    self.prefetch.disclose_complete(history=pending['history'],terminal=pending['terminal'])
                except Exception as error:
                    from .release_ack import ReleasePending
                    pending['release_pending'] = isinstance(error, ReleasePending)
                    raise LifecycleError([failure('prefetch.disclose_complete', error)]) from error
            self._pending_completions.pop(0)

    def close(self):
        if self._closed:
            return
        self._close_attempts += 1
        errors = []
        if self.prefetch is not None:
            try:
                self._drain_prefetch()
            except Exception as error:
                errors.append(failure('prefetch.pending',error))
        recovering_release = any(p.get('release_pending') for p in self._pending_completions)
        if self.prefetch is not None and not self._prefetch_closed and not recovering_release:
            try:
                self.prefetch.close()
                self._prefetch_closed = True
            except Exception as error:
                errors.append(failure('prefetch.close',error))
        snapshot = {'observed_until':time.time()}
        if self.enabled:
            try:
                snapshot = self.publisher.publish(ttl_s=3600)
            except Exception as error:
                errors.append(failure('telemetry.publish',error))
        name = 'telemetry_publisher.json' if self._close_attempts==1 else f'telemetry_publisher_retry_{self._close_attempts}.json'
        try:
            save(self.root/name,dict(namespace=self.config.get('telemetry_namespace'),
                publishers=[dataclasses.asdict(self.publisher.ref)] if self.publisher is not None else [],
                enabled=self.enabled,start=self.history_start,end=snapshot['observed_until'],errors=errors,
                pending_completions=copy.deepcopy(self._pending_completions),
                expected_reuse_success=sum(v.get('kind')=='reuse_success' for v in self.observations.values())))
        except Exception as error:
            errors.append(failure('close.receipt',error))
        self._closed = not errors
        if errors:
            raise LifecycleError(errors)
