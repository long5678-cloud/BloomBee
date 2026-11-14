"""Ordinary token-input BloomBee generation with optional final-request adapters.

This adapter does not map trace hashes, load models, select GPUs or launch a
worker. Its caller supplies a normally loaded model and explicit service config.
"""
import json
import os
from pathlib import Path
import time
import uuid
from .lifecycle import LifecycleError, boolean, failure


def write_new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


class OrdinaryClient:
    def __init__(self, model, *, output_root, context=None, config=None, job=None):
        self.model = model
        self.root = Path(output_root)
        self.root.mkdir(exist_ok=False, parents=True)
        self.config = dict(config or {})
        self.config['distributed_telemetry'] = boolean(self.config, 'distributed_telemetry')
        self.config['cache_evidence'] = boolean(self.config, 'cache_evidence')
        self.job = dict(job or {})
        self.context = context
        self.history = []
        self.attempt_sequence = 0
        self.ledger = Path(self.config.get('terminal_ledger', self.root/'history.jsonl'))
        with self.ledger.open('x'):
            pass
        self._pending_local_terminal = None
        self.closed = False
        self.closing = False
        self.adapter = None
        if self.config.get('distributed_telemetry', False) or self.config.get('prefetch_adapter') or self.config.get('cache_evidence', False):
            from .completion import Adapter
            self.adapter = Adapter(model=model, job=self.job, config=self.config, output_root=self.root)
        from .client_compat import install_logit_argument_forwarding
        self.compatibility = install_logit_argument_forwarding(model)

    def generate(self, input_ids, *, max_new_tokens, request_id=None, priorities=None, purpose=None, **kwargs):
        import torch
        if self.closed or self.closing:
            raise RuntimeError('client is closed')
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError('ordinary page integration requires batch size one')
        if type(max_new_tokens) is not int or max_new_tokens <= 0:
            raise ValueError('positive max_new_tokens required')
        if purpose is None:
            purpose = self.config.get('purpose', 'measured')
        if not isinstance(purpose, str) or purpose not in ('measured', 'warmup', 'prefetch'):
            raise ValueError('unknown invocation purpose')
        forbidden = {'dynamo_cache', 'max_length', 'max_new_tokens'} & kwargs.keys()
        if forbidden:
            raise ValueError('session/cache options belong to the adapter')
        from bloombee.kv_cache.retention_completion import prefix_key
        ids = input_ids.reshape(-1).tolist()
        request_id = request_id or uuid.uuid4().hex
        if not isinstance(request_id, str) or not request_id:
            raise ValueError('request_id must be a nonempty string')
        if any(h['request_id'] == request_id for h in self.history):
            raise ValueError('duplicate successful logical request')
        self._drain_local_terminal()
        if self.adapter is not None and getattr(self.adapter.prefetch, 'owns_page_terminal', False):
            self.adapter.resume_pending_completion()
        attempt = uuid.uuid4().hex
        index = self.attempt_sequence
        self.attempt_sequence += 1
        terminal = dict(request_id=request_id, attempt_id=attempt, index=index, purpose=purpose,
                        requested_input_tokens=len(ids), requested_output_tokens=max_new_tokens,
                        actual_ingress_ns=time.monotonic_ns())
        options = dict(max_length=len(ids)+max_new_tokens)
        if self.context is not None:
            options['dynamo_cache'] = dict(context=self.context, token_ids=ids,
                attempt_id=attempt)
            if priorities is not None:
                options['dynamo_cache']['priorities'] = priorities
        output = None
        primary_error = None
        errors = []
        model_started = False
        try:
            if self.adapter is not None:
                terminal['prefetch_claim'] = self.adapter.before_request(
                    attempt_id=attempt,token_ids=ids,actual_ingress_ns=terminal['actual_ingress_ns'],
                    clock_domain=self.config.get('ingress_clock_domain'))
            model_started = True
            with torch.inference_mode(), self.model.inference_session(**options):
                output = self.model.generate(input_ids,max_new_tokens=max_new_tokens,**kwargs)
            out = output.reshape(-1).tolist()
            if out[:len(ids)] != ids or len(out) != len(ids)+max_new_tokens:
                raise ValueError('full fixed-length completion contract not met')
            terminal.update(status='complete_success',actual_output_tokens=max_new_tokens,
                            output_ids=out,model_complete_ns=time.monotonic_ns())
        except Exception as error:
            primary_error = error
            terminal.update(status='failed',actual_output_tokens=0,error_type=type(error).__name__,
                error=str(error),model_started=model_started,model_complete_ns=time.monotonic_ns())
        ref = self.root/(attempt+'.json')
        receipt_saved = False
        try:
            write_new(ref,terminal)
            receipt_saved = True
        except Exception as error:
            errors.append(failure('model.receipt',error))
        observation = None
        observation_ok = self.adapter is None
        if self.adapter is not None:
            try:
                observation = self.adapter.observe_terminal(terminal=terminal,input_ids=ids,output_ids=output)
                observation_ok = True
            except Exception as error:
                errors.append(failure('observe_terminal',error))
        committed = False
        if terminal['status']=='complete_success' and receipt_saved and observation_ok:
            event = dict(request_id=request_id,attempt_id=attempt,status='complete_success',
                actual_output_tokens=max_new_tokens,requested_output_tokens=max_new_tokens,
                terminal_ref=str(ref),token_ids=ids,prefix_key=prefix_key(ids),
                completed_monotonic=terminal['model_complete_ns']/1e9,completion_event_seq=len(self.history)+1,
                kv_reuse_success=observation['kv_reuse_success'] if observation is not None else False,
                cache_observation_known=observation is not None)
            try:
                self._commit_history(event)
                committed = True
            except Exception as error:
                errors.append(failure('history.commit',error))
        # Cache commit is a separate fact from model success. Without a valid
        # committed observation/ledger, explicitly abort its staged transaction.
        cache_status = ('complete_success' if committed else
                        ('cancelled' if terminal['status']=='complete_success' else terminal['status']))
        terminal['cache_terminal_status'] = cache_status
        control = self.config.get('control_endpoint')
        if control and not getattr(getattr(self.adapter, "prefetch", None), "owns_page_terminal", False):
            payload = dict(attempt_id=attempt, status=cache_status, model_status=terminal['status'],
                           cache_transaction='commit' if cache_status=='complete_success' else 'abort')
            try:
                self._post_page_terminal(control, payload)
            except Exception as error:
                if self.adapter is None or self.adapter.prefetch is None:
                    self._pending_local_terminal = (control, payload)
                errors.append(failure('cache.terminal',error))
        if self.adapter is not None:
            try:
                terminal['completion_adapter'] = self.adapter.finish(terminal=terminal,input_ids=ids,
                    output_ids=output,history=tuple(self.history),cache_status=cache_status,
                    disclose=committed,observe=observation_ok)
            except LifecycleError as error:
                errors.extend(error.errors)
            except Exception as error:
                errors.append(failure('completion.finish',error))
        if errors:
            try:
                write_new(self.root/(attempt+'_lifecycle_errors.json'),dict(
                    model_terminal=terminal,errors=errors,model_error=str(primary_error) if primary_error else None))
            except Exception as error:
                errors.append(failure('lifecycle.receipt',error))
        if primary_error is not None:
            primary_error.kv_lifecycle_errors = errors
            primary_error.kv_terminal = terminal
            if errors and hasattr(primary_error,'add_note'):
                primary_error.add_note('KV lifecycle errors: '+str(errors))
            raise primary_error
        if errors:
            raise LifecycleError(errors,terminal=terminal)
        return output,terminal

    def _post_page_terminal(self, endpoint, payload):
        import httpx
        with httpx.Client(trust_env=False, timeout=self.config.get('control_timeout_s', 120)) as client:
            response = client.post(endpoint.rstrip('/') + '/terminal', json=payload)
            response.raise_for_status()

    def _drain_local_terminal(self):
        if self._pending_local_terminal is None:
            return
        try:
            self._post_page_terminal(*self._pending_local_terminal)
        except Exception as error:
            raise LifecycleError([failure('cache.terminal', error)]) from error
        self._pending_local_terminal = None

    def _commit_history(self, event):
        # Readers must see either complete prior history or complete next
        # history. A partially written or unsynced append is never visible.
        next_history = [*self.history, event]
        payload = ''.join(json.dumps(row,separators=(',',':'))+'\n' for row in next_history)
        draft = self.ledger.with_name('history_draft_'+event['attempt_id']+'.jsonl')
        with draft.open('x') as stream:
            if stream.write(payload) != len(payload):
                raise OSError('incomplete history draft write')
            stream.flush()
            os.fsync(stream.fileno())
        # Keep both the draft and old generation as evidence. The publication
        # hardlink is atomically renamed; no half-generation reaches readers.
        previous = self.ledger.with_name('history_before_'+event['attempt_id']+'.jsonl')
        os.link(self.ledger, previous)
        publication = self.ledger.with_name('history_publish_'+event['attempt_id']+'.jsonl')
        os.link(draft, publication)
        os.replace(publication, self.ledger)
        self.history = next_history

    def close(self):
        if not self.closed:
            self.closing = True
            errors = []
            try:
                self._drain_local_terminal()
            except LifecycleError as error:
                errors.extend(error.errors)
            if self.adapter is not None:
                try:
                    self.adapter.close()
                except LifecycleError as error:
                    errors.extend(error.errors)
            if errors:
                raise LifecycleError(errors)
            self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        try:
            self.close()
        except Exception as cleanup:
            if exc is None:
                raise
            exc.kv_close_error = cleanup
            if hasattr(exc, 'add_note'):
                exc.add_note('KV close also failed: '+str(cleanup))
        return False
