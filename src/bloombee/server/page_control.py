"""Node-local CPU metadata control for owned ordinary-page runtime states.

This endpoint cannot run a model, load files supplied by a caller, or issue
arbitrary CUDA operations. The inference worker owns all tensor reads/copies.
"""
import http.server
from contextlib import contextmanager
from functools import wraps
import json
import os
from pathlib import Path
import threading
import time

_clock_pid = None
_clock_domain = None

def local_clock_domain():
    """Actual local ordering domain; non-Linux fallback is process-scoped.

    A configured common domain must be supplied explicitly by local deployment.
    No invented boot ID or implicit cross-host monotonic comparability.
    """
    global _clock_pid, _clock_domain
    pid = os.getpid()
    if _clock_pid != pid:
        try:
            domain = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
            if not domain:
                raise ValueError('empty kernel clock domain')
        except FileNotFoundError:
            explicit = os.environ.get('BLOOMBEE_CLOCK_DOMAIN')
            if explicit is not None:
                domain = explicit.strip()
                if not domain:
                    raise ValueError('explicit clock domain must be nonempty')
            else:
                import uuid
                domain = 'process_'+str(pid)+'_'+uuid.uuid4().hex
        _clock_pid, _clock_domain = pid, domain
    return _clock_domain


_lock = threading.RLock()
_states = {}
_server = None
_terminal_hook = None
_ready_hook = None
_request_terminal_hook = None
_bundle_registry = None
_unset = object()
_startup_retained_gpu_blocks = _unset
_remote_allowed = os.environ.get('BLOOMBEE_PAGE_ALLOW_REMOTE', '1') == '1'
_demand_reservations = {}
_control_epoch = 0
_page_epoch = 0
_mutation_lock = threading.RLock()


@contextmanager
def metadata_guard(*, write=False):
    global _page_epoch
    with _mutation_lock:
        try:
            yield
        finally:
            if write:
                _page_epoch += 1


def atomic_metadata(write=False):
    def decorate(function):
        @wraps(function)
        def call(*args, **kwargs):
            with metadata_guard(write=write):
                return function(*args, **kwargs)
        return call
    return decorate


def set_hooks(*, terminal=None, ready=None, request_terminal=None):
    global _terminal_hook, _ready_hook, _request_terminal_hook
    with _lock:
        if terminal is not None:
            _terminal_hook = terminal
        if ready is not None:
            _ready_hook = ready
        if request_terminal is not None:
            _request_terminal_hook = request_terminal


def bundle_registry():
    global _bundle_registry
    with _lock:
        if _bundle_registry is None:
            from bloombee.server.page_bundle_reader import RequestBundleRegistry
            _bundle_registry = RequestBundleRegistry(max_active=1, max_bytes=64*1024**2,
                clock_domain=local_clock_domain())
        return _bundle_registry


def remote_allowed():
    return _remote_allowed


@atomic_metadata()
def configure_group(value):
    """Switch only a preconfigured frozen setting, with all old leases drained."""
    config_path = os.environ.get('BLOOMBEE_PAGE_GROUP_CONFIGS')
    if not config_path:
        raise ValueError('frozen per-group configurations missing')
    plan = json.loads(Path(config_path).read_text())
    group = plan[value['group_id']]
    if group['policy'] not in ('dynamo', 'lru', 'hotness', 'decay') or group['tier'] not in ('cpu', 'gpu', 'both'):
        raise ValueError('invalid frozen retention setting')
    if group['mode'] not in ('local', 'remote'):
        raise ValueError('invalid frozen group mode')
    limit = group.get('retained_gpu_blocks')
    if 'retained_gpu_blocks' in group and (type(limit) is not int or limit < 0):
        raise ValueError('retained_gpu_blocks must be a nonnegative integer')
    global _startup_retained_gpu_blocks
    # Capture at first successful switch; the CLI applies server_env after import.
    inherited_limit = _startup_retained_gpu_blocks
    if inherited_limit is _unset:
        inherited_limit = os.environ.get('BLOOMBEE_RETAINED_GPU_BLOCKS')
    selected_limit = str(limit) if limit is not None else inherited_limit
    effective = int(selected_limit) if selected_limit is not None else None
    for key in ('warm_ledger', 'formal_ledger'):
        target = Path(group[key])
        if not target.is_file() or not str(target).startswith('/dev/shm/'):
            raise ValueError('owned node-local committed ledger required')
    with _lock:
        if _demand_reservations:
            raise ValueError('active demand reservation while switching group')
        states = list(_states.items())
    if len(states) != 32:
        raise ValueError('initialize32 actual backend states first')
    for uid, (_, state) in states:
        with state.lock, state.pool.lock:
            if effective is not None and not 0 <= effective <= state.pool.capacity_pages:
                raise ValueError(f'retained_gpu_blocks budget exceeds actual capacity for {uid}')
            if state.sessions or getattr(state, 'retention_pending', {}) or any(p.refs for p in state.pool.pages):
                raise ValueError('drain model-thread free queue before switching group')
    _startup_retained_gpu_blocks = inherited_limit
    if selected_limit is None:
        os.environ.pop('BLOOMBEE_RETAINED_GPU_BLOCKS', None)
    else:
        os.environ['BLOOMBEE_RETAINED_GPU_BLOCKS'] = selected_limit
    os.environ['BLOOMBEE_RETENTION_POLICY'] = group['policy']
    os.environ['BLOOMBEE_HISTORY_TIER'] = group['tier']
    os.environ['BLOOMBEE_RETENTION_TERMINALS'] = group['warm_ledger']
    os.environ['BLOOMBEE_PAGE_FORMAL_LEDGER'] = group['formal_ledger']
    for uid, (_, state) in states:
        with state.lock, state.pool.lock:
            state.cache.entries.clear()
            state.remote_cpu_sources.clear()
            state.pool.lookup.clear()
            for page in state.pool.pages:
                page.key = None  # Preserve all original completion fences.
            state.remote_gpu_sources.clear()
            if hasattr(state, 'retention_done'):
                state.retention_done.clear()
    global _remote_allowed, _page_epoch
    _page_epoch += 1
    _remote_allowed = group['mode'] == 'local'  # Local preparation only; R starts empty.
    return dict(configured=True, group_id=value['group_id'], frozen_setting=group,
                actual_runtime_pid=os.getpid(), states=32, remote_allowed_during_preparation=_remote_allowed)


@atomic_metadata(write=True)
def begin_formal(value):
    """Explicit owned-state initialization; no tensor copies or file deletion."""
    global _remote_allowed
    if value.get('mode') not in ('local', 'remote') or type(value.get('clear_history')) is not bool:
        raise ValueError('explicit frozen group mode and reset choice required')
    path = os.environ.get('BLOOMBEE_PAGE_FORMAL_LEDGER')
    if not path or not Path(path).is_file():
        raise ValueError('preconfigured immutable formal history ledger missing')
    with _lock:
        states = list(_states.items())
    if len(states) != 32:
        raise ValueError('must initialize exactly32 actual model states first')
    for uid, (_, state) in states:
        with state.lock, state.pool.lock:
            if state.sessions or getattr(state, 'retention_pending', {}):
                raise ValueError('active or pending page lease during reset')
            if any(page.refs for page in state.pool.pages):
                raise ValueError('live physical page references during reset')
    previous_ledger = os.environ.get('BLOOMBEE_RETENTION_TERMINALS')
    os.environ['BLOOMBEE_RETENTION_TERMINALS'] = path
    from bloombee.kv_cache.retention_completion import snapshot
    try:
        historical = snapshot()
    except Exception:
        if previous_ledger is None:
            os.environ.pop('BLOOMBEE_RETENTION_TERMINALS', None)
        else:
            os.environ['BLOOMBEE_RETENTION_TERMINALS'] = previous_ledger
        raise
    rows = []
    for uid, (_, state) in states:
        with state.lock, state.pool.lock:
            if value['clear_history']:
                state.cache.entries.clear()
                state.remote_cpu_sources.clear()
                for page in state.pool.pages:
                    page.key = None
                    # Keep completion fences: zero references is not proof
                    # that an asynchronous CUDA stream has finished.
                state.pool.lookup.clear()
                state.remote_gpu_sources.clear()
            if hasattr(state, 'retention_done'):
                state.retention_done.clear()
            rows.append(dict(uid=uid, cpu_blocks=len(state.cache.entries),
                             gpu_blocks=len(state.pool.lookup)))
    _remote_allowed = value['mode'] == 'remote'
    return dict(initialized=True, states=rows, remote_allowed=_remote_allowed,
                history_sha256=historical['sha256'], history_events=historical['sequence'],
                monotonic_ns=time.monotonic_ns())


@atomic_metadata(write=True)
def reserve_current(value):
    attempt, tokens = value.get('attempt_id'), value.get('tokens')
    if not isinstance(attempt, str) or not attempt or not isinstance(tokens, list) or len(tokens) != 16:
        raise ValueError('actual arrived16token demand reservation required')
    with _lock:
        if _demand_reservations and attempt not in _demand_reservations:
            raise ValueError('single model worker already owns another demand reservation')
        old = _demand_reservations.setdefault(attempt, tuple(tokens))
        if old != tuple(tokens):
            raise ValueError('attempt reservation prefix changed')
    return dict(reserved=True, attempt_id=attempt, runtime_pid=os.getpid(),
                clock_domain=local_clock_domain(), monotonic_ns=time.monotonic_ns())


@atomic_metadata(write=False)
def probe_pages(value):
    from bloombee.server.dynamo_remote import Settings, block_identity
    tokens = value.get('tokens')
    identities = value.get('identities_by_block')
    if not isinstance(tokens, list) or len(tokens) != 16 or not isinstance(identities, list) or len(identities) != 1 or len(identities[0]) != 32:
        raise ValueError('probe requires exact known16token32layer bundle')
    with _lock:
        states = list(_states.items())
        inflight = [a for a, prefix in _demand_reservations.items() if prefix == tuple(tokens)]
    if len(states) != 32:
        return dict(known=False, complete=False, demand_inflight=bool(inflight), reason='actual_states_not_initialized', evidence=[])
    rows = []
    for uid, (backend, state) in states:
        layer = getattr(backend, 'block_index', None)
        if layer is None:
            layer = int(uid.rsplit('.', 1)[1])
        with state.lock, state.pool.lock:
            actual = block_identity(Settings.from_env(identity_only=True), backend, state, tokens, 16).to_dict()
            if actual != identities[0][layer]:
                return dict(known=False, complete=False, reason='actual_execution_identity_mismatch', evidence=rows)
            key = state.binding, tuple(tokens)
            slot = state.pool.lookup.get(key)
            gpu = False
            generation = None
            if slot is not None:
                page = state.pool.pages[slot]
                generation = page.generation
                gpu = page.key == key and page.valid == 16 and state.aux_generation[slot] == generation
            cpu = key in state.cache.entries
            rows.append(dict(uid=uid, layer=layer, identity=actual, gpu_complete=gpu,
                             slot=slot, generation=generation, cpu_complete=cpu,
                             cpu_digest=state.cache.entries[key].digest if cpu else None,
                             gpu_source=state.remote_gpu_sources.get((slot,generation)) if gpu else None,
                             cpu_source=state.remote_cpu_sources.get(key) if cpu else None))
    rows.sort(key=lambda row: row['layer'])
    if [r['layer'] for r in rows] != list(range(32)):
        return dict(known=False, complete=False, reason='layer_coverage_mismatch', evidence=rows)
    return dict(known=True, complete=all(r['gpu_complete'] or r['cpu_complete'] for r in rows),
                demand_inflight=bool(inflight), demand_reservation_attempts=inflight,
                evidence=rows, runtime_pid=os.getpid(), page_epoch=_page_epoch,
                snapshot_atomic=True, snapshot_lock='model_metadata_guard_v1', monotonic_ns=time.monotonic_ns())


def register(backend, state):
    global _server
    port = os.environ.get('BLOOMBEE_PAGE_CONTROL_PORT')
    if not port:
        return
    with _lock:
        from bloombee.kv_cache import page_retention
        registry = bundle_registry()
        set_hooks(terminal=page_retention.terminal, ready=registry.accept_ready_payload,
                  request_terminal=registry.terminal)
        _states[str(backend.name)] = (backend, state)
        if _server is None:
            _server = http.server.ThreadingHTTPServer((os.environ.get('BLOOMBEE_PAGE_CONTROL_BIND', '127.0.0.1'), int(port)), Handler)
            threading.Thread(target=_server.serve_forever, daemon=True,
                             name='page-control').start()


def bundle_inventory():
    registry = _bundle_registry
    if registry is None:
        return dict(schema=1, epoch=_control_epoch, runtime_pid=os.getpid(), leases=[])
    with registry.lock:
        rows = []
        for attempt, lease in registry.leases.items():
            rows.append(dict(attempt_id=attempt, object_id=lease.selection.object_id,
                             selection_id=lease.selection.selection_id,
                             active=attempt in registry.active, prepared=attempt not in registry.active,
                             decoded_bytes=lease.decoded_bytes, closed=lease.closed))
        return dict(schema=1, epoch=_control_epoch, runtime_pid=os.getpid(), leases=rows)


@atomic_metadata(write=False)
def status():
    # Never hold the registry lock while taking state/pool locks: the model
    # thread legitimately resolves its request bundle under its state lock.
    with _lock:
        snapshot = list(_states.items())
    rows = []
    for uid, (_, state) in snapshot:
        with state.lock, state.pool.lock:
            rows.append(dict(uid=uid, active_sessions=len(state.sessions),
                             active_pages=sum(page.refs > 0 for page in state.pool.pages),
                             page_refs=sum(page.refs for page in state.pool.pages),
                             continuation_calls=getattr(state, 'continuation_calls', 0),
                             continuation_tokens=getattr(state, 'continuation_tokens', 0),
                             last_continuation_length=getattr(state, 'last_continuation_length', None),
                             gpu_keys=len(state.pool.lookup), gpu_slots=state.pool.capacity_pages,
                             cpu_blocks=len(state.cache.entries),
                             cpu_payload_bytes=state.cache.resident_bytes(),
                             gpu_allocation_bytes=state.payload_bytes))
    return dict(pid=os.getpid(), states=rows, page_epoch=_page_epoch,
                snapshot_atomic=True, snapshot_lock='model_metadata_guard_v1', bundle_registry=bundle_inventory(), terminal_hook=_terminal_hook is not None,
                ready_hook=_ready_hook is not None,
                clock_domain=local_clock_domain(),
                monotonic_ns=time.monotonic_ns())


def _owner_groups():
    with _lock:
        backends = [backend for backend, _ in _states.values()]
    groups = {}
    for backend in backends:
        groups.setdefault(backend.cache_manager.cache, set()).add(str(backend.name))
    return groups


def release_expected(attempt):
    owners = []
    for memory, names in _owner_groups().items():
        journal = getattr(memory, "release_journal", None)
        row = journal.expected(attempt) if journal is not None else None
        if row is None or set(row["backend_uids"]) != names:
            return None
        owners.append(row)
    return dict(attempt_id=attempt, runtime_pid=os.getpid(), clock_domain=local_clock_domain(),
                owners=owners) if owners else None


def release_ack(attempt):
    if not isinstance(attempt, str) or not attempt:
        raise ValueError("attempt_id required")
    owners = []
    for memory, names in _owner_groups().items():
        journal = getattr(memory, "release_journal", None)
        row = journal.status(attempt) if journal is not None else {"status": "unknown"}
        if row.get("status") in ("pending", "released") and set(row["backend_uids"]) != names:
            row = {"status": "unknown"}
        owners.append(row)
    statuses = {row["status"] for row in owners}
    status = "released" if statuses == {"released"} else (
        "failed" if "failed" in statuses else "unknown" if not owners or "unknown" in statuses else "pending")
    return dict(attempt_id=attempt, status=status, runtime_pid=os.getpid(),
                clock_domain=local_clock_domain(), owners=owners)


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *unused):
        pass

    def send(self, code, value):
        raw = json.dumps(value, sort_keys=True).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path != '/status':
            return self.send(404, dict(error='unknown endpoint'))
        self.send(200, status())

    def do_POST(self):
        global _control_epoch
        try:
            n = int(self.headers.get('Content-Length', '0'))
            if not 0 < n <= 64 * 1024 ** 2:
                raise ValueError('bounded JSON body required')
            value = json.loads(self.rfile.read(n))
            if self.path == '/release_ack':
                self.send(200, release_ack(value.get('attempt_id')))
            elif self.path == '/terminal':
                if not isinstance(value.get('attempt_id'), str) or value.get('status') not in ('complete_success', 'failed', 'cancelled'):
                    raise ValueError('explicit final attempt terminal required')
                if _terminal_hook is None:
                    raise ValueError('terminal hook not installed; no false acknowledgement')
                results = []
                with _lock:
                    snapshot = list(_states.items())
                with metadata_guard(write=True):
                    expected = release_expected(value["attempt_id"])
                    for uid, (_, state) in snapshot:
                        with state.lock, state.pool.lock:
                            results.append(dict(uid=uid, result=_terminal_hook(state, value)))
                with _lock:
                    _demand_reservations.pop(value['attempt_id'], None)
                bundle_terminal = (_request_terminal_hook(None, value)
                                   if _request_terminal_hook is not None else None)
                with _lock:
                    _control_epoch += 1
                self.send(200, dict(results=results, bundle_terminal=bundle_terminal,
                                    release_expected=expected, monotonic_ns=time.monotonic_ns()))
            elif self.path == '/abort_ready':
                # Abort a failed preparation envelope, not the actual model
                # request: leave its real demand reservation and H untouched.
                result = bundle_registry().terminal(None, dict(attempt_id=value['attempt_id'], status='cancelled'))
                with _lock:
                    _control_epoch += 1
                self.send(200, result)
            elif self.path == '/reserve_current':
                self.send(200, reserve_current(value))
            elif self.path == '/page_probe':
                self.send(200, probe_pages(value))
            elif self.path == '/configure_group':
                self.send(200, configure_group(value))
            elif self.path == '/begin_formal':
                self.send(200, begin_formal(value))
            elif self.path == '/ready':
                if _ready_hook is None:
                    raise ValueError('ready hook not installed')
                result = _ready_hook(value)
                with _lock:
                    _control_epoch += 1
                self.send(200, result)
            else:
                self.send(404, dict(error='unknown endpoint'))
        except Exception as error:
            self.send(409, dict(error=type(error).__name__, detail=str(error)))
