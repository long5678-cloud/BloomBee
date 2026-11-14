"""接入真实 TransformerBackend 的显式前缀会话 API（非 RPC 集成声明）。"""
from __future__ import annotations

from dataclasses import replace, dataclass
import hashlib
import json
import uuid
import types
from collections import deque
from contextlib import contextmanager
import time
import torch

from bloombee.kv_cache.transport import pack_snapshot, unpack_snapshot, fetch_snapshot
from bloombee.kv_cache.identity import CacheIdentity
from bloombee.kv_cache.directory import CacheRecord


def _raw(tensor):
    return tensor if isinstance(tensor, torch.Tensor) else tensor.data


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


_VARIABLE_FIELDS = {"layer_start", "layer_end", "prefix_length", "prefix_tokens_digest", "position_digest"}


@contextmanager
def _timed(metrics, field):
    started = time.perf_counter()
    try:
        yield
    finally:
        metrics[field] += time.perf_counter() - started


def install_runtime_probe(backend, model_binding: dict):
    """在可信模型loader处绑定身份并观察真实写回。binding不能来自请求/目录。

    loader负责从本地已核权重revision、tokenizer和执行配置构造binding。
    本函数独立检查实际config摘要、层号、头数、dtype，并观测真实forward头布局。
    """
    if hasattr(backend, "_remote_prefix_binding"):
        raise ValueError("运行时绑定已存在")
    binding = {k: v for k, v in model_binding.items() if k not in _VARIABLE_FIELDS}
    config = backend.config
    if (binding.get("config_digest") != _digest(config.to_dict())
            or binding.get("num_attention_heads") != config.num_attention_heads
            or binding.get("num_key_value_heads") != config.num_key_value_heads
            or binding.get("dtype") != str(backend.dtype).removeprefix("torch.")):
        raise ValueError("可信loader绑定与实际backend不匹配")
    backend._remote_prefix_binding = binding
    backend._remote_prefix_writes = {}
    backend._remote_prefix_step_events = deque(maxlen=4096)
    backend._remote_prefix_event_counter = 0
    original = backend._finalize_cache_update
    original_inference = backend.inference_step

    def observe(self, new_kvs, cache_len, inference_info, kv_cache_position_ids, cache_tensors):
        result = original(new_kvs, cache_len, inference_info, kv_cache_position_ids, cache_tensors)
        pair = new_kvs.kvs if hasattr(new_kvs, "kvs") else new_kvs
        key, value = map(_raw, pair)
        # 当前scope只接受cache manager标准写入格式K[BH,D,S]/V[BH,S,D]。
        if (key.ndim == 3 and tuple(value.shape) == (key.shape[0], key.shape[2], key.shape[1])
                and not inference_info.batch_offset and inference_info.full_batch_size in (0, 1)):
            self._remote_prefix_writes[tuple(inference_info.cache_handles)] = (
                key.shape[0], cache_len + key.shape[2])
        return result
    backend._finalize_cache_update = types.MethodType(observe, backend)

    def inference(self, hidden_states, hypo_ids, inference_info):
        result = original_inference(hidden_states, hypo_ids, inference_info)
        self._remote_prefix_event_counter += 1
        self._remote_prefix_step_events.append(dict(layer=self.block_index,
            sequence=self._remote_prefix_event_counter,
            epoch=getattr(self, '_remote_prefix_epoch', None),
            computed_tokens=int(hidden_states.shape[1]), prefix_length=int(inference_info.prefix_length)))
        return result
    backend.inference_step = types.MethodType(inference, backend)


def _check_binding(backend, identity):
    expected = getattr(backend, "_remote_prefix_binding", None)
    if expected is None or expected != {k: v for k, v in identity.items() if k not in _VARIABLE_FIELDS}:
        raise ValueError("缓存身份与可信loader运行时绑定不一致")
    if (backend.block_index != identity.get("layer_start")
            or identity.get("layer_end") != backend.block_index + 1):
        raise ValueError("缓存层与运行时backend不一致")


def _slab(backend, info):
    manager = backend.cache_manager
    if getattr(backend.config, "model_type", None) != "llama":
        raise ValueError("首版只支持标准 Llama KV")
    policy = manager.offloading_policy
    if (policy.compress_cache or policy.cpu_cache_compute or policy.cache_gpu_percent != 100
            or policy.cache_cpu_percent or policy.cache_disk_percent
            or getattr(manager, "_paged_kv_enabled", False)):
        raise ValueError("首版不支持压缩、offload 或 paged KV")
    if (info.batch_offset or info.full_batch_size not in (0, 1)
            or info.micro_batch_size not in (0, 1)
            or info.active_adapter or info.kv_cache_position_ids is not None
            or info.tree_attention_mask is not None
            or (info.is_spec_dec is not None and bool(info.is_spec_dec))):
        raise ValueError("首版要求 batch1、无适配器、无 speculative decoding")
    return manager


def export_prefix(backend, info, identity: dict, prefix_length: int) -> bytes:
    """调用方只可在实际成功计算 prefix 后调用；不能推测句柄中有效长度。"""
    identity = identity.to_dict() if isinstance(identity, CacheIdentity) else CacheIdentity.from_dict(identity).to_dict()
    manager = _slab(backend, info)
    _check_binding(backend, identity)
    valid_heads, valid_length = getattr(backend, "_remote_prefix_writes", {}).get(
        tuple(info.cache_handles), (0, 0))
    if valid_length < prefix_length:
        raise ValueError("当前句柄缺少真实已完成forward的有效长度证据")
    manager.wait_for_pending_reorder()
    with manager.use_cache(*info.cache_handles) as cache:
        (key, value), = cache
        key, value = _raw(key), _raw(value)
        if (key.ndim != 3 or key.shape != value.shape or not 0 < prefix_length <= key.shape[0]
                or key.shape[1] != backend.config.num_attention_heads):
            raise ValueError("缓存 slab 布局或批量错误")
        # 必须由实际forward观测证明头数；不能从config推测packed/broadcast。
        if valid_heads != key.shape[1] or identity.get("layout") != "bhsd_v1":
            raise ValueError("首版只接受运行时已确认的全有效 broadcast 头布局")
        if identity.get("prefix_length") != prefix_length:
            raise ValueError("身份前缀长度不匹配")
        return pack_snapshot(identity, {"key": key[:prefix_length].permute(1, 0, 2).unsqueeze(0),
                                        "value": value[:prefix_length].permute(1, 0, 2).unsqueeze(0)})


@dataclass(frozen=True)
class ImportReceipt:
    session_id: str
    prefix_length: int
    handles: tuple
    identity_digests: tuple
    epochs: tuple


class RemotePrefixSession:
    """全层准备后才跳前缀；失败继续完整计算，绝不将部分层命中当整体命中。

    backends/infos 顺序必须等于模型层顺序。身份对象由 B 构造，外层必须先
    验证模型 revision/token前缀/层号/布局；这里只接受完整身份逐字一致的数据。
    lookup(identity) 返回对象 endpoint/sha256，目录和遥测可通过回调接入。
    """
    def __init__(self, backends, infos, identities, prefix_length, lookup,
                 record_fetch=None, record_reuse=None, *, token_ids, timeout_s=10,
                 max_total_bytes=2 * 1024**3, allowed_endpoints=(), fetcher=fetch_snapshot):
        if not (len(backends) == len(infos) == len(identities)) or not backends:
            raise ValueError("层数不一致")
        if type(prefix_length) is not int or prefix_length <= 0:
            raise ValueError("前缀长度错误")
        identities = [i.to_dict() if isinstance(i, CacheIdentity) else CacheIdentity.from_dict(i).to_dict()
                      for i in identities]
        self.backends, self.infos, self.identities = backends, infos, identities
        self.prefix_length, self.lookup = prefix_length, lookup
        self.record_fetch, self.record_reuse = record_fetch, record_reuse
        self.fetcher = fetcher
        if (len(token_ids) <= prefix_length or any(type(t) is not int or t < 0 for t in token_ids)):
            raise ValueError("要求实际token IDs且前缀后至少一个token")
        token_digest = hashlib.sha256(json.dumps(list(token_ids[:prefix_length]),
                                      separators=(",", ":")).encode()).hexdigest()
        previous = None
        for index, (backend, identity) in enumerate(zip(backends, identities)):
            _check_binding(backend, identity)
            if (identity.get("prefix_tokens_digest") != token_digest
                    or identity.get("prefix_length") != prefix_length
                    or identity.get("layer_start") != index or identity.get("layer_end") != index + 1
                    or identity.get("layout") != "bhsd_v1" or identity.get("batch_size") != 1
                    or getattr(backend, "block_index", index) != index):
                raise ValueError("token前缀、层覆盖或布局不匹配")
            common = {k: v for k, v in identity.items() if k not in ("layer_start", "layer_end")}
            if previous is not None and common != previous:
                raise ValueError("各层计算身份不一致")
            previous = common
        if len(backends) != backends[0].config.num_hidden_layers:
            raise ValueError("必须覆盖完整模型层")
        self.timeout_s, self.max_total_bytes = timeout_s, max_total_bytes
        self.allowed_endpoints = frozenset(allowed_endpoints)
        # 实际目录使用其DHT时钟；独立CPU替身可使用普通wall clock。
        self.directory_clock = getattr(lookup, 'clock', None) or getattr(
            getattr(lookup, '__self__', None), 'clock', time.time)
        self.session_id = uuid.uuid4().hex
        self.epochs = tuple(getattr(b, '_remote_prefix_epoch', None) for b in backends)
        if any(epoch is None for epoch in self.epochs):
            raise ValueError("新会话必须在其分配上下文绑定活动epoch")
        self.receipt = None
        self.metrics = dict(hit=False, fetched_bytes=0, network_bytes=0,
                            source_kind=getattr(fetcher, 'source_kind', 'http'),
                            fetch_seconds=0., lookup_seconds=0.,
                            import_seconds=0., prepare_seconds=0.,
                            imported_layers=0, skipped_tokens_per_layer=0, fallback=None)
        self.ready = False

    def prepare(self):
        if self.ready or self.receipt is not None:
            raise RuntimeError("不能重复导入同一会话")
        started = time.perf_counter()
        try:
            staged = []
            for backend, info, identity in zip(self.backends, self.infos, self.identities):
                if info.prefix_length != 0:
                    raise ValueError("远程导入要求新会话")
                manager = _slab(backend, info)
                remaining = self.timeout_s - (time.perf_counter() - started)
                if remaining <= 0:
                    raise TimeoutError("查询获取导入总预算耗尽")
                with _timed(self.metrics, 'lookup_seconds'):
                    location = self.lookup(CacheIdentity.from_dict(identity), timeout_s=remaining)
                if isinstance(location, list):
                    location = location[0] if location else None
                if location is None:
                    raise LookupError("目录未命中")
                location = location if isinstance(location, dict) else vars(location)
                location = CacheRecord.from_dict(location).to_dict()
                if (location["identity_digest"] != CacheIdentity.from_dict(identity).digest()
                        or location["object_id"] != location["payload_sha256"]
                        or location["expires_at"] <= self.directory_clock()):
                    raise ValueError("目录身份或对象标识不匹配")
                remaining = self.timeout_s - (time.perf_counter() - started)
                if remaining <= 0:
                    raise TimeoutError("查询获取导入总预算耗尽")
                size = location["payload_nbytes"]
                if type(size) is not int or not 0 < size <= min(256 * 1024**2, self.max_total_bytes - self.metrics["fetched_bytes"]):
                    raise ValueError("目录字节数超限")
                with _timed(self.metrics, 'fetch_seconds'):
                    blob = self.fetcher(location["endpoint"], location["payload_sha256"],
                                          timeout=remaining, max_bytes=size,
                                          allowed_endpoints=self.allowed_endpoints)
                self.metrics["fetched_bytes"] += len(blob)
                if self.metrics['source_kind'] == 'http':
                    self.metrics['network_bytes'] += len(blob)
                with _timed(self.metrics, 'import_seconds'):
                    tensors = unpack_snapshot(blob, identity)
                    if set(tensors) != {"key", "value"}:
                        raise ValueError("缓存张量集合错误")
                    manager.wait_for_pending_reorder()
                    with manager.use_cache(*info.cache_handles) as cache:
                        (key, value), = cache
                        key, value = _raw(key), _raw(value)
                        shape = (1, key.shape[1], self.prefix_length, key.shape[-1])
                        if (key.ndim != 3 or value.shape != key.shape or key.shape[0] < self.prefix_length
                                or key.shape[1] != backend.config.num_attention_heads
                                or any(tuple(t.shape) != shape or t.dtype != key.dtype for t in tensors.values())):
                            raise ValueError("缓存形状、容量或 dtype 不兼容")
                        # 先完成所有验证和设备分配，之后再动目标 slab。
                        staged.append((key, value, tensors["key"][0].permute(1, 0, 2).to(key.device),
                                       tensors["value"][0].permute(1, 0, 2).to(value.device)))
                if self.record_fetch:
                    self.record_fetch(identity, nbytes=len(blob))
            with _timed(self.metrics, 'import_seconds'):
                with torch.inference_mode():
                    for key, value, source_k, source_v in staged:
                        if time.perf_counter() - started > self.timeout_s:
                            raise TimeoutError("导入总预算耗尽")
                        key[:self.prefix_length].copy_(source_k)
                        value[:self.prefix_length].copy_(source_v)
                # .to/copy_默认同步；显式完成CUDA写入后才发receipt。
                for key, _, _, _ in staged:
                    if key.device.type == "cuda":
                        torch.cuda.synchronize(key.device)
                self.metrics['kv_copy_integrity'] = []
                for layer_index, (key, value, source_k, source_v) in enumerate(staged):
                    check = dict(layer_index=layer_index, source_dtype=str(source_k.dtype),
                        destination_dtype=str(key.dtype), source_shape=list(source_k.shape),
                        destination_prefix_shape=list(key[:self.prefix_length].shape),
                        key_exact=bool(torch.equal(key[:self.prefix_length], source_k)),
                        value_exact=bool(torch.equal(value[:self.prefix_length], source_v)))
                    self.metrics['kv_copy_integrity'].append(check)
                    if not check['key_exact'] or not check['value_exact']:
                        raise ValueError('KV import copy is not bit-exact')
            if time.perf_counter() - started > self.timeout_s:
                raise TimeoutError("导入完成时总预算耗尽")
            self.ready = True
            self.receipt = ImportReceipt(self.session_id, self.prefix_length,
                tuple(tuple(info.cache_handles) for info in self.infos),
                tuple(hashlib.sha256(json.dumps(i, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
                      for i in self.identities), self.epochs)
            self.metrics.update(hit=True, imported_layers=len(staged))
        except Exception as error:
            # 调用方必须退出此上下文；run禁止在失败会话继续推理。
            self.ready = False
            self.metrics["fallback"] = type(error).__name__ + ": " + str(error)
        finally:
            self.metrics["prepare_seconds"] = time.perf_counter() - started
        return self.ready

    def run(self, full_hidden_states, hypo_ids):
        if not self.ready or self.receipt is None or self.receipt.session_id != self.session_id:
            raise RuntimeError("缺少当前会话导入receipt，必须另开干净会话回退")
        if self.receipt.handles != tuple(tuple(info.cache_handles) for info in self.infos):
            raise RuntimeError("句柄已改变，receipt失效")
        if self.receipt.epochs != tuple(getattr(b, '_remote_prefix_epoch', None) for b in self.backends):
            raise RuntimeError("分配上下文已释放或重建，receipt失效")
        if full_hidden_states.shape[0] != 1 or full_hidden_states.shape[1] <= self.prefix_length:
            raise ValueError("要求 batch1 且前缀后至少一个 token")
        prefix = self.prefix_length
        hidden = full_hidden_states[:, prefix:]
        for backend, info, identity in zip(self.backends, self.infos, self.identities):
            hidden = backend.inference_step(hidden, hypo_ids, replace(info, prefix_length=prefix))[0]
        if self.record_reuse:
            for identity in self.identities:
                self.record_reuse(identity)
        self.metrics["skipped_tokens_per_layer"] = prefix
        self.ready = False  # receipt只允许首个suffix消费一次。
        return hidden


def run_prefix_request(fresh_session, identities, prefix_length, token_ids, full_hidden_states,
                       hypo_ids, lookup, **options):
    """fresh_session()必须新分配句柄并在退出时正常释放；失败不复用污染会话。"""
    metrics = {}
    with fresh_session() as (backends, infos):
        try:
            session = RemotePrefixSession(backends, infos, identities, prefix_length, lookup,
                                          token_ids=token_ids, **options)
            if session.prepare():
                return session.run(full_hidden_states, hypo_ids), session.metrics
            metrics.update(session.metrics)
        except Exception as error:
            metrics.update(hit=False, fallback=type(error).__name__ + ": " + str(error))
    # 上一个分配上下文已结束，完整prompt只在新的干净句柄上计算。
    with fresh_session() as (backends, infos):
        hidden = full_hidden_states
        for backend, info in zip(backends, infos):
            if info.prefix_length != 0:
                raise ValueError("回退会话非空")
            hidden = backend.inference_step(hidden, hypo_ids, info)[0]
        return hidden, metrics
