"""有界、不可变、无 pickle 的 KV 数据面；目录和身份由公共模块提供。"""
from __future__ import annotations

import hashlib
import ctypes
import http.server
import http.client
import ipaddress
import json
import math
import re
import struct
import threading
import socket
import time
from urllib.parse import urlsplit
from pathlib import Path

import torch

MAGIC = b"KVSLAB01"
MAX_HEADER = 65536
DEFAULT_LIMIT = 256 * 1024 * 1024
DTYPES = {str(d).removeprefix("torch."): d for d in
          (torch.float16, torch.bfloat16, torch.float32, torch.float64)}


def pack_snapshot(identity: dict, tensors: dict[str, torch.Tensor], *, max_bytes=DEFAULT_LIMIT) -> bytes:
    """编码 CPU 连续张量；身份完整保存在校验范围内，不重新定义公共缓存键。"""
    parts, specs = [], []
    total = 0
    for name, tensor in sorted(tensors.items()):
        if tensor.dtype not in DTYPES.values() or tensor.layout != torch.strided:
            raise ValueError("不支持的缓存张量格式")
        size = tensor.numel() * tensor.element_size()
        total += size
        if total > max_bytes:
            raise ValueError("缓存超过单对象上限")
        tensor = tensor.detach().to("cpu").contiguous()
        raw = ctypes.string_at(tensor.data_ptr(), size)
        parts.append(raw)
        specs.append(dict(name=name, dtype=str(tensor.dtype).removeprefix("torch."),
                          shape=list(tensor.shape), nbytes=size))
    header = json.dumps(dict(identity=identity, tensors=specs), sort_keys=True,
                        separators=(",", ":"), allow_nan=False).encode()
    if len(header) > MAX_HEADER:
        raise ValueError("缓存头过长")
    result = MAGIC + struct.pack("!I", len(header)) + header + b"".join(parts)
    if len(result) > max_bytes:
        raise ValueError("缓存超过单对象上限")
    return result


def unpack_snapshot(blob: bytes, expected_identity: dict, *, max_bytes=DEFAULT_LIMIT) -> dict:
    if len(blob) > max_bytes or len(blob) < 12 or blob[:8] != MAGIC:
        raise ValueError("缓存长度或魔数错误")
    length, = struct.unpack("!I", blob[8:12])
    if length > MAX_HEADER or 12 + length > len(blob):
        raise ValueError("缓存头长度错误")
    header = json.loads(blob[12:12 + length])
    if header.get("identity") != expected_identity:
        raise ValueError("缓存身份不匹配")
    specs = header.get("tensors")
    if not isinstance(specs, list) or not 1 <= len(specs) <= 4:
        raise ValueError("缓存张量数量错误")
    offset, output = 12 + length, {}
    for spec in specs:
        dtype = DTYPES.get(spec.get("dtype"))
        shape, name = spec.get("shape"), spec.get("name")
        if (dtype is None or not isinstance(shape, list) or not 1 <= len(shape) <= 4
                or any(type(d) is not int or d <= 0 for d in shape)
                or not isinstance(name, str) or name in output):
            raise ValueError("缓存张量描述错误")
        size = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
        if size != spec.get("nbytes") or size > len(blob) - offset:
            raise ValueError("缓存张量字节数错误")
        output[name] = torch.frombuffer(bytearray(blob[offset:offset + size]), dtype=dtype).reshape(shape)
        offset += size
    if offset != len(blob):
        raise ValueError("缓存含多余字节")
    return output


class SnapshotStore:
    """本地持久服务；只新增对象，不覆盖、不自动删除；预算耗尽显式失败。"""
    def __init__(self, root, *, max_total_bytes=2 * 1024**3, max_object_bytes=DEFAULT_LIMIT):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_total_bytes, self.max_object_bytes = max_total_bytes, max_object_bytes
        self.lock = threading.Lock()

    def put(self, blob: bytes) -> str:
        if len(blob) > self.max_object_bytes:
            raise ValueError("缓存对象过大")
        digest = hashlib.sha256(blob).hexdigest()
        path = self.path(digest)
        with self.lock:
            if path.exists():
                if path.read_bytes() != blob:
                    raise ValueError("缓存对象冲突")
                return digest
            used = sum(p.stat().st_size for p in self.root.iterdir() if p.is_file())
            if used + len(blob) > self.max_total_bytes:
                raise ValueError("缓存持久存储预算耗尽")
            with path.open("xb") as stream:
                stream.write(blob)
        return digest

    def path(self, digest):
        if not isinstance(digest, str) or re.fullmatch(r"[a-f0-9]{64}", digest) is None:
            raise ValueError("对象摘要错误")
        return self.root / (digest + ".bin")


class LocalSnapshotFetcher:
    """本地CPU不可变对象基线；与HTTP使用相同校验及全层导入，不产生网络流量。"""
    source_kind = 'local_store'

    def __init__(self, store):
        self.store = store

    def __call__(self, endpoint, digest, *, timeout=5, max_bytes=DEFAULT_LIMIT, allowed_endpoints=()):
        if endpoint not in allowed_endpoints:
            raise ValueError('端点未获准')
        deadline = time.monotonic() + timeout
        path = self.store.path(digest)
        size = path.stat().st_size
        if not 0 < size <= min(max_bytes, self.store.max_object_bytes):
            raise ValueError('本地缓存大小错误')
        chunks, received = [], 0
        with path.open('rb') as stream:
            while received < size:
                if time.monotonic() >= deadline:
                    raise TimeoutError('本地缓存读取预算耗尽')
                chunk = stream.read(min(1024 * 1024, size - received))
                if not chunk:
                    raise ValueError('本地缓存截断')
                chunks.append(chunk)
                received += len(chunk)
        blob = b''.join(chunks)
        if hashlib.sha256(blob).hexdigest() != digest:
            raise ValueError('本地缓存校验和错误')
        return blob


class SnapshotHTTPServer(http.server.HTTPServer):
    """单请求串行服务，首版绑定环回地址；跨进程 CPU 服务不需要第二张 GPU。"""
    def __init__(self, store, address=("127.0.0.1", 0)):
        self.store = store
        super().__init__(address, _SnapshotHandler)


class _SnapshotHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            digest = self.path.removeprefix("/objects/")
            if self.path != "/objects/" + digest:
                raise ValueError("对象路径错误")
            path = self.server.store.path(digest)
            size = path.stat().st_size
            if size > self.server.store.max_object_bytes:
                raise ValueError("缓存对象过大")
            self.send_response(200)
            self.send_header("Content-Length", str(size))
            self.send_header("Content-Type", "application/octet-stream")
            self.end_headers()
            with path.open("rb") as stream:
                while chunk := stream.read(65536):
                    self.wfile.write(chunk)
        except FileNotFoundError:
            self.send_error(404)
        except ValueError:
            self.send_error(400)

    def log_message(self, *args):
        pass


def fetch_snapshot(endpoint: str, digest: str, *, timeout=5, max_bytes=DEFAULT_LIMIT,
                   allowed_endpoints=(), on_read=None) -> bytes:
    if endpoint not in allowed_endpoints:
        raise ValueError("端点未获准")
    parsed = urlsplit(endpoint)
    if (parsed.scheme != "http" or not parsed.hostname or parsed.username or parsed.password
            or parsed.query or parsed.fragment or parsed.path not in ("", "/")):
        raise ValueError("端点格式错误")
    # 首版只接受显式IP，避免DNS解析脱离有界传输预算。
    ipaddress.ip_address(parsed.hostname)
    if re.fullmatch(r"[a-f0-9]{64}", digest) is None:
        raise ValueError("对象摘要错误")
    if not 0 < timeout < float('inf') or type(max_bytes) is not int or max_bytes <= 0:
        raise ValueError("传输预算错误")
    deadline = time.monotonic() + timeout
    connection = http.client.HTTPConnection(parsed.hostname, parsed.port or 80, timeout=timeout)
    timer = None
    try:
        connection.connect()
        transport_socket = connection.sock
        def abort():
            try:
                transport_socket.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        # 除每次read更新剩余超时外，墙钟deadline守护还中止慢速持续header/body。
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("传输总预算耗尽")
        timer = threading.Timer(remaining, abort)
        timer.daemon = True
        timer.start()
        connection.request('GET', '/objects/' + digest)
        response = connection.getresponse()
        length = response.getheader('Content-Length')
        if response.status != 200 or response.getheader('Transfer-Encoding') is not None:
            raise ValueError("缓存响应状态或传输编码错误")
        if length is None or not 0 < int(length) <= max_bytes:
            raise ValueError("缓存响应大小未知或超限")
        chunks, received, size = [], 0, int(length)
        while received < size:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("传输总预算耗尽")
            transport_socket.settimeout(remaining)
            chunk = response.read1(min(65536, size - received))
            if not chunk:
                raise ValueError("缓存响应截断")
            chunks.append(chunk)
            received += len(chunk)
            if on_read is not None:
                on_read(len(chunk))
        blob = b''.join(chunks)
        if time.monotonic() >= deadline:
            raise TimeoutError("传输总预算耗尽")
    finally:
        if timer is not None:
            timer.cancel()
        connection.close()
    if hashlib.sha256(blob).hexdigest() != digest:
        raise ValueError("缓存校验和不匹配")
    return blob
