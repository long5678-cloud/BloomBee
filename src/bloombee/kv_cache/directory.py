"""Small metadata on the existing Hivemind DHT, with explicit endpoint policy."""
import asyncio
from dataclasses import asdict, dataclass, fields
import hashlib
import json
import logging
import math
from typing import Mapping
from urllib.parse import urlsplit

from .identity import CacheIdentity, _sha256

logger = logging.getLogger(__name__)


def _dht_clock():
    from hivemind.utils import get_dht_time
    return get_dht_time()


@dataclass(frozen=True)
class CacheRecord:
    identity_digest: str
    holder_id: str
    endpoint: str
    object_id: str
    payload_sha256: str
    payload_nbytes: int
    expires_at: float
    record_version: int = 1

    def __post_init__(self):
        if type(self.record_version) is not int or self.record_version != 1:
            raise ValueError('unsupported record schema')
        if not _sha256(self.identity_digest) or not _sha256(self.payload_sha256):
            raise ValueError('invalid record digest')
        for name in ('holder_id', 'object_id'):
            value = getattr(self, name)
            if not isinstance(value, str) or not 0 < len(value) <= 256 or any(ord(c) < 32 for c in value):
                raise ValueError('invalid ' + name)
        if not isinstance(self.endpoint, str) or len(self.endpoint) > 2048:
            raise ValueError('invalid endpoint')
        parsed = urlsplit(self.endpoint)
        if parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.username or parsed.password or parsed.fragment or parsed.query:
            raise ValueError('endpoint must be a credential-free HTTP(S) service address')
        if type(self.payload_nbytes) is not int or self.payload_nbytes <= 0:
            raise ValueError('invalid payload size')
        if isinstance(self.expires_at, bool) or not isinstance(self.expires_at, (int, float)) or not math.isfinite(self.expires_at):
            raise ValueError('invalid expiration')

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, Mapping) or set(value) != {f.name for f in fields(cls)}:
            raise ValueError('record fields must match schema exactly')
        return cls(**value)

    def subkey(self):
        return hashlib.sha256(json.dumps([self.holder_id, self.object_id], separators=(',', ':')).encode()).hexdigest()


class KVCacheDirectory:
    """Synchronous calls outside async loops; async wrappers use a worker thread.

    The allowlist is exact service endpoint matching, never implicit arbitrary URL
    access. Payload availability remains the data plane's responsibility. A lookup
    result is an advisory candidate, not proof that bytes can be fetched.
    """
    def __init__(self, dht, *, clock=_dht_clock, allowed_endpoints=(), max_payload_bytes=2 * 1024**3,
                 operation_timeout_s=5.0):
        if type(max_payload_bytes) is not int or max_payload_bytes <= 0:
            raise ValueError('invalid payload bound')
        self._validate_timeout(operation_timeout_s)
        self.dht, self.clock = dht, clock
        self.allowed_endpoints = frozenset(allowed_endpoints)
        self.max_payload_bytes = max_payload_bytes
        self.operation_timeout_s = operation_timeout_s

    @staticmethod
    def _validate_timeout(timeout):
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError('timeout must be finite and positive')

    @staticmethod
    def _check_sync():
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return
        raise RuntimeError('use aregister/alookup or a worker thread from an async loop')

    def _valid_record(self, identity, record, now):
        return (record.identity_digest == identity.digest() and record.expires_at > now
                and record.endpoint in self.allowed_endpoints and record.payload_nbytes <= self.max_payload_bytes)

    def register(self, identity: CacheIdentity, record: CacheRecord, *, ttl_seconds: float) -> bool:
        self._check_sync()
        self._validate_timeout(ttl_seconds)
        now = self.clock()
        if not self._valid_record(identity, record, now):
            raise ValueError('incompatible, expired, unapproved, or oversized record')
        peer = self.dht.peer_id
        peer = peer.to_base58() if hasattr(peer, 'to_base58') else str(peer)
        if record.holder_id != peer:
            raise ValueError('holder must be the publishing DHT peer')
        expiry = min(now + ttl_seconds, record.expires_at)
        try:
            return bool(self.dht.store(key=identity.dht_key(), subkey=record.subkey(), value=record.to_dict(),
                                       expiration_time=expiry, return_future=True).result(timeout=self.operation_timeout_s))
        except TimeoutError:
            logger.warning('KV directory registration timed out')
            return False

    def lookup(self, identity: CacheIdentity, *, timeout_s: float = 5.0):
        self._check_sync()
        self._validate_timeout(timeout_s)
        try:
            found = self.dht.get(identity.dht_key(), latest=True, return_future=True).result(timeout=timeout_s)
        except TimeoutError:
            logger.warning('KV directory lookup timed out')
            return []
        now = self.clock()
        if found is None or not isinstance(found.value, Mapping) or found.expiration_time <= now:
            return []
        result = []
        for subkey, entry in found.value.items():
            try:
                record = CacheRecord.from_dict(entry.value)
                if entry.expiration_time > now and self._valid_record(identity, record, now) and subkey == record.subkey():
                    result.append(record)
            except (ValueError, TypeError, AttributeError):
                logger.warning('ignored invalid KV directory subrecord')
        return sorted(result, key=lambda r: (r.holder_id, r.object_id))

    async def aregister(self, identity, record, *, ttl_seconds):
        return await asyncio.to_thread(self.register, identity, record, ttl_seconds=ttl_seconds)

    async def alookup(self, identity, *, timeout_s=5.0):
        return await asyncio.to_thread(self.lookup, identity, timeout_s=timeout_s)
