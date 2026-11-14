"""Bounded real HTTP holder: store native snapshots and advertise through real DHT."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import time

from .directory import CacheRecord, KVCacheDirectory
from .identity import CacheIdentity
from .transport import SnapshotHTTPServer, SnapshotStore, _SnapshotHandler, MAGIC


class HolderHandler(_SnapshotHandler):
    def do_GET(self):
        if self.path == '/status':
            data = json.dumps(dict(peer_id=self.server.directory.dht.peer_id.to_base58())).encode()
            self.send_response(200)
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        return super().do_GET()

    def do_POST(self):
        if self.path != '/publish':
            self.send_error(404)
            return
        try:
            size = int(self.headers.get('Content-Length', 0))
            if not 12 < size <= self.server.store.max_object_bytes:
                raise ValueError('bounded content length required')
            self.connection.settimeout(30)
            blob = self.rfile.read(size)
            if len(blob) != size or blob[:8] != MAGIC:
                raise ValueError('truncated snapshot')
            header_size, = struct.unpack('!I', blob[8:12])
            if not 0 < header_size <= 65536 or 12 + header_size > size:
                raise ValueError('invalid header')
            identity = CacheIdentity.from_dict(json.loads(blob[12:12 + header_size])['identity'])
            digest = self.server.store.put(blob)
            peer = self.server.directory.dht.peer_id.to_base58()
            record = CacheRecord(identity.digest(), peer, self.server.public_endpoint,
                digest, digest, size, self.server.directory.clock() + self.server.ttl)
            if not self.server.directory.register(identity, record, ttl_seconds=self.server.ttl):
                raise RuntimeError('DHT publication failed')
            data = json.dumps(record.to_dict()).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except Exception:
            self.send_error(400)


def main():
    from hivemind import DHT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    dht = DHT(initial_peers=cfg.get('initial_peers', []),
              host_maddrs=cfg.get('host_maddrs', ['/ip4/0.0.0.0/tcp/0']), start=True)
    store = SnapshotStore(cfg['store_path'], max_total_bytes=cfg['store_capacity_bytes'],
                          max_object_bytes=cfg.get('max_object_bytes', 256 * 1024**2))
    directory = KVCacheDirectory(dht, allowed_endpoints=[cfg['public_endpoint']])
    # CPU holder binds only configured interface; uploader must use approved private network.
    server = SnapshotHTTPServer(store, tuple(cfg['listen']))
    server.RequestHandlerClass = HolderHandler
    server.directory, server.public_endpoint, server.ttl = directory, cfg['public_endpoint'], cfg.get('ttl_s', 3600)
    ready = dict(peer_id=dht.peer_id.to_base58(), endpoint=cfg['public_endpoint'],
                 visible_maddrs=[str(x) for x in dht.get_visible_maddrs()])
    with open(cfg['ready_path'], 'x') as stream:
        json.dump(ready, stream)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        dht.shutdown()


if __name__ == '__main__':
    main()
