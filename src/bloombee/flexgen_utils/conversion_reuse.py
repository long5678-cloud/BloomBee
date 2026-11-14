"""Explicit reuse of previously verified immutable np-v1 assets.

This verifies the prior full-hash receipt and current file identities, without
rehashing large tensors or changing the source/conversion directories. It is
not a new full-content audit or protection against a malicious filesystem.
"""
import hashlib,json,os
from pathlib import Path,PurePosixPath
from functools import lru_cache


def _read(path):
    raw=Path(path).read_bytes()
    return json.loads(raw),hashlib.sha256(raw).hexdigest()


def _identity(path):
    s=path.stat()
    return dict(device=s.st_dev,inode=s.st_ino,size=s.st_size,mtime_ns=s.st_mtime_ns,
        ctime_ns=s.st_ctime_ns,mode=s.st_mode,uid=s.st_uid,gid=s.st_gid)


def _safe(root,name):
    p=PurePosixPath(name)
    if p.is_absolute() or not p.parts or '..' in p.parts:raise ValueError('unsafe conversion receipt path')
    return root.joinpath(*p.parts)


@lru_cache(maxsize=4)
def _verified(source,revision,conversion_root,receipt_path,attestation_path):
    source=Path(source).resolve();root=Path(conversion_root).resolve()
    proof,_=_read(receipt_path);attestation,a_sha=_read(attestation_path);binding,b_sha=_read(root/'binding.json')
    if proof.get('schema')!=1 or binding.get('schema')!=1:raise ValueError('unsupported prior conversion receipt schema')
    if a_sha!=proof.get('attestation_sha256') or b_sha!=proof.get('binding_sha256'):
        raise ValueError('prior asset receipt control SHA mismatch')
    if not revision or attestation.get('revision')!=revision:
        raise ValueError('preconverted source revision mismatch; supply its full pinned revision')
    if proof.get('source_hashes')!=attestation.get('files') or proof.get('conversion_hashes')!=binding.get('files'):
        raise ValueError('prior full-hash inventories disagree')
    for kind,folder in [('source',source),('conversion',root)]:
        identities=proof.get(kind)
        if not isinstance(identities,dict) or not identities or set(identities)!=set(proof[kind+'_hashes']):
            raise ValueError('incomplete prior '+kind+' identity inventory')
        for name,expected in identities.items():
            if _identity(_safe(folder,name))!=expected:
                raise ValueError('immutable '+kind+' asset changed: '+name)
    top={PurePosixPath(name).parts[0] for name in binding['files']}
    if len(top)!=1:raise ValueError('one complete conversion directory required')
    result=root/next(iter(top))
    if not (result/'embed_tokens.weight').is_file() or not (result/'.bloombee_np_converted').is_file():
        raise ValueError('incomplete prior np-v1 conversion')
    return str(result)


def configured_conversion(source,revision):
    names=['BLOOMBEE_PRECONVERTED_ROOT','BLOOMBEE_ASSET_REUSE_RECEIPT','BLOOMBEE_SOURCE_ATTESTATION']
    values=[os.environ.get(name) for name in names]
    if not any(values):return None
    if not all(values):raise ValueError('all three explicit preconverted asset settings are required')
    return _verified(str(Path(source).resolve()),revision,*values)
