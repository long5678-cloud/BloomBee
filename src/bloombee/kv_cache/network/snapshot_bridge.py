"""Independent exact-prefix bridge. Uses genuine LMCache SDK; no predictor.

Original snapshot identity, checksum, dtype and physical-head layout stay strict.
One stable model namespace describes static execution/layout. Full prefix
identities are isolated by native cache_salt; tokens still determine chunk keys.
"""
from pathlib import Path
import hashlib,importlib.util,json,os
import torch
from bloombee.kv_cache import identity as identity_api, transport

class SnapshotLayout:
    def __init__(self,identities,tokens,*,max_bytes=256*1024**2):
        self.identities=[identity_api.CacheIdentity.from_dict(i).to_dict() for i in identities]
        self.tokens=list(tokens);self.max_bytes=max_bytes
        if not self.identities or len(self.tokens)%16 or len(self.tokens)==0:
            raise ValueError('requires nonempty complete 16-token chunks')
        base={k:v for k,v in self.identities[0].items() if k not in ('layer_start','layer_end')}
        for offset,i in enumerate(self.identities):
            if ({k:v for k,v in i.items() if k not in ('layer_start','layer_end')}!=base or
                i['layer_start']!=self.identities[0]['layer_start']+offset or i['layer_end']!=i['layer_start']+1 or
                i['prefix_length']!=len(self.tokens) or i['prefix_tokens_digest']!=identity_api.token_ids_digest(self.tokens)):
                raise ValueError('incompatible layer bundle or token identity')
        self.heads=base['num_attention_heads'];self.dim=base['head_dim'];self.dtype=getattr(torch,base['dtype'])
        self.shape=(2,len(self.identities),len(self.tokens),self.heads*self.dim)
        if torch.empty((),dtype=self.dtype).element_size()*__import__('math').prod(self.shape)>max_bytes:
            raise ValueError('native tensor exceeds byte budget')
        variable={'layer_start','layer_end','prefix_tokens_digest','prefix_length','position_digest'}
        static={k:v for k,v in self.identities[0].items() if k not in variable}
        self.model_name='bb_native_'+identity_api.canonical_digest(dict(static=static,layers=len(self.identities)))
        self.cache_salt=identity_api.canonical_digest(self.identities)

    def to_native(self,blobs,digests):
        if len(blobs)!=len(self.identities) or len(digests)!=len(blobs) or sum(map(len,blobs))>self.max_bytes:
            raise ValueError('incomplete bundle or byte budget')
        layers=[]
        for blob,digest,i in zip(blobs,digests,self.identities):
            if hashlib.sha256(blob).hexdigest()!=digest:raise ValueError('snapshot checksum mismatch')
            t=transport.unpack_snapshot(blob,i,max_bytes=self.max_bytes)
            shape=(1,self.heads,len(self.tokens),self.dim)
            if set(t)!={'key','value'} or any(tuple(x.shape)!=shape or x.dtype!=self.dtype for x in t.values()):
                raise ValueError('snapshot physical geometry/dtype mismatch')
            layers.append(torch.stack([t[n][0].permute(1,0,2).reshape(len(self.tokens),-1) for n in ('key','value')]))
        return torch.stack(layers,dim=1).contiguous()

    def from_native(self,tensor):
        if tuple(tensor.shape)!=self.shape or tensor.dtype!=self.dtype:
            raise ValueError('native geometry/dtype mismatch')
        blobs=[]
        for layer,i in enumerate(self.identities):
            tensors={name:tensor[plane,layer].reshape(len(self.tokens),self.heads,self.dim).permute(1,0,2).unsqueeze(0).contiguous() for plane,name in enumerate(('key','value'))}
            blobs.append(transport.pack_snapshot(i,tensors,max_bytes=self.max_bytes))
        return blobs

class LMCacheSnapshotBridge:
    def __init__(self,layout,sdk_context):
        self.layout=layout;self.context=sdk_context
    def store(self,blobs,digests):
        return self.context.store(self.layout.to_native(blobs,digests),self.layout.tokens,cache_salt=self.layout.cache_salt)
    def retrieve(self):
        tensor=self.context.retrieve(self.layout.tokens,cache_salt=self.layout.cache_salt)
        return None if tensor is None else self.layout.from_native(tensor)
