"""Actual TransformerBackend automatic chunks and production Llama TP on CPU.

The weights are deterministic local miniature Llama weights, loaded by the real
FlexGen production loader. No forward stub, custom test model, patched chunk
estimator or fake page data provider. CPU component evidence, not ordinary RPC
or the fixed 8B GPU acceptance owned by 13G.
"""
import asyncio
from dataclasses import replace

import numpy as np
import pytest
import torch

from bloombee.data_structures import InferenceMetadata
from bloombee.server.backend import TransformerBackend
from bloombee.server.dynamo_pages import EnginePagePool
from bloombee.server.memory_cache_manager import KVCacheManager
from bloombee.server.flexgen_tensor_parallel import FlexgenLlamaTensorParallel
from bloombee.flexgen_utils.ExecutionEnv import ExecutionEnv
from bloombee.flexgen_utils.pytorch_backend import TorchDevice
from bloombee.flexgen_utils.policy import Policy
from bloombee.flexgen_utils.compression import CompressionConfig
from bloombee.utils.hivemind_compat import BatchTensorDescriptor
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer


def build_backend(tmp_path):
    torch.manual_seed(18)
    config = LlamaConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                         num_attention_heads=4, num_key_value_heads=4, vocab_size=32)
    config.name = "local-llama"
    template = LlamaDecoderLayer(config, layer_idx=0).eval()
    cpu = TorchDevice('cpu')
    env = ExecutionEnv(gpu=cpu, cpu=cpu, disk=None, mixed=None)
    template.env, template.path = env, str(tmp_path)
    weights = tmp_path/'local-llama-np'
    weights.mkdir()
    def save(name, value):
        with (weights/name).open('xb') as f:
            np.save(f, value.detach().cpu().half().numpy())
    save('embed_tokens.weight', torch.randn(32,16)*.02)
    for name, value in template.state_dict().items():
        save('layers.0.'+name, value)
    save('layers.0.self_attn.rotary_emb.inv_freq', 1/(10000**(torch.arange(0,4,2).float()/4)))
    # Latest upstream resolves an actual local HF source and writes an
    # identity/revision-bound conversion manifest; do not bypass its loader.
    source = tmp_path/'local-llama'
    source.mkdir()
    config.save_pretrained(source)
    state = {'model.layers.0.'+name:value for name,value in template.state_dict().items()}
    state['model.embed_tokens.weight'] = torch.randn(32,16)*.02
    state['model.layers.0.self_attn.rotary_emb.inv_freq'] = 1/(10000**(torch.arange(0,4,2).float()/4))
    torch.save(state,source/'pytorch_model.bin')
    config._name_or_path = str(source)
    compression = CompressionConfig(num_bits=4, group_size=64, group_dim=0, symmetric=False)
    policy = Policy(gpu_batch_size=1, num_gpu_batches=1, w_gpu_percent=100, w_cpu_percent=0,
                    cache_gpu_percent=100, cache_cpu_percent=0, act_gpu_percent=100,
                    act_cpu_percent=0, overlap=False, sep_layer=True, pin_weight=False,
                    cpu_cache_compute=False, attn_sparsity=1., compress_weight=False,
                    comp_weight_config=compression, compress_cache=False, comp_cache_config=compression)
    module = FlexgenLlamaTensorParallel(template, config, [torch.device('cpu')]*2,
                                        torch.device('cpu'), policy=policy).eval()
    manager = KVCacheManager(256, 30, policy, env, config)
    descriptor = BatchTensorDescriptor(1,16,dtype=torch.float16)
    backend = TransformerBackend('ordinary-local-layer.0', module,
                                 args_schema=(descriptor,), outputs_schema=(descriptor,),
                                 max_batch_size=1, config=config, cache_manager=manager,
                                 pruner_manager=None, is_last_block=False,
                                 backend_dtype=torch.float16, max_chunk_size_bytes=96)
    descriptors = backend.get_inference_cache_descriptors(1,16)
    handles = asyncio.run(manager.cache._schedule_alloc(manager.get_allocation_size_tokens(*descriptors),
                                                       *descriptors, timeout=0))
    info = InferenceMetadata('ordinary-local-layer.0',0,handles,None,
                              full_batch_size=1,micro_batch_size=1)
    pool = EnginePagePool(8,2,4,4,dtype=torch.float16,device='cpu')
    return backend, manager, module, pool, info


def full_reference(module, inputs):
    length = inputs.shape[1]
    mask = torch.zeros(1,1,length,length)
    mask.masked_fill_(torch.triu(torch.ones(length,length,dtype=torch.bool),1)[None,None], -65504.)
    with torch.inference_mode():
        return module(inputs.clone(), attention_mask=mask,
                      rotary_position_ids=torch.arange(length)[None], use_cache=True)[0]


def run(backend, manager, session, info, inputs, callback=None):
    # Budget selects two tokens per chunk through the ACTUAL memory estimator.
    worst = info.prefix_length + inputs.shape[1]
    backend.max_chunk_size_bytes = 2 * backend._max_shard_num_heads * backend.dtype_bytes * worst
    assert backend._estimate_max_chunk_length(inputs, info) == 2
    with manager.bind_dynamo_pages(info.cache_handles,session,on_chunk_complete=callback):
        return backend.inference_step(inputs.clone(),torch.tensor([0]),info)[0]


def test_ordinary_backend_at_least_three_chunks_high_low_high_and_hot_suffix(tmp_path):
    backend, manager, module, pool, info = build_backend(tmp_path)
    torch.manual_seed(21)
    inputs = torch.randn(1,12,16,dtype=torch.float16)*.2
    first = pool.begin()
    progress = []
    priorities = [80,0,80]  # explicit known input; callback never reads future KV
    def completed(*,session,start,end,inputs,outputs):
        assert session.length == end
        assert inputs.shape == outputs.shape == (1,2,16)
        blocks = [session.block(i) for i in range(end//2)]
        with pytest.raises(IndexError):
            session.block(end//2)  # no uncomputed next block may be observed
        progress.append(dict(start=start,end=end,completed_priorities=priorities[:end//2],
                             physical=[(b['slot'],b['generation']) for b in blocks]))
    cold = run(backend,manager,first,info,inputs[:,:6],completed)
    torch.testing.assert_close(cold,full_reference(module,inputs[:,:6]),atol=2e-3,rtol=2e-3)
    assert [(p['start'],p['end']) for p in progress] == [(0,2),(2,4),(4,6)]
    assert [p['completed_priorities'] for p in progress] == [[80],[80,0],[80,0,80]]
    assert first.length == 6  # no duplicate final append
    first.publish([('request',2),('request',4),('request',6)],priorities)
    first.release()
    hot = pool.begin([('request',2),('request',4),('request',6)])
    before = [hot.block(i)['key'].data_ptr() for i in range(3)]
    pool.drain_events()
    hot_progress = []
    def hot_completed(**event):
        hot_progress.append((event['start'],event['end']))
    suffix = run(backend,manager,hot,replace(info,prefix_length=6),inputs[:,6:],hot_completed)
    torch.testing.assert_close(suffix,full_reference(module,inputs)[:,6:],atol=2e-3,rtol=2e-3)
    assert hot_progress == [(6,8),(8,10),(10,12)]
    assert [hot.block(i)['key'].data_ptr() for i in range(3)] == before
    attention = [e for e in pool.events if e['event']=='attention_direct_pages']
    assert [e['prefix_tokens'] for e in attention] == [6,6,8,8,10,10]  # two real TP shards
    assert all(e['history_kv_gather_bytes']==0 for e in attention)
    assert hot.length == 12
    hot.release()
    manager.cache.force_free(*info.cache_handles)


def test_chunk_callback_failure_cleans_binding_without_computing_future_chunks(tmp_path):
    backend, manager, module, pool, info = build_backend(tmp_path)
    session = pool.begin()
    seen=[]
    def stop(**event):
        seen.append((event['start'],event['end']))
        raise ValueError('consumer failure')
    with pytest.raises(ValueError,match='consumer failure'):
        run(backend,manager,session,info,torch.randn(1,6,16,dtype=torch.float16),stop)
    assert seen == [(0,2)] and session.length == 2
    assert manager.dynamo_page_session() is None
    session.release()
    assert pool.describe()['active_pages']==0
    manager.cache.force_free(*info.cache_handles)
