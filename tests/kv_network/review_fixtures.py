"""Unmodified latest-upstream real CPU manager fixture for isolated review."""
import pytest
@pytest.fixture
def make_cpu_cache_manager():
    """Real cache managers without CUDA or disk copy threads."""
    from bloombee.flexgen_utils.compression import CompressionConfig
    from bloombee.flexgen_utils.ExecutionEnv import ExecutionEnv
    from bloombee.flexgen_utils.policy import Policy
    from bloombee.flexgen_utils.pytorch_backend import TorchDevice, fix_recursive_import
    from bloombee.server.memory_cache_manager import KVCacheManager
    managers = []

    def make(config):
        fix_recursive_import()
        device = TorchDevice('cpu')
        compression = CompressionConfig(num_bits=4, group_size=64, group_dim=0, symmetric=False)
        policy = Policy(gpu_batch_size=1, num_gpu_batches=1, w_gpu_percent=100, w_cpu_percent=0, cache_gpu_percent=0, cache_cpu_percent=100, act_gpu_percent=100, act_cpu_percent=0, overlap=False, sep_layer=True, pin_weight=False, cpu_cache_compute=False, attn_sparsity=1.0, compress_weight=False, comp_weight_config=compression, compress_cache=False, comp_cache_config=compression)
        manager = KVCacheManager(4096, None, policy, ExecutionEnv(gpu=device, cpu=device), config)
        managers.append(manager)
        return manager
    yield make
    for manager in managers:
        manager.wait_for_pending_reorder()
        manager._reorder_executor.shutdown(wait=True)
