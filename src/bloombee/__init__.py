"""BloomBee public API with model dependencies loaded on first model access.

KV policy and telemetry imports do not import Transformers or start DHT/CUDA.
The original model initialization remains in _runtime_init.
"""
import importlib
import threading
__version__ = "2.3.0.dev2"
_LOCK = threading.RLock()
_RUNTIME_LOADING = False
__all__ = ['DistributedLoRA', 'LoRAConfig', 'AutoConfig', 'AutoDistributedConfig', 'AutoDistributedModel', 'AutoDistributedModelForCausalLM', 'AutoDistributedModelForSequenceClassification', 'AutoDistributedSpeculativeModel', 'ClientConfig', 'DistributedBloomConfig', 'DistributedBloomForCausalLM', 'DistributedBloomForSequenceClassification', 'DistributedBloomModel', 'DistributedDeepseekV3Config', 'DistributedDeepseekV3ForCausalLM', 'DistributedDeepseekV3ForSequenceClassification', 'DistributedDeepseekV3Model', 'DistributedFalconConfig', 'DistributedFalconForCausalLM', 'DistributedFalconForSequenceClassification', 'DistributedFalconModel', 'DistributedGemma4Config', 'DistributedGemma4ForCausalLM', 'DistributedGemma4ForSequenceClassification', 'DistributedGemma4Model', 'DistributedGlm4MoeLiteConfig', 'DistributedGlm4MoeLiteForCausalLM', 'DistributedGlm4MoeLiteModel', 'DistributedLlamaConfig', 'DistributedLlamaForCausalLM', 'DistributedLlamaForSequenceClassification', 'DistributedLlamaForSpeculativeGeneration', 'DistributedLlamaModel', 'DistributedMixtralConfig', 'DistributedMixtralForCausalLM', 'DistributedMixtralForSequenceClassification', 'DistributedMixtralModel', 'DistributedNemotronHConfig', 'DistributedNemotronHForCausalLM', 'DistributedNemotronHModel', 'DistributedQwen3Config', 'DistributedQwen3ForCausalLM', 'DistributedQwen3ForSequenceClassification', 'DistributedQwen3Model', 'InferenceSession', 'NoSpendingPolicy', 'RemoteSequenceManager', 'RemoteSequential', 'SpendingPolicyBase', 'WrappedBloomBlock', 'WrappedDeepseekV3Block', 'WrappedFalconBlock', 'WrappedGemma4Block', 'WrappedGlm4MoeLiteBlock', 'WrappedLlamaBlock', 'WrappedMixtralBlock', 'WrappedNemotronHBlock', 'WrappedQwen3Block', 'declare_active_modules', 'get_remote_module_infos', 'register_model_classes']

def __getattr__(name):
    global _RUNTIME_LOADING
    if name.startswith("_") or name not in __all__:
        raise AttributeError(name)
    with _LOCK:
        runtime = importlib.import_module("bloombee._runtime_init")
        for key, value in vars(runtime).items():
            if not key.startswith("_"):
                globals()[key] = value
        if name not in globals():
            raise AttributeError(name)
        return globals()[name]
