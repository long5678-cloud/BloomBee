"""Public utility exports loaded on demand, avoiding metadata import cycles."""
from importlib import import_module
_EXPORTS = {
    'AutoDistributedConfig':'auto_config',
    'AutoDistributedModel':'auto_config',
    'AutoDistributedModelForCausalLM':'auto_config',
    'AutoDistributedModelForSequenceClassification':'auto_config',
    'AutoDistributedSpeculativeModel':'auto_config',
    'declare_active_modules':'dht',
    'get_remote_module_infos':'dht',
}
__all__ = list(_EXPORTS)

def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module('.'+_EXPORTS[name], __name__), name)
    globals()[name] = value
    return value
