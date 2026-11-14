"""Independent cache APIs. CUDA and model packages are loaded only when used."""
from importlib import import_module
from .placement import Group, Layer, Placement
_EXPORTS = {
 "HistoryCache": "history", "CacheKey": "tiered", "TieredCache": "tiered",
 "TokenCache": "token_cache", "DynamoCache": "dynamo", "RetentionRange": "token_rules",
 "create_cache": "api", "describe_options": "api", "make_tensor_cache": "tensor_utils",
 "move_tensor_tree": "tensor_utils", "tensor_nbytes": "tensor_utils",
}
__all__ = ["Group", "Layer", "Placement", *_EXPORTS]
def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module("." + _EXPORTS[name], __name__), name)
    globals()[name] = value
    return value
