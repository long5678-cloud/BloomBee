"""Scoped bridge to the existing FlexGen in-place dense attention branch."""
from contextlib import contextmanager
from contextvars import ContextVar

_native = ContextVar("bloombee_dynamo_native", default=None)


@contextmanager
def native_execution(key, value):
    observation = {"key": key, "value": value, "observed": False}
    token = _native.set(observation)
    try:
        yield observation
    finally:
        _native.reset(token)


def native_attention_storage(past_key, past_value, src_tokens):
    """Return full capacity only if past really aliases this leased slab.

    The standard past shape must still report history length to positional
    logic. Only FlexGen's cache_read_buf receives full capacity; its existing
    mha_gen_llama branch then writes new suffix in place instead of torch.cat.
    """
    state = _native.get()
    if state is None:
        return past_key, past_value
    key, value = state["key"], state["value"]
    for view, backing in ((past_key, key), (past_value, value)):
        if (view.untyped_storage().data_ptr() != backing.untyped_storage().data_ptr()
                or view.device != backing.device or view.dtype != backing.dtype
                or view.storage_offset() != backing.storage_offset()
                or backing.shape[0] < src_tokens or view.shape[1:] != backing.shape[1:]):
            raise RuntimeError("attention history does not alias leased native storage")
    state["observed"] = True
    state["capacity"] = int(key.shape[0])
    return key, value
