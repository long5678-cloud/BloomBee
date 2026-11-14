"""Real torch tensor-tree callbacks for the historical tiered cache.

PyTorch is imported lazily. These helpers never load a model or open a network
connection. Transfers create independent contiguous snapshots, so a retained
slice does not keep its entire original backing allocation alive.
"""

from .tiered import TieredCache

__all__ = ["tree_map", "tree_leaves", "tensor_nbytes", "move_tensor_tree", "make_tensor_cache"]


def tree_map(value, operation):
    """Map tensor leaves in a dict/list/tuple tree, preserving container shape."""
    if isinstance(value, dict):
        return {key: tree_map(child, operation) for key, child in value.items()}
    if isinstance(value, tuple):
        return tuple(tree_map(child, operation) for child in value)
    if isinstance(value, list):
        return [tree_map(child, operation) for child in value]
    return operation(value)


def tree_leaves(value):
    if isinstance(value, dict):
        return [leaf for child in value.values() for leaf in tree_leaves(child)]
    if isinstance(value, (tuple, list)):
        return [leaf for child in value for leaf in tree_leaves(child)]
    return [value]


def tensor_nbytes(value):
    """Sum logical tensor payload bytes, not allocator reserved/RSS/workspace.

    The transfer helper makes independent contiguous leaves before retention.
    Repeated leaves count separately, because they are independently copied.
    Arbitrary caller-owned views must not be treated as an allocation bound.
    """
    import torch

    leaves = tree_leaves(value)
    if not leaves or not all(torch.is_tensor(leaf) for leaf in leaves):
        raise ValueError("expected a nonempty tree containing only tensors")
    if any(leaf.layout != torch.strided for leaf in leaves):
        raise ValueError("only dense strided tensor snapshots are supported")
    count = sum(leaf.numel() * leaf.element_size() for leaf in leaves)
    if count <= 0:
        raise ValueError("tensor snapshots must contain payload bytes")
    return count


def move_tensor_tree(value, tier, *, device="cpu"):
    """Copy an immutable snapshot to the request device or CPU slow tier.

    CPU is a real, runnable default; it is not described as a GPU fast tier.
    CUDA callers explicitly select a device. All CUDA devices involved in the
    transfer are synchronized before returning. This is serial, not an async
    stream/concurrency adapter.
    """
    import torch

    if tier not in ("fast", "slow"):
        raise ValueError("tier must be fast or slow")
    tensor_nbytes(value)
    target = torch.device(device if tier == "fast" else "cpu")
    copied = tree_map(value, lambda tensor: tensor.detach().to(target, copy=True).contiguous())
    devices = {leaf.device for leaf in tree_leaves(value) + tree_leaves(copied) if leaf.is_cuda}
    for cuda_device in devices:
        torch.cuda.synchronize(cuda_device)
    return copied


def make_tensor_cache(
    method,
    fast_bytes,
    slow_bytes=0,
    *,
    validate,
    device="cpu",
    admission_guard=False,
    context=None,
):
    """Construct a TieredCache using actual tensor size and copy callbacks.

    validate remains mandatory: tensor shape alone cannot prove prefix/model
    identity or tensor integrity. With context, use tiered.CacheKey instances.
    """
    return TieredCache(
        method,
        fast_bytes,
        slow_bytes,
        size=tensor_nbytes,
        move=lambda value, tier: move_tensor_tree(value, tier, device=device),
        validate=validate,
        admission_guard=admission_guard,
        context=context,
    )
