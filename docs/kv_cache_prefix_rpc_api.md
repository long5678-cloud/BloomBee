# Legacy prefix RPC

The inherited native prefix-RPC extension remains available separately from the ordinary page-bundle integration. Its metadata key is `remote_prefix`; the new ordinary page path uses `dynamo_cache`. A request must not mix both extensions. Compatibility code remains in `prefix_generation.py`, `prefix_context.py`, `prefix_holder.py`, `prefix_rpc.py` and `remote_prefix_session.py`. This delivery does not claim new GPU acceptance of the legacy standalone extension.
