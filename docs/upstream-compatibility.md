# Upstream compatibility

This integration is based on [BloomBee commit 0028a5e](https://github.com/ai-decentralized/BloomBee/commit/0028a5e89ddb634106a2d86fed0197105a377b97). The original local retention policies were published in fork commit `9612af3`.

Three manual conflicts were resolved:

1. Backend chunk processing: physical page sessions append only new K/V per chunk, refresh page operands and invoke the completion callback. Dense latest-upstream GLM MLA record/concat logic remains under the mutually exclusive nonpage branch. Dense finalization is skipped for page sessions to prevent duplicate writes. The Dynamo owner independently compared the retained GLM/dense AST branches with latest HEAD and found no blocking issue in this limited static audit.
2. Client session history: retain latest `n_new_tokens`, append/recovery history logic and current upstream fixes while preserving the legacy `remote_prefix` extension and new explicit `dynamo_cache` metadata. These are separate opt-in protocols; mixed use is rejected.
3. Package initialization: defer heavy model initialization for independent KV APIs, retain the latest upstream initialization body and new `DistributedLoRA`/`LoRAConfig` exports. A real `AutoDistributedModelForCausalLM` import was checked locally; no model package was replaced by a stub.

Page binding continues to restrict the ordinary native page implementation to supported Llama execution. GLM remains on its normal dense MLA path; this change does not add Dynamo paging to GLM. CPU mini-model/allocator tests and the latest upstream GLM/recovery tests are run against this checkout, with exact scope in validation.

The 32-layer Llama integration was exercised on two H20 hosts. See [validation](validation.md) for the model, workload and measured results.
