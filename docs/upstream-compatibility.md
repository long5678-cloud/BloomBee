# Latest upstream compatibility

The upstream main read on 2026-11-11 was `0028a5e89ddb634106a2d86fed0197105a377b97`. The fork main was `bb99922071de23f2bf4ccf204c4a30ea64ac1c32`; its original module branch was `9612af3f4002a69f74ec317b7e3e97af61bcadf6`. Upstream added prompt-tuning fixes and standard LoRA/backward work in 93 changed files. This review branch starts at the current upstream SHA and applies the KV integration as local uncommitted changes.

The earlier fork-based assembled candidate was retained separately. The original framework working tree and all historical experiment assets were left intact.

Three manual conflicts were resolved:

1. Backend chunk processing: physical page sessions append only new K/V per chunk, refresh page operands and invoke the completion callback. Dense latest-upstream GLM MLA record/concat logic remains under the mutually exclusive nonpage branch. Dense finalization is skipped for page sessions to prevent duplicate writes. The Dynamo owner independently compared the retained GLM/dense AST branches with latest HEAD and found no blocking issue in this limited static audit.
2. Client session history: retain latest `n_new_tokens`, append/recovery history logic and current upstream fixes while preserving the legacy `remote_prefix` extension and new explicit `dynamo_cache` metadata. These are separate opt-in protocols; mixed use is rejected.
3. Package initialization: defer heavy model initialization for independent KV APIs, retain the latest upstream initialization body and new `DistributedLoRA`/`LoRAConfig` exports. A real `AutoDistributedModelForCausalLM` import was checked locally; no model package was replaced by a stub.

Page binding continues to restrict the ordinary native page implementation to supported Llama execution. GLM remains on its normal dense MLA path; this change does not add Dynamo paging to GLM. CPU mini-model/allocator tests and the latest upstream GLM/recovery tests are run against this checkout, with exact scope in validation.

An actual 32-layer GPU/native model acceptance on this merged SHA is not covered by local CPU tests. Its numerical output, cross-host native extension build, model/cache assets, tensor copy and real ordinary RPC adoption need a separate explicitly scoped runtime check. Existing older-SHA GPU evidence is preserved but does not certify this merge.
