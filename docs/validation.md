# Validation

The tested runtime is based on upstream BloomBee commit `0028a5e89ddb634106a2d86fed0197105a377b97`. The exact snapshot is local Git commit `8930ce9` and its [source manifest](../provenance/source_manifest.json) has SHA-256 `52573d9149e4534b6abd74e5f878e04f970041a1b514c6d0977ca64467f258d8`.

## Replayed workload

The two-host test used DeepSeek-R1-Distill-Llama-8B, FP16, and 32 layers. The three consecutive requests correspond to one-based lines 39004 through 39006 of the [Qwen Coder block trace](https://github.com/alibaba-edu/qwen-bailian-usagetraces-anon/blob/5f7439c51ec248a0c585f7d90a41a6f57773b912/qwen_coder_blksz_16.jsonl). Anonymous block IDs were mapped to tokens while preserving shared prefixes. The original input lengths were 395, 4518 and 701 tokens and output lengths were 181, 21 and 32.

## Results

- All 234 output tokens matched the cold reference. All nine sampled logit vectors passed the absolute and relative tolerances of 0.05 and 0.01.
- The first request supplied predictor history. For the third request, LRFU triggered one 16 MiB LMCache transfer to consumer CPU memory. The GPU used the prefix across all 32 layers and skipped 16 input tokens. An independent DHT reader observed one successful reuse.
- The prefix became ready after the original trace arrival and before service began. All three requests took longer than their cold references. The final request took 9.46 seconds with reuse and 2.40 seconds cold.
- In a separate H20 comparison with 773 contiguous pages, the layer-zero attention step took 2.28 ms versus 67.54 ms with the original page loop. This timing covers one layer and one decode step.
- The local regression suite passed 404 tests and skipped seven GPU tests. A separate H20 check passed nine CUDA cases.

The two-host run covers this model, trace segment and LRFU configuration. Additional model families, the full strategy matrix, and fresh environment installation need separate validation. See [installation](installation.md) and the [ordinary request workflow](ordinary-runtime.md).
