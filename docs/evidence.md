# Evidence boundaries

## Historical real-model integration

The selected integrated snapshot matched 193 source files in the final execution archive (168 BloomBee, 7 native, 14 common, 4 strategy). The coordinator separately checked 198 selected files including five ordinary entrypoints. These are source-identity checks, not proof that a reorganized directory starts correctly.

The final short batch had ten groups, 60/60 complete requests, 180 generated tokens, matching complete outputs and first-three-step score vectors. Eight Remote configurations each had one B→A official LMCache 16 MiB transfer consumed by the ordinary 32-layer model, then two local retained reuses. Three Telemetry successful uses do not mean three new transfers. The model was DeepSeek-R1-Distill-Llama-8B at revision `6a6f4aa4197940add57724a7707d069478df56b1`, FP16.

R1–R6 used controller c1; R7/R8 used d1. d1 fixes a worker-done/file-arrival/exit race. Current working group_config differs from the archived short-run version; historical provenance is unchanged.

This was a mapped short workload (33 input/3 output tokens), not original full coding requests or 43,011 full-model requests. Local configurations disabled distributed Telemetry and the predictor while Remote enabled them, so throughput is a whole-configuration comparison, not a pure cache-location causal experiment or hardware limit.

## What remains negative or unproved

The trace strategies did not establish a winner or general acceleration. Natural queue-head arrival before any history remains cold; successful prediction ready and adopted before natural arrival was not demonstrated. The selected warm in the final short batch happened after ingress and its wait was timed. The endpoint currently prefetches into CPU, not GPU.

Earlier v0.5.6 explicit-known-prefix tests showed prearrival transfer when the correct prefix was supplied. Later v0.5.7 trace/strategy tests showed candidate selection and model adoption but zero naturally arriving, history-backed queue-head requests with ready-and-adopted prefetch. These are different experiments and must not be combined into a universal prediction claim.

Old failures, numeric non-bit-exact outcomes, IPC repairs and evidence-only receipt recovery remain in the internal archive. They are not relabeled as successful model reruns. A measured reuse event and numerical correctness are separate fields.

See [validation](validation.md) for the current two-host run.
