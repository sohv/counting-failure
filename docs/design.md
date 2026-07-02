# Experimental Design — Counting Failure Study

## Design choices

**Uniform pipeline.** The same core suite (Blocks A–F) runs identically on all five models (Llama-3.2-1B, Llama-3.2-3B, Qwen2.5-1.5B, Qwen2.5-3B, Qwen2.5-7B) so cross-family comparison is like for like. Model-specific follow-ups branch off only after Block A reveals each model's failure mode.

**Determinism.** Greedy decoding, temperature 0, ten seeds. Since greedy output is seed-invariant, seeds test pipeline determinism and accuracy is reported as the deterministic output per condition.

**The probe is the anchor.** Block B (linear probe) carries the central claim and is margin-independent, so it runs first and every mechanistic result is judged against it.

**Logit difference, not argmax.** Every layer-wise readout in Blocks D and F reports the wrong-answer logit minus the correct-answer logit, not the winning digit. Near-ties become small signed numbers instead of coin-flip winners. This removes argmax fragility in low-margin regimes.

**Graded causality.** Every correlational localization (Block D) is matched by a causal test (Block E), and each intervention is scored by how far it moves the logit difference toward the correct answer, not by whether the top digit flips.

**Numerical robustness is explicit.** Block F re-runs localization in float32 and across three seeds to confirm the writer layer holds, closing the reproducibility objection with data rather than a disclaimer.

**Tokenization guards.** Before any counting inference, verify the payload tokenizes to exactly n tokens, so a byte-pair merge cannot masquerade as a counting error.

**Branch by failure mode.** Models that fail counting get the full writer-layer localization and causal treatment. Models that solve counting but fail anomaly detection get the anomaly-detection sweep instead. Block A decides the branch.

## Output and logging

**Every experiment writes its full results to a JSON file, including all diagnostics.** No result lives only in stdout. Specifically:

- Each block saves a JSON keyed by model, named `results/<model>/<block>.json`.
- Behavioral blocks save per-seed raw output, parsed answer, expected value, and correctness, not just the summary accuracy.
- The probe blocks save per-layer R² and MAE for both conditions, plus the embedding-control values.
- Logit-lens and decomposition blocks save the full per-layer wrong-minus-correct logit difference, the top-5 tokens, and the per-component (attention vs MLP) contributions at every layer, not only the writer layer.
- Every intervention saves per-n pre- and post-intervention logit differences and the resulting output.
- Every diagnostic table printed to console is also serialized to JSON, including the paraphrase per-layer sweeps, the rank-of-attractor diagnostics, and the per-head attention breakdowns.
- The float32/seed robustness runs save alongside their bfloat16 counterparts under distinct keys so the two can be compared directly.

The rule is simple: if it printed, it is in a JSON file. This guarantees every number in the paper traces to a committed file and survives a rerun check.

## Console output style

Never use decorative separators in experiment output. No lines of `#`, `*`, `=`, `-`, or any other repeated character. No banners like `print("#" * 70)`. Each experiment stage prints its heading as a plain line followed by its results. Nothing else.