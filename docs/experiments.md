## Behavioral characterization

**A1. Three-phase counting.** Present each model with three prompt conditions. P1 is ten identical tokens (apple ×10), correct answer 10. P2 is nine identical plus one intruder (banana at position 5), correct answer 9. P3 is ten distinct words, correct answer 10. Run each under space-separated and comma-separated delimiters. Ten seeds, greedy decoding, temperature 0. Because decoding is greedy the ten seeds test only pipeline determinism, so report accuracy as the deterministic output per condition. Purpose: fix what each model does and whether format changes it.

**A2. Sequence-length sweep.** Run P1 (repeated) and P3 (unique) across n = 5, 6, 7, 8, 9, 10, 11, 12, 15, 20. Single greedy call per n. Record the output digit at each n. Purpose: reveal the attractor structure, the fixed wrong value each model collapses to and the n at which it sets in. Before inference, verify the payload tokenizes to exactly n tokens at every length, so a byte-pair merge cannot masquerade as a counting error.

## Representation probe (the anchor)

**B1. Count probe across layers.** Generate repeated-token prompts for n = 3 to 15 and unique-token prompts for n = 3 to 13. For each prompt, extract the last-token residual stream at every layer. Train a ridge regression probe per layer to predict n, using leave-one-out cross-validation. Report R² and mean absolute error per layer, for the repeated and unique conditions separately. Purpose: establish that the count is linearly present inside the model at each depth, independent of what the model outputs. This block is margin-independent and carries the paper's central claim, so it runs first and everything else is judged against it.

**B2. Embedding control.** Include the embedding layer (layer 0) in the probe sweep. It should give chance-level R². Purpose: confirm the count signal is computed by the transformer, not already present in the token embeddings.

## Attention control

**C1. Word-span attention analysis.** Run a single forward pass per phase with attention output enabled. Restrict attention to the word-list token positions, masking the beginning-of-sequence token and the instruction tokens. Compute per-layer entropy and a uniformity score over that span, for P1, P2, and P3. Purpose: test and rule out the hypothesis that repeated tokens cause attention to collapse. If entropy and uniformity are similar across the failing and non-failing conditions, attention collapse is not the cause. This is a control that closes an alternative explanation, not a positive finding.

## Localization, reported as logit difference

The defining choice for this block: at every layer, do not report the winning digit. Report the logit of the wrong answer minus the logit of the correct answer. A near-tie becomes a small signed number, a decisive write becomes a large one. This removes the fragility of argmax readouts in low-margin regimes.

**D1. Logit lens across layers.** Project the last-token hidden state at each layer through the final layernorm and unembedding. Record the wrong-minus-correct logit difference at every layer for P1. Purpose: show the depth trajectory of the wrong answer taking over, with an honest magnitude at each step rather than a coin-flip winner.

**D2. MLP versus attention decomposition.** At each layer capture three residual-stream states: entering the layer, after the attention sublayer, after the full layer. Project each through the logit lens and record the wrong-minus-correct difference at each of the three points. Attribute the change to attention or to the MLP by which sublayer moved the difference, and by how much. Purpose: identify which component writes the wrong answer, with a graded contribution rather than a binary label.

**D3. Per-n writer input.** At the identified writer layer, record the incoming residual-stream state (as a logit difference) across n = 7 to 15. Purpose: test whether the writer is counting or pattern-matching. If the writer's input does not change with n, it cannot be estimating sequence length, it is responding to a fixed format signature.

## Causal confirmation, graded

Every intervention in this block is scored by how far it moves the wrong-minus-correct logit difference toward the correct answer, not by whether the top digit flips. A patch that moves the correct answer from rank 2 to rank 1, or even closes half the gap, is real causal evidence that a flip criterion would discard.

**E1. Zero-ablation.** Zero the writer MLP's output and measure the resulting logit difference across n. Purpose: test necessity. A large move toward correct means the writer is necessary, a partial move means the mechanism is distributed.

**E2. Mean-ablation.** Replace the writer MLP's output with its mean output computed over the unique-token (P3) condition at matched n. Purpose: a cleaner necessity test than zeroing, because it removes the format-specific signal while preserving on-task activation statistics. Using P3 as the reference holds task, length, and answer magnitude fixed and removes only the repeated-token signature.

**E3. Denoising patch.** Inject the writer MLP's output from a correct case into a failing case at the same layer, norm-matched to avoid injecting an off-distribution activation. Two variants: patch from the unique-token condition into the repeated condition at matched n, and patch from a correct low-n case into a failing high-n case holding the prompt format fixed. Purpose: test sufficiency. If the writer's output alone moves the failing case toward correct, the writer is where the failure enters.

## Numerical robustness

**F1. Precision and seed check.** Re-run Block D (logit lens and decomposition) once in float32 in addition to the default bfloat16, and across three independent seeds. Report whether the identified writer layer and the depth band are stable. Purpose: directly answer the objection that low-margin layer attributions could reorder under a different numerical environment. If the writer holds in float32, the localization is robust. If it moves, you report that honestly and lean on the probe, which is precision-independent.

## Model-conditional branch (decided by Block A, not designed now)

After Block A reveals each model's failure mode, the localization and causal blocks (D and E) attach to whichever failure the model actually exhibits.

Models that fail counting (produce a wrong count on P1) get the full D and E treatment at their counting-writer layer, as described.

Models that solve counting but fail anomaly detection get an anomaly-detection branch instead: sweep the intruder position from 0 to 9 and the intruder count from 1 to 5, record detection as a function of position and quantity, and run the word-span attention analysis on the intruder token specifically, including per-head, to locate where the anomaly signal is present but not expressed. These models have no wrong-count writer to trace, so their mechanistic interest is entirely in the anomaly pathway.

This branch is not specified per model in advance. Block A decides which branch each of the five models takes, which keeps the design honest, you characterize the failure each model actually has rather than forcing every model through the same mechanistic pipeline.