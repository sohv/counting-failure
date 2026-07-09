# Usage

All commands use `uv run -m` from the project root.

## Models

| Key | HuggingFace model | Layers | Ablation sites (`cfg.ablation_sites`) |
|---|---|---|---|
| `llama-1b` | `meta-llama/Llama-3.2-1B-Instruct` | 16 | `[14]` |
| `llama-3b` | `meta-llama/Llama-3.2-3B-Instruct` | 28 | `[22, 26]` |
| `qwen-1.5b` | `Qwen/Qwen2.5-1.5B-Instruct` | 28 | `[22, 24]` |
| `qwen-3b` | `Qwen/Qwen2.5-3B-Instruct` | 36 | n/a (solves P1) |
| `qwen-7b` | `Qwen/Qwen2.5-7B-Instruct` | 28 | n/a (solves P1) |

Qwen models have no pre-set critical layers — `mechanistic.py` auto-discovers them from the logit lens output. `llama-3b` and `qwen-1.5b` have more than one ablation site, so their causal experiments run individual-site and joint (all-sites-at-once) interventions; `llama-1b` has a single site, so joint is skipped.

## Experiment structure

```
src/experiments/
├── behavioral.py          ← all 5 models
├── n_sweep_extended.py    ← all 5 models (extended range n=25-40 mainly meaningful for the 3 prior-active models)
├── token_generality.py    ← all 5 models
├── interleaved_noise.py   ← llama-1b, llama-3b, qwen-1.5b only (prior-active models)
├── llama/                 ← Llama-specific mechanistic pipeline
│   ├── attention.py
│   ├── linear_probe.py
│   ├── logit_lens.py
│   ├── causal.py
│   ├── robustness.py
│   ├── interleaved_probe.py  ← probe + decomposition for the interleaved-noise variants
│   └── run_all.py
└── qwen/                  ← Qwen-specific mechanistic pipeline
    ├── mechanistic.py
    ├── diagnostics.py
    ├── causal.py              ← qwen-1.5b only (only Qwen model with a counting-writer)
    ├── robustness.py          ← qwen-1.5b only
    ├── interleaved_probe.py   ← qwen-1.5b only, same role as the Llama version
    └── run_all.py
```

---

## Behavioral (all models)

Phases 1–3, n-sweep, fixed prompts, paraphrase robustness, language robustness, repeated symbols (with tokenization guard + fallback substitution), chain-of-thought probes.

```bash
uv run -m src.experiments.behavioral --model llama-1b
uv run -m src.experiments.behavioral --model llama-3b
uv run -m src.experiments.behavioral --model qwen-1.5b
uv run -m src.experiments.behavioral --model qwen-3b
uv run -m src.experiments.behavioral --model qwen-7b
```

Output: `results/<model>/behavioral.json`

---

## Extended n-sweep (all models)

Fills n = 13, 14, 16-19 around `behavioral.py`'s n-sweep (5-12, 15, 20), and extends past n=20 to n = 25, 30, 35, 40. The far range only tests attractor persistence for the three prior-active models (llama-1b, llama-3b, qwen-1.5b) — on qwen-3b/7b, which solve counting, it tests where correct counting degrades instead, a different question. Single greedy call per n (deterministic decoding). Depends on `behavioral.json`.

```bash
uv run -m src.experiments.n_sweep_extended --model llama-1b
uv run -m src.experiments.n_sweep_extended --model llama-3b
uv run -m src.experiments.n_sweep_extended --model qwen-1.5b
uv run -m src.experiments.n_sweep_extended --model qwen-3b
uv run -m src.experiments.n_sweep_extended --model qwen-7b
```

Output: `results/<model>/n_sweep_extended.json`

---

## Token generality (all models)

Tests whether the counting prior is a property of the repeated-word-list format or specific to the token "apple". Validates 2 replacement nouns as single-token (with leading space) across all 5 tokenizers before running, then re-runs P1/P2/P3 under space- and comma-separated formats, keeping banana as the P2 intruder. Depends on `behavioral.json`.

```bash
uv run -m src.experiments.token_generality --model llama-1b
uv run -m src.experiments.token_generality --model llama-3b
uv run -m src.experiments.token_generality --model qwen-1.5b
uv run -m src.experiments.token_generality --model qwen-3b
uv run -m src.experiments.token_generality --model qwen-7b
```

Output: `results/<model>/token_generality.json`

---

## Interleaved noise (llama-1b, llama-3b, qwen-1.5b only)

Tests whether newline/pipe delimiters (with and without a matching "-separated list" instruction hint) suppress the repeated-token counting prior on phase1_baseline, compared against the existing space and comma baselines. Scoped to the three prior-active models — on qwen-3b/7b the prior is benign, so this tests nothing. Depends on `behavioral.json`.

```bash
uv run -m src.experiments.interleaved_noise --model llama-1b
uv run -m src.experiments.interleaved_noise --model llama-3b
uv run -m src.experiments.interleaved_noise --model qwen-1.5b
```

Output: `results/<model>/interleaved_noise.json`, `results/<model>/interleaved_noise.png`

---

## Llama pipeline

### Run all Llama stages at once

```bash
uv run -m src.experiments.llama.run_all --model llama-1b
uv run -m src.experiments.llama.run_all --model llama-3b
```

Runs all 10 stages (4 common + 6 Llama-specific) in order as subprocesses. Stops on first failure. To run a subset:

```bash
uv run -m src.experiments.llama.run_all --model llama-1b --only behavioral logit_lens causal
```

Stage names: `behavioral`, `n_sweep_extended`, `token_generality`, `interleaved_noise`, `linear_probe`, `attention`, `logit_lens`, `causal`, `robustness`, `interleaved_probe`.

### linear_probe

Ridge regression probe (LOO cross-val) on residual stream activations across all layers.

```bash
uv run -m src.experiments.llama.linear_probe --model llama-1b
uv run -m src.experiments.llama.linear_probe --model llama-3b
```

Output: `results/<model>/linear_probe.json`

### attention

Per-head attention from last token to word-list span — entropy and uniformity per layer.

```bash
uv run -m src.experiments.llama.attention --model llama-1b
uv run -m src.experiments.llama.attention --model llama-3b
```

Output: `results/<model>/attention.json`, `results/<model>/attention.png`

### logit_lens

Logit lens across layers, MLP/attention decomposition, per-n writer input, paraphrase decomposition. Wrong-minus-correct logit difference. Depends on `behavioral.json`.

```bash
uv run -m src.experiments.llama.logit_lens --model llama-1b
uv run -m src.experiments.llama.logit_lens --model llama-3b
```

Output: `results/<model>/logit_lens.json`

### causal

Zero-ablation (E1), mean-ablation (E2), denoising patch (E3), residual patching sweep, steering vector — all at `cfg.lockin_layer`. Also runs mean-ablation and targeted patch individually at each site in `cfg.ablation_sites`, plus jointly across all sites when there is more than one (llama-3b: L22 and L26). Depends on `behavioral.json` and `logit_lens.json`.

```bash
uv run -m src.experiments.llama.causal --model llama-1b
uv run -m src.experiments.llama.causal --model llama-3b
```

Output: `results/<model>/causal.json` (includes `mean_ablation_sites` and `patch_sites` keyed by `L<n>` per site plus `joint`)

### robustness

Logit lens + decomposition in bfloat16 and float32 across 3 seeds. Checks writer-layer stability across numerical environments. Depends on `behavioral.json` and `logit_lens.json`.

```bash
uv run -m src.experiments.llama.robustness --model llama-1b
uv run -m src.experiments.llama.robustness --model llama-3b
```

Output: `results/<model>/robustness.json`

### interleaved_probe

Count-probe generalization (fit on space-separated prompts, evaluated out-of-distribution on each interleaved-noise variant) and MLP/attention decomposition at the fixed lock-in layer (never re-detected per condition). Regenerates the interleaved-noise prompts directly from `INTERLEAVED_VARIANTS` rather than depending on `interleaved_noise.json`. Depends on `behavioral.json` and `logit_lens.json`.

```bash
uv run -m src.experiments.llama.interleaved_probe --model llama-1b
uv run -m src.experiments.llama.interleaved_probe --model llama-3b
```

Output: `results/<model>/interleaved_mechanistic.json`

### Llama stage dependencies

```
behavioral.json ──┬──> logit_lens ──┬──> causal
                  │                 ├──> robustness
                  │                 └──> interleaved_probe
                  └── (linear_probe and attention are independent)
```

---

## Qwen pipeline

Run `behavioral` first (shared), or use `qwen.run_all` to run every stage in order.

### Run all Qwen stages at once

```bash
uv run -m src.experiments.qwen.run_all --model qwen-1.5b
uv run -m src.experiments.qwen.run_all --model qwen-3b
uv run -m src.experiments.qwen.run_all --model qwen-7b
```

Runs all 9 stages (4 common + 5 Qwen-specific) in order as subprocesses. Stops on first failure. To run a subset:

```bash
uv run -m src.experiments.qwen.run_all --model qwen-1.5b --only behavioral mechanistic causal
```

Stage names: `behavioral`, `n_sweep_extended`, `token_generality`, `interleaved_noise`, `mechanistic`, `diagnostics`, `causal`, `robustness`, `interleaved_probe`.

### mechanistic

Attention analysis, linear probes, logit lens, MLP decomposition (1.5B only — fails P1), anomaly sweep with intruder-token and base-length variation (3B/7B only — passes P1; varies the intruder across banana/car/seven/xyz at n=10, and the base length across n=8/10/12 with banana), paraphrase decomposition at the writer sites (1.5B only). Depends on `behavioral.json`.

```bash
uv run -m src.experiments.qwen.mechanistic --model qwen-1.5b
uv run -m src.experiments.qwen.mechanistic --model qwen-3b
uv run -m src.experiments.qwen.mechanistic --model qwen-7b
```

Output: `results/<model>/mechanistic_qwen.json`
Also writes: `attention_qwen.json`, `probe_results_qwen.json`, `logit_lens_qwen.json`
1.5B also writes: `mlp_decomp_qwen.json`, `paraphrase_decomp_qwen.json`
3B/7B also write: `anomaly_sweep_qwen.json` (token and length variation grids + threshold summaries)

### diagnostics

Probe dissociation check (1.5B only), tokenizer limitation analysis, direct logit check at final layer. Depends on `behavioral.json`, `probe_results_qwen.json`, and `mechanistic_qwen.json`.

```bash
uv run -m src.experiments.qwen.diagnostics --model qwen-1.5b
uv run -m src.experiments.qwen.diagnostics --model qwen-3b
uv run -m src.experiments.qwen.diagnostics --model qwen-7b
```

Output: `results/<model>/diagnostics_qwen.json`

### causal (qwen-1.5b only)

Mean-ablation and targeted (P3→P1) patching at the writer sites in `cfg.ablation_sites` (L22, L24), individually and jointly. Mirrors `llama.causal`'s E2/E3 but has no zero-ablation and no `logit_lens.json` dependency — reads `behavioral.json` only. Exits immediately for qwen-3b/7b (they solve P1, so there is no writer to ablate).

```bash
uv run -m src.experiments.qwen.causal --model qwen-1.5b
```

Output: `results/<model>/causal_qwen.json` (`mean_ablation_sites` and `patch_sites` keyed by `L<n>` per site plus `joint`)

### robustness

Re-runs the phase1_baseline logit lens in bfloat16 and float32 across 3 seeds and checks whether the auto-discovered writer layer is stable. Only meaningful for models with a counting-writer to find (1.5B — fails P1); 3B/7B solve P1 and the script exits after printing a skip message. Depends on `behavioral.json` and `mechanistic_qwen.json`.

```bash
uv run -m src.experiments.qwen.robustness --model qwen-1.5b
```

Output: `results/<model>/robustness_qwen.json`

### interleaved_probe

Count-probe generalization and MLP/attention decomposition at the fixed writer layer (recomputed once via the persistence backward-scan, never re-detected per condition) for the interleaved-noise variants. "10" is multi-token for this tokenizer, so the graded logit-difference uses "1" (its leading digit) as the correct-answer proxy. 1.5B only — 3B/7B have no counting-writer to test. Regenerates the interleaved-noise prompts directly from `INTERLEAVED_VARIANTS` rather than depending on `interleaved_noise.json`. Depends on `behavioral.json` and `mechanistic_qwen.json`.

```bash
uv run -m src.experiments.qwen.interleaved_probe --model qwen-1.5b
```

Output: `results/<model>/interleaved_mechanistic_qwen.json`

### Qwen stage dependencies

```
behavioral.json ──> mechanistic ──> diagnostics
                                ├─> causal (1.5B only)
                                ├─> robustness (1.5B only)
                                └─> interleaved_probe (1.5B only)
```

---

## Run all models end-to-end

```bash
# Llama
for model in llama-1b llama-3b; do
    uv run -m src.experiments.llama.run_all --model "$model"
done

# Qwen
for model in qwen-1.5b qwen-3b qwen-7b; do
    uv run -m src.experiments.qwen.run_all --model "$model"
done
```

Each `run_all` already includes the common experiments (`behavioral`, `n_sweep_extended`, `token_generality`, `interleaved_noise`) alongside the family-specific ones, so no separate loop is needed for those.

---

## Output structure

```
results/
├── Llama-3.2-1B-Instruct/
│   ├── behavioral.json
│   ├── n_sweep_extended.json
│   ├── token_generality.json
│   ├── interleaved_noise.json / .png
│   ├── linear_probe.json
│   ├── attention.json / .png
│   ├── logit_lens.json
│   ├── causal.json
│   ├── robustness.json
│   └── interleaved_mechanistic.json
├── Llama-3.2-3B-Instruct/
│   └── ... (same as above; causal.json also has multi-site mean_ablation_sites/patch_sites)
├── Qwen2.5-1.5B-Instruct/
│   ├── behavioral.json
│   ├── n_sweep_extended.json
│   ├── token_generality.json
│   ├── interleaved_noise.json / .png
│   ├── mechanistic_qwen.json
│   ├── attention_qwen.json
│   ├── probe_results_qwen.json
│   ├── logit_lens_qwen.json
│   ├── mlp_decomp_qwen.json
│   ├── paraphrase_decomp_qwen.json
│   ├── diagnostics_qwen.json
│   ├── causal_qwen.json
│   ├── robustness_qwen.json
│   └── interleaved_mechanistic_qwen.json
├── Qwen2.5-3B-Instruct/
│   ├── behavioral.json
│   ├── n_sweep_extended.json
│   ├── token_generality.json
│   ├── mechanistic_qwen.json
│   ├── attention_qwen.json
│   ├── probe_results_qwen.json
│   ├── logit_lens_qwen.json
│   ├── anomaly_sweep_qwen.json
│   └── diagnostics_qwen.json
└── Qwen2.5-7B-Instruct/
    └── ... (same as 3B)
```
