# Usage

All commands use `uv run -m` from the project root.

## Models

| Key | HuggingFace model | Layers |
|---|---|---|
| `llama-1b` | `meta-llama/Llama-3.2-1B-Instruct` | 16 |
| `llama-3b` | `meta-llama/Llama-3.2-3B-Instruct` | 28 |
| `qwen-1.5b` | `Qwen/Qwen2.5-1.5B-Instruct` | 28 |
| `qwen-3b` | `Qwen/Qwen2.5-3B-Instruct` | 36 |
| `qwen-7b` | `Qwen/Qwen2.5-7B-Instruct` | 28 |

Qwen models have no pre-set critical layers — `mechanistic.py` auto-discovers them from the logit lens output.

## Experiment structure

```
src/experiments/
├── behavioral.py         ← all 5 models
├── interleaved_noise.py  ← llama-1b, llama-3b, qwen-1.5b only (prior-active models)
├── llama/                ← Llama-specific mechanistic pipeline
│   ├── attention.py
│   ├── linear_probe.py
│   ├── logit_lens.py
│   ├── causal.py
│   ├── robustness.py
│   ├── interleaved_probe.py  ← probe + decomposition for the interleaved-noise variants
│   └── run_all.py
└── qwen/                 ← Qwen-specific mechanistic pipeline
    ├── mechanistic.py
    ├── diagnostics.py
    ├── robustness.py         ← qwen-1.5b only (only Qwen model with a counting-writer)
    └── interleaved_probe.py  ← qwen-1.5b only, same role as the Llama version
```

---

## Behavioral (all models)

Phases 1–3, n-sweep, fixed prompts, paraphrase robustness, language robustness, repeated symbols, chain-of-thought probes.

```bash
uv run -m src.experiments.behavioral --model llama-1b
uv run -m src.experiments.behavioral --model llama-3b
uv run -m src.experiments.behavioral --model qwen-1.5b
uv run -m src.experiments.behavioral --model qwen-3b
uv run -m src.experiments.behavioral --model qwen-7b
```

Output: `results/<model>/behavioral.json`

---

## Extended n-sweep (qwen-3b, qwen-7b — used to characterize length-onset degradation)

Fills in n = 13, 14, 16-19 around `behavioral.py`'s n-sweep (5-12, 15, 20), purely to check whether a model's degradation past n=12 is a clean threshold or intermittent. Behavioral only, out of scope for any mechanistic follow-up. Depends on `behavioral.json`.

```bash
uv run -m src.experiments.n_sweep_extended --model qwen-3b
uv run -m src.experiments.n_sweep_extended --model qwen-7b
```

Output: `results/<model>/n_sweep_extended.json`

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

Runs all 6 stages in order as subprocesses. Stops on first failure. To run a subset:

```bash
uv run -m src.experiments.llama.run_all --model llama-1b --only behavioral logit_lens
```

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

Logit lens across layers, MLP/attention decomposition, per-n writer input. Wrong-minus-correct logit difference. Depends on `behavioral.json`.

```bash
uv run -m src.experiments.llama.logit_lens --model llama-1b
uv run -m src.experiments.llama.logit_lens --model llama-3b
```

Output: `results/<model>/logit_lens.json`

### causal

Zero-ablation (E1), mean-ablation (E2), denoising patch (E3), residual patching sweep, steering vector. Depends on `behavioral.json` and `logit_lens.json`.

```bash
uv run -m src.experiments.llama.causal --model llama-1b
uv run -m src.experiments.llama.causal --model llama-3b
```

Output: `results/<model>/causal.json`

### robustness

Logit lens + decomposition in bfloat16 and float32 across 3 seeds. Checks writer-layer stability across numerical environments. Depends on `behavioral.json` and `logit_lens.json`.

```bash
uv run -m src.experiments.llama.robustness --model llama-1b
uv run -m src.experiments.llama.robustness --model llama-3b
```

Output: `results/<model>/robustness.json`

### interleaved_probe

Count-probe generalization (fit on space-separated prompts, evaluated out-of-distribution on each interleaved-noise variant) and MLP/attention decomposition at the fixed lock-in layer (never re-detected per condition). Depends on `behavioral.json`, `logit_lens.json`, and `interleaved_noise.json`.

```bash
uv run -m src.experiments.llama.interleaved_probe --model llama-1b
uv run -m src.experiments.llama.interleaved_probe --model llama-3b
```

Output: `results/<model>/interleaved_mechanistic.json`

### Llama stage dependencies

```
behavioral.json ──┬──> logit_lens ──┬──> causal
                  │                 ├──> robustness
                  │                 └──> interleaved_probe (also needs interleaved_noise.json)
                  └── (linear_probe and attention are independent)
```

---

## Qwen pipeline

Run `behavioral` first (shared), then `mechanistic`, then `diagnostics`.

### mechanistic

Attention analysis, linear probes, logit lens, MLP decomposition (1.5B only — fails P1), anomaly sweep (3B/7B only — passes P1). Depends on `behavioral.json`.

```bash
uv run -m src.experiments.qwen.mechanistic --model qwen-1.5b
uv run -m src.experiments.qwen.mechanistic --model qwen-3b
uv run -m src.experiments.qwen.mechanistic --model qwen-7b
```

Output: `results/<model>/mechanistic_qwen.json`  
Also writes: `attention_qwen.json`, `probe_results_qwen.json`, `logit_lens_qwen.json`  
1.5B also writes: `mlp_decomp_qwen.json`  
3B/7B also write: `anomaly_sweep_qwen.json`

### diagnostics

Probe dissociation check (1.5B only), tokenizer limitation analysis, direct logit check at final layer. Depends on `behavioral.json`, `probe_results_qwen.json`, and `mechanistic_qwen.json`.

```bash
uv run -m src.experiments.qwen.diagnostics --model qwen-1.5b
uv run -m src.experiments.qwen.diagnostics --model qwen-3b
uv run -m src.experiments.qwen.diagnostics --model qwen-7b
```

Output: `results/<model>/diagnostics_qwen.json`

### robustness

Re-runs the phase1_baseline logit lens in bfloat16 and float32 across 3 seeds and checks whether the auto-discovered writer layer is stable. Only meaningful for models with a counting-writer to find (1.5B — fails P1); 3B/7B solve P1 and the script exits after printing a skip message. Depends on `behavioral.json` and `mechanistic_qwen.json`.

```bash
uv run -m src.experiments.qwen.robustness --model qwen-1.5b
```

Output: `results/<model>/robustness_qwen.json`

### interleaved_probe

Count-probe generalization and MLP/attention decomposition at the fixed writer layer (recomputed once via the persistence backward-scan, never re-detected per condition) for the interleaved-noise variants. "10" is multi-token for this tokenizer, so the graded logit-difference uses "1" (its leading digit) as the correct-answer proxy. 1.5B only — 3B/7B have no counting-writer to test. Depends on `behavioral.json`, `mechanistic_qwen.json`, and `interleaved_noise.json`.

```bash
uv run -m src.experiments.qwen.interleaved_probe --model qwen-1.5b
```

Output: `results/<model>/interleaved_mechanistic_qwen.json`

### Qwen stage dependencies

```
behavioral.json ──> mechanistic ──> diagnostics
                                ├─> robustness (1.5B only)
                                └─> interleaved_probe (1.5B only, also needs interleaved_noise.json)
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
    uv run -m src.experiments.behavioral --model "$model"
    uv run -m src.experiments.qwen.mechanistic --model "$model"
    uv run -m src.experiments.qwen.diagnostics --model "$model"
done
```

---

## Output structure

```
results/
├── Llama-3.2-1B-Instruct/
│   ├── behavioral.json
│   ├── linear_probe.json
│   ├── attention.json
│   ├── attention.png
│   ├── logit_lens.json
│   ├── causal.json
│   └── robustness.json
├── Llama-3.2-3B-Instruct/
│   └── ...
├── Qwen2.5-1.5B-Instruct/
│   ├── behavioral.json
│   ├── mechanistic_qwen.json
│   ├── attention_qwen.json
│   ├── probe_results_qwen.json
│   ├── logit_lens_qwen.json
│   ├── mlp_decomp_qwen.json
│   └── diagnostics_qwen.json
├── Qwen2.5-3B-Instruct/
│   ├── behavioral.json
│   ├── mechanistic_qwen.json
│   ├── attention_qwen.json
│   ├── probe_results_qwen.json
│   ├── logit_lens_qwen.json
│   ├── anomaly_sweep_qwen.json
│   └── diagnostics_qwen.json
└── Qwen2.5-7B-Instruct/
    └── ...
```
