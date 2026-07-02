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

Qwen models have no pre-set critical layers — the logit_lens stage auto-discovers them via a full layer sweep.

## Run all stages for a model

```bash
uv run -m src.experiments.run_all --model llama-1b
```

This runs all 6 stages in order as subprocesses. If a stage fails, execution stops. To run a subset:

```bash
uv run -m src.experiments.run_all --model llama-1b --only behavioral logit_lens
```

## Individual stages

### behavioral

Behavioral n-sweep, diagnostic interpretation, tokenization analysis, language robustness, repeated symbols, chain-of-thought probes.

```bash
uv run -m src.experiments.behavioral --model llama-1b
uv run -m src.experiments.behavioral --model llama-3b
uv run -m src.experiments.behavioral --model qwen-1.5b
uv run -m src.experiments.behavioral --model qwen-3b
uv run -m src.experiments.behavioral --model qwen-7b
```

Output: `results/<model>/behavioral.json`

### linear_probe

Ridge regression probe on residual stream activations with leave-one-out cross-validation.

```bash
uv run -m src.experiments.linear_probe --model llama-1b
uv run -m src.experiments.linear_probe --model llama-3b
uv run -m src.experiments.linear_probe --model qwen-1.5b
uv run -m src.experiments.linear_probe --model qwen-3b
uv run -m src.experiments.linear_probe --model qwen-7b
```

Output: `results/<model>/linear_probe.json`

### attention

Word-span attention analysis: per-head attention from last token, entropy, uniformity over word-list span.

```bash
uv run -m src.experiments.attention --model llama-1b
uv run -m src.experiments.attention --model llama-3b
uv run -m src.experiments.attention --model qwen-1.5b
uv run -m src.experiments.attention --model qwen-3b
uv run -m src.experiments.attention --model qwen-7b
```

Output: `results/<model>/attention.json`, `results/<model>/attention.png`

### logit_lens

Logit lens across layers, MLP/attention decomposition, per-n writer input. Reports wrong-minus-correct logit difference. Depends on `behavioral.json`.

```bash
uv run -m src.experiments.logit_lens --model llama-1b
uv run -m src.experiments.logit_lens --model llama-3b
uv run -m src.experiments.logit_lens --model qwen-1.5b
uv run -m src.experiments.logit_lens --model qwen-3b
uv run -m src.experiments.logit_lens --model qwen-7b
```

Output: `results/<model>/logit_lens.json`

### causal

Zero-ablation (E1), mean-ablation (E2), denoising patch (E3), residual patching sweep, steering vector. Depends on `behavioral.json` and `logit_lens.json`.

```bash
uv run -m src.experiments.causal --model llama-1b
uv run -m src.experiments.causal --model llama-3b
uv run -m src.experiments.causal --model qwen-1.5b
uv run -m src.experiments.causal --model qwen-3b
uv run -m src.experiments.causal --model qwen-7b
```

Output: `results/<model>/causal.json`

### robustness

Re-runs logit lens and decomposition in float32 (alongside bfloat16) across 3 seeds. Reports whether the writer layer is stable across numerical environments. Depends on `behavioral.json` and `logit_lens.json`.

```bash
uv run -m src.experiments.robustness --model llama-1b
uv run -m src.experiments.robustness --model llama-3b
uv run -m src.experiments.robustness --model qwen-1.5b
uv run -m src.experiments.robustness --model qwen-3b
uv run -m src.experiments.robustness --model qwen-7b
```

Output: `results/<model>/robustness.json`

## Stage dependencies

```
behavioral ─────┬──> logit_lens ──┬──> causal
                │                 └──> robustness
linear_probe    (independent)
attention       (independent)
```

The orchestrator (`run_all`) handles this ordering automatically.

## Run all models

```bash
for model in llama-1b llama-3b qwen-1.5b qwen-3b qwen-7b; do
    uv run -m src.experiments.run_all --model "$model"
done
```

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
│   └── ...
├── Qwen2.5-3B-Instruct/
│   └── ...
└── Qwen2.5-7B-Instruct/
    └── ...
```
