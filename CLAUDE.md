# CLAUDE.md

Check for a project-level CLAUDE.md in the repo root and follow it. Project-level rules override these.

---

# Core principles

- All Python runs via `uv run -m ...`. Never use `python -m ...` directly.
- Do or do not. There is no try. Avoid try-except blocks, especially around data creation. Silent failure is worse than a crash. If something can fail, let it fail loudly.
- When in doubt about what went wrong, add print statements. Use `print(f"{varname=}")` to print name and value together.
- JSONL is the default format for any dataset or experiment output. JSON for config dumps, with indentation.
- All floats written to JSONL files get rounded to 4 decimal places.
- Write best code with minimal token usage. Avoid unnecessary loops, string concatenation, and repeated API calls. Use vectorized operations and batch calls when possible.
- Avoid unnecessary complexity. If a simple solution works, use it. Do not over-engineer. Do not under-engineer.
---

# Python conventions

- Use type hints on all function signatures. Use `dict`, `list`, `tuple`, `X | None` instead of `typing.Dict`, `typing.Optional`, etc.
- Use `async def` for any function that has LLM API calls downstream of it.
- Prefer Pydantic models over raw dicts for structured data. Use `model_dump()` not `.dict()`.
- Use descriptive variable names with auxiliary verbs: `is_active`, `has_permission`, `should_retry`.
- Lowercase with underscores for all file and directory names.
- Imports at the top of every file.
- No unnecessary curly braces in conditionals. One-line syntax for simple conditionals: `if condition: do_something()`.

---

# Writing code

- No decorators in scripts meant to run from the console. Keep the entry point a plain function call.
- No `try`/`except` unless the program genuinely cannot continue without it (e.g., graceful shutdown of a server, or retrying a flaky network call). Let errors crash loudly.
- File-level comment at the top of each script: one line saying what the experiment does, followed by the run command. Nothing else.
- Never use decorative separators in experiment output. No lines of `#`, `*`, `=`, `-`, or any other repeated character. No banners like `print("#" * 70)`. Each experiment stage prints its heading as a plain line followed by its results. Nothing else.
- Keep inline comments short, one line max, plain words. Only comment on non-obvious logic. No comment is better than a redundant comment. Start comments with a lowercase letter.
    ```python
    # measures counting accuracy across paraphrased prompts for n=1..20.
    # uv run -m src.experiments.robustness --dataset_path data/prompts.jsonl --output_dir results/robustness --model_id claude-sonnet-4-6 --seed 42
    ```
- Do not comment imports, config fields with obvious names, or standard library calls.
  
## Experimental Logging & Reproducibility Guidelines

- Every experiment must write its full results and diagnostics to a structured, persistent file. 
- Never leave results exclusively in `stdout` or text logs. This guarantees every number in final reports traces back to a committed file and survives rerun checks.
- Rule of thumb: If a metric, diagnostic, or figure is printed to the console, it **must** exist inside a structured output file.

- Format Selection by Scale:    
  - Standard JSON (`.json`) for Small-Scale Results (< 1000 records)
    - Use for single-run summaries, hyperparameters, hardware configurations, final evaluation metrics, and short diagnostic outputs.
    - Store data as a single structured object or list.

  - JSON Lines (`.jsonl`) for Large-Scale Results (≥ 1000 records)
    - Use for epoch-by-epoch training logs, step-level loss tracking, large batches of model predictions, or stream-based outputs where each line represents a separate record.
    - Write one valid JSON object per line. 
    - Append incrementally to prevent memory bloat and protect data if the run crashes mid-way.

  - Parquet (`.parquet`) or CSV (`.csv`) for Highly Tabular Data
    - Use for massive data frames or dense matrices where JSON serialization causes severe performance bottlenecks.


**Decision-Making & Ambiguity:** Feel free to use your common sense to select the best format based on the structure of the data. If there is any ambiguity or edge-case context that makes choosing a format unclear, ask me before proceeding.


## Logging

Every script gets:

```python
import logging
LOGGER = logging.getLogger(__name__)
```

Use `LOGGER.info` for normal progress. `LOGGER.warning` for suspicious events. `LOGGER.error` for failures.

Don't print too much to stdout. Use logging for internal state. Reserve stdout for output filenames and final results the user needs to see.

---

# LLM API calls

Use the Anthropic and OpenAI SDKs directly, or LiteLLM if the project needs multiple providers.

Always use the project's caching wrapper if one exists. If not, use this pattern:

```python
import hashlib, json
from pathlib import Path

async def cached_llm_call(client, model: str, messages: list[dict], cache_dir: str = "cache") -> str:
    Path(cache_dir).mkdir(exist_ok=True)
    key = hashlib.md5(f"{model}{json.dumps(messages, sort_keys=True)}".encode()).hexdigest()
    path = Path(cache_dir) / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text())["response"]
    response = await client.messages.create(model=model, messages=messages, max_tokens=2048)
    result = response.content[0].text
    path.write_text(json.dumps({"response": result, "model": model}))
    return result
```

For concurrent calls, use `asyncio.gather` with a semaphore:

```python
import asyncio

async def run_batch(prompts: list[str], model: str, max_concurrent: int = 20) -> list[str]:
    sem = asyncio.Semaphore(max_concurrent)
    async def call(prompt):
        async with sem:
            return await cached_llm_call(client, model, [{"role": "user", "content": prompt}])
    return await asyncio.gather(*[call(p) for p in prompts])
```

Log failed requests prominently. Never let them fail silently. Processing should continue on individual failures when possible.

Default models:
- Debug/testing: `claude-haiku-4-5-20251001` or `gpt-4o-mini`
- Production: `claude-sonnet-4-6` or `gpt-4o`

Never set temperature or max_tokens unless the experiment explicitly requires it and the project CLAUDE.md says so.

---

# LiteLLM

Use LiteLLM when a project needs to call multiple providers with the same interface. Install with `uv add litellm`.

```python
from litellm import completion

response = completion(model="claude-sonnet-4-6", messages=[{"role": "user", "content": "hello"}])
response = completion(model="gpt-4o", messages=[{"role": "user", "content": "hello"}])
response = completion(model="gemini/gemini-pro", messages=[{"role": "user", "content": "hello"}])
```

Same interface for everything. Use this instead of rewriting API call code per project when running experiments across multiple models.

For async:

```python
from litellm import acompletion

response = await acompletion(model="claude-sonnet-4-6", messages=[{"role": "user", "content": "hello"}])
```

---

# Weights and Biases

Use W&B for any experiment that runs more than a handful of API calls or has multiple conditions to compare. Install with `uv add wandb`.

Sign up at https://wandb.ai, then authenticate once:

```bash
wandb login
```

Add to every experiment script:

```python
import wandb

wandb.init(
    project="project-name",
    config={
        "model_id": config.model_id,
        "num_tasks": config.num_tasks,
        "seed": config.seed,
    }
)

# inside your loop:
wandb.log({"accuracy": acc, "loss": loss, "step": i})

# at the end:
wandb.finish()
```

Log the key metric at the end of each run so runs are comparable across experiments:

```python
wandb.summary["final_accuracy"] = final_acc
```

Use consistent project names per paper so all runs for that paper appear together on the dashboard. Example: `"nonidentifiability"`, `"cot-flip"`, `"sycophancy-control"`.

---

# Experiment scripts

Use `simple_parsing` with a dataclass config for every experiment script:

```python
from dataclasses import dataclass
import simple_parsing
import logging

LOGGER = logging.getLogger(__name__)

@dataclass
class Config:
    model_id: str = "claude-sonnet-4-6"
    dataset_path: str = ""          # always required, no default path
    output_dir: str = "results"
    num_tasks: int | None = None    # None = use all
    n_repeats: int = 1
    seed: int = 42

def main():
    config = simple_parsing.parse(Config)
    ...
```

Rules:
- Never use default paths for input data. All input paths must be explicit CLI args.
- Always print the output filename to stdout when saving results.
- When a script produces data that will be plotted separately, print the plot command at the end.
- Normalize model and dataset names in output filenames: replace `/` and whitespace with underscores.
- Experiment folder structure: `experiments/<topic>/YYMMDD_description_v1/`
- Number scripts in execution order: `1_generate_data.py`, `2_run_experiment.py`, `3_analyze.ipynb`

## Calling scripts

Every script call must include `--output_dir`, `--seed 42`, and a `--model_id`. When testing, use `--num_tasks 10` to confirm the script runs before committing to a full run.

```bash
uv run -m src.experiments.steering.run \
  --dataset_path data/prompts.jsonl \
  --output_dir results/steering_v1 \
  --model_id claude-sonnet-4-6 \
  --num_tasks 10 \
  --seed 42
```

---

# Derisking workflow order

Before writing any code, validate the idea manually. Follow this order and only move to the next step when the current one confirms the idea is worth pursuing:

1. **Chat interface first** — send 10-100 messages in Claude.ai or ChatGPT. Manually test the behavior you're trying to measure or produce. Update the prompt based on what you see. This costs nothing and takes 30 minutes. If it doesn't work here it won't work in code.
2. **Few-shot prompting** — add 1-10 gold examples of the behavior you want and test manually. If a few examples in the prompt don't improve the behavior, reconsider the approach before scaling.
3. **Small-scale code** — write a script, run on 10-50 examples, confirm the result matches what you saw manually. Use the debug model (`claude-haiku-4-5-20251001` or `gpt-4o-mini`).
4. **Full-scale run** — only after step 3 confirms the experiment works. Use production model, full dataset, tmux overnight.

Never skip to step 3 or 4 without doing steps 1 and 2. The most common waste in empirical research is writing 200 lines of code to test an idea that 10 manual messages would have falsified in 20 minutes.

---

# Two-mode workflow

**De-risk mode** — use when asking "does this even work?"
- Notebooks are fine. Hardcoded paths are fine. Copy-paste is fine.
- Goal is one question answered fast, not clean code.
- 75% of experiments stay here permanently.

**Extended project mode** — use when the experiment works and needs to scale or be shared.
- Refactor notebook into numbered scripts.
- Add CLI args, caching, logging, proper output paths.
- Add pre-commit hooks if collaborating.
- Switch modes when: compute cost is high, collaborators need to run it, or you're writing it into a paper.

Do not over-engineer de-risk experiments. Do not under-engineer extended ones.

---

# Project structure

Every project follows the standard template found [here](https://github.com/sohv/research-template). Clone the repository at the start, not halfway through. To clone the template:

```bash
git clone https://github.com/sohv/research-template.git
cd research-template
```

Copy the contents of the template once cloned into the project folder (project root directory).

Alternatively, use the following structure (remember to prefer the repo over this structure):
```
my-project/
├── src/                        # all reusable code lives here, never in experiments/
│   ├── __init__.py
│   ├── common/                 # shared utilities used across experiments
│   │   ├── __init__.py
│   │   ├── cache.py            # LLM response caching wrapper
│   │   └── utils.py            # any other shared helpers
│   └── experiments/            # one subfolder per experiment type
│       └── baseline/
│           ├── run.py          # main experiment script
│           └── plot.py         # plotting script for this experiment
├── experiments/                # numbered scripts and notebooks for each run
│   └── topic_name/             # group by paper section or research question
│       └── YYMMDD_description_v1/   # one folder per experiment run, date-prefixed
│           ├── 1_prepare_data.py    # data preparation
│           ├── 2_run_experiment.py  # calls src/ code with specific args
│           ├── 3_analyze.ipynb      # analysis and plots for this run
│           └── results/             # outputs from this specific run
├── data/                       # all input datasets, never commit large files
│   └── raw/                    # original unmodified datasets
├── results/                    # experiment outputs, never commit large files
│   └── figures/                # saved plots
├── tests/                      # unit tests for src/ code
│   └── test_cache.py
├── cache/                      # LLM response cache, gitignored
├── .env                        # API keys, always gitignored
├── .gitignore                  # must include .env, cache/, results/large_files
├── .pre-commit-config.yaml     # ruff and nbstripout hooks
├── CLAUDE.md                   # project-level instructions for Claude Code
├── pyproject.toml              # uv project config and dependencies
├── research_log.md             # running log of what was run and what was found
└── README.md                   # project overview and experiment index
```

Key rules:
- `src/` is for reusable code. `experiments/` is for the specific runs. Never put a one-off hardcoded script in `src/`.
- `experiments/` folders are append-only. Never edit a past experiment folder. Create a new versioned one instead.
- `data/` and `results/` hold files, not code. Large files go in `.gitignore`.
- `cache/` is always gitignored. It is local only.
- `.env` is always gitignored. Never commit API keys.
- Save the git commit hash alongside every experiment output so you can reproduce it exactly later:

```python
import subprocess

def get_git_hash() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()[:8]

# save in your config dump or results metadata
metadata = {
    "git_hash": get_git_hash(),
    "model_id": config.model_id,
    "seed": config.seed,
}
```

Write this to a `config.json` in the experiment output folder alongside the results jsonl.

---

# Documentation requirements

Every script must have an entry in its README with:
1. One sentence describing what the experiment tests.
2. The exact bash command to run it, including all required args.
3. What the script expects as input (file format, fields).
4. What the script produces as output (file format, fields).

Never write vague descriptions like "loads data" or "runs experiment". Specify the exact file paths and formats. You will need this during a rebuttal three months later when you've forgotten everything.

Example README entry:

```
## Experiment: steering vector non-identifiability baseline

Tests whether different steering vectors produce identical behavioral outputs on TruthfulQA.

**Input:** `data/truthfulqa_prompts.jsonl` — fields: `id`, `prompt`, `category`
**Output:** `results/baseline/outputs.jsonl` — fields: `id`, `prompt`, `vector_id`, `response`, `behavioral_score`

**Run:**
uv run -m src.experiments.nonidentifiability.run \
  --dataset_path data/truthfulqa_prompts.jsonl \
  --output_dir results/baseline \
  --model_id claude-sonnet-4-6 \
  --num_tasks 100 \
  --seed 42
```

---

# Data and output conventions

- JSONL for datasets and experiment outputs. One JSON object per line. First field is `id`.
- JSON with indentation for config dumps.
- Round floats to 4 decimal places before writing to JSONL.
- Always output a results file even for de-risk experiments. You will want it during the rebuttal.

---

# Visualization

Default to stdout when there are only a few numbers to display. Only create a plot or HTML when there is genuine structure to show.

## Plots

- Use matplotlib as default. Use seaborn for distributions and multi-condition comparisons.
- Save all figures to `output_dir/figures/`. Print the figure path to stdout after saving.
- Titles and axis labels in sentence case.
- Include model name and key config params in the title. Use linebreaks if the title is long.
- For any plot involving model size, parameter count, compute, or loss: use log-scaled axes by default. Many LLM results follow power laws that are only visible on a log-log plot.
- When a script produces plottable data, print the plot command at the end of the run:

```
Results saved to results/baseline/outputs.jsonl
Plot with: uv run -m src.experiments.nonidentifiability.plot --results_path results/baseline/outputs.jsonl --output_dir results/baseline
```



---

# Testing

Run tests with:

```bash
uv run -m pytest tests/ -v -s
```

For a specific test:

```bash
uv run -m pytest tests/test_file.py::test_name -v -s
```

Rules:
- Write tests for any non-trivial function in `src/`.
- Do not mock LLM calls. Use a real API call with a real small input.
- Use a real small datapoint, call with the debug model (`claude-haiku-4-5-20251001` or `gpt-4o-mini`), assert on structure not exact content.
- Tests should be fast. If a test needs a full experiment run, it is not a unit test.
- Before implementing a function, write the test first and confirm it fails. Then implement.
- Make sure tests pass before committing.

Example test:

```python
import pytest
from src.common.cache import cached_llm_call

@pytest.mark.asyncio
async def test_cached_llm_call_returns_string():
    import anthropic
    client = anthropic.AsyncAnthropic()
    result = await cached_llm_call(
        client=client,
        model="claude-haiku-4-5-20251001",
        messages=[{"role": "user", "content": "Say hello."}],
        cache_dir="/tmp/test_cache"
    )
    assert isinstance(result, str)
    assert len(result) > 0
```

---

# Debugging

No debugger. Use print statements and iteration.

When something is unclear, add `print(f"{varname=}")` at the relevant point. If the state is complex, use:

```python
import code; code.interact(local=dict(globals(), **locals()))
```

This drops into an interactive shell where you can inspect everything.

---

# Pre-commit hooks

Every project that has a collaborator or runs serious compute gets this `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.4.4
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
  - repo: https://github.com/kynan/nbstripout
    rev: 0.7.1
    hooks:
      - id: nbstripout
```

Install with `pre-commit install`. Run `pre-commit run --all-files` before every commit.

Ruff ignores: E501, E402, E741, F841, F403, F401. Line length 120.

---

# Tmux

Always run long experiments in tmux so they survive disconnects.

```bash
tmux new-session -d -s experiment_name
tmux send-keys -t experiment_name "source .env" Enter
tmux send-keys -t experiment_name "uv run -m src.experiment --args" Enter
```

Start the session first, then send keys. Always source `.env` first for API keys. Use descriptive session names.

---

# Git and GitHub

- `git status` before staging anything. Add only the files relevant to this change.
- Run `pre-commit run --all-files` before every commit.
- Commit messages are short and descriptive. No emoji.
- Push after committing.
- Only create private repositories. User changes to public if needed.
- Use `gh` CLI for GitHub interactions.
- Use git worktrees for parallel work on multiple papers or issues. One worktree per GitHub issue.

For worktrees, symlink `.venv`, `.cache`, `.pytest_cache`, and `uv.lock` to avoid duplicate installs. Copy `.env`.

---

# Research log

After finishing a significant experiment or a round of results, append a short entry to `research_log.md` in the repo root:

```markdown
## YYMMDD — short description

**What:** one sentence on what was tested
**Result:** one sentence on what was found
**Command:**
uv run -m src.experiments... --args
**Output:** path to results file
```

This is the thing you will read when writing the paper or preparing a rebuttal. Keep entries short. Write them immediately after the run, not later.

---

# File management

- Use `trash` instead of `rm`.
- Use `rg` instead of `grep`.
- Use `tree` to understand directory structure, not `ls`.

---

# Research principles

After any experiment:
1. Write one sentence in the README describing what the experiment tests.
2. Write the full bash command used to run it.
3. Write what the script expects as input and what it produces.
4. Create at least one visualization of the results. If you ran something, something gets plotted.

Before starting an experiment, ask:
- Have I already run something similar? Check `research_log.md` before writing any code. The most common waste is re-running something you ran three weeks ago with slightly different wording.
- What result do I expect and why?
- Is this the highest-priority question right now?
- Am I changing too many variables at once?
- Will this add real value to the paper or rebuttal?

Start with a small `--num_tasks` (10) to confirm the script runs before committing to a full run.
