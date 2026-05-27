# The Odometer Hypothesis in LLMs

Code for this paper - [Repeated-Token Counting Reveals a Dissociation Between Representations and Outputs](https://arxiv.org/abs/2605.09239)

## Project Structure

```
counting-failure/
├── config/
│   ├── __init__.py
│   ├── config.json
│   ├── prompts.json
│   └── utils.py
├── scripts/
│   ├── main.py
│   ├── experiments.py
│   ├── attention_analysis.py
│   ├── hidden_states.py
│   ├── probing.py
│   ├── logit_lens.py
│   ├── intervention.py
│   ├── hooks.py
│   └── prompts_utils.py
└── README.md
```

## Experiments

- core - 3-phase baseline
- tokenization - Token analysis
- n-sweep - Count variation (5-20)
- prompts - Robustness testing
- attention - Attention patterns
- linear-probe - Hidden state decoding
- logit-lens - Layer analysis
- patching - Activation intervention

## Setup

```bash
pip install transformers torch accelerate matplotlib scikit-learn
huggingface-cli login
```

## Run

```bash
python3 scripts/main.py --model llama-1b --exp core
python3 scripts/main.py --model qwen-7b --exp core tokenization n-sweep --n-runs 5
python3 scripts/main.py --model llama-3b --exp all --save-results
```

