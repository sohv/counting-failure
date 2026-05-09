# The Odometer Hypothesis in LLMs

## Project Structure

```
counting-failure/
├── notebooks/
│   ├── llama-1B.ipynb
│   ├── llama-3B.ipynb
│   ├── qwen-1.5B.ipynb
│   ├── qwen-3B.ipynb
│   └── qwen-7B.ipynb
└── scripts/
    ├── main.py
    ├── utils.py
    ├── experiments_code.py
    └── prompts.json
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
cd scripts
```

## Run

```bash
python main.py --model llama-1b --exp core
python main.py --model qwen-7b --exp core tokenization n-sweep --n-runs 5
python main.py --model llama-3b --exp all --save-results
```
