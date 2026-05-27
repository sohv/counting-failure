import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import Tuple, List
from config.utils import PhaseSummary, set_seed, extract_count


def run_single(
    phase_key: str,
    prompts: dict,
    pipe,
    config,
    model,
    tokenizer,
    seed: int = 0
) -> dict:
    entry = prompts[phase_key]
    set_seed(seed)
    
    raw = pipe(
        [{"role": "user", "content": entry["text"]}],
        max_new_tokens=config.max_new_tokens,
        do_sample=False,
        temperature=None,
        pad_token_id=tokenizer.eos_token_id,
        return_full_text=False,
        max_length=None,
    )[0]["generated_text"].strip()
    
    predicted = extract_count(raw)
    correct = (predicted == entry["expected"]) if predicted is not None else False
    
    return {
        "seed": seed,
        "raw": raw,
        "predicted": predicted,
        "expected": entry["expected"],
        "correct": correct,
    }


def run_phase(
    phase_key: str,
    prompts: dict,
    pipe,
    config,
    model,
    tokenizer,
    verbose: bool = True
) -> Tuple[list, PhaseSummary]:
    entry = prompts[phase_key]
    seeds = list(range(config.n_runs))
    
    if verbose:
        print(f"\n{chr(9472)*60}")
        print(f"  {phase_key.upper():<25} | {entry['description']}")
        print(f"{chr(9472)*60}")
    
    phase_results = []
    for seed in seeds:
        r = run_single(phase_key, prompts, pipe, config, model, tokenizer, seed)
        status = "✓" if r["correct"] else "✗"
        
        if verbose:
            raw_preview = repr(r['raw'][:50])
            print(f"  seed={seed:02d}  pred={str(r['predicted']):>4}  exp={r['expected']}  {status}  {raw_preview}")
        
        phase_results.append(r)
    
    summary = PhaseSummary(
        phase=phase_key,
        description=entry['description'],
        expected=entry['expected'],
        n_runs=config.n_runs,
    )
    for r in phase_results:
        summary.add(r)
    
    return phase_results, summary


def n_sweep_analysis(
    phase_key: str,
    prompts: dict,
    pipe,
    config,
    model,
    tokenizer,
    n_range: List[int] = None,
    unique_vocab: List[str] = None,
    n_trials_per_n: int = 3,
) -> dict:
    if n_range is None:
        n_range = range(5, 21)
    
    print(f"\nN-sweep for {phase_key} (n_range={min(n_range)}-{max(n_range)})...")
    
    results = {}
    for n in n_range:
        if unique_vocab:
            words = unique_vocab[:n]
            text = f"Count the number of words in this list: {' '.join(words)}. " \
                   "Respond only with the integer, nothing else."
        else:
            word = "apple"
            text = f'Count the number of times "{word}" appears in this list: ' \
                   f"{' '.join([word]*n)}. Respond only with the integer, nothing else."
        
        predictions = []
        for trial in range(n_trials_per_n):
            raw = pipe(
                [{"role": "user", "content": text}],
                max_new_tokens=config.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                return_full_text=False,
            )[0]["generated_text"].strip()
            
            pred = extract_count(raw)
            predictions.append(pred if pred is not None else -1)
        
        results[n] = predictions
        print(f"  n={n:2d}: {predictions}")
    
    return results
