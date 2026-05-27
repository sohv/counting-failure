import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import List
from sklearn.linear_model import LogisticRegression
from config.utils import set_seed
from hidden_states import get_hidden_states


def run_loo_probe(
    model, tokenizer, prompts: dict, phase_keys: List[str] = None
) -> dict:
    if phase_keys is None:
        phase_keys = ["phase1_baseline", "phase3_control"]
    
    results = {}
    
    for test_phase in phase_keys:
        train_phases = [p for p in phase_keys if p != test_phase]
        
        X_train, y_train = [], []
        for phase_key in train_phases:
            for seed in range(10):
                set_seed(seed)
                hs = get_hidden_states(
                    model, tokenizer, prompts[phase_key]["text"], layer_idx=-1
                )
                X_train.append(hs[0, -1, :].cpu().numpy())
                y_train.append(prompts[phase_key]["expected"])
        
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train, y_train)
        
        X_test, y_test = [], []
        for seed in range(10):
            set_seed(seed)
            hs = get_hidden_states(
                model, tokenizer, prompts[test_phase]["text"], layer_idx=-1
            )
            X_test.append(hs[0, -1, :].cpu().numpy())
            y_test.append(prompts[test_phase]["expected"])
        
        accuracy = clf.score(X_test, y_test)
        results[test_phase] = {"accuracy": accuracy, "classifier": clf}
    
    return results
