import os
import joblib
import numpy as np
import xgboost as xgb
from collections import defaultdict

try:
    from .evaluate import compute_macro_f05
    from .features import FEATURE_NAMES
except ImportError:
    from evaluate import compute_macro_f05
    from features import FEATURE_NAMES


def resolve_mutual_exclusivity(pred_dict: dict[str, set[str]], s1_to_probs: dict[str, list[tuple[str, float]]]) -> dict[str, set[str]]:
    """Enforce Global Mutual Exclusivity: Every target in S2/S3 matches at most one S1 entity.
    
    If multiple S1 entities claim the same candidate CID, award it strictly to the S1 entity
    with the highest confidence prob (argmax), breaking ties deterministically.
    """
    # 1. Invert mapping: cid -> list of (sid, prob)
    target_claims = defaultdict(list)
    for sid, matched in pred_dict.items():
        prob_map = dict(s1_to_probs.get(sid, []))
        for cid in matched:
            target_claims[cid].append((sid, prob_map.get(cid, 0.5)))
            
    # 2. Assign each cid to argmax(prob)
    target_winner = {}
    for cid, claims in target_claims.items():
        if len(claims) == 1:
            target_winner[cid] = claims[0][0]
        else:
            # Sort by prob descending
            best_sid = sorted(claims, key=lambda x: x[1], reverse=True)[0][0]
            target_winner[cid] = best_sid
            
    # 3. Rebuild conflict-free predictions
    clean_dict = {sid: set() for sid in pred_dict}
    for cid, winner_sid in target_winner.items():
        clean_dict[winner_sid].add(cid)
        
    return clean_dict


class MatchClassifierV2:
    """Version 2.0: Asymmetric Cost-Sensitive GBDT with 3-Tier Filter and Global Mutual Exclusivity."""
    
    def __init__(self, n_estimators: int = 300, max_depth: int = 6, learning_rate: float = 0.08):
        # scale_pos_weight = 0.40 heavily penalizes False Positives (aligned with F0.5)
        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=0.40,
            eval_metric='logloss',
            random_state=42,
            n_jobs=-1
        )
        self.best_threshold = 0.70
        self.singleton_threshold = 0.75
        self.margin_delta = 0.15

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the cost-sensitive XGBoost classifier."""
        self.model.fit(X, y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return positive class match probabilities."""
        if len(X) == 0:
            return np.array([])
        return self.model.predict_proba(X)[:, 1]

    def optimize_threshold(
        self, 
        val_pairs: list[tuple[str, str]], 
        val_probs: np.ndarray, 
        val_ground_truth: dict[str, set[str]]
    ) -> tuple[float, float, float]:
        """Find the optimal (tau, singleton_tau, margin_delta) that maximizes Macro F0.5."""
        best_score = -1.0
        best_tau = 0.70
        best_single_tau = 0.75
        best_delta = 0.15
        
        # Build lookup table from s1_id to list of (cand_id, prob)
        s1_to_preds = {sid: [] for sid in val_ground_truth}
        for (s1_id, cand_id), prob in zip(val_pairs, val_probs):
            if s1_id in s1_to_preds:
                s1_to_preds[s1_id].append((cand_id, prob))
                
        # Grid search over candidate thresholds, singleton thresholds, and margin deltas
        thresholds = [0.65, 0.70, 0.72, 0.75]
        single_thresholds = [0.72, 0.75, 0.78, 0.80]
        deltas = [0.12, 0.15, 0.18]
        
        for tau in thresholds:
            for s_tau in single_thresholds:
                if s_tau < tau: continue
                for d in deltas:
                    pred_dict = {}
                    for s1_id, cands in s1_to_preds.items():
                        if not cands:
                            pred_dict[s1_id] = set()
                            continue
                            
                        max_p = max(p for _, p in cands)
                        # Singleton protection filter
                        if max_p < s_tau:
                            pred_dict[s1_id] = set()
                            continue
                            
                        # Margin filter
                        matched = {cid for cid, p in cands if p >= tau and p >= (max_p - d)}
                        pred_dict[s1_id] = matched
                        
                    # Apply Mutual Exclusivity conflict resolution
                    pred_dict = resolve_mutual_exclusivity(pred_dict, s1_to_preds)
                    score = compute_macro_f05(val_ground_truth, pred_dict)
                    if score > best_score:
                        best_score = score
                        best_tau = tau
                        best_single_tau = s_tau
                        best_delta = d

        print(f"Optimal V2.0 Parameters:")
        print(f"  tau*: {best_tau:.3f} | singleton_tau*: {best_single_tau:.3f} | margin_delta*: {best_delta:.3f}")
        print(f"  Validation Macro F0.5: {best_score:.4f} ({best_score*100:.2f}%)")
        
        self.best_threshold = best_tau
        self.singleton_threshold = best_single_tau
        self.margin_delta = best_delta
        return best_tau, best_single_tau, best_delta

    def save(self, filepath: str):
        """Save model and tuned hyperparameters."""
        data = {
            'model': self.model,
            'threshold': self.best_threshold,
            'singleton_threshold': self.singleton_threshold,
            'margin_delta': self.margin_delta
        }
        joblib.dump(data, filepath)

    def load(self, filepath: str):
        """Load model and tuned hyperparameters."""
        data = joblib.load(filepath)
        if isinstance(data, dict):
            self.model = data['model']
            self.best_threshold = data.get('threshold', 0.70)
            self.singleton_threshold = data.get('singleton_threshold', 0.75)
            self.margin_delta = data.get('margin_delta', 0.15)
        else:
            self.model = data
