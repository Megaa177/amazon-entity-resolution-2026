"""
Version 2.3 Cost-Sensitive Matching Model with Adaptive Distinctiveness Thresholding
Includes:
- XGBoost classifier with cost-sensitive objective
- Adaptive length-calibrated thresholding
- Sibling triangulation with acronym support
- Global anti-stealing mutual exclusivity
"""

import os
import joblib
import numpy as np
import xgboost as xgb
from collections import defaultdict

try:
    from .evaluate import compute_macro_f05
    from .features_v23 import FEATURE_NAMES_V23, token_sort_jaccard
except ImportError:
    from evaluate import compute_macro_f05
    from features_v23 import FEATURE_NAMES_V23, token_sort_jaccard


def triangulate_siblings_v23(
    s1_id: str,
    candidates: list[tuple[str, float]],
    target_records: dict[str, dict],
    high_conf_threshold: float = 0.86,
    sub_threshold_min: float = 0.50,
    target_tau: float = 0.70
) -> list[tuple[str, float]]:
    """Version 2.3 Sibling Triangulation with Acronym and Address verification."""
    if not candidates:
        return candidates
        
    s2_high = []
    s3_high = []
    s2_sub = []
    s3_sub = []
    
    for cid, p in candidates:
        if cid.startswith('S2-'):
            if p >= high_conf_threshold:
                s2_high.append((cid, p))
            elif p >= sub_threshold_min:
                s2_sub.append((cid, p))
        elif cid.startswith('S3-'):
            if p >= high_conf_threshold:
                s3_high.append((cid, p))
            elif p >= sub_threshold_min:
                s3_sub.append((cid, p))
                
    if (not s2_high or not s3_sub) and (not s3_high or not s2_sub):
        return candidates
        
    boosted = dict(candidates)
    
    if s2_high and s3_sub:
        for s2_id, _ in s2_high:
            rec_s2 = target_records.get(s2_id)
            if not rec_s2: continue
            d_s2 = set(rec_s2.get('addr_digits', []))
            tok_s2 = rec_s2.get('name_tokens', [])
            acr_s2 = rec_s2.get('acronym', '')
            
            for s3_id, p3 in s3_sub:
                rec_s3 = target_records.get(s3_id)
                if not rec_s3: continue
                d_s3 = set(rec_s3.get('addr_digits', []))
                tok_s3 = rec_s3.get('name_tokens', [])
                acr_s3 = rec_s3.get('acronym', '')
                
                sibling_name_sim = token_sort_jaccard(tok_s2, tok_s3)
                acr_match = (acr_s2 and acr_s2 == acr_s3)
                
                if (d_s2 and d_s3 and len(d_s2 & d_s3) > 0 and (sibling_name_sim >= 0.45 or acr_match)):
                    boosted[s3_id] = max(boosted[s3_id], target_tau + 0.02)
                    
    if s3_high and s2_sub:
        for s3_id, _ in s3_high:
            rec_s3 = target_records.get(s3_id)
            if not rec_s3: continue
            d_s3 = set(rec_s3.get('addr_digits', []))
            tok_s3 = rec_s3.get('name_tokens', [])
            acr_s3 = rec_s3.get('acronym', '')
            
            for s2_id, p2 in s2_sub:
                rec_s2 = target_records.get(s2_id)
                if not rec_s2: continue
                d_s2 = set(rec_s2.get('addr_digits', []))
                tok_s2 = rec_s2.get('name_tokens', [])
                acr_s2 = rec_s2.get('acronym', '')
                
                sibling_name_sim = token_sort_jaccard(tok_s3, tok_s2)
                acr_match = (acr_s3 and acr_s3 == acr_s2)
                
                if (d_s3 and d_s2 and len(d_s3 & d_s2) > 0 and (sibling_name_sim >= 0.45 or acr_match)):
                    boosted[s2_id] = max(boosted[s2_id], target_tau + 0.02)
                    
    return list(boosted.items())


def resolve_mutual_exclusivity_v23(
    pred_dict: dict[str, set[str]], 
    s1_to_probs: dict[str, list[tuple[str, float]]],
    min_margin: float = 0.06
) -> dict[str, set[str]]:
    """Version 2.3 Global Mutual Exclusivity with Anti-Stealing Ambiguity Rejection."""
    target_claims = defaultdict(list)
    for sid, matched in pred_dict.items():
        prob_map = dict(s1_to_probs.get(sid, []))
        for cid in matched:
            target_claims[cid].append((sid, prob_map.get(cid, 0.5)))
            
    clean_dict = {sid: set() for sid in pred_dict}
    
    for cid, claims in target_claims.items():
        if len(claims) == 1:
            clean_dict[claims[0][0]].add(cid)
        else:
            sorted_claims = sorted(claims, key=lambda x: x[1], reverse=True)
            top1_sid, top1_p = sorted_claims[0]
            top2_sid, top2_p = sorted_claims[1]
            gap = top1_p - top2_p
            if gap >= min_margin:
                clean_dict[top1_sid].add(cid)
                
    return clean_dict


class MatchClassifierV23:
    """Version 2.3 Cost-Sensitive GBDT with Adaptive Distinctiveness Thresholding."""
    
    def __init__(self, n_estimators: int = 380, max_depth: int = 6, learning_rate: float = 0.06):
        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=0.28,
            eval_metric='logloss',
            random_state=42,
            n_jobs=-1
        )
        self.best_threshold = 0.71
        self.singleton_threshold = 0.75
        self.margin_delta = 0.17
        self.ambiguity_margin = 0.06

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if len(X) == 0:
            return np.array([])
        return self.model.predict_proba(X)[:, 1]

    def optimize_threshold(
        self, 
        val_pairs: list[tuple[str, str]], 
        val_probs: np.ndarray, 
        val_ground_truth: dict[str, set[str]],
        s1_distinctiveness: dict[str, float] = None,
        target_records: dict[str, dict] = None
    ) -> tuple[float, float, float, float]:
        best_score = -1.0
        best_tau = 0.71
        best_single_tau = 0.75
        best_delta = 0.17
        best_ambiguity = 0.06
        
        s1_to_preds = {sid: [] for sid in val_ground_truth}
        for (s1_id, cand_id), prob in zip(val_pairs, val_probs):
            if s1_id in s1_to_preds:
                s1_to_preds[s1_id].append((cand_id, float(prob)))
                
        thresholds = [0.68, 0.70, 0.72]
        single_thresholds = [0.74, 0.76, 0.78]
        deltas = [0.15, 0.17]
        ambiguity_margins = [0.05, 0.06]
        
        s1_dist = s1_distinctiveness or {}
        
        for tau in thresholds:
            for s_tau in single_thresholds:
                if s_tau < tau: continue
                for d in deltas:
                    base_preds = {}
                    for sid, cands in s1_to_preds.items():
                        if not cands:
                            base_preds[sid] = set()
                            continue
                            
                        # Adaptive threshold adjustment based on distinctiveness
                        dist = s1_dist.get(sid, 0.5)
                        tau_eff = tau + (0.5 - dist) * 0.08
                        s_tau_eff = s_tau + (0.5 - dist) * 0.06

                        active_cands = cands
                        if target_records:
                            active_cands = triangulate_siblings_v23(sid, cands, target_records, target_tau=tau_eff)
                            
                        max_p = max(p for _, p in active_cands)
                        if max_p < s_tau_eff:
                            base_preds[sid] = set()
                            continue
                            
                        matched = {cid for cid, p in active_cands if p >= tau_eff and p >= (max_p - d)}
                        base_preds[sid] = matched
                        
                    for amb in ambiguity_margins:
                        pred_dict = resolve_mutual_exclusivity_v23(base_preds, s1_to_preds, min_margin=amb)
                        score = compute_macro_f05(val_ground_truth, pred_dict)
                        if score > best_score:
                            best_score = score
                            best_tau = tau
                            best_single_tau = s_tau
                            best_delta = d
                            best_ambiguity = amb

        print(f"Optimal V2.3 Parameters:")
        print(f"  tau*: {best_tau:.3f} | singleton_tau*: {best_single_tau:.3f} | margin_delta*: {best_delta:.3f} | ambiguity_margin*: {best_ambiguity:.3f}")
        print(f"  Validation Macro F0.5: {best_score:.4f} ({best_score*100:.2f}%)")
        
        self.best_threshold = best_tau
        self.singleton_threshold = best_single_tau
        self.margin_delta = best_delta
        self.ambiguity_margin = best_ambiguity
        return best_tau, best_single_tau, best_delta, best_ambiguity

    def save(self, filepath: str):
        data = {
            'model': self.model,
            'threshold': self.best_threshold,
            'singleton_threshold': self.singleton_threshold,
            'margin_delta': self.margin_delta,
            'ambiguity_margin': self.ambiguity_margin
        }
        joblib.dump(data, filepath)

    def load(self, filepath: str):
        data = joblib.load(filepath)
        if isinstance(data, dict):
            self.model = data['model']
            self.best_threshold = data.get('threshold', 0.71)
            self.singleton_threshold = data.get('singleton_threshold', 0.75)
            self.margin_delta = data.get('margin_delta', 0.17)
            self.ambiguity_margin = data.get('ambiguity_margin', 0.06)
        else:
            self.model = data
