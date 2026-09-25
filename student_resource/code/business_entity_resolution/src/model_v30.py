"""
Version 3.0 Tri-Model GBDT Ensemble & Graph Closure Engine
Includes:
- Tri-Model Ensemble: XGBoost + CatBoost + HistGradientBoosting
- Graph Triangle Closure (S1 <-> S2 <-> S3 tripartite concordance)
- Anti-Stealing Bipartite Mutual Exclusivity with Ambiguity Rejection
- Entity Distinctiveness-Adaptive Thresholding
"""

import os
import joblib
import numpy as np
import xgboost as xgb
import catboost as cb
from sklearn.ensemble import HistGradientBoostingClassifier
from collections import defaultdict

try:
    from .features_v30 import token_sort_jaccard, char_dice_2gram
    from .preprocessing_v30 import jaro_winkler
except ImportError:
    from features_v30 import token_sort_jaccard, char_dice_2gram
    from preprocessing_v30 import jaro_winkler


def triangulate_triangle_closure_v30(
    s1_id: str,
    candidates: list[tuple[str, float]],
    target_records: dict[str, dict],
    target_tau: float = 0.68,
    sub_threshold_min: float = 0.40
) -> list[tuple[str, float]]:
    """Version 3.0 Graph Tripartite Closure between S1, S2, and S3."""
    s2_high = []
    s3_high = []
    s2_sub = []
    s3_sub = []
    
    for cid, p in candidates:
        if cid.startswith('S2-'):
            if p >= target_tau:
                s2_high.append((cid, p))
            elif p >= sub_threshold_min:
                s2_sub.append((cid, p))
        elif cid.startswith('S3-'):
            if p >= target_tau:
                s3_high.append((cid, p))
            elif p >= sub_threshold_min:
                s3_sub.append((cid, p))
                
    if (not s2_high or not s3_sub) and (not s3_high or not s2_sub):
        return candidates
        
    boosted = dict(candidates)
    
    # 1. S2 is confident -> check if any S3 candidate agrees with S2
    if s2_high and s3_sub:
        for s2_id, _ in s2_high:
            rec_s2 = target_records.get(s2_id)
            if not rec_s2: continue
            d_s2 = set(rec_s2.get('addr_digits', []))
            tok_s2 = rec_s2.get('name_tokens', [])
            cn_s2 = rec_s2.get('clean_name', '')
            acr_s2 = rec_s2.get('acronym', '')
            ph_s2 = rec_s2.get('phone', '')
            
            for s3_id, p3 in s3_sub:
                rec_s3 = target_records.get(s3_id)
                if not rec_s3: continue
                d_s3 = set(rec_s3.get('addr_digits', []))
                tok_s3 = rec_s3.get('name_tokens', [])
                cn_s3 = rec_s3.get('clean_name', '')
                acr_s3 = rec_s3.get('acronym', '')
                ph_s3 = rec_s3.get('phone', '')
                
                sibling_name_sim = token_sort_jaccard(tok_s2, tok_s3)
                sibling_jw = jaro_winkler(cn_s2, cn_s3)
                acr_match = bool(acr_s2 and acr_s2 == acr_s3)
                phone_match = bool(ph_s2 and ph_s3 and ph_s2 == ph_s3)
                digits_match = bool(d_s2 and d_s3 and len(d_s2 & d_s3) > 0)
                
                if phone_match or (digits_match and (sibling_name_sim >= 0.40 or sibling_jw >= 0.70 or acr_match)):
                    boosted[s3_id] = max(boosted[s3_id], target_tau + 0.03)
                    
    # 2. S3 is confident -> check if any S2 candidate agrees with S3
    if s3_high and s2_sub:
        for s3_id, _ in s3_high:
            rec_s3 = target_records.get(s3_id)
            if not rec_s3: continue
            d_s3 = set(rec_s3.get('addr_digits', []))
            tok_s3 = rec_s3.get('name_tokens', [])
            cn_s3 = rec_s3.get('clean_name', '')
            acr_s3 = rec_s3.get('acronym', '')
            ph_s3 = rec_s3.get('phone', '')
            
            for s2_id, p2 in s2_sub:
                rec_s2 = target_records.get(s2_id)
                if not rec_s2: continue
                d_s2 = set(rec_s2.get('addr_digits', []))
                tok_s2 = rec_s2.get('name_tokens', [])
                cn_s2 = rec_s2.get('clean_name', '')
                acr_s2 = rec_s2.get('acronym', '')
                ph_s2 = rec_s2.get('phone', '')
                
                sibling_name_sim = token_sort_jaccard(tok_s3, tok_s2)
                sibling_jw = jaro_winkler(cn_s3, cn_s2)
                acr_match = bool(acr_s3 and acr_s3 == acr_s2)
                phone_match = bool(ph_s3 and ph_s2 and ph_s3 == ph_s2)
                digits_match = bool(d_s3 and d_s2 and len(d_s3 & d_s2) > 0)
                
                if phone_match or (digits_match and (sibling_name_sim >= 0.40 or sibling_jw >= 0.70 or acr_match)):
                    boosted[s2_id] = max(boosted[s2_id], target_tau + 0.03)
                    
    return list(boosted.items())


def resolve_mutual_exclusivity_v30(
    pred_dict: dict[str, set[str]], 
    s1_to_probs: dict[str, list[tuple[str, float]]],
    min_margin: float = 0.05
) -> dict[str, set[str]]:
    """Version 3.0 Global Mutual Exclusivity with Anti-Stealing Ambiguity Rejection."""
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


class TriModelEnsembleV30:
    """Version 3.0 Tri-Model GBDT Ensemble: XGBoost + CatBoost + HistGradientBoosting."""
    
    def __init__(self, w_xgb: float = 0.45, w_cat: float = 0.35, w_hist: float = 0.20):
        self.w_xgb = w_xgb
        self.w_cat = w_cat
        self.w_hist = w_hist
        
        self.model_xgb = xgb.XGBClassifier(
            n_estimators=380,
            max_depth=6,
            learning_rate=0.06,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=0.20,
            eval_metric='logloss',
            random_state=42,
            n_jobs=-1
        )
        self.model_cat = cb.CatBoostClassifier(
            iterations=350,
            depth=6,
            learning_rate=0.06,
            l2_leaf_reg=3.0,
            random_seed=42,
            verbose=0,
            thread_count=-1
        )
        self.model_hist = HistGradientBoostingClassifier(
            max_iter=300,
            max_depth=6,
            learning_rate=0.06,
            l2_regularization=1.0,
            random_state=42
        )
        self.best_threshold = 0.68
        self.singleton_threshold = 0.74
        self.margin_delta = 0.17
        self.ambiguity_margin = 0.05

    def fit(self, X: np.ndarray, y: np.ndarray):
        print("  [1/3] Fitting XGBoost model...")
        self.model_xgb.fit(X, y)
        print("  [2/3] Fitting CatBoost model...")
        self.model_cat.fit(X, y)
        print("  [3/3] Fitting HistGradientBoosting model...")
        self.model_hist.fit(X, y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if len(X) == 0:
            return np.array([])
        p_xgb = self.model_xgb.predict_proba(X)[:, 1]
        p_cat = self.model_cat.predict_proba(X)[:, 1]
        p_hist = self.model_hist.predict_proba(X)[:, 1]
        return self.w_xgb * p_xgb + self.w_cat * p_cat + self.w_hist * p_hist

    def save(self, path: str):
        joblib.dump({
            'model': self,
            'threshold': self.best_threshold,
            'singleton_threshold': self.singleton_threshold,
            'margin_delta': self.margin_delta,
            'ambiguity_margin': self.ambiguity_margin
        }, path, compress=3)

    @classmethod
    def load(cls, path: str):
        return joblib.load(path)
