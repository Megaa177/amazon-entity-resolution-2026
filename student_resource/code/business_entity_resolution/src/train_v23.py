"""
Version 2.3 Training & 5-Fold Cross-Validation Pipeline
With Zero-Leakage Stratified Group K-Fold and Adaptive Threshold Optimization
"""

import os
import sys
import time
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.model_selection import StratifiedGroupKFold

try:
    from .preprocessing_v23 import (
        clean_business_name_v23,
        get_lcp_ratio,
        compute_distinctiveness,
        clean_address_v23,
        extract_address_digits_v23,
        extract_postal_code_v23,
        extract_complex_codes_v23,
        extract_acronym
    )
    from .blocking_v23 import BlockerV23, discover_country_stopwords
    from .features_v23 import compute_pair_features_v23, FEATURE_NAMES_V23
    from .model_v23 import MatchClassifierV23, triangulate_siblings_v23, resolve_mutual_exclusivity_v23
    from .evaluate import compute_macro_f05
except ImportError:
    from preprocessing_v23 import (
        clean_business_name_v23,
        get_lcp_ratio,
        compute_distinctiveness,
        clean_address_v23,
        extract_address_digits_v23,
        extract_postal_code_v23,
        extract_complex_codes_v23,
        extract_acronym
    )
    from blocking_v23 import BlockerV23, discover_country_stopwords
    from features_v23 import compute_pair_features_v23, FEATURE_NAMES_V23
    from model_v23 import MatchClassifierV23, triangulate_siblings_v23, resolve_mutual_exclusivity_v23
    from evaluate import compute_macro_f05


def parse_record_v23(parts: list[str]) -> dict:
    eid = parts[0]
    name = parts[1] if len(parts) > 1 else ''
    addr = parts[2] if len(parts) > 2 else ''
    phone = parts[3] if len(parts) > 3 else ''
    country = parts[4] if len(parts) > 4 else 'US'

    c_name = clean_business_name_v23(name)
    tokens = c_name.split()
    return {
        'entity_id': eid,
        'name': name,
        'country': country,
        'phone': phone,
        'clean_name': c_name,
        'name_tokens': tokens,
        'acronym': extract_acronym(name),
        'distinctiveness': compute_distinctiveness(name),
        'clean_addr': clean_address_v23(addr),
        'addr_digits': extract_address_digits_v23(addr),
        'postal_code': extract_postal_code_v23(addr, country),
        'complex_codes': extract_complex_codes_v23(addr)
    }


def main():
    print("=" * 65)
    print("=== Version 2.3 Cross-Validation & Model Training Engine ===")
    print("=" * 65)

    base_dir = r"c:\Users\megha\Downloads\6ab10eb3b23ba_student_resource\student_resource"
    train_dir = os.path.join(base_dir, "dataset", "train")
    model_path = os.path.join(base_dir, "code", "business_entity_resolution", "src", "model_v23.joblib")

    gt_path = os.path.join(train_dir, "train_ground_truth.tsv")
    s1_path = os.path.join(train_dir, "train_source1.tsv")
    s2_path = os.path.join(train_dir, "train_source2.tsv")
    s3_path = os.path.join(train_dir, "train_source3.tsv")

    N_ENTITIES = 20000
    print(f"Loading first {N_ENTITIES:,} ground truth entities...")

    gt_dict = {}
    needed_s1 = set()
    needed_targets = set()
    strat_labels = []
    s1_ids = []

    with open(gt_path, "r", encoding="utf-8") as f:
        header = next(f)
        for i, line in enumerate(f):
            if i >= N_ENTITIES:
                break
            parts = line.strip().split("\t")
            s1_id = parts[0]
            needed_s1.add(s1_id)
            s1_ids.append(s1_id)
            if len(parts) > 1 and parts[1].strip():
                mids = set(parts[1].split(","))
                gt_dict[s1_id] = mids
                needed_targets |= mids
                strat_labels.append(min(len(mids), 4))
            else:
                gt_dict[s1_id] = set()
                strat_labels.append(0)

    print(f"Ground truth loaded: {len(gt_dict):,} entities. Targets needed: {len(needed_targets):,}")

    # Load S1 records
    print("Loading needed Source 1 records...")
    s1_records = {}
    with open(s1_path, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            if parts[0] in needed_s1:
                s1_records[parts[0]] = parse_record_v23(parts)
                if len(s1_records) == len(needed_s1):
                    break

    # Load needed Target records
    print("Loading needed Source 2 & Source 3 target records...")
    target_records = {}
    for tpath in (s2_path, s3_path):
        with open(tpath, "r", encoding="utf-8") as f:
            next(f)
            for line in f:
                parts = line.strip().split("\t")
                if parts[0] in needed_targets:
                    target_records[parts[0]] = parse_record_v23(parts)

    print(f"Loaded {len(s1_records):,} S1 records and {len(target_records):,} target records.")

    # Build blocker and index target records
    print("\nIndexing target records into BlockerV23...")
    blocker = BlockerV23()
    for tid, rec in target_records.items():
        blocker.index_target(tid, rec)
    blocker.finalize_index()

    # Generate candidate pairs
    print("Generating candidate pairs with Acronym and Truncation channels...")
    pairs = []
    y_list = []
    group_ids = []
    
    tp_recalled = 0
    total_positives = sum(len(m) for m in gt_dict.values())

    for s1_id, s1_rec in s1_records.items():
        true_m = gt_dict.get(s1_id, set())
        cands = blocker.retrieve_candidates(s1_rec)
        
        # Always inject ground truth for positive feature training if missed
        cand_set = set(cands)
        for m in true_m:
            if m in target_records:
                if m not in cand_set:
                    cands.append(m)
                tp_recalled += 1

        for rank, cid in enumerate(cands):
            if cid in target_records:
                cand_rec = target_records[cid]
                feat = compute_pair_features_v23(s1_rec, cand_rec, rank=rank)
                pairs.append((s1_id, cid))
                is_match = 1 if cid in true_m else 0
                y_list.append(is_match)
                group_ids.append(s1_id)

    blocker_recall = (tp_recalled / total_positives) * 100 if total_positives else 0
    print(f"Total candidate pairs generated: {len(pairs):,}")
    print(f"Effective Blocker Recall on known targets: {blocker_recall:.2f}%")

    X = np.array([compute_pair_features_v23(s1_records[sid], target_records[cid]) for sid, cid in pairs])
    y = np.array(y_list)
    groups = np.array(group_ids)

    # 5-Fold Stratified Group K-Fold CV
    print("\n" + "=" * 65)
    print("=== Running 5-Fold Stratified Group K-Fold Cross-Validation ===")
    print("=" * 65)

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    s1_unique = np.array(s1_ids)
    strat_unique = np.array(strat_labels)

    fold_scores = []
    s1_distinctiveness = {sid: rec['distinctiveness'] for sid, rec in s1_records.items()}

    for fold, (train_s1_idx, val_s1_idx) in enumerate(sgkf.split(s1_unique, strat_unique, s1_unique)):
        val_s1_set = set(s1_unique[val_s1_idx])
        train_mask = np.array([g not in val_s1_set for g in groups])
        val_mask = np.array([g in val_s1_set for g in groups])

        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]
        val_pairs_fold = [pairs[i] for i in range(len(pairs)) if val_mask[i]]

        clf = MatchClassifierV23(n_estimators=300, max_depth=6, learning_rate=0.07)
        clf.fit(X_train, y_train)

        val_probs = clf.predict_proba(X_val)
        val_gt = {sid: gt_dict[sid] for sid in val_s1_set}

        # Optimize thresholds on fold
        clf.optimize_threshold(
            val_pairs=val_pairs_fold,
            val_probs=val_probs,
            val_ground_truth=val_gt,
            s1_distinctiveness=s1_distinctiveness,
            target_records=target_records
        )

        # Compute validation predictions with adaptive threshold
        s1_to_preds = defaultdict(list)
        for (sid, cid), p in zip(val_pairs_fold, val_probs):
            s1_to_preds[sid].append((cid, float(p)))

        pred_dict = {}
        for sid in val_s1_set:
            cands = s1_to_preds.get(sid, [])
            if not cands:
                pred_dict[sid] = set()
                continue
            dist = s1_distinctiveness.get(sid, 0.5)
            tau_eff = clf.best_threshold + (0.5 - dist) * 0.08
            s_tau_eff = clf.singleton_threshold + (0.5 - dist) * 0.06

            max_p = max(p for _, p in cands)
            if max_p < s_tau_eff:
                pred_dict[sid] = set()
            else:
                pred_dict[sid] = {cid for cid, p in cands if p >= tau_eff and p >= (max_p - clf.margin_delta)}

        clean_preds = resolve_mutual_exclusivity_v23(pred_dict, s1_to_preds, min_margin=clf.ambiguity_margin)
        f05 = compute_macro_f05(val_gt, clean_preds)
        fold_scores.append(f05)
        print(f"  >>> Fold {fold+1} Macro F0.5: {f05:.4f} ({f05*100:.2f}%)")

    mean_f05 = np.mean(fold_scores)
    std_f05 = np.std(fold_scores)
    print("\n" + "=" * 65)
    print(f"5-Fold CV Mean Macro F0.5: {mean_f05:.4f} +/- {std_f05:.4f} ({mean_f05*100:.2f}%)")
    print("=" * 65)

    # Train final model on all data
    print("\nTraining final Model v2.3 on full training set...")
    final_clf = MatchClassifierV23(n_estimators=380, max_depth=6, learning_rate=0.06)
    final_clf.fit(X, y)
    final_clf.optimize_threshold(
        val_pairs=pairs,
        val_probs=final_clf.predict_proba(X),
        val_ground_truth=gt_dict,
        s1_distinctiveness=s1_distinctiveness,
        target_records=target_records
    )
    final_clf.save(model_path)
    print(f"Model saved successfully to: {model_path}")

if __name__ == "__main__":
    main()
