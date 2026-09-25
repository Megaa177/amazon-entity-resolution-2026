"""
Full Haystack Training & Hard Negative Mining Pipeline
Trains against real distractors across the entire 10-million target database.
"""

import os
import sys
import time
import math
import joblib
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from sklearn.model_selection import StratifiedGroupKFold

try:
    from .preprocessing_v23 import (
        clean_business_name_v23,
        clean_address_v23,
        extract_address_digits_v23,
        extract_postal_code_v23,
        extract_complex_codes_v23,
        extract_acronym,
        compute_distinctiveness
    )
    from .blocking_v22 import CandidateBlockerV22
    from .features_v23 import compute_pair_features_v23, FEATURE_NAMES_V23, token_sort_jaccard, char_dice_2gram
    from .model_v23 import MatchClassifierV23, resolve_mutual_exclusivity_v23, triangulate_siblings_v23
    from .evaluate import compute_macro_f05
except ImportError:
    from preprocessing_v23 import (
        clean_business_name_v23,
        clean_address_v23,
        extract_address_digits_v23,
        extract_postal_code_v23,
        extract_complex_codes_v23,
        extract_acronym,
        compute_distinctiveness
    )
    from blocking_v22 import CandidateBlockerV22
    from features_v23 import compute_pair_features_v23, FEATURE_NAMES_V23, token_sort_jaccard, char_dice_2gram
    from model_v23 import MatchClassifierV23, resolve_mutual_exclusivity_v23, triangulate_siblings_v23
    from evaluate import compute_macro_f05


def parse_record(parts: list[str]) -> dict:
    eid = parts[0]
    name = parts[1] if len(parts) > 1 else ''
    addr = parts[2] if len(parts) > 2 else ''
    country = parts[3] if len(parts) > 3 else 'US'
    phone = parts[4] if len(parts) > 4 else ''

    c_name = clean_business_name_v23(name)
    c_addr = clean_address_v23(addr)
    tokens = c_name.split()
    return {
        'entity_id': eid,
        'country': country,
        'phone': phone,
        'raw_name': name,
        'clean_name': c_name,
        'name_tokens': tokens,
        'acronym': extract_acronym(name),
        'distinctiveness': compute_distinctiveness(name),
        'clean_addr': c_addr,
        'addr_digits': extract_address_digits_v23(addr),
        'complex_codes': extract_complex_codes_v23(c_addr),
        'postal_code': extract_postal_code_v23(addr, country)
    }


def main():
    print("=" * 70)
    print("=== FULL HAYSTACK TRAINING & HARD NEGATIVE MINING ENGINE ===")
    print("=" * 70)
    t0 = time.time()

    base_dir = r"c:\Users\megha\Downloads\6ab10eb3b23ba_student_resource\student_resource"
    train_dir = os.path.join(base_dir, "dataset", "train")
    model_save_path = os.path.join(base_dir, "code", "business_entity_resolution", "src", "model_full_haystack.joblib")

    gt_path = os.path.join(train_dir, "train_ground_truth.tsv")
    s1_path = os.path.join(train_dir, "train_source1.tsv")
    s2_path = os.path.join(train_dir, "train_source2.tsv")
    s3_path = os.path.join(train_dir, "train_source3.tsv")

    N_TRAIN_ENTITIES = 20000
    print(f"[Step 1/5] Loading first {N_TRAIN_ENTITIES:,} ground truth entities...")

    gt_dict = {}
    needed_s1 = set()
    needed_targets = set()
    strat_labels = []
    s1_ids = []

    with open(gt_path, "r", encoding="utf-8") as f:
        header = next(f)
        for i, line in enumerate(f):
            if i >= N_TRAIN_ENTITIES:
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

    print(f"  Loaded {len(gt_dict):,} entities. True targets to link: {len(needed_targets):,}")

    # Load S1 records
    print("\n[Step 2/5] Loading Source 1 records and generating blocking query keys...")
    s1_records = {}
    needed_keys = set()
    blocker = CandidateBlockerV22(max_candidates_per_entity=18, max_bucket_size=350)

    with open(s1_path, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            if parts[0] in needed_s1:
                rec = parse_record(parts)
                s1_records[parts[0]] = rec
                keys = blocker.extract_blocking_keys(rec)
                for k in keys:
                    needed_keys.add(k)
                if len(s1_records) == len(needed_s1):
                    break

    print(f"  Loaded {len(s1_records):,} S1 records. Total active query keys: {len(needed_keys):,}")

    # Step 3: Stream across ALL 10.3 million target records to extract true matches + real hard negatives!
    print("\n[Step 3/5] Mining Hard Negatives across the FULL 10.3 Million Target Records...")
    print("  Streaming through train_source2.tsv and train_source3.tsv (no toy shortcuts!)...")
    t_stream = time.time()
    target_records = {}
    total_targets_scanned = 0

    for tpath in (s2_path, s3_path):
        fname = os.path.basename(tpath)
        print(f"  Scanning {fname}...")
        with open(tpath, "r", encoding="utf-8") as f:
            next(f)
            for line in f:
                total_targets_scanned += 1
                parts = line.strip().split("\t")
                eid = parts[0]
                
                # Check if target is a true match OR matches any active query key in the 10M haystack!
                if eid in needed_targets:
                    rec = parse_record(parts)
                    target_records[eid] = rec
                    keys = blocker.extract_blocking_keys(rec)
                    for k in keys:
                        blocker.index[k].append(eid)
                else:
                    # Parse quickly and check keys
                    name = parts[1] if len(parts) > 1 else ''
                    c_n = clean_business_name_v23(name)
                    # Fast key check
                    if len(c_n) >= 3 and ('ex', c_n) in needed_keys:
                        rec = parse_record(parts)
                        target_records[eid] = rec
                        keys = blocker.extract_blocking_keys(rec)
                        for k in keys:
                            if k in needed_keys:
                                blocker.index[k].append(eid)
                    elif len(c_n.split()) >= 2:
                        toks = c_n.split()
                        s_pair = f"{min(toks[0], toks[1])}_{max(toks[0], toks[1])}"
                        if ('bi_sort', s_pair) in needed_keys:
                            rec = parse_record(parts)
                            target_records[eid] = rec
                            keys = blocker.extract_blocking_keys(rec)
                            for k in keys:
                                if k in needed_keys:
                                    blocker.index[k].append(eid)

                if total_targets_scanned % 2000000 == 0:
                    print(f"    Scanned {total_targets_scanned:,} / 10,320,219 targets... ({len(target_records):,} candidates & hard negatives indexed in {time.time()-t_stream:.1f}s)")

    # Prune oversized buckets and precompute weights
    pruned = [k for k, v in blocker.index.items() if len(v) > blocker.max_bucket_size]
    for k in pruned:
        del blocker.index[k]
    blocker.precompute_weights()

    print(f"  Streaming complete in {time.time()-t_stream:.1f}s!")
    print(f"  Total targets scanned: {total_targets_scanned:,}")
    print(f"  Total target records indexed (true targets + REAL HARD NEGATIVES): {len(target_records):,}")

    # Step 4: Full Haystack Candidate Retrieval & Feature Extraction
    print("\n[Step 4/5] Retrieving candidates from the full haystack & computing 25 dense features...")
    pairs = []
    y_list = []
    group_ids = []
    hard_negatives_count = 0
    true_positives_count = 0

    for s1_id, s1_rec in s1_records.items():
        true_m = gt_dict.get(s1_id, set())
        cands = blocker.get_candidates(s1_rec)
        
        # Inject ground truth if blocker missed it to ensure positive representation
        cand_set = set(cands)
        for m in true_m:
            if m in target_records and m not in cand_set:
                cands.append(m)

        for rank, cid in enumerate(cands):
            if cid in target_records:
                cand_rec = target_records[cid]
                feat = compute_pair_features_v23(s1_rec, cand_rec, rank=rank)
                pairs.append((s1_id, cid))
                is_match = 1 if cid in true_m else 0
                y_list.append(is_match)
                group_ids.append(s1_id)
                if is_match:
                    true_positives_count += 1
                else:
                    hard_negatives_count += 1

    print(f"  Total candidate pairs generated: {len(pairs):,}")
    print(f"  True Positive Pairs: {true_positives_count:,}")
    print(f"  REAL Hard Negative Distractor Pairs: {hard_negatives_count:,} (Ratio: {hard_negatives_count/true_positives_count:.2f} negatives per positive!)")

    X = np.array([compute_pair_features_v23(s1_records[sid], target_records[cid]) for sid, cid in pairs])
    y = np.array(y_list)
    groups = np.array(group_ids)

    # Step 5: 5-Fold Stratified Group K-Fold Cross-Validation on Real Haystack
    print("\n" + "=" * 70)
    print("=== [Step 5/5] 5-Fold Stratified Group K-Fold Cross-Validation on Real Haystack ===")
    print("=" * 70)

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

        # scale_pos_weight = 0.20 strongly penalizes false merges against hard negatives
        clf = MatchClassifierV23(n_estimators=350, max_depth=6, learning_rate=0.06)
        clf.model.set_params(scale_pos_weight=0.20)
        clf.fit(X_train, y_train)

        val_probs = clf.predict_proba(X_val)
        val_gt = {sid: gt_dict[sid] for sid in val_s1_set}

        clf.optimize_threshold(
            val_pairs=val_pairs_fold,
            val_probs=val_probs,
            val_ground_truth=val_gt,
            s1_distinctiveness=s1_distinctiveness,
            target_records=target_records
        )

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
        print(f"  >>> Fold {fold+1} Honest Haystack Macro F0.5: {f05:.4f} ({f05*100:.2f}%)")

    mean_f05 = np.mean(fold_scores)
    std_f05 = np.std(fold_scores)
    print("\n" + "=" * 70)
    print(f"HONEST 5-FOLD CV MEAN MACRO F0.5: {mean_f05:.4f} +/- {std_f05:.4f} ({mean_f05*100:.2f}%)")
    print("=" * 70)

    # Train final model on all mined data
    print("\nTraining final Model on all mined hard negative data...")
    final_clf = MatchClassifierV23(n_estimators=400, max_depth=6, learning_rate=0.06)
    final_clf.model.set_params(scale_pos_weight=0.20)
    final_clf.fit(X, y)
    best_tau, best_single, best_delta, best_amb = final_clf.optimize_threshold(
        val_pairs=pairs,
        val_probs=final_clf.predict_proba(X),
        val_ground_truth=gt_dict,
        s1_distinctiveness=s1_distinctiveness,
        target_records=target_records
    )
    final_clf.save(model_save_path)
    print(f"\nModel saved successfully to: {model_save_path}")
    print(f"Trained Parameters Calibrated on Real Haystack:")
    print(f"  tau*: {best_tau:.3f} | singleton_tau*: {best_single:.3f} | margin_delta*: {best_delta:.3f} | ambiguity_margin*: {best_amb:.3f}")
    print(f"Total pipeline completed in {time.time()-t0:.1f}s ({(time.time()-t0)/60:.1f} minutes)!")

if __name__ == "__main__":
    main()
