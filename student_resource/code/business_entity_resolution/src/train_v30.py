"""
Version 3.0 Tri-Model Full Haystack Training Engine
Trains XGBoost + CatBoost + HistGradientBoosting on real distractors mined from 10.3M targets.
"""

import os
import sys
import time
import math
import joblib
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import StratifiedGroupKFold

try:
    from .preprocessing_v30 import (
        clean_business_name_v30,
        clean_address_v30,
        extract_address_digits_v30,
        extract_postal_code_v30,
        extract_complex_codes_v30,
        extract_acronym,
        compute_distinctiveness
    )
    from .blocking_v30 import CandidateBlockerV30
    from .features_v30 import compute_pair_features_v30, FEATURE_NAMES_V30
    from .model_v30 import TriModelEnsembleV30, resolve_mutual_exclusivity_v30, triangulate_triangle_closure_v30
    from .evaluate import compute_macro_f05
except ImportError:
    from preprocessing_v30 import (
        clean_business_name_v30,
        clean_address_v30,
        extract_address_digits_v30,
        extract_postal_code_v30,
        extract_complex_codes_v30,
        extract_acronym,
        compute_distinctiveness
    )
    from blocking_v30 import CandidateBlockerV30
    from features_v30 import compute_pair_features_v30, FEATURE_NAMES_V30
    from model_v30 import TriModelEnsembleV30, resolve_mutual_exclusivity_v30, triangulate_triangle_closure_v30
    from evaluate import compute_macro_f05


def parse_record(parts: list[str]) -> dict:
    eid = parts[0]
    name = parts[1] if len(parts) > 1 else ''
    addr = parts[2] if len(parts) > 2 else ''
    country = parts[3] if len(parts) > 3 else 'US'
    phone = parts[4] if len(parts) > 4 else ''

    c_name = clean_business_name_v30(name)
    c_addr = clean_address_v30(addr)
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
        'addr_digits': extract_address_digits_v30(addr),
        'complex_codes': extract_complex_codes_v30(c_addr),
        'postal_code': extract_postal_code_v30(addr, country)
    }


def main():
    print("=" * 70)
    print("=== VERSION 3.0 TRI-MODEL FULL HAYSTACK TRAINING ENGINE ===")
    print("=" * 70)
    t0 = time.time()

    base_dir = r"c:\Users\megha\Downloads\6ab10eb3b23ba_student_resource\student_resource"
    train_dir = os.path.join(base_dir, "dataset", "train")
    model_save_path = os.path.join(base_dir, "code", "business_entity_resolution", "src", "model_v30_ensemble.joblib")

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
    blocker = CandidateBlockerV30(max_candidates_per_entity=25, max_bucket_size=350)

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

    # Step 3: Stream across ALL 10.3 million target records
    print("\n[Step 3/5] Mining Hard Negatives across the FULL 10.3 Million Target Records...")
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
                
                if eid in needed_targets:
                    rec = parse_record(parts)
                    target_records[eid] = rec
                    keys = blocker.extract_blocking_keys(rec)
                    for k in keys:
                        blocker.index[k].append(eid)
                else:
                    name = parts[1] if len(parts) > 1 else ''
                    c_n = clean_business_name_v30(name)
                    if len(c_n) >= 3 and ('ex', c_n) in needed_keys:
                        rec = parse_record(parts)
                        target_records[eid] = rec
                        keys = blocker.extract_blocking_keys(rec)
                        for k in keys:
                            if k in needed_keys: blocker.index[k].append(eid)
                    elif len(c_n.split()) >= 2:
                        toks = c_n.split()
                        s_pair = f"{min(toks[0], toks[1])}_{max(toks[0], toks[1])}"
                        if ('bi_sort', s_pair) in needed_keys:
                            rec = parse_record(parts)
                            target_records[eid] = rec
                            keys = blocker.extract_blocking_keys(rec)
                            for k in keys:
                                if k in needed_keys: blocker.index[k].append(eid)

                if total_targets_scanned % 2000000 == 0:
                    print(f"    Scanned {total_targets_scanned:,} / 10,320,219 targets... ({len(target_records):,} indexed in {time.time()-t_stream:.1f}s)", flush=True)

    pruned = [k for k, v in blocker.index.items() if len(v) > blocker.max_bucket_size]
    for k in pruned:
        del blocker.index[k]
    blocker.precompute_weights()

    print(f"  Streaming complete in {time.time()-t_stream:.1f}s!")
    print(f"  Total target records indexed: {len(target_records):,}")

    # Step 4: Candidate Retrieval & Feature Extraction (35 features)
    print("\n[Step 4/5] Retrieving candidates from the full haystack & computing 35 dense features...")
    pairs = []
    y_list = []
    group_ids = []
    hard_negatives_count = 0
    true_positives_count = 0

    for s1_id, s1_rec in s1_records.items():
        true_m = gt_dict.get(s1_id, set())
        cands = blocker.get_candidates(s1_rec)
        
        cand_set = set(cands)
        for m in true_m:
            if m in target_records and m not in cand_set:
                cands.append(m)

        for rank, cid in enumerate(cands):
            if cid in target_records:
                cand_rec = target_records[cid]
                feat = compute_pair_features_v30(s1_rec, cand_rec, rank=rank)
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
    print(f"  REAL Hard Negative Distractor Pairs: {hard_negatives_count:,} (Ratio: {hard_negatives_count/true_positives_count:.2f} negatives/pos)")

    X = np.array([compute_pair_features_v30(s1_records[sid], target_records[cid]) for sid, cid in pairs], dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    groups = np.array(group_ids)

    # Step 5: Train Tri-Model Ensemble
    print("\n" + "=" * 70)
    print("=== [Step 5/5] Training Tri-Model Ensemble (XGBoost + CatBoost + HistGradient) ===")
    print("=" * 70)

    ensemble = TriModelEnsembleV30(w_xgb=0.45, w_cat=0.35, w_hist=0.20)
    ensemble.fit(X, y)
    
    # Calibrated decision thresholds
    ensemble.best_threshold = 0.68
    ensemble.singleton_threshold = 0.74
    ensemble.margin_delta = 0.17
    ensemble.ambiguity_margin = 0.05
    
    ensemble.save(model_save_path)
    print(f"\nModel saved successfully to: {model_save_path}")
    print(f"Total Version 3.0 training completed in {time.time()-t0:.1f}s ({(time.time()-t0)/60:.1f} minutes)!")

if __name__ == "__main__":
    main()
