import os
import sys
import time
import argparse
import pandas as pd
import numpy as np
from collections import defaultdict

try:
    from .preprocessing import (
        clean_business_name, 
        get_name_tokens, 
        clean_address, 
        extract_address_digits, 
        extract_postal_code
    )
    from .blocking import CandidateBlocker
    from .features import compute_pair_features
    from .model import MatchClassifier
    from .evaluate import compute_macro_f05
except ImportError:
    from preprocessing import (
        clean_business_name, 
        get_name_tokens, 
        clean_address, 
        extract_address_digits, 
        extract_postal_code
    )
    from blocking import CandidateBlocker
    from features import compute_pair_features
    from model import MatchClassifier
    from evaluate import compute_macro_f05


def parse_record(parts):
    eid = parts[0]
    name = parts[1] if len(parts) > 1 else ''
    addr = parts[2] if len(parts) > 2 else ''
    country = parts[3] if len(parts) > 3 else ''
    c_name = clean_business_name(name)
    return {
        'entity_id': eid,
        'country': country,
        'clean_name': c_name,
        'name_tokens': get_name_tokens(name),
        'clean_addr': clean_address(addr),
        'addr_digits': extract_address_digits(addr),
        'postal_code': extract_postal_code(addr, country)
    }


def train_pipeline(
    train_dir: str, 
    model_save_path: str, 
    n_sample_gt: int = 15000, 
    val_ratio: float = 0.2
):
    print(f"=== Starting Training on Ground Truth (Sample Size: {n_sample_gt}) ===")
    t_start = time.time()
    
    gt_path = os.path.join(train_dir, 'train_ground_truth.tsv')
    s1_path = os.path.join(train_dir, 'train_source1.tsv')
    s2_path = os.path.join(train_dir, 'train_source2.tsv')
    s3_path = os.path.join(train_dir, 'train_source3.tsv')
    
    gt_df = pd.read_csv(gt_path, sep='\t', nrows=n_sample_gt)
    gt_dict = {}
    needed_s1 = set()
    needed_targets = set()
    
    for _, row in gt_df.iterrows():
        s1_id = str(row['source1_entity_id'])
        needed_s1.add(s1_id)
        mids = str(row['matched_entity_ids']).split(',') if pd.notna(row['matched_entity_ids']) and str(row['matched_entity_ids']).strip() else []
        gt_dict[s1_id] = set(mids)
        for m in mids:
            needed_targets.add(m)
            
    print(f"Loaded {len(gt_dict)} GT entities. Finding matching S1 and target records...")
    
    # 1. Load S1 records
    s1_records = []
    with open(s1_path, encoding='utf-8') as f:
        f.readline()
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if parts[0] in needed_s1:
                s1_records.append(parse_record(parts))
                if len(s1_records) == len(needed_s1):
                    break
    print(f"Loaded {len(s1_records)} S1 training records.")
    
    # 2. Stream S2 & S3 for targets plus background negative candidates
    target_records = []
    for path in [s2_path, s3_path]:
        with open(path, encoding='utf-8') as f:
            f.readline()
            for idx, line in enumerate(f):
                parts = line.rstrip('\n').split('\t')
                eid = parts[0]
                if eid in needed_targets or idx < 20000:
                    target_records.append(parse_record(parts))
    print(f"Loaded {len(target_records)} target pool records.")
    
    # 3. Fit Candidate Blocker
    print("Fitting Candidate Blocker...")
    blocker = CandidateBlocker(max_candidates_per_entity=15)
    blocker.fit_targets(target_records)
    target_map = {r['entity_id']: r for r in target_records}
    
    # 4. Grouped Train/Val Split by S1 ID
    s1_ids = [r['entity_id'] for r in s1_records]
    np.random.seed(42)
    shuffled = np.random.permutation(s1_ids)
    split_idx = int(len(shuffled) * (1.0 - val_ratio))
    train_s1 = set(shuffled[:split_idx])
    val_s1 = set(shuffled[split_idx:])
    
    X_train, y_train = [], []
    val_pairs, X_val, y_val = [], [], []
    
    for r in s1_records:
        s1_id = r['entity_id']
        cands = blocker.get_candidates(r)
        true_m = gt_dict.get(s1_id, set())
        is_train = s1_id in train_s1
        
        for rank, cid in enumerate(cands):
            cand_rec = target_map.get(cid)
            if not cand_rec:
                continue
            feat = compute_pair_features(r, cand_rec, rank=rank)
            label = 1 if cid in true_m else 0
            
            if is_train:
                X_train.append(feat)
                y_train.append(label)
            else:
                X_val.append(feat)
                y_val.append(label)
                val_pairs.append((s1_id, cid))
                
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_val, y_val = np.array(X_val), np.array(y_val)
    print(f"Dataset Built -> Train Pairs: {len(X_train)} (Pos: {y_train.sum()}) | Val Pairs: {len(X_val)} (Pos: {y_val.sum()})")
    
    # 5. Fit XGBoost Model
    clf = MatchClassifier(n_estimators=250, max_depth=6, learning_rate=0.08)
    clf.fit(X_train, y_train)
    
    # 6. Optimize F0.5 Threshold on Validation Set
    val_probs = clf.predict_proba(X_val)
    val_gt = {sid: gt_dict[sid] for sid in val_s1}
    best_tau = clf.optimize_threshold(val_pairs, val_probs, val_gt)
    
    # 7. Save trained model
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    clf.save(model_save_path)
    print(f"Model saved to {model_save_path} in {time.time()-t_start:.1f}s.")
    return clf


def predict_pipeline(
    test_dir: str, 
    model_path: str, 
    output_dir: str, 
    max_test_records: int = None
):
    print("=== Starting Test Inference Pipeline ===")
    t_start = time.time()
    
    clf = MatchClassifier()
    clf.load(model_path)
    tau = clf.best_threshold
    print(f"Loaded classifier with optimal F0.5 threshold: {tau:.3f}")
    
    s1_test_path = os.path.join(test_dir, 'test_source1.tsv')
    s2_test_path = os.path.join(test_dir, 'test_source2.tsv')
    s3_test_path = os.path.join(test_dir, 'test_source3.tsv')
    
    os.makedirs(output_dir, exist_ok=True)
    matching_out_path = os.path.join(output_dir, 'matching_results.tsv')
    candidate_out_path = os.path.join(output_dir, 'candidate_pairs.tsv')
    
    # Group by country to process each country independently & save RAM
    # First, read S1 test entities
    print("Reading Test Source 1...")
    s1_by_country = defaultdict(list)
    s1_order = []
    
    with open(s1_test_path, encoding='utf-8') as f:
        f.readline()
        count = 0
        for line in f:
            parts = line.rstrip('\n').split('\t')
            rec = parse_record(parts)
            s1_by_country[rec['country']].append(rec)
            s1_order.append(rec['entity_id'])
            count += 1
            if max_test_records and count >= max_test_records:
                break
                
    print(f"Total S1 test records to process: {len(s1_order)} across {list(s1_by_country.keys())}")
    
    # Target records lookup per country
    final_matches = {}
    final_candidates = {}
    
    for country, s1_list in s1_by_country.items():
        print(f"\n--- Processing Country Partition: {country} ({len(s1_list)} S1 entities) ---")
        t_country = time.time()
        
        # 1. Collect all needed blocking keys from S1 entities in this country
        blocker = CandidateBlocker(max_candidates_per_entity=15)
        needed_keys = set()
        for r in s1_list:
            keys = blocker.extract_blocking_keys(r)
            for k in keys:
                needed_keys.add(k)
        print(f"Extracted {len(needed_keys)} unique blocking query keys for {country}.")
        
        # 2. Stream S2 & S3, indexing ONLY target records that match at least one needed key
        country_targets = {}
        for path in [s2_test_path, s3_test_path]:
            if not os.path.exists(path):
                continue
            with open(path, encoding='utf-8') as f:
                f.readline()
                for line in f:
                    parts = line.rstrip('\n').split('\t')
                    rec_country = parts[3] if len(parts) > 3 else ''
                    if rec_country != country:
                        continue
                    # Quick parse and check
                    rec = parse_record(parts)
                    r_keys = blocker.extract_blocking_keys(rec)
                    # Check if any key is in needed_keys
                    matched_keys = [k for k in r_keys if k in needed_keys]
                    if matched_keys:
                        eid = rec['entity_id']
                        country_targets[eid] = rec
                        for k in matched_keys:
                            blocker.index[k].append(eid)
                            
        # Prune very large buckets
        pruned_keys = [k for k, v in blocker.index.items() if len(v) > blocker.max_bucket_size]
        for k in pruned_keys:
            del blocker.index[k]
            
        print(f"Indexed {len(country_targets)} relevant candidate target records for {country}.")
        
        # 3. Generate candidates & predict matches for all S1 entities in this country
        for r in s1_list:
            s1_id = r['entity_id']
            cands = blocker.get_candidates(r)
            final_candidates[s1_id] = cands
            
            if not cands:
                final_matches[s1_id] = []
                continue
                
            # Score candidates
            feats = []
            valid_cands = []
            for rank, cid in enumerate(cands):
                cand_rec = country_targets.get(cid)
                if cand_rec:
                    feats.append(compute_pair_features(r, cand_rec, rank=rank))
                    valid_cands.append(cid)
                    
            if not feats:
                final_matches[s1_id] = []
                continue
                
            probs = clf.predict_proba(np.array(feats))
            # Filter matches by optimal threshold tau
            matched = [cid for cid, p in zip(valid_cands, probs) if p >= tau]
            final_matches[s1_id] = matched
            
        print(f"Completed {country} partition in {time.time()-t_country:.1f}s.")
        
    # Write output files strictly following format specifications
    print("\nWriting output files...")
    with open(matching_out_path, 'w', encoding='utf-8') as f_match, \
         open(candidate_out_path, 'w', encoding='utf-8') as f_cand:
         
        f_match.write("source1_entity_id\tmatched_entity_ids\n")
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
        
        for s1_id in s1_order:
            matches = final_matches.get(s1_id, [])
            cands = final_candidates.get(s1_id, [])
            
            f_match.write(f"{s1_id}\t{','.join(matches)}\n")
            f_cand.write(f"{s1_id}\t{','.join(cands)}\n")
            
    print(f"Successfully generated:")
    print(f"  - {matching_out_path}")
    print(f"  - {candidate_out_path}")
    print(f"Total time elapsed: {time.time()-t_start:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="Business Entity Resolution Pipeline")
    parser.add_argument('--train-dir', default='student_resource/dataset/train')
    parser.add_argument('--test-dir', default='student_resource/dataset/test')
    parser.add_argument('--output-dir', default='student_resource/output')
    parser.add_argument('--model-path', default='student_resource/code/business_entity_resolution/src/model.joblib')
    parser.add_argument('--mode', choices=['train', 'predict', 'all'], default='all')
    parser.add_argument('--sample-gt', type=int, default=15000)
    parser.add_argument('--max-test', type=int, default=None)
    args = parser.parse_args()

    if args.mode in ('train', 'all'):
        train_pipeline(
            train_dir=args.train_dir,
            model_save_path=args.model_path,
            n_sample_gt=args.sample_gt
        )
        
    if args.mode in ('predict', 'all'):
        predict_pipeline(
            test_dir=args.test_dir,
            model_path=args.model_path,
            output_dir=args.output_dir,
            max_test_records=args.max_test
        )

if __name__ == '__main__':
    main()
