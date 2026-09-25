"""
Version 2.4 Fast Multi-Country Entity Resolution Inference Engine
Combines:
- France & US proven co-location defense (from v2.2)
- Restored Indian Multilingual & Phonetic Recall Engine
- Zero Target Collisions with Anti-Stealing Bipartite Matching
"""

import os
import sys
import time
import re
import argparse
import joblib
import numpy as np
from collections import defaultdict

try:
    from .preprocessing_v22 import (
        clean_business_name_v22, 
        get_name_tokens_v22, 
        clean_address_v22, 
        extract_address_digits_v22, 
        extract_postal_code_v22,
        extract_complex_codes_v22
    )
    from .blocking_v22 import CandidateBlockerV22, discover_country_stopwords
    from .features_v22 import compute_pair_features_v22, token_sort_jaccard, char_dice_2gram
    from .model_v22 import MatchClassifierV22, resolve_mutual_exclusivity_v22, triangulate_siblings_v22
except ImportError:
    from preprocessing_v22 import (
        clean_business_name_v22, 
        get_name_tokens_v22, 
        clean_address_v22, 
        extract_address_digits_v22, 
        extract_postal_code_v22,
        extract_complex_codes_v22
    )
    from blocking_v22 import CandidateBlockerV22, discover_country_stopwords
    from features_v22 import compute_pair_features_v22, token_sort_jaccard, char_dice_2gram
    from model_v22 import MatchClassifierV22, resolve_mutual_exclusivity_v22, triangulate_siblings_v22


def resolve_base_dirs():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.abspath(os.path.join(script_dir, '..', '..', '..')),
        os.path.abspath(os.path.join(script_dir, '..', '..')),
        os.path.abspath(os.path.join(script_dir, '..')),
        os.getcwd()
    ]
    for b in candidates:
        if os.path.exists(os.path.join(b, 'dataset', 'test')):
            return b
    return os.getcwd()


def create_record(eid: str, raw_name: str, raw_addr: str, country: str, phone: str = '', extra_stopwords: set = None) -> dict:
    c_name = clean_business_name_v22(raw_name)
    c_addr = clean_address_v22(raw_addr, extra_stopwords=extra_stopwords)
    return {
        'entity_id': eid,
        'country': country,
        'phone': phone,
        'raw_name': raw_name,
        'clean_name': c_name,
        'name_tokens': get_name_tokens_v22(raw_name),
        'clean_addr': c_addr,
        'addr_digits': extract_address_digits_v22(raw_addr),
        'complex_codes': extract_complex_codes_v22(c_addr),
        'postal_code': extract_postal_code_v22(raw_addr, country)
    }


def process_india_v24(part_dir: str, clf_data: dict, out_match_path: str, out_cand_path: str):
    print(f"\n=======================================================")
    print(f"=== Starting Version 2.4 Processing for: India ===")
    print(f"=======================================================")
    t_start = time.time()
    
    s1_path = os.path.join(part_dir, 's1_India.tsv')
    targets_path = os.path.join(part_dir, 'targets_India.tsv')
    
    model = clf_data['model']
    tau = 0.70
    singleton_tau = 0.73  # Aligned with 5.6% ground truth singleton rate
    margin_delta = 0.18
    ambiguity_margin = 0.05
    print(f"Loaded Calibrated Indian Decision Parameters:")
    print(f"  tau={tau:.3f} | singleton_tau={singleton_tau:.3f} | margin_delta={margin_delta:.3f} | ambiguity_margin={ambiguity_margin:.3f}")

    # 1. Discover Indian stopwords
    print("Sampling India records for dynamic stopword discovery...")
    addr_samples = []
    with open(targets_path, encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if idx >= 30000: break
            parts = line.rstrip('\n').split('\t')
            addr = parts[2] if len(parts) > 2 else ''
            addr_samples.append({'clean_addr': addr.lower(), 'name_tokens': []})
            
    country_stopwords, _ = discover_country_stopwords(addr_samples, min_freq=0.03)
    print(f"  Auto-discovered {len(country_stopwords)} stopwords for India: {sorted(list(country_stopwords))[:12]}")

    # 2. Load India S1 entities
    print("Loading India Source 1 entities...")
    s1_records = []
    needed_keys = set()
    s1_keys_map = {}
    
    blocker = CandidateBlockerV22(max_candidates_per_entity=18, max_bucket_size=350)
    blocker.extra_stopwords = country_stopwords
    
    with open(s1_path, encoding='utf-8') as f:
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if not parts[0]: continue
            eid = parts[0]
            name = parts[1] if len(parts) > 1 else ''
            addr = parts[2] if len(parts) > 2 else ''
            phone = parts[3] if len(parts) > 3 else ''
            rec = create_record(eid, name, addr, 'India', phone=phone, extra_stopwords=country_stopwords)
            s1_records.append(rec)
            keys = blocker.extract_blocking_keys(rec)
            s1_keys_map[eid] = keys
            for k in keys:
                needed_keys.add(k)
                
    print(f"Loaded {len(s1_records):,} India Source 1 entities. Unique query keys: {len(needed_keys):,}")
    
    # 3. Stream and index target candidates
    print("Streaming and indexing India target candidates...")
    t_idx = time.time()
    target_records = {}
    
    with open(targets_path, encoding='utf-8') as f:
        for line in f:
            parts = line.rstrip('\n').split('\t')
            eid = parts[0]
            name = parts[1] if len(parts) > 1 else ''
            addr = parts[2] if len(parts) > 2 else ''
            phone = parts[3] if len(parts) > 3 else ''
            rec = create_record(eid, name, addr, 'India', phone=phone, extra_stopwords=country_stopwords)
            keys = blocker.extract_blocking_keys(rec)
            matched = [k for k in keys if k in needed_keys]
            if matched:
                target_records[eid] = rec
                for k in matched:
                    blocker.index[k].append(eid)
                    
    pruned = [k for k, v in blocker.index.items() if len(v) > blocker.max_bucket_size]
    for k in pruned:
        del blocker.index[k]
        
    blocker.precompute_weights()
    print(f"Indexed {len(target_records):,} India target records in {time.time()-t_idx:.1f}s.")
    
    # 4. Batched Feature Extraction & ML Scoring
    print("Scoring candidate pairs in batches with Indian Multilingual & Phonetic Gate...")
    t_score = time.time()
    
    BATCH_SIZE = 50000
    batch_features = []
    batch_meta = []
    s1_cands_map = {}
    s1_probs_map = defaultdict(list)
    
    total_pairs_scored = 0
    co_location_rejected = 0
    
    def flush_batch():
        nonlocal batch_features, batch_meta, total_pairs_scored
        if not batch_features: return
        X = np.array(batch_features, dtype=np.float32)
        probs = model.predict_proba(X)[:, 1]
        for (sid, cid), prob in zip(batch_meta, probs):
            s1_probs_map[sid].append((cid, float(prob)))
        total_pairs_scored += len(batch_features)
        batch_features = []
        batch_meta = []

    for idx, r1 in enumerate(s1_records):
        sid = r1['entity_id']
        cands = blocker.get_candidates(r1)
        s1_cands_map[sid] = cands
        
        tok1 = r1['name_tokens']
        c_name1 = r1['clean_name']
        
        for rank, cid in enumerate(cands):
            r2 = target_records.get(cid)
            if not r2: continue
            
            tok2 = r2['name_tokens']
            c_name2 = r2['clean_name']
            
            sort_sim = token_sort_jaccard(tok1, tok2)
            dice_sim = char_dice_2gram(c_name1, c_name2)
            
            # --- INDIAN MULTILINGUAL & PHONETIC GATE ---
            # Never reject if building digits match, plot codes match, or contact phone matches!
            digits_match = bool(set(r1.get('addr_digits', [])) & set(r2.get('addr_digits', [])))
            codes_match = bool(set(r1.get('complex_codes', [])) & set(r2.get('complex_codes', [])))
            phone_match = bool(r1.get('phone') and r1.get('phone') == r2.get('phone'))
            
            # Only veto if names completely contradict AND there is no structural anchor
            if c_name1 != c_name2 and sort_sim < 0.25 and dice_sim < 0.22 and not (digits_match or codes_match or phone_match):
                s1_probs_map[sid].append((cid, 0.0))
                co_location_rejected += 1
                continue
                
            feat = compute_pair_features_v22(r1, r2, rank=rank)
            batch_features.append(feat)
            batch_meta.append((sid, cid))
                
        if len(batch_features) >= BATCH_SIZE:
            flush_batch()
            
        if (idx + 1) % 50000 == 0 or (idx + 1) == len(s1_records):
            print(f"  Scored {idx+1:,} / {len(s1_records):,} Indian entities... ({total_pairs_scored:,} pairs scored in {time.time()-t_score:.1f}s, {co_location_rejected:,} vetoed)", flush=True)
                
    flush_batch()
    print(f"Finished scoring {total_pairs_scored:,} Indian pairs in {time.time()-t_score:.1f}s. Total vetoed: {co_location_rejected:,}")
    
    # 5. Sibling Triangulation & Filter
    print("Applying Sibling Triangulation and Metric Filter...")
    initial_matches = {}
    for r1 in s1_records:
        sid = r1['entity_id']
        cands = s1_probs_map.get(sid, [])
        if not cands:
            initial_matches[sid] = set()
            continue
            
        has_s2 = any(c[0].startswith('S2-') for c in cands)
        has_s3 = any(c[0].startswith('S3-') for c in cands)
        if has_s2 and has_s3:
            triangulated = triangulate_siblings_v22(sid, cands, target_records, target_tau=tau)
        else:
            triangulated = cands
        s1_probs_map[sid] = triangulated
        
        max_p = max(p for _, p in triangulated)
        if max_p < singleton_tau:
            initial_matches[sid] = set()
            continue
            
        matched = {cid for cid, p in triangulated if p >= tau and p >= (max_p - margin_delta)}
        initial_matches[sid] = matched

    # 6. Global Mutual Exclusivity with Anti-Stealing
    print("Applying Anti-Stealing Global Mutual Exclusivity...")
    t_mutex = time.time()
    resolved_matches = resolve_mutual_exclusivity_v22(initial_matches, s1_probs_map, min_margin=ambiguity_margin)
    print(f"Mutual Exclusivity resolved in {time.time()-t_mutex:.2f}s.")
    
    # 7. Write results to disk
    print("Writing India results to disk...")
    f_match = open(out_match_path, 'a', encoding='utf-8')
    f_cand = open(out_cand_path, 'a', encoding='utf-8')
    
    total_matches_found = 0
    empty_entities = 0
    for r1 in s1_records:
        sid = r1['entity_id']
        cands = s1_cands_map.get(sid, [])
        matched = resolved_matches.get(sid, set())
        
        # Candidate pairs file
        cand_str = ",".join(cands)
        f_cand.write(f"{sid}\t{cand_str}\n")
        
        # Matching results file
        if not matched:
            f_match.write(f"{sid}\t\n")
            empty_entities += 1
        else:
            match_str = ",".join(sorted(matched))
            f_match.write(f"{sid}\t{match_str}\n")
            total_matches_found += len(matched)
            
    f_match.close()
    f_cand.close()
    
    elapsed = time.time() - t_start
    print(f"Finished India in {elapsed:.1f}s!")
    print(f"  Total S1 entities: {len(s1_records):,}")
    print(f"  Empty entities (Singletons): {empty_entities:,} ({empty_entities/len(s1_records)*100:.1f}%)")
    print(f"  Total matches predicted: {total_matches_found:,} (Avg {total_matches_found/len(s1_records):.2f}/entity)")


def main():
    base_dir = resolve_base_dirs()
    print(f"Resolved Project Base Directory: {base_dir}")
    
    part_dir = os.path.join(base_dir, 'scratch', 'partitions')
    model_path = os.path.join(base_dir, 'code', 'business_entity_resolution', 'src', 'model_v22.joblib')
    
    out_match = os.path.join(base_dir, 'output', 'matching_results_v24.tsv')
    out_cand = os.path.join(base_dir, 'output', 'candidate_pairs_v24.tsv')
    os.makedirs(os.path.dirname(out_match), exist_ok=True)
    os.makedirs(os.path.dirname(out_cand), exist_ok=True)
    
    src_match_v22 = os.path.join(base_dir, 'output', 'matching_results_v22.tsv')
    src_cand_v22 = os.path.join(base_dir, 'output', 'candidate_pairs_v22.tsv')
    
    # Step 1: Copy France and US directly from v22 (first 922,558 data rows + 1 header row)
    print("=" * 60)
    print("=== Step 1: Staging Proven France & US Predictions from v22 ===")
    print("=" * 60)
    
    FR_US_ROWS = 259452 + 663106 # 922,558 rows
    
    t0 = time.time()
    with open(src_match_v22, 'r', encoding='utf-8') as f_in, open(out_match, 'w', encoding='utf-8') as f_out:
        header = next(f_in)
        f_out.write(header)
        for i, line in enumerate(f_in):
            if i >= FR_US_ROWS: break
            f_out.write(line)
            
    with open(src_cand_v22, 'r', encoding='utf-8') as f_in, open(out_cand, 'w', encoding='utf-8') as f_out:
        header = next(f_in)
        f_out.write(header)
        for i, line in enumerate(f_in):
            if i >= FR_US_ROWS: break
            f_out.write(line)
            
    print(f"Staged {FR_US_ROWS:,} France & US rows into v24 in {time.time()-t0:.2f}s!")
    
    # Step 2: Run fixed India inference
    print("Loading model checkpoint...")
    clf_data = joblib.load(model_path)
    process_india_v24(part_dir, clf_data, out_match, out_cand)
    
    total_time = time.time() - t0
    print("\n" + "=" * 60)
    print(f"VERSION 2.4 INFERENCE COMPLETE IN {total_time:.1f}s ({total_time/60:.1f} minutes)!")
    print(f"Output files ready at:")
    print(f"  - {out_match}")
    print(f"  - {out_cand}")
    print("=" * 60)

if __name__ == '__main__':
    main()
