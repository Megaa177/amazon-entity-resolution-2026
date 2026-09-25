"""
Version 3.0 Fast Universal Multi-Country Entity Resolution Inference Engine
Powered by Tri-Model GBDT Ensemble (XGBoost + CatBoost + HistGradientBoosting)
with Phonetic Soundex, Acronym Cross-Blocking, and Graph Triangle Closure.
"""

import os
import sys
import time
import math
import gc
import joblib
import numpy as np
from collections import defaultdict

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
    from .blocking_v30 import CandidateBlockerV30, discover_country_stopwords
    from .features_v30 import compute_pair_features_v30
    from .model_v30 import TriModelEnsembleV30, resolve_mutual_exclusivity_v30, triangulate_triangle_closure_v30
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
    from blocking_v30 import CandidateBlockerV30, discover_country_stopwords
    from features_v30 import compute_pair_features_v30
    from model_v30 import TriModelEnsembleV30, resolve_mutual_exclusivity_v30, triangulate_triangle_closure_v30


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
    c_name = clean_business_name_v30(raw_name)
    c_addr = clean_address_v30(raw_addr, extra_stopwords=extra_stopwords)
    tokens = c_name.split()
    return {
        'entity_id': eid,
        'country': country,
        'phone': phone,
        'raw_name': raw_name,
        'clean_name': c_name,
        'name_tokens': tokens,
        'acronym': extract_acronym(raw_name),
        'distinctiveness': compute_distinctiveness(raw_name),
        'clean_addr': c_addr,
        'addr_digits': extract_address_digits_v30(raw_addr),
        'complex_codes': extract_complex_codes_v30(c_addr),
        'postal_code': extract_postal_code_v30(raw_addr, country)
    }


def process_country_v30(country: str, part_dir: str, clf_data: dict, out_match_path: str, out_cand_path: str):
    print(f"\n" + "=" * 65)
    print(f"=== Starting Version 3.0 Processing for: {country} ===")
    print(f"=" * 65)
    t_start = time.time()
    
    s1_path = os.path.join(part_dir, f's1_{country}.tsv')
    targets_path = os.path.join(part_dir, f'targets_{country}.tsv')
    
    if not os.path.exists(s1_path) or not os.path.exists(targets_path):
        print(f"Partition files missing for {country}, skipping.")
        return
        
    model = clf_data['model']
    tau = clf_data.get('threshold', 0.68)
    singleton_tau = clf_data.get('singleton_threshold', 0.74)
    margin_delta = clf_data.get('margin_delta', 0.17)
    ambiguity_margin = clf_data.get('ambiguity_margin', 0.05)
    print(f"Loaded Tri-Model Decision Parameters:")
    print(f"  tau={tau:.3f} | singleton_tau={singleton_tau:.3f} | margin_delta={margin_delta:.3f} | ambiguity_margin={ambiguity_margin:.3f}")

    # 1. Dynamically discover country-specific stopwords from target sample
    print(f"Sampling {country} records for dynamic stopword discovery...")
    addr_samples = []
    with open(targets_path, encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if idx >= 30000: break
            parts = line.rstrip('\n').split('\t')
            addr = parts[2] if len(parts) > 2 else ''
            addr_samples.append({'clean_addr': addr.lower(), 'name_tokens': []})
            
    country_stopwords, _ = discover_country_stopwords(addr_samples, min_freq=0.03)
    print(f"  Auto-discovered {len(country_stopwords)} stopwords for {country}: {sorted(list(country_stopwords))[:12]}")

    # 2. Load Source 1 entities & extract blocking query keys
    print(f"Loading {country} Source 1 entities...")
    s1_records = []
    needed_keys = set()
    
    blocker = CandidateBlockerV30(max_candidates_per_entity=25, max_bucket_size=350)
    blocker.extra_stopwords = country_stopwords
    
    with open(s1_path, encoding='utf-8') as f:
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if not parts[0]: continue
            eid = parts[0]
            name = parts[1] if len(parts) > 1 else ''
            addr = parts[2] if len(parts) > 2 else ''
            phone = parts[4] if len(parts) > 4 else ''
            rec = create_record(eid, name, addr, country, phone=phone, extra_stopwords=country_stopwords)
            s1_records.append(rec)
            keys = blocker.extract_blocking_keys(rec)
            for k in keys:
                needed_keys.add(k)
                
    print(f"Loaded {len(s1_records):,} {country} Source 1 entities. Unique query keys: {len(needed_keys):,}")
    
    # 3. Stream and index target candidates matching query keys
    print(f"Streaming and indexing {country} target candidates from targets_{country}.tsv...")
    t_idx = time.time()
    target_records = {}
    total_targets_scanned = 0
    
    with open(targets_path, encoding='utf-8') as f:
        for line in f:
            total_targets_scanned += 1
            parts = line.rstrip('\n').split('\t')
            eid = parts[0]
            name = parts[1] if len(parts) > 1 else ''
            addr = parts[2] if len(parts) > 2 else ''
            phone = parts[4] if len(parts) > 4 else ''
            rec = create_record(eid, name, addr, country, phone=phone, extra_stopwords=country_stopwords)
            keys = blocker.extract_blocking_keys(rec)
            matched = [k for k in keys if k in needed_keys]
            if matched:
                target_records[eid] = rec
                for k in matched:
                    blocker.index[k].append(eid)
                    
            if total_targets_scanned % 1000000 == 0:
                print(f"    Scanned {total_targets_scanned:,} targets... ({len(target_records):,} candidates indexed in {time.time()-t_idx:.1f}s)", flush=True)

    pruned = [k for k, v in blocker.index.items() if len(v) > blocker.max_bucket_size]
    for k in pruned:
        del blocker.index[k]
        
    blocker.precompute_weights()
    print(f"Indexed {len(target_records):,} relevant target candidates from {total_targets_scanned:,} total targets in {time.time()-t_idx:.1f}s.")
    
    # 4. Batched Feature Extraction & Model Inference
    print(f"Extracting 35 dense features & scoring pairs with Tri-Model Ensemble...")
    t_score = time.time()
    
    BATCH_SIZE = 50000
    batch_features = []
    batch_meta = []
    s1_cands_map = {}
    s1_probs_map = defaultdict(list)
    
    total_pairs_scored = 0
    
    def flush_batch():
        nonlocal batch_features, batch_meta, total_pairs_scored
        if not batch_features: return
        X = np.array(batch_features, dtype=np.float32)
        probs = model.predict_proba(X)
        if hasattr(probs, 'ndim') and probs.ndim > 1:
            probs = probs[:, 1]
        for (sid, cid), prob in zip(batch_meta, probs):
            s1_probs_map[sid].append((cid, float(prob)))
        total_pairs_scored += len(batch_features)
        batch_features = []
        batch_meta = []

    for idx, r1 in enumerate(s1_records):
        sid = r1['entity_id']
        cands = blocker.get_candidates(r1)
        s1_cands_map[sid] = cands
        
        for rank, cid in enumerate(cands):
            r2 = target_records.get(cid)
            if not r2: continue
            
            feat = compute_pair_features_v30(r1, r2, rank=rank)
            batch_features.append(feat)
            batch_meta.append((sid, cid))
                
        if len(batch_features) >= BATCH_SIZE:
            flush_batch()
            
        if (idx + 1) % 50000 == 0 or (idx + 1) == len(s1_records):
            print(f"  Scored {idx+1:,} / {len(s1_records):,} entities... ({total_pairs_scored:,} pairs scored in {time.time()-t_score:.1f}s)", flush=True)
                
    flush_batch()
    print(f"Finished scoring {total_pairs_scored:,} {country} pairs in {time.time()-t_score:.1f}s.")
    
    # 5. Graph Triangle Closure & Distinctiveness-Adaptive Thresholding
    print("Applying Graph Triangle Closure and Distinctiveness-Adaptive Thresholding...")
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
            triangulated = triangulate_triangle_closure_v30(sid, cands, target_records, target_tau=tau)
        else:
            triangulated = cands
        s1_probs_map[sid] = triangulated
        
        # Adaptive thresholding based on entity distinctiveness
        dist = r1.get('distinctiveness', 0.5)
        tau_eff = tau + (0.5 - dist) * 0.08
        s_tau_eff = singleton_tau + (0.5 - dist) * 0.06
        
        max_p = max(p for _, p in triangulated)
        if max_p < s_tau_eff:
            initial_matches[sid] = set()
        else:
            matched = {cid for cid, p in triangulated if p >= tau_eff and p >= (max_p - margin_delta)}
            initial_matches[sid] = matched

    # 6. Global Mutual Exclusivity with Anti-Stealing
    print("Applying Anti-Stealing Global Mutual Exclusivity (Zero Collision Guarantee)...")
    t_mutex = time.time()
    resolved_matches = resolve_mutual_exclusivity_v30(initial_matches, s1_probs_map, min_margin=ambiguity_margin)
    print(f"Mutual Exclusivity resolved in {time.time()-t_mutex:.2f}s.")
    
    # 7. Append results to disk
    print(f"Appending {country} results to output files...")
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
    print(f"Finished {country} in {elapsed:.1f}s ({elapsed/60:.1f} minutes)!")
    print(f"  Total S1 entities: {len(s1_records):,}")
    print(f"  Empty entities (Singletons): {empty_entities:,} ({empty_entities/len(s1_records)*100:.1f}%)")
    print(f"  Total matches predicted: {total_matches_found:,} (Avg {total_matches_found/len(s1_records):.2f}/entity)")
    
    del s1_records, target_records, blocker, s1_probs_map, initial_matches, resolved_matches, s1_cands_map
    gc.collect()


def main():
    base_dir = resolve_base_dirs()
    print(f"Resolved Project Base Directory: {base_dir}")
    
    part_dir = os.path.join(base_dir, 'scratch', 'partitions')
    model_path = os.path.join(base_dir, 'code', 'business_entity_resolution', 'src', 'model_v30_ensemble.joblib')
    
    out_match = os.path.join(base_dir, 'output', 'matching_results_v30.tsv')
    out_cand = os.path.join(base_dir, 'output', 'candidate_pairs_v30.tsv')
    os.makedirs(os.path.dirname(out_match), exist_ok=True)
    os.makedirs(os.path.dirname(out_cand), exist_ok=True)
    
    with open(out_match, 'w', encoding='utf-8') as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        
    with open(out_cand, 'w', encoding='utf-8') as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        
    print(f"Output files initialized with competition headers:")
    print(f"  - {out_match}")
    print(f"  - {out_cand}")
    
    print(f"Loading Tri-Model Ensemble Checkpoint: {model_path}...")
    clf_data = joblib.load(model_path)
    
    t0 = time.time()
    
    for country in ['France', 'US', 'India']:
        process_country_v30(country, part_dir, clf_data, out_match, out_cand)
        
    total_time = time.time() - t0
    print("\n" + "=" * 70)
    print(f"VERSION 3.0 TRI-MODEL INFERENCE COMPLETE IN {total_time:.1f}s ({total_time/60:.1f} minutes)!")
    print(f"Output files ready at:")
    print(f"  - {out_match}")
    print(f"  - {out_cand}")
    print("=" * 70)


if __name__ == '__main__':
    main()
