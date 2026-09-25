"""
Version 2.3 High-Performance Test Inference Engine
With Region-Adaptive Co-Location Defense & Multilingual Indian Recall Recovery
"""

import os
import sys
import time
import re
import argparse
import joblib
import numpy as np
from collections import defaultdict, Counter

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
    from .blocking_v23 import BlockerV23, discover_country_stopwords
    from .features_v23 import compute_pair_features_v23, token_sort_jaccard, char_dice_2gram
    from .model_v23 import MatchClassifierV23, resolve_mutual_exclusivity_v23, triangulate_siblings_v23
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
    from blocking_v23 import BlockerV23, discover_country_stopwords
    from features_v23 import compute_pair_features_v23, token_sort_jaccard, char_dice_2gram
    from model_v23 import MatchClassifierV23, resolve_mutual_exclusivity_v23, triangulate_siblings_v23


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


def create_record_v23(eid: str, raw_name: str, raw_addr: str, country: str, phone: str = '', extra_stopwords: set = None) -> dict:
    c_name = clean_business_name_v23(raw_name)
    c_addr = clean_address_v23(raw_addr, extra_stopwords=extra_stopwords)
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
        'addr_digits': extract_address_digits_v23(raw_addr),
        'complex_codes': extract_complex_codes_v23(c_addr),
        'postal_code': extract_postal_code_v23(raw_addr, country)
    }


def process_country_v23(country: str, part_dir: str, clf_data: dict, out_match_path: str, out_cand_path: str):
    print(f"\n=======================================================")
    print(f"=== Starting Version 2.3 Processing for: {country} ===")
    print(f"=======================================================")
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
    print(f"Loaded Decision Parameters:")
    print(f"  tau={tau:.3f} | singleton_tau={singleton_tau:.3f} | margin_delta={margin_delta:.3f} | ambiguity_margin={ambiguity_margin:.3f}")

    # 1. Discover country stopwords
    print(f"Sampling {country} records for dynamic stopword discovery...")
    addr_samples = []
    with open(targets_path, encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if idx >= 30000: break
            parts = line.rstrip('\n').split('\t')
            addr = parts[2] if len(parts) > 2 else ''
            addr_samples.append({'address': addr})
            
    country_stopwords = discover_country_stopwords(addr_samples, min_freq=0.03)
    print(f"  Auto-discovered {len(country_stopwords)} stopwords for {country}: {sorted(list(country_stopwords))[:12]}")

    # 2. Load S1 entities
    print(f"Loading {country} Source 1 entities...")
    s1_records = []
    blocker = BlockerV23(country=country, extra_stopwords=country_stopwords)
    
    with open(s1_path, encoding='utf-8') as f:
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if not parts[0]: continue
            eid = parts[0]
            name = parts[1] if len(parts) > 1 else ''
            addr = parts[2] if len(parts) > 2 else ''
            phone = parts[3] if len(parts) > 3 and country == 'India' else ''
            rec = create_record_v23(eid, name, addr, country, phone=phone, extra_stopwords=country_stopwords)
            s1_records.append(rec)
                
    print(f"Loaded {len(s1_records):,} Source 1 entities.")
    
    # 3. Stream and index target candidates
    print(f"Streaming and indexing {country} target candidates...")
    t_idx = time.time()
    target_records = {}
    
    with open(targets_path, encoding='utf-8') as f:
        for line in f:
            parts = line.rstrip('\n').split('\t')
            eid = parts[0]
            name = parts[1] if len(parts) > 1 else ''
            addr = parts[2] if len(parts) > 2 else ''
            phone = parts[3] if len(parts) > 3 and country == 'India' else ''
            rec = create_record_v23(eid, name, addr, country, phone=phone, extra_stopwords=country_stopwords)
            target_records[eid] = rec
            blocker.index_target(eid, rec)
                    
    blocker.finalize_index()
    print(f"Indexed {len(target_records):,} target records in {time.time()-t_idx:.1f}s.")
    
    # 4. Batched Feature Extraction & ML Scoring
    print(f"Scoring candidate pairs in batches with Region-Adaptive Gate...")
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
        cands = blocker.retrieve_candidates(r1)
        s1_cands_map[sid] = cands
        
        tok1 = r1['name_tokens']
        c_name1 = r1['clean_name']
        
        for rank, cid in enumerate(cands):
            r2 = target_records.get(cid)
            if not r2: continue
            
            tok2 = r2['name_tokens']
            c_name2 = r2['clean_name']
            
            sort_sim = token_sort_jaccard(tok1, tok2)
            dice = char_dice_2gram(c_name1, c_name2)
            
            # --- REGION-ADAPTIVE GATE ---
            if country in ('France', 'US'):
                # In France and US: Strong co-location defense against multi-tenant commercial centers
                if c_name1 != c_name2 and sort_sim < 0.40 and dice < 0.35 and len(tok1) >= 2 and len(tok2) >= 2:
                    s1_probs_map[sid].append((cid, 0.0))
                    co_location_rejected += 1
                    continue
            elif country == 'India':
                # In India: Multilingual script & OCR typo protection
                # NEVER reject if phone matches or plot code matches or char dice >= 0.28
                digits_match = bool(set(r1.get('addr_digits', [])) & set(r2.get('addr_digits', [])))
                codes_match = bool(set(r1.get('complex_codes', [])) & set(r2.get('complex_codes', [])))
                phone_match = bool(r1.get('phone') and r1.get('phone') == r2.get('phone'))
                
                if c_name1 != c_name2 and sort_sim < 0.25 and dice < 0.22 and not (digits_match or codes_match or phone_match):
                    s1_probs_map[sid].append((cid, 0.0))
                    co_location_rejected += 1
                    continue
                
            feat = compute_pair_features_v23(r1, r2, rank=rank)
            batch_features.append(feat)
            batch_meta.append((sid, cid))
                
        if len(batch_features) >= BATCH_SIZE:
            flush_batch()
            
        if (idx + 1) % 50000 == 0 or (idx + 1) == len(s1_records):
            print(f"  Scored {idx+1:,} / {len(s1_records):,} S1 entities... ({total_pairs_scored:,} pairs scored in {time.time()-t_score:.1f}s, {co_location_rejected:,} vetoed)", flush=True)
                
    flush_batch()
    print(f"Finished scoring {total_pairs_scored:,} pairs in {time.time()-t_score:.1f}s. Total vetoed: {co_location_rejected:,}")
    
    # 5. Sibling Triangulation & Adaptive Distinctiveness Filter
    print("Applying Sibling Triangulation and Adaptive Distinctiveness Filter...")
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
            triangulated = triangulate_siblings_v23(sid, cands, target_records, target_tau=tau)
        else:
            triangulated = cands
        s1_probs_map[sid] = triangulated
        
        # Adaptive thresholding
        dist = r1.get('distinctiveness', 0.5)
        tau_eff = tau + (0.5 - dist) * 0.08
        s_tau_eff = singleton_tau + (0.5 - dist) * 0.06

        max_p = max(p for _, p in triangulated)
        if max_p < s_tau_eff:
            initial_matches[sid] = set()
            continue
            
        matched = {cid for cid, p in triangulated if p >= tau_eff and p >= (max_p - margin_delta)}
        initial_matches[sid] = matched

    # 6. Global Mutual Exclusivity with Anti-Stealing
    print("Applying Anti-Stealing Global Mutual Exclusivity...")
    t_mutex = time.time()
    resolved_matches = resolve_mutual_exclusivity_v23(initial_matches, s1_probs_map, min_margin=ambiguity_margin)
    print(f"Mutual Exclusivity resolved in {time.time()-t_mutex:.2f}s.")
    
    # 7. Write results to disk
    print(f"Writing {country} results to disk...")
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
    print(f"Finished {country} in {elapsed:.1f}s!")
    print(f"  Total S1 entities: {len(s1_records):,}")
    print(f"  Empty entities (Singletons): {empty_entities:,} ({empty_entities/len(s1_records)*100:.1f}%)")
    print(f"  Total matches predicted: {total_matches_found:,} (Avg {total_matches_found/len(s1_records):.2f}/entity)")


def main():
    parser = argparse.ArgumentParser(description="Run Version 2.3 Fast Multi-Country Entity Resolution Inference")
    parser.add_argument('--output_match', type=str, default='output/matching_results_v23.tsv')
    parser.add_argument('--output_cand', type=str, default='output/candidate_pairs_v23.tsv')
    args = parser.parse_args()
    
    base_dir = resolve_base_dirs()
    print(f"Resolved Project Base Directory: {base_dir}")
    
    part_dir = os.path.join(base_dir, 'scratch', 'partitions')
    model_path = os.path.join(base_dir, 'code', 'business_entity_resolution', 'src', 'model_v23.joblib')
    
    # Strip any redundant 'student_resource' prefix if present
    match_rel = args.output_match
    if match_rel.startswith('student_resource/') or match_rel.startswith('student_resource\\'):
        match_rel = match_rel[len('student_resource/'):]
    cand_rel = args.output_cand
    if cand_rel.startswith('student_resource/') or cand_rel.startswith('student_resource\\'):
        cand_rel = cand_rel[len('student_resource/'):]

    out_match = os.path.join(base_dir, match_rel)
    out_cand = os.path.join(base_dir, cand_rel)
    os.makedirs(os.path.dirname(out_match), exist_ok=True)
    os.makedirs(os.path.dirname(out_cand), exist_ok=True)
    
    # Initialize TSV files with headers
    with open(out_match, 'w', encoding='utf-8') as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
    with open(out_cand, 'w', encoding='utf-8') as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        
    print(f"Loading model checkpoint from {model_path}...")
    clf_data = joblib.load(model_path)
    
    t0 = time.time()
    # Process all three countries
    for country in ['France', 'US', 'India']:
        process_country_v23(country, part_dir, clf_data, out_match, out_cand)
        
    total_time = time.time() - t0
    print("\n" + "=" * 55)
    print(f"ALL COUNTRIES COMPLETED IN {total_time:.1f}s ({total_time/60:.1f} minutes)!")
    print(f"Output files ready at:")
    print(f"  - {out_match}")
    print(f"  - {out_cand}")
    print("=" * 55)

if __name__ == '__main__':
    main()
