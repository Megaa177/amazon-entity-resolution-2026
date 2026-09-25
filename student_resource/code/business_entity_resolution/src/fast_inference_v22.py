import os
import sys
import time
import re
import argparse
import joblib
import numpy as np
from collections import defaultdict, Counter

try:
    from .preprocessing_v22 import (
        clean_business_name_v22, 
        get_name_tokens_v22, 
        clean_address_v22, 
        extract_address_digits_v22, 
        extract_postal_code_v22,
        extract_complex_codes_v22,
        clean_mojibake,
        transliterate_indic,
        UNIVERSAL_STREET_WORDS,
        UNIVERSAL_LEGAL_SUFFIXES
    )
    from .blocking_v22 import CandidateBlockerV22, discover_country_stopwords
    from .features_v22 import compute_pair_features_v22, token_sort_jaccard
    from .model_v22 import MatchClassifierV22, resolve_mutual_exclusivity_v22, triangulate_siblings_v22
except ImportError:
    from preprocessing_v22 import (
        clean_business_name_v22, 
        get_name_tokens_v22, 
        clean_address_v22, 
        extract_address_digits_v22, 
        extract_postal_code_v22,
        extract_complex_codes_v22,
        clean_mojibake,
        transliterate_indic,
        UNIVERSAL_STREET_WORDS,
        UNIVERSAL_LEGAL_SUFFIXES
    )
    from blocking_v22 import CandidateBlockerV22, discover_country_stopwords
    from features_v22 import compute_pair_features_v22, token_sort_jaccard
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


def create_record_v22(eid: str, raw_name: str, raw_addr: str, country: str, extra_stopwords: set = None) -> dict:
    c_name = clean_business_name_v22(raw_name)
    c_addr = clean_address_v22(raw_addr, extra_stopwords=extra_stopwords)
    return {
        'entity_id': eid,
        'country': country,
        'raw_name': raw_name,
        'clean_name': c_name,
        'name_tokens': get_name_tokens_v22(raw_name),
        'clean_addr': c_addr,
        'addr_digits': extract_address_digits_v22(raw_addr),
        'complex_codes': extract_complex_codes_v22(c_addr),
        'postal_code': extract_postal_code_v22(raw_addr, country)
    }


def process_country_v22(country: str, part_dir: str, clf_data: dict, out_match_path: str, out_cand_path: str):
    print(f"\n=======================================================")
    print(f"=== Starting Version 2.2 Processing for: {country} ===")
    print(f"=======================================================")
    t_start = time.time()
    
    s1_path = os.path.join(part_dir, f's1_{country}.tsv')
    targets_path = os.path.join(part_dir, f'targets_{country}.tsv')
    
    if not os.path.exists(s1_path) or not os.path.exists(targets_path):
        print(f"Partition files missing for {country}, skipping.")
        return
        
    model = clf_data['model']
    tau = clf_data.get('threshold', 0.70)
    singleton_tau = clf_data.get('singleton_threshold', 0.76)
    margin_delta = clf_data.get('margin_delta', 0.16)
    ambiguity_margin = clf_data.get('ambiguity_margin', 0.06)
    print(f"Loaded Decision Parameters:")
    print(f"  tau={tau:.3f} | singleton_tau={singleton_tau:.3f} | margin_delta={margin_delta:.3f} | ambiguity_margin={ambiguity_margin:.3f}")

    # 1. First-pass sample to dynamically discover country-specific stopwords
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

    # 2. Load S1 entities
    print(f"Loading {country} Source 1 entities...")
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
            rec = create_record_v22(eid, name, addr, country, extra_stopwords=country_stopwords)
            s1_records.append(rec)
            keys = blocker.extract_blocking_keys(rec)
            s1_keys_map[eid] = keys
            for k in keys:
                needed_keys.add(k)
                
    print(f"Loaded {len(s1_records):,} Source 1 entities. Unique query keys: {len(needed_keys):,}")
    
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
            rec = create_record_v22(eid, name, addr, country, extra_stopwords=country_stopwords)
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
    print(f"Indexed {len(target_records):,} target records in {time.time()-t_idx:.1f}s.")
    
    # 4. Batched Feature Extraction & ML Scoring
    print(f"Scoring candidate pairs in batches with Strict Name Gate...")
    t_score = time.time()
    
    BATCH_SIZE = 50000
    batch_features = []
    batch_meta = [] # (sid, cid)
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
            
            # --- STRICT NAME GATE ---
            # If two businesses have conflicting names (< 0.40 token sort similarity),
            # reject immediately to prevent multi-tenant building false merges!
            tok2 = r2['name_tokens']
            c_name2 = r2['clean_name']
            
            sort_sim = token_sort_jaccard(tok1, tok2)
            if c_name1 != c_name2 and sort_sim < 0.40 and len(tok1) >= 2 and len(tok2) >= 2:
                # Suppress probability directly to 0.0 (Hard Name Gate)
                s1_probs_map[sid].append((cid, 0.0))
                co_location_rejected += 1
                continue
                
            feat = compute_pair_features_v22(r1, r2, rank=rank)
            batch_features.append(feat)
            batch_meta.append((sid, cid))
                
        if len(batch_features) >= BATCH_SIZE:
            flush_batch()
            
        if (idx + 1) % 50000 == 0 or (idx + 1) == len(s1_records):
            print(f"  Scored {idx+1:,} / {len(s1_records):,} S1 entities... ({total_pairs_scored:,} pairs scored in {time.time()-t_score:.1f}s, {co_location_rejected:,} co-locations rejected)", flush=True)
                
    flush_batch()
    print(f"Finished scoring {total_pairs_scored:,} pairs in {time.time()-t_score:.1f}s. Total co-location false merges blocked: {co_location_rejected:,}")
    
    # 5. Sibling Triangulation & 3-Tier Filter
    print("Applying Sibling Triangulation and 3-Tier Metric Filter...")
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
    print(f"Writing {country} results to disk...")
    f_match = open(out_match_path, 'a', encoding='utf-8')
    f_cand = open(out_cand_path, 'a', encoding='utf-8')
    
    total_matches_found = 0
    empty_entities = 0
    for r1 in s1_records:
        sid = r1['entity_id']
        m_set = resolved_matches.get(sid, set())
        m_list = sorted(list(m_set))
        c_list = s1_cands_map.get(sid, [])
        
        if not m_list:
            empty_entities += 1
        else:
            total_matches_found += len(m_list)
            
        f_match.write(f"{sid}\t{','.join(m_list)}\n")
        f_cand.write(f"{sid}\t{','.join(c_list)}\n")
        
    f_match.close()
    f_cand.close()
    
    print(f"Finished {country} in {time.time()-t_start:.1f}s!")
    print(f"  Total S1 entities: {len(s1_records):,}")
    print(f"  Empty entities (Singletons): {empty_entities:,} ({empty_entities/len(s1_records)*100:.1f}%)")
    print(f"  Total matches predicted: {total_matches_found:,} (Avg {total_matches_found/len(s1_records):.2f}/entity)")


def main():
    parser = argparse.ArgumentParser(description="Version 2.2 Universal Entity Resolution Inference")
    parser.add_argument('--output_match', type=str, default=None, help="Path for matching_results TSV")
    parser.add_argument('--output_cand', type=str, default=None, help="Path for candidate_pairs TSV")
    parser.add_argument('--model_path', type=str, default=None, help="Path to model_v22.joblib")
    args = parser.parse_args()

    print("=== Starting Version 2.2 Universal Multi-Country Inference Pipeline ===")
    t_all = time.time()
    
    base_dir = resolve_base_dirs()
    print(f"Resolved Project Base Directory: {base_dir}")
    
    part_dir = os.path.join(base_dir, 'scratch', 'partitions')
    out_dir = os.path.join(base_dir, 'output')
    os.makedirs(out_dir, exist_ok=True)
    
    model_path = args.model_path or os.path.join(os.path.dirname(__file__), 'model_v22.joblib')
    if not os.path.exists(model_path):
        model_path = os.path.join(base_dir, 'code', 'business_entity_resolution', 'src', 'model_v22.joblib')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Version 2.2 Model not found at {model_path}!")
        
    print(f"Loading model checkpoint from {model_path}...")
    clf_data = joblib.load(model_path)
    
    out_match = args.output_match or os.path.join(out_dir, 'matching_results_v22.tsv')
    out_cand = args.output_cand or os.path.join(out_dir, 'candidate_pairs_v22.tsv')
    
    with open(out_match, 'w', encoding='utf-8') as fm, open(out_cand, 'w', encoding='utf-8') as fc:
        fm.write("source1_entity_id\tmatched_entity_ids\n")
        fc.write("source1_entity_id\tcandidate_entity_ids\n")
        
    for country in ['France', 'US', 'India']:
        process_country_v22(country, part_dir, clf_data, out_match, out_cand)
        
    print(f"\n=======================================================")
    print(f"ALL COUNTRIES COMPLETED IN {time.time()-t_all:.1f}s ({(time.time()-t_all)/60:.1f} minutes)!")
    print(f"Output files ready at:")
    print(f"  - {out_match}")
    print(f"  - {out_cand}")
    print(f"=======================================================")


if __name__ == '__main__':
    main()
