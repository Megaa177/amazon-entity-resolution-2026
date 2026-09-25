import os
import sys
import time
import re
import argparse
import joblib
import numpy as np
from collections import defaultdict

try:
    from .preprocessing import (
        clean_business_name, 
        get_name_tokens, 
        clean_address, 
        extract_address_digits, 
        extract_postal_code,
        extract_complex_codes
    )
    from .blocking_v21 import CandidateBlockerV21, GENERIC_STREET_WORDS
    from .features_v21 import compute_pair_features_v21
    from .model_v21 import MatchClassifierV21, resolve_mutual_exclusivity_v21, triangulate_siblings
except ImportError:
    from preprocessing import (
        clean_business_name, 
        get_name_tokens, 
        clean_address, 
        extract_address_digits, 
        extract_postal_code,
        extract_complex_codes
    )
    from blocking_v21 import CandidateBlockerV21, GENERIC_STREET_WORDS
    from features_v21 import compute_pair_features_v21
    from model_v21 import MatchClassifierV21, resolve_mutual_exclusivity_v21, triangulate_siblings


def resolve_base_dirs():
    """Dynamically determine project base directory."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.abspath(os.path.join(script_dir, '..', '..', '..')), # student_resource/
        os.path.abspath(os.path.join(script_dir, '..', '..')),
        os.path.abspath(os.path.join(script_dir, '..')),
        os.getcwd()
    ]
    for b in candidates:
        if os.path.exists(os.path.join(b, 'dataset', 'test')):
            return b
    return os.getcwd()


def ensure_partitions(test_dir: str, part_dir: str):
    """Automatically partition test data by country if partition files do not exist."""
    os.makedirs(part_dir, exist_ok=True)
    needed = [
        os.path.join(part_dir, f's1_{c}.tsv') for c in ['France', 'US', 'India']
    ] + [
        os.path.join(part_dir, f'targets_{c}.tsv') for c in ['France', 'US', 'India']
    ]
    if all(os.path.exists(p) for p in needed):
        return

    print("=== Auto-partitioning test data by country... ===")
    t_part = time.time()
    
    # 1. Stream S1
    s1_src = os.path.join(test_dir, 'test_source1.tsv')
    if not os.path.exists(s1_src):
        raise FileNotFoundError(f"Missing {s1_src}! Ensure test dataset is present in {test_dir}")
        
    s1_files = {c: open(os.path.join(part_dir, f's1_{c}.tsv'), 'w', encoding='utf-8') for c in ['France', 'US', 'India']}
    with open(s1_src, encoding='utf-8') as f:
        f.readline() # header
        for line in f:
            p = line.rstrip('\n').split('\t')
            c = p[3] if len(p) > 3 else 'US'
            if c in s1_files:
                s1_files[c].write(line)
    for fh in s1_files.values(): fh.close()
    
    # 2. Stream Targets (Source 2 and Source 3)
    target_files = {c: open(os.path.join(part_dir, f'targets_{c}.tsv'), 'w', encoding='utf-8') for c in ['France', 'US', 'India']}
    for src_name in ['test_source2.tsv', 'test_source3.tsv']:
        src_path = os.path.join(test_dir, src_name)
        if not os.path.exists(src_path): continue
        with open(src_path, encoding='utf-8') as f:
            f.readline() # header
            for line in f:
                p = line.rstrip('\n').split('\t')
                c = p[3] if len(p) > 3 else 'US'
                if c in target_files:
                    target_files[c].write(line)
    for fh in target_files.values(): fh.close()
    print(f"Country partitions generated in {time.time()-t_part:.1f}s.")


def create_record(eid: str, raw_name: str, raw_addr: str, country: str) -> dict:
    c_name = clean_business_name(raw_name)
    c_addr = clean_address(raw_addr)
    norm_addr = re.sub(r'[-/]', ' ', c_addr)
    st_words = [
        w for w in re.findall(r'[a-z]+', norm_addr) 
        if len(w) >= 3 and w not in GENERIC_STREET_WORDS
    ]
    return {
        'entity_id': eid,
        'country': country,
        'raw_name': raw_name,
        'clean_name': c_name,
        'name_tokens': get_name_tokens(raw_name),
        'clean_addr': c_addr,
        'addr_digits': extract_address_digits(raw_addr),
        'complex_codes': extract_complex_codes(c_addr),
        'postal_code': extract_postal_code(raw_addr, country),
        'street_words': st_words
    }


def process_country_v21(country: str, part_dir: str, clf_data: dict, out_match_path: str, out_cand_path: str):
    print(f"\n=======================================================")
    print(f"=== Starting Version 2.1 Processing for: {country} ===")
    print(f"=======================================================")
    t_start = time.time()
    
    s1_path = os.path.join(part_dir, f's1_{country}.tsv')
    targets_path = os.path.join(part_dir, f'targets_{country}.tsv')
    
    if not os.path.exists(s1_path) or not os.path.exists(targets_path):
        print(f"Partition files missing for {country}, skipping.")
        return
        
    model = clf_data['model']
    tau = clf_data.get('threshold', 0.68)
    singleton_tau = clf_data.get('singleton_threshold', 0.76)
    margin_delta = clf_data.get('margin_delta', 0.16)
    ambiguity_margin = clf_data.get('ambiguity_margin', 0.05)
    print(f"Loaded Decision Parameters:")
    print(f"  tau={tau:.3f} | singleton_tau={singleton_tau:.3f} | margin_delta={margin_delta:.3f} | ambiguity_margin={ambiguity_margin:.3f}")

    # 1. Load S1 entities
    print(f"Loading {country} Source 1 entities...")
    s1_records = []
    needed_keys = set()
    s1_keys_map = {}
    
    blocker = CandidateBlockerV21(max_candidates_per_entity=18, max_bucket_size=350)
    
    with open(s1_path, encoding='utf-8') as f:
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if not parts[0]: continue
            eid = parts[0]
            name = parts[1] if len(parts) > 1 else ''
            addr = parts[2] if len(parts) > 2 else ''
            rec = create_record(eid, name, addr, country)
            s1_records.append(rec)
            keys = blocker.extract_blocking_keys(rec)
            s1_keys_map[eid] = keys
            for k in keys:
                needed_keys.add(k)
                
    print(f"Loaded {len(s1_records):,} Source 1 entities. Unique query keys: {len(needed_keys):,}")
    
    # 2. Stream and index target records matching needed keys
    print(f"Streaming and indexing {country} target candidates with IDF damping...")
    t_idx = time.time()
    target_records = {}
    
    with open(targets_path, encoding='utf-8') as f:
        for line in f:
            parts = line.rstrip('\n').split('\t')
            eid = parts[0]
            name = parts[1] if len(parts) > 1 else ''
            addr = parts[2] if len(parts) > 2 else ''
            rec = create_record(eid, name, addr, country)
            keys = blocker.extract_blocking_keys(rec)
            matched = [k for k in keys if k in needed_keys]
            if matched:
                target_records[eid] = rec
                for k in matched:
                    blocker.index[k].append(eid)
                    
    # Prune keys larger than max_bucket_size
    pruned = [k for k, v in blocker.index.items() if len(v) > blocker.max_bucket_size]
    for k in pruned:
        del blocker.index[k]
        
    blocker.precompute_weights()
    print(f"Indexed {len(target_records):,} target records in {time.time()-t_idx:.1f}s.")
    
    # 3. Batched ML Feature Extraction and Inference
    print(f"Scoring candidate pairs in batches...")
    t_score = time.time()
    
    BATCH_SIZE = 50000
    batch_features = []
    batch_meta = [] # (sid, cid)
    s1_cands_map = {}
    s1_probs_map = defaultdict(list)
    
    total_pairs_scored = 0
    
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
        
        for rank, cid in enumerate(cands):
            r2 = target_records.get(cid)
            if r2:
                feat = compute_pair_features_v21(r1, r2, rank=rank)
                batch_features.append(feat)
                batch_meta.append((sid, cid))
                
        if len(batch_features) >= BATCH_SIZE:
            flush_batch()
            
        if (idx + 1) % 50000 == 0 or (idx + 1) == len(s1_records):
            print(f"  Scored {idx+1:,} / {len(s1_records):,} S1 entities... ({total_pairs_scored:,} pairs scored in {time.time()-t_score:.1f}s)", flush=True)
                
    flush_batch()
    print(f"Finished scoring {total_pairs_scored:,} candidate pairs in {time.time()-t_score:.1f}s.")
    
    # 4. S2 <-> S3 Sibling Triangulation & 3-Tier Metric Filter
    print("Applying S2 <-> S3 Sibling Triangulation and 3-Tier Metric Filter...")
    initial_matches = {}
    for r1 in s1_records:
        sid = r1['entity_id']
        cands = s1_probs_map.get(sid, [])
        if not cands:
            initial_matches[sid] = set()
            continue
            
        # Triangulate S2 and S3 siblings
        triangulated = triangulate_siblings(sid, cands, target_records, target_tau=tau)
        s1_probs_map[sid] = triangulated
        
        max_p = max(p for _, p in triangulated)
        if max_p < singleton_tau:
            initial_matches[sid] = set()
            continue
            
        matched = {cid for cid, p in triangulated if p >= tau and p >= (max_p - margin_delta)}
        initial_matches[sid] = matched

    # 5. Global Mutual Exclusivity with Anti-Stealing Ambiguity Rejection
    print("Applying Anti-Stealing Global Mutual Exclusivity...")
    t_mutex = time.time()
    resolved_matches = resolve_mutual_exclusivity_v21(initial_matches, s1_probs_map, min_margin=ambiguity_margin)
    print(f"Mutual Exclusivity resolved in {time.time()-t_mutex:.2f}s.")
    
    # 6. Stream results to output files
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
    parser = argparse.ArgumentParser(description="Version 2.1 Fast Multi-Country Entity Resolution Inference")
    parser.add_argument('--output_match', type=str, default=None, help="Path for matching_results TSV")
    parser.add_argument('--output_cand', type=str, default=None, help="Path for candidate_pairs TSV")
    parser.add_argument('--model_path', type=str, default=None, help="Path to model_v21.joblib")
    args = parser.parse_args()

    print("=== Starting Version 2.1 Multi-Country High-Recall Inference Pipeline ===")
    t_all = time.time()
    
    base_dir = resolve_base_dirs()
    print(f"Resolved Project Base Directory: {base_dir}")
    
    test_dir = os.path.join(base_dir, 'dataset', 'test')
    part_dir = os.path.join(base_dir, 'scratch', 'partitions')
    out_dir = os.path.join(base_dir, 'output')
    os.makedirs(out_dir, exist_ok=True)
    
    ensure_partitions(test_dir, part_dir)
    
    model_path = args.model_path or os.path.join(os.path.dirname(__file__), 'model_v21.joblib')
    if not os.path.exists(model_path):
        model_path = os.path.join(base_dir, 'code', 'business_entity_resolution', 'src', 'model_v21.joblib')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Version 2.1 Model not found at {model_path}!")
        
    print(f"Loading model checkpoint from {model_path}...")
    clf_data = joblib.load(model_path)
    
    out_match = args.output_match or os.path.join(out_dir, 'matching_results_v21.tsv')
    out_cand = args.output_cand or os.path.join(out_dir, 'candidate_pairs_v21.tsv')
    
    # Initialize files with competition headers
    with open(out_match, 'w', encoding='utf-8') as fm, open(out_cand, 'w', encoding='utf-8') as fc:
        fm.write("source1_entity_id\tmatched_entity_ids\n")
        fc.write("source1_entity_id\tcandidate_entity_ids\n")
        
    for country in ['France', 'US', 'India']:
        process_country_v21(country, part_dir, clf_data, out_match, out_cand)
        
    print(f"\n=======================================================")
    print(f"ALL COUNTRIES COMPLETED IN {time.time()-t_all:.1f}s ({(time.time()-t_all)/60:.1f} minutes)!")
    print(f"Output files ready at:")
    print(f"  - {out_match}")
    print(f"  - {out_cand}")
    print(f"=======================================================")


if __name__ == '__main__':
    main()
