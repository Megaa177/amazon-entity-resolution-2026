import os
import sys
import time
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
    from .blocking import CandidateBlocker
    from .features import compute_pair_features
    from .model import MatchClassifierV2, resolve_mutual_exclusivity
except ImportError:
    from preprocessing import (
        clean_business_name, 
        get_name_tokens, 
        clean_address, 
        extract_address_digits, 
        extract_postal_code,
        extract_complex_codes
    )
    from blocking import CandidateBlocker
    from features import compute_pair_features
    from model import MatchClassifierV2, resolve_mutual_exclusivity


def resolve_base_dirs():
    """Dynamically determine directory paths regardless of where the script is executed from."""
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

    print("=== Partition files not found. Auto-partitioning test data by country... ===")
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
    return {
        'entity_id': eid,
        'country': country,
        'clean_name': c_name,
        'name_tokens': get_name_tokens(raw_name),
        'clean_addr': c_addr,
        'addr_digits': extract_address_digits(raw_addr),
        'complex_codes': extract_complex_codes(c_addr),
        'postal_code': extract_postal_code(raw_addr, country)
    }


def process_country(country: str, part_dir: str, clf_data: dict, out_match_path: str, out_cand_path: str):
    print(f"\n=======================================================")
    print(f"=== Starting Processing for Country: {country} ===")
    print(f"=======================================================")
    t_start = time.time()
    
    s1_path = os.path.join(part_dir, f's1_{country}.tsv')
    targets_path = os.path.join(part_dir, f'targets_{country}.tsv')
    
    if not os.path.exists(s1_path) or not os.path.exists(targets_path):
        print(f"Partition files missing for {country}, skipping.")
        return
        
    model = clf_data['model']
    tau = clf_data.get('threshold', 0.65)
    singleton_tau = clf_data.get('singleton_threshold', 0.78)
    margin_delta = clf_data.get('margin_delta', 0.18)
    print(f"Loaded Decision Parameters: tau={tau:.3f}, singleton_tau={singleton_tau:.3f}, margin_delta={margin_delta:.3f}")

    # 1. Load S1 entities
    print(f"Loading {country} Source 1 entities...")
    s1_records = []
    needed_keys = set()
    s1_keys_map = {}
    
    blocker = CandidateBlocker(max_candidates_per_entity=15, max_bucket_size=200)
    
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
    
    # 2. Stream and index target records that match needed keys
    print(f"Streaming and indexing {country} target candidates...")
    t_idx = time.time()
    index = defaultdict(list)
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
                    index[k].append(eid)
                    
    # Prune overgrown buckets (> 200 records) to suppress generic commercial noise
    pruned = [k for k, v in index.items() if len(v) > blocker.max_bucket_size]
    for k in pruned:
        del index[k]
        
    print(f"Indexed {len(target_records):,} target records in {time.time()-t_idx:.1f}s.")
    
    # 3. Candidate generation & Batched ML Inference
    print(f"Scoring candidate pairs in batches...")
    t_score = time.time()
    
    weights = {
        'ex': 6, 'bi': 5, 'code': 5, 'comp': 4,
        'addr_co': 4, 'addr_num_st': 4, 'post_num': 3,
        'addr_num_st2': 2, 'tk': 1
    }
    BATCH_SIZE = 50000
    
    batch_features = []
    batch_meta = [] # (sid, cid)
    s1_cands_map = {} # sid -> list of cands
    s1_probs_map = defaultdict(list) # sid -> list of (cid, prob)
    
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
        keys = s1_keys_map.get(sid, [])
        score_map = defaultdict(int)
        for k in keys:
            w = weights.get(k[0], 1)
            for cid in index.get(k, []):
                score_map[cid] += w
                
        if not score_map:
            s1_cands_map[sid] = []
            continue
            
        ranked = sorted(score_map.items(), key=lambda x: x[1], reverse=True)
        cands = [cid for cid, _ in ranked[:blocker.max_candidates]]
        s1_cands_map[sid] = cands
        
        for rank, cid in enumerate(cands):
            r2 = target_records.get(cid)
            if r2:
                feat = compute_pair_features(r1, r2, rank=rank)
                batch_features.append(feat)
                batch_meta.append((sid, cid))
                
        if len(batch_features) >= BATCH_SIZE:
            flush_batch()
            if (idx + 1) % 50000 == 0:
                print(f"  Processed {idx+1:,} / {len(s1_records):,} S1 entities... ({total_pairs_scored:,} pairs scored)")
                
    flush_batch()
    print(f"Finished scoring {total_pairs_scored:,} candidate pairs in {time.time()-t_score:.1f}s.")
    
    # 4. Apply 3-Tier Metric Filter
    print("Applying 3-Tier Metric Filter (Singleton Shield + Margin Filter)...")
    initial_matches = {}
    for r1 in s1_records:
        sid = r1['entity_id']
        cands = s1_probs_map.get(sid, [])
        if not cands:
            initial_matches[sid] = set()
            continue
        max_p = max(p for _, p in cands)
        if max_p < singleton_tau:
            initial_matches[sid] = set()
            continue
        matched = {cid for cid, p in cands if p >= tau and p >= (max_p - margin_delta)}
        initial_matches[sid] = matched

    # 5. Apply Global Mutual Exclusivity (Anti-Stealing Resolution)
    print("Applying Global Mutual Exclusivity (resolving target collisions)...")
    t_mutex = time.time()
    resolved_matches = resolve_mutual_exclusivity(initial_matches, s1_probs_map)
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
    print("=== Starting Version 2.0 High-Performance Multi-Country Inference ===")
    t_all = time.time()
    
    base_dir = resolve_base_dirs()
    print(f"Resolved Project Base Directory: {base_dir}")
    
    test_dir = os.path.join(base_dir, 'dataset', 'test')
    part_dir = os.path.join(base_dir, 'scratch', 'partitions')
    out_dir = os.path.join(base_dir, 'output')
    os.makedirs(out_dir, exist_ok=True)
    
    ensure_partitions(test_dir, part_dir)
    
    model_path = os.path.join(os.path.dirname(__file__), 'model.joblib')
    if not os.path.exists(model_path):
        model_path = os.path.join(os.path.dirname(__file__), 'model_v2.joblib')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at {model_path}!")
        
    print(f"Loading model checkpoint from {model_path}...")
    clf_data = joblib.load(model_path)
    
    out_match = os.path.join(out_dir, 'matching_results.tsv')
    out_cand = os.path.join(out_dir, 'candidate_pairs.tsv')
    
    # Initialize files with headers
    with open(out_match, 'w', encoding='utf-8') as fm, open(out_cand, 'w', encoding='utf-8') as fc:
        fm.write("source1_entity_id\tmatched_entity_ids\n")
        fc.write("source1_entity_id\tcandidate_entity_ids\n")
        
    # Process sequentially country by country to keep RAM strictly < 1.5 GB
    for country in ['France', 'US', 'India']:
        process_country(country, part_dir, clf_data, out_match, out_cand)
        
    print(f"\n=======================================================")
    print(f"ALL COUNTRIES COMPLETED IN {time.time()-t_all:.1f}s ({(time.time()-t_all)/60:.1f} minutes)!")
    print(f"Output files ready at:")
    print(f"  - {out_match}")
    print(f"  - {out_cand}")
    print(f"=======================================================")


if __name__ == '__main__':
    main()
