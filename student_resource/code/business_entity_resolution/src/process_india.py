import os
import sys
import time
import unicodedata
import re
import joblib
import numpy as np
from collections import defaultdict

try:
    from .model import MatchClassifier
except ImportError:
    from model import MatchClassifier


LEGAL_SUFFIXES = {
    'inc', 'incorporated', 'corp', 'corporation', 'llc', 'ltd', 'limited',
    'pvt', 'private', 'co', 'company', 'group', 'holdings', 'enterprise', 'enterprises',
    'pllc', 'llp', 'lp', 'associates', 'association', 'services', 'solutions',
    'sa', 'sarl', 'sas', 'sasu', 'eurl', 'sci', 'snc', 'gie', 'societe',
    'and', 'or', 'of', 'in', 'at', 'the', 'a', 'an', '&'
}

ADDR_ABBR = {
    'rd': 'road', 'st': 'street', 'ave': 'avenue', 'blvd': 'boulevard',
    'dr': 'drive', 'ln': 'lane', 'ct': 'court', 'pl': 'place', 'sq': 'square',
    'pkwy': 'parkway', 'hwy': 'highway', 'apt': 'apartment', 'ste': 'suite',
    'fl': 'floor', 'bldg': 'building', 'dept': 'department', 'no': 'number',
    'sec': 'sector', 'po': 'post'
}

def remove_accents(text: str) -> str:
    if not text: return ""
    return "".join(c for c in unicodedata.normalize('NFKD', str(text)) if not unicodedata.combining(c))

def clean_name(name: str) -> str:
    if not name: return ""
    s = remove_accents(name).lower()
    s = re.sub(r'\b(d/?b/?a|formerly|f/?k/?a|a/?k/?a)\b', ' ', s)
    s = re.sub(r'\.(com|in|org|net|fr|co|us|gov|edu)\b', '', s)
    s = re.sub(r'[^a-z0-9]+', ' ', s)
    tokens = s.split()
    filtered = [t for t in tokens if t not in LEGAL_SUFFIXES]
    return " ".join(filtered) if filtered else " ".join(tokens)

def clean_addr(addr: str) -> str:
    if not addr: return ""
    s = remove_accents(addr).lower()
    s = re.sub(r'[^a-z0-9]+', ' ', s)
    tokens = [ADDR_ABBR.get(t, t) for t in s.split()]
    return " ".join(tokens)

def char_ngrams(s: str, n: int = 3) -> list[str]:
    if not s: return []
    s = s.replace(' ', '_')
    if len(s) < n: return [s]
    return [s[i:i+n] for i in range(len(s) - n + 1)]

def extract_postal(addr: str) -> str:
    if not addr: return ""
    m = re.findall(r'\b[1-9]\d{5}\b', str(addr))
    return m[-1] if m else ""

def common_prefix_ratio(s1: str, s2: str) -> float:
    min_len = min(len(s1), len(s2))
    if min_len == 0: return 0.0
    count = 0
    for c1, c2 in zip(s1, s2):
        if c1 == c2: count += 1
        else: break
    return count / min_len

class CompactTarget:
    __slots__ = ('c_name', 'c_addr', 'digits', 'postal', 'is_s2')
    def __init__(self, c_name: str, c_addr: str, digits: str, postal: str, is_s2: float):
        self.c_name = c_name
        self.c_addr = c_addr
        self.digits = digits
        self.postal = postal
        self.is_s2 = is_s2

def get_blocking_keys(c_name: str, digits: str, postal: str):
    keys = []
    tokens = c_name.split()
    if c_name and len(c_name) >= 3:
        keys.append(('ex', c_name))
    if len(tokens) >= 2:
        keys.append(('bi', f"{tokens[0]}_{tokens[1]}"))
    for tok in tokens[:2]:
        if len(tok) >= 5:
            keys.append(('tk', tok))
    if digits and c_name:
        keys.append(('np', f"{digits}_{c_name[:2]}"))
    if postal and c_name:
        keys.append(('pp', f"{postal}_{c_name[:2]}"))
    return keys

def compute_pair_features(r1: dict, r2: CompactTarget, rank: int):
    # 1. Exact
    name_exact = 1.0 if r1['c_name'] and r1['c_name'] == r2.c_name else 0.0
    # 2. Containment
    name_containment = 1.0 if (r1['c_name'] and r2.c_name and (r1['c_name'] in r2.c_name or r2.c_name in r1['c_name'])) else 0.0
    
    # 3. Token Jaccard
    s1 = r1['tok_set']
    s2 = set(r2.c_name.split())
    name_tok_jaccard = len(s1 & s2) / len(s1 | s2) if (s1 or s2) else 0.0
    
    # 4. Char Jaccard
    c1 = r1['char_set']
    c2 = set(char_ngrams(r2.c_name))
    name_char_jaccard = len(c1 & c2) / len(c1 | c2) if (c1 or c2) else 0.0
    
    # 5. Prefix
    name_prefix = common_prefix_ratio(r1['c_name'], r2.c_name)
    
    # 6. Length diff
    l1, l2 = len(r1['c_name']), len(r2.c_name)
    max_l = max(l1, l2)
    name_len_diff = abs(l1 - l2) / max_l if max_l > 0 else 0.0
    
    # 7. Digits
    num_match = 1.0 if (r1['digits'] and r2.digits and r1['digits'] == r2.digits) else 0.5
    
    # 8. Postal
    postal_match = 1.0 if (r1['postal'] and r2.postal and r1['postal'] == r2.postal) else 0.5
    
    # 9. Addr token Jaccard
    a1 = r1['addr_tok_set']
    a2 = set(r2.c_addr.split())
    addr_tok_jaccard = len(a1 & a2) / len(a1 | a2) if (a1 or a2) else 0.0
    
    # 10. Addr char Jaccard
    ac1 = r1['addr_char_set']
    ac2 = set(char_ngrams(r2.c_addr))
    addr_char_jaccard = len(ac1 & ac2) / len(ac1 | ac2) if (ac1 or ac2) else 0.0
    
    # 11. Sources
    is_s2 = r2.is_s2
    is_s3 = 1.0 - is_s2
    rank_feat = float(rank)
    name_addr_interaction = name_char_jaccard * addr_char_jaccard

    return [
        name_exact, name_containment, name_tok_jaccard, name_char_jaccard,
        name_prefix, name_len_diff, num_match, postal_match,
        addr_tok_jaccard, addr_char_jaccard, is_s2, is_s3, rank_feat, name_addr_interaction
    ]

def run_india():
    print("=== Starting Memory-Optimized India Partition Inference ===")
    t_start = time.time()
    
    clf = MatchClassifier()
    clf.load('student_resource/code/business_entity_resolution/src/model.joblib')
    tau = clf.best_threshold
    print(f"Loaded classifier with optimal F0.5 threshold: {tau:.3f}")
    
    s1_path = 'scratch/partitions/s1_India.tsv'
    targets_path = 'scratch/partitions/targets_India.tsv'
    out_match = 'student_resource/output/matching_results.tsv'
    out_cand = 'student_resource/output/candidate_pairs.tsv'
    
    # 1. Load S1 records
    print("Loading India Source 1 records...")
    s1_records = []
    needed_keys = set()
    s1_keys_map = {}
    
    with open(s1_path, encoding='utf-8') as f:
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if not parts[0]: continue
            eid = parts[0]
            raw_n = parts[1] if len(parts) > 1 else ''
            raw_a = parts[2] if len(parts) > 2 else ''
            
            c_name_val = clean_name(raw_n)
            c_addr_val = clean_addr(raw_a)
            digits = re.findall(r'\b\d+\b', raw_a)
            d_val = digits[0] if digits else ""
            p_val = extract_postal(raw_a)
            
            rec = {
                'eid': eid,
                'c_name': c_name_val,
                'digits': d_val,
                'postal': p_val,
                'tok_set': set(c_name_val.split()),
                'char_set': set(char_ngrams(c_name_val)),
                'addr_tok_set': set(c_addr_val.split()),
                'addr_char_set': set(char_ngrams(c_addr_val)),
            }
            s1_records.append(rec)
            keys = get_blocking_keys(c_name_val, d_val, p_val)
            s1_keys_map[eid] = keys
            for k in keys:
                needed_keys.add(k)
                
    print(f"Loaded {len(s1_records):,} India S1 records. Unique query keys: {len(needed_keys):,}")
    
    # 2. Stream targets with ultra-compact storage
    print("Streaming targets into compact index...")
    t_idx = time.time()
    index = defaultdict(list)
    target_records = {}
    
    with open(targets_path, encoding='utf-8') as f:
        for line in f:
            parts = line.rstrip('\n').split('\t')
            eid = parts[0]
            raw_n = parts[1] if len(parts) > 1 else ''
            raw_a = parts[2] if len(parts) > 2 else ''
            
            c_n = clean_name(raw_n)
            digits = re.findall(r'\b\d+\b', raw_a)
            d_val = digits[0] if digits else ""
            p_val = extract_postal(raw_a)
            
            keys = get_blocking_keys(c_n, d_val, p_val)
            matched = [k for k in keys if k in needed_keys]
            if matched:
                c_a = clean_addr(raw_a)
                is_s2 = 1.0 if eid.startswith('S2-') else 0.0
                target_records[eid] = CompactTarget(c_n, c_a, d_val, p_val, is_s2)
                for k in matched:
                    index[k].append(eid)
                    
    # Prune heavy buckets (> 250)
    pruned = [k for k, v in index.items() if len(v) > 250]
    for k in pruned:
        del index[k]
        
    print(f"Indexed {len(target_records):,} compact target records in {time.time()-t_idx:.1f}s.")
    
    # 3. Score candidate pairs in batches
    print("Scoring candidate pairs in batches...")
    weights = {'ex': 5, 'bi': 4, 'np': 3, 'pp': 2, 'tk': 1}
    BATCH_SIZE = 50000
    
    f_match = open(out_match, 'a', encoding='utf-8')
    f_cand = open(out_cand, 'a', encoding='utf-8')
    
    batch_features = []
    batch_meta = []
    s1_results = defaultdict(list)
    s1_cands_map = {}
    
    total_pairs = 0
    total_matches = 0
    
    def flush():
        nonlocal batch_features, batch_meta, total_pairs, total_matches
        if not batch_features: return
        X = np.array(batch_features, dtype=np.float32)
        probs = clf.predict_proba(X)
        for (sid, cid), prob in zip(batch_meta, probs):
            if prob >= tau:
                s1_results[sid].append(cid)
                total_matches += 1
        total_pairs += len(batch_features)
        batch_features = []
        batch_meta = []

    for idx, r1 in enumerate(s1_records):
        sid = r1['eid']
        keys = s1_keys_map.get(sid, [])
        score_map = defaultdict(int)
        for k in keys:
            w = weights.get(k[0], 1)
            for cid in index.get(k, []):
                score_map[cid] += w
                
        if not score_map:
            s1_cands_map[sid] = []
            continue
            
        cands = [cid for cid, _ in sorted(score_map.items(), key=lambda x: x[1], reverse=True)[:15]]
        s1_cands_map[sid] = cands
        
        for rank, cid in enumerate(cands):
            r2 = target_records.get(cid)
            if r2:
                feat = compute_pair_features(r1, r2, rank)
                batch_features.append(feat)
                batch_meta.append((sid, cid))
                
        if len(batch_features) >= BATCH_SIZE:
            flush()
            if (idx + 1) % 100000 == 0:
                print(f"  Processed {idx+1:,} / {len(s1_records):,} India S1... ({total_pairs:,} pairs scored)")
                
    flush()
    
    # 4. Stream to output files
    print("Appending India results to output files...")
    for r1 in s1_records:
        sid = r1['eid']
        m_list = s1_results.get(sid, [])
        c_list = s1_cands_map.get(sid, [])
        f_match.write(f"{sid}\t{','.join(m_list)}\n")
        f_cand.write(f"{sid}\t{','.join(c_list)}\n")
        
    f_match.close()
    f_cand.close()
    
    print(f"\n=======================================================")
    print(f"India completed in {time.time()-t_start:.1f}s!")
    print(f"  Total India S1 entities: {len(s1_records):,}")
    print(f"  Pairs scored: {total_pairs:,}")
    print(f"  Total matches predicted: {total_matches:,}")
    print(f"=======================================================")

if __name__ == '__main__':
    run_india()
