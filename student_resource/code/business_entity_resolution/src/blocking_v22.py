import re
import math
from collections import defaultdict, Counter
try:
    from .preprocessing_v22 import (
        clean_business_name_v22, 
        get_name_tokens_v22, 
        clean_address_v22, 
        extract_address_digits_v22, 
        extract_postal_code_v22,
        extract_complex_codes_v22,
        UNIVERSAL_STREET_WORDS,
        UNIVERSAL_LEGAL_SUFFIXES
    )
except ImportError:
    from preprocessing_v22 import (
        clean_business_name_v22, 
        get_name_tokens_v22, 
        clean_address_v22, 
        extract_address_digits_v22, 
        extract_postal_code_v22,
        extract_complex_codes_v22,
        UNIVERSAL_STREET_WORDS,
        UNIVERSAL_LEGAL_SUFFIXES
    )

BASE_KEY_WEIGHTS = {
    'ex': 8.0,
    'bi_sort': 6.5,
    'code': 6.0,
    'phone': 8.0,
    'addr_st': 4.5,
    'post_num': 3.5,
    'tk': 1.5,
}


def discover_country_stopwords(records: list[dict], min_freq: float = 0.02) -> tuple[set, set]:
    """Dynamically discover frequent street words and corporate suffixes for ANY country."""
    addr_token_counts = Counter()
    name_suffix_counts = Counter()
    n = len(records)
    if n == 0:
        return set(), set()
        
    for r in records[:50000]: # Sample 50k records
        c_addr = r.get('clean_addr', '')
        tokens = set(re.findall(r'[a-z]{3,}', c_addr.lower()))
        for t in tokens:
            addr_token_counts[t] += 1
                
        tokens = r.get('name_tokens', [])
        if tokens:
            name_suffix_counts[tokens[-1]] += 1
            
    sample_len = min(n, 50000)
    street_stopwords = {w for w, cnt in addr_token_counts.items() if cnt / sample_len >= min_freq}
    legal_suffixes = {w for w, cnt in name_suffix_counts.items() if cnt / sample_len >= min_freq}
    return street_stopwords, legal_suffixes


class CandidateBlockerV22:
    """Version 2.2 Universal Candidate Blocker.
    
    Key Features:
    1. Order-invariant bigram keys (bi_sort) so 'team ecole' and 'ecole team' produce identical blocking key.
    2. Dynamic stopword suppression for arbitrary unseen countries.
    3. French house number modifier handling (bis, ter ignored in street words).
    4. Precomputed logarithmic IDF frequency damping across buckets up to 350 items.
    """

    def __init__(self, max_candidates_per_entity: int = 18, max_bucket_size: int = 350):
        self.max_candidates = max_candidates_per_entity
        self.max_bucket_size = max_bucket_size
        self.index = defaultdict(list)
        self.key_weights = {}
        self.target_records = {}
        self.extra_stopwords = set()

    def extract_blocking_keys(self, record: dict) -> list[tuple[str, str]]:
        keys = []
        c_name = record.get('clean_name', '')
        tokens = record.get('name_tokens', [])
        digits = record.get('addr_digits', [])
        postal = record.get('postal_code', '')
        c_addr = record.get('clean_addr', '')
        
        # 1. Exact clean name key
        if c_name and len(c_name) >= 3:
            keys.append(('ex', c_name))
            
        # 2. Order-Invariant Token Bigram Key (bi_sort)
        if len(tokens) >= 2:
            t1, t2 = tokens[0], tokens[1]
            s_pair = f"{min(t1, t2)}_{max(t1, t2)}"
            keys.append(('bi_sort', s_pair))
            
        # 3. Informative distinctive name tokens (len >= 3)
        for tok in tokens[:3]:
            if len(tok) >= 3 and tok not in UNIVERSAL_STREET_WORDS and tok not in self.extra_stopwords:
                keys.append(('tk', tok))
                
        # 4. Pure address keys: House number + clean alphabetic street words
        if digits:
            d0 = digits[0]
            st_words = [
                w for w in c_addr.split() 
                if not w.isdigit() and len(w) >= 3 and w not in UNIVERSAL_STREET_WORDS and w not in self.extra_stopwords
            ]
            for st in st_words[:2]:
                keys.append(('addr_st', f"{d0}_{st}"))
                    
        # 5. Complex plot / shop codes
        for code in record.get('complex_codes', []):
            keys.append(('code', code))

        # 6. Postal code + house number
        if postal and digits:
            keys.append(('post_num', f"{postal}_{digits[0]}"))

        # 7. Contact phone numbers (10 digits)
        raw_full = f"{record.get('raw_name', '')} {c_addr}"
        phones = re.findall(r'\b[6-9]\d{9}\b|\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b', raw_full)
        for p in phones[:1]:
            d_p = re.sub(r'\D', '', p)
            if len(d_p) >= 10:
                keys.append(('phone', d_p[-10:]))

        return keys

    def precompute_weights(self):
        """Precompute logarithmic IDF weights for all indexed keys."""
        self.key_weights.clear()
        log_max = math.log(self.max_bucket_size + 1)
        for k, bucket in self.index.items():
            base_w = BASE_KEY_WEIGHTS.get(k[0], 1.0)
            b_len = len(bucket)
            idf = max(0.20, 1.0 - (math.log(b_len) / log_max))
            self.key_weights[k] = base_w * idf

    def fit_targets(self, records: list[dict], auto_stopwords: bool = True):
        """Index target records (from Source 2 and Source 3)."""
        self.index.clear()
        self.target_records.clear()
        
        if auto_stopwords and records:
            extra_sw, _ = discover_country_stopwords(records, min_freq=0.03)
            self.extra_stopwords = extra_sw
        
        for rec in records:
            eid = rec['entity_id']
            self.target_records[eid] = rec
            keys = self.extract_blocking_keys(rec)
            for k in keys:
                self.index[k].append(eid)
                
        # Drop overgrown buckets
        pruned_keys = [k for k, v in self.index.items() if len(v) > self.max_bucket_size]
        for k in pruned_keys:
            del self.index[k]

        self.precompute_weights()

    def get_candidates(self, query_rec: dict) -> list[str]:
        """Retrieve top candidate entity IDs for an S1 record."""
        keys = self.extract_blocking_keys(query_rec)
        score_map = defaultdict(float)
        
        for k in keys:
            bucket = self.index.get(k)
            if not bucket:
                continue
            w = self.key_weights.get(k, 1.0)
            for cand_id in bucket:
                score_map[cand_id] += w
                
        if not score_map:
            return []
            
        ranked = sorted(score_map.items(), key=lambda x: x[1], reverse=True)
        return [cid for cid, _ in ranked[:self.max_candidates]]
