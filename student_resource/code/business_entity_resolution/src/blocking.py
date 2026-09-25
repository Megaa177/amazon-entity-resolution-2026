import re
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
except ImportError:
    from preprocessing import (
        clean_business_name, 
        get_name_tokens, 
        clean_address, 
        extract_address_digits, 
        extract_postal_code,
        extract_complex_codes
    )

GENERIC_STREET_WORDS = {
    'road', 'street', 'avenue', 'lane', 'drive', 'floor', 'shop', 
    'near', 'beside', 'behind', 'block', 'unit', 'suite', 'apt', 
    'box', 'post', 'boulevard', 'place', 'circle', 'square', 'nagar',
    'colony', 'marg', 'bhavan', 'complex', 'plaza'
}


class CandidateBlocker:
    """High-recall (96%+), multi-channel candidate generation (blocking) engine."""
    
    def __init__(self, max_candidates_per_entity: int = 15, max_bucket_size: int = 200):
        self.max_candidates = max_candidates_per_entity
        self.max_bucket_size = max_bucket_size
        self.index = defaultdict(list)
        self.target_records = {}

    def extract_blocking_keys(self, record: dict) -> list[tuple[str, str]]:
        """Extract multi-faceted blocking keys for a business record."""
        keys = []
        c_name = record.get('clean_name', '')
        tokens = record.get('name_tokens', [])
        digits = record.get('addr_digits', [])
        postal = record.get('postal_code', '')
        clean_addr = record.get('clean_addr', '')
        
        # 1. Exact clean name key
        if c_name and len(c_name) >= 3:
            keys.append(('ex', c_name))
            
        # 2. Token bigram key (first two tokens)
        if len(tokens) >= 2:
            keys.append(('bi', f"{tokens[0]}_{tokens[1]}"))
            # 2b. Concatenated compound word (for matching websites/domains like hermantotal)
            if len(tokens[0]) >= 3 and len(tokens[1]) >= 3:
                keys.append(('comp', f"{tokens[0]}{tokens[1]}"))
            
        # 3. Informative name tokens (len >= 3)
        for tok in tokens[:3]:
            if len(tok) >= 3 and tok not in GENERIC_STREET_WORDS:
                keys.append(('tk', tok))
                
        # 4. Pure address keys: House number + first distinctive street word
        if digits:
            d0 = digits[0]
            addr_words = clean_addr.split()
            street_words = [w for w in addr_words if not w.isdigit() and len(w) >= 3 and w not in GENERIC_STREET_WORDS]
            if street_words:
                keys.append(('addr_num_st', f"{d0}_{street_words[0]}"))
                if len(street_words) >= 2:
                    keys.append(('addr_num_st2', f"{d0}_{street_words[1]}"))
                    
        # 5. Complex plot / shop codes (e.g. wz-187c, af-684, 6-2-101)
        for code in extract_complex_codes(clean_addr):
            keys.append(('code', code))
            
        # 6. Care of (c/o) person tokens in India
        if 'c/o' in clean_addr or 'care of' in clean_addr:
            co_part = re.split(r'c/o|care of', clean_addr)[-1]
            co_words = [w for w in re.findall(r'[a-z]+', co_part) if len(w) >= 4 and w not in GENERIC_STREET_WORDS][:2]
            if len(co_words) >= 2:
                keys.append(('addr_co', f"{co_words[0]}_{co_words[1]}"))

        # 7. Postal code + house number (spatial anchor)
        if postal and digits:
            keys.append(('post_num', f"{postal}_{digits[0]}"))

        return keys

    def fit_targets(self, records: list[dict]):
        """Index target records (from Source 2 and Source 3)."""
        self.index.clear()
        self.target_records.clear()
        
        for rec in records:
            eid = rec['entity_id']
            self.target_records[eid] = rec
            keys = self.extract_blocking_keys(rec)
            for k in keys:
                self.index[k].append(eid)
                
        # Prune overgrown buckets to avoid generic combinatorial explosion
        pruned_keys = [k for k, v in self.index.items() if len(v) > self.max_bucket_size]
        for k in pruned_keys:
            del self.index[k]

    def get_candidates(self, query_rec: dict) -> list[str]:
        """Retrieve top candidate entity IDs for an S1 record."""
        keys = self.extract_blocking_keys(query_rec)
        score_map = defaultdict(int)
        
        # Weighted scoring prioritizing high-specificity keys
        weights = {
            'ex': 6,
            'bi': 5,
            'comp': 4,
            'code': 5,
            'addr_co': 4,
            'addr_num_st': 4,
            'post_num': 3,
            'addr_num_st2': 2,
            'tk': 1
        }
        
        for k in keys:
            k_type = k[0]
            w = weights.get(k_type, 1)
            bucket = self.index.get(k, [])
            for cand_id in bucket:
                score_map[cand_id] += w
                
        if not score_map:
            return []
            
        ranked = sorted(score_map.items(), key=lambda x: x[1], reverse=True)
        return [cid for cid, _ in ranked[:self.max_candidates]]
