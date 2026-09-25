"""
Version 2.3 Candidate Generation (Blocking) Engine
Includes:
- Dynamic country-adaptive stopword discovery
- Order-invariant bigram sort keys (bi_sort)
- Acronym & Initialism matching channel (acronym)
- Truncation prefix channel (trunc)
- Logarithmic IDF damping
"""

import math
import re
from collections import defaultdict, Counter
from preprocessing_v23 import (
    clean_business_name_v23,
    clean_address_v23,
    extract_address_digits_v23,
    extract_postal_code_v23,
    extract_complex_codes_v23,
    extract_acronym
)

MAX_BUCKET_SIZE = 350
CANDIDATE_POOL_LIMIT = 18

BASE_CHANNEL_WEIGHTS = {
    'ex': 3.5,
    'acronym': 3.2,
    'bi_sort': 3.0,
    'trunc': 2.8,
    'comp': 2.5,
    'phone': 3.2,
    'code': 3.0,
    'addr_st': 2.4,
    'post_num': 2.0,
    'tk': 1.2
}


def discover_country_stopwords(sample_records: list[dict], min_freq: float = 0.01) -> set:
    """Sample records from a country dataset to dynamically discover high-frequency tokens."""
    token_counts = Counter()
    total = len(sample_records)
    if total == 0:
        return set()
    
    for r in sample_records:
        addr = str(r.get('address', '')).lower()
        words = re.findall(r'[a-z]{3,}', addr)
        token_counts.update(set(words))
    
    threshold = int(total * min_freq)
    return {w for w, c in token_counts.items() if c >= threshold}


def generate_blocking_keys_v23(record: dict, country: str = "US", extra_stopwords: set = None) -> list[tuple[str, str]]:
    """Generate multi-channel blocking keys for a business entity."""
    keys = []
    name = record.get('name', '')
    address = record.get('address', '')
    phone = record.get('phone', '')

    clean_n = clean_business_name_v23(name)
    tokens = clean_n.split()

    # 1. Exact clean name (len >= 3)
    if len(clean_n) >= 3:
        keys.append(('ex', clean_n))

    # 2. Acronym channel
    # (a) If name has >= 2 words, generate acronym
    acr = extract_acronym(name)
    if acr and len(acr) >= 2:
        keys.append(('acronym', acr))
    # (b) If name itself is short (2-5 characters), treat it as an acronym lookup key
    if 2 <= len(clean_n) <= 5 and clean_n.isalnum():
        keys.append(('acronym', clean_n))

    # 3. Order-invariant sorted bigram
    if len(tokens) >= 2:
        sorted_bi = "_".join(sorted(tokens[:2]))
        keys.append(('bi_sort', sorted_bi))
        keys.append(('comp', "".join(sorted(tokens[:2]))))

    # 4. Truncation prefix (first 12 characters if string is long)
    if len(clean_n) >= 14:
        keys.append(('trunc', clean_n[:12]))

    # 5. Distinctive name tokens
    for t in tokens:
        if len(t) >= 4 and not t.isdigit():
            keys.append(('tk', t))

    # 6. Address house number + clean street words
    digits = extract_address_digits_v23(address)
    postal = extract_postal_code_v23(address, country)
    clean_addr = clean_address_v23(address, extra_stopwords=extra_stopwords)
    street_words = [w for w in clean_addr.split() if not w.isdigit() and len(w) >= 3]

    if digits and street_words:
        # Pair first digit with first 2 street words
        st_key = f"{digits[0]}_{'_'.join(street_words[:2])}"
        keys.append(('addr_st', st_key))

    # 7. Postal code + building number
    if postal and digits:
        keys.append(('post_num', f"{postal}_{digits[0]}"))

    # 8. Complex plot/shop codes
    codes = extract_complex_codes_v23(address)
    for c in codes:
        keys.append(('code', c))

    # 9. Phone number
    if phone:
        digits_phone = re.sub(r'\D', '', str(phone))
        if len(digits_phone) >= 10:
            keys.append(('phone', digits_phone[-10:]))

    return keys


class BlockerV23:
    """Inverted index blocking engine with IDF damping & acronym support."""
    def __init__(self, country: str = "US", extra_stopwords: set = None):
        self.country = country
        self.extra_stopwords = extra_stopwords or set()
        self.index = defaultdict(list)
        self.bucket_sizes = {}

    def index_target(self, target_id: str, record: dict):
        keys = generate_blocking_keys_v23(record, self.country, self.extra_stopwords)
        for channel, k_val in keys:
            composite_key = f"{channel}:{k_val}"
            self.index[composite_key].append(target_id)

    def finalize_index(self):
        """Prune oversized buckets and compute IDF damping weights."""
        pruned_index = {}
        for k, b in self.index.items():
            sz = len(b)
            if sz <= MAX_BUCKET_SIZE:
                pruned_index[k] = b
                self.bucket_sizes[k] = sz
        self.index = pruned_index

    def retrieve_candidates(self, s1_record: dict) -> list[str]:
        keys = generate_blocking_keys_v23(s1_record, self.country, self.extra_stopwords)
        scores = defaultdict(float)

        for channel, k_val in keys:
            composite_key = f"{channel}:{k_val}"
            bucket = self.index.get(composite_key)
            if not bucket:
                continue

            sz = self.bucket_sizes.get(composite_key, len(bucket))
            # Logarithmic IDF damping
            idf_factor = max(0.20, 1.0 - (math.log(sz) / math.log(MAX_BUCKET_SIZE + 1)))
            weight = BASE_CHANNEL_WEIGHTS.get(channel, 1.0) * idf_factor

            for target_id in bucket:
                scores[target_id] += weight

        if not scores:
            return []

        sorted_cand = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [cid for cid, _ in sorted_cand[:CANDIDATE_POOL_LIMIT]]
