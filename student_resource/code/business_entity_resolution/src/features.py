import difflib
import re
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

def fast_jaccard(tokens1: list[str], tokens2: list[str]) -> float:
    s1, s2 = set(tokens1), set(tokens2)
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    return len(s1 & s2) / len(s1 | s2)

def char_ngrams(s: str, n: int = 3) -> list[str]:
    if not s:
        return []
    s = s.replace(' ', '_')
    if len(s) < n:
        return [s]
    return [s[i:i+n] for i in range(len(s) - n + 1)]

def common_prefix_ratio(s1: str, s2: str) -> float:
    min_len = min(len(s1), len(s2))
    if min_len == 0:
        return 0.0
    count = 0
    for c1, c2 in zip(s1, s2):
        if c1 == c2:
            count += 1
        else:
            break
    return count / min_len

def compute_pair_features(
    s1_rec: dict, 
    cand_rec: dict, 
    rank: int = 0
) -> list[float]:
    """Compute dense similarity feature vector between an S1 entity and a candidate entity."""
    c_name1 = s1_rec.get('clean_name', '')
    c_name2 = cand_rec.get('clean_name', '')
    
    # 1. Name Exact Match
    name_exact = 1.0 if c_name1 and c_name1 == c_name2 else 0.0
    
    # 2. Name Containment
    name_containment = 1.0 if (c_name1 and c_name2 and (c_name1 in c_name2 or c_name2 in c_name1)) else 0.0
    
    # 3. Name Word Jaccard
    tokens1 = s1_rec.get('name_tokens', [])
    tokens2 = cand_rec.get('name_tokens', [])
    name_tok_jaccard = fast_jaccard(tokens1, tokens2)
    
    # 4. Name Char 3-gram Jaccard
    name_char_jaccard = fast_jaccard(char_ngrams(c_name1), char_ngrams(c_name2))
    
    # 5. Name Prefix Ratio
    name_prefix = common_prefix_ratio(c_name1, c_name2)
    
    # 6. Name Length Difference Ratio
    l1, l2 = len(c_name1), len(c_name2)
    max_l = max(l1, l2)
    name_len_diff = abs(l1 - l2) / max_l if max_l > 0 else 0.0
    
    # 7. Address Numbers Match (House numbers, unit numbers)
    d1 = set(s1_rec.get('addr_digits', []))
    d2 = set(cand_rec.get('addr_digits', []))
    if d1 and d2:
        num_match = 1.0 if len(d1 & d2) > 0 else 0.0
        house_num_mismatch_veto = 1.0 if len(d1 & d2) == 0 else 0.0
    else:
        num_match = 0.5
        house_num_mismatch_veto = 0.0
        
    # 8. Postal Code Match & Veto
    p1 = s1_rec.get('postal_code', '')
    p2 = cand_rec.get('postal_code', '')
    if p1 and p2:
        postal_match = 1.0 if p1 == p2 else 0.0
        postal_mismatch_veto = 1.0 if p1 != p2 else 0.0
    else:
        postal_match = 0.5
        postal_mismatch_veto = 0.0
        
    # 9. Address Word Jaccard
    c_addr1 = s1_rec.get('clean_addr', '')
    c_addr2 = cand_rec.get('clean_addr', '')
    addr_tok1 = c_addr1.split()
    addr_tok2 = c_addr2.split()
    addr_tok_jaccard = fast_jaccard(addr_tok1, addr_tok2)
    
    # 10. Address Char 3-gram Jaccard
    addr_char_jaccard = fast_jaccard(char_ngrams(c_addr1), char_ngrams(c_addr2))
    
    # 11. Pure Address High-Precision Match (house number match + high address Jaccard)
    pure_address_match = 1.0 if (num_match == 1.0 and addr_tok_jaccard >= 0.65) else 0.0
    
    # 12. Non-ASCII script flag (indicates native Indian script where name lexical overlap is 0)
    has_non_ascii = 1.0 if any(ord(c) > 127 for c in c_name2) else 0.0
    
    # 13. Domain / Compound Substring Match (e.g. herman inside hermantotal.com)
    c2_merged = "".join(tokens2)
    domain_stem_match = 0.0
    for t in tokens1:
        if len(t) >= 4 and t in c2_merged:
            domain_stem_match = 1.0
            break
            
    # 14. Complex Plot Code Match (e.g. wz-187c, af-684, 6-2-101)
    codes1 = set(s1_rec.get('complex_codes', []))
    codes2 = set(cand_rec.get('complex_codes', []))
    code_match = 1.0 if (codes1 and codes2 and len(codes1 & codes2) > 0) else 0.0
    
    # 15. Source Origin Indicators
    cand_id = cand_rec.get('entity_id', '')
    is_s2 = 1.0 if cand_id.startswith('S2-') else 0.0
    is_s3 = 1.0 if cand_id.startswith('S3-') else 0.0
    
    # 16. Retrieval Rank
    rank_feat = float(rank)
    
    # 17. Interaction Signal
    name_addr_interaction = name_char_jaccard * addr_char_jaccard

    return [
        name_exact,
        name_containment,
        name_tok_jaccard,
        name_char_jaccard,
        name_prefix,
        name_len_diff,
        num_match,
        postal_match,
        addr_tok_jaccard,
        addr_char_jaccard,
        pure_address_match,
        has_non_ascii,
        domain_stem_match,
        code_match,
        is_s2,
        is_s3,
        rank_feat,
        name_addr_interaction,
        postal_mismatch_veto,
        house_num_mismatch_veto
    ]

FEATURE_NAMES = [
    'name_exact',
    'name_containment',
    'name_tok_jaccard',
    'name_char_jaccard',
    'name_prefix',
    'name_len_diff',
    'num_match',
    'postal_match',
    'addr_tok_jaccard',
    'addr_char_jaccard',
    'pure_address_match',
    'has_non_ascii',
    'domain_stem_match',
    'code_match',
    'is_s2',
    'is_s3',
    'rank_feat',
    'name_addr_interaction',
    'postal_mismatch_veto',
    'house_num_mismatch_veto'
]
