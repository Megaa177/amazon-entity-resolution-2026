"""
Version 2.3 Feature Extraction Engine
Includes:
- 25 dense pairwise features
- Acronym & Initialism matching
- Longest Common Prefix (LCP) truncation detection
- Anti-Co-Location penalty
- Entity distinctiveness scoring
"""

import re
try:
    from .preprocessing_v23 import (
        clean_business_name_v23, 
        clean_address_v23, 
        extract_address_digits_v23, 
        extract_postal_code_v23,
        extract_complex_codes_v23,
        extract_acronym,
        get_lcp_ratio,
        compute_distinctiveness
    )
except ImportError:
    from preprocessing_v23 import (
        clean_business_name_v23, 
        clean_address_v23, 
        extract_address_digits_v23, 
        extract_postal_code_v23,
        extract_complex_codes_v23,
        extract_acronym,
        get_lcp_ratio,
        compute_distinctiveness
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


def char_dice_2gram(s1: str, s2: str) -> float:
    if not s1 or not s2:
        return 0.0
    g1 = set(char_ngrams(s1, n=2))
    g2 = set(char_ngrams(s2, n=2))
    if not g1 and not g2:
        return 1.0
    if not g1 or not g2:
        return 0.0
    return (2.0 * len(g1 & g2)) / (len(g1) + len(g2))


def token_sort_jaccard(tokens1: list[str], tokens2: list[str]) -> float:
    """Alphabetically sorted token Jaccard (order-invariant)."""
    if not tokens1 and not tokens2:
        return 1.0
    if not tokens1 or not tokens2:
        return 0.0
    s1, s2 = sorted(tokens1), sorted(tokens2)
    return fast_jaccard(s1, s2)


def compute_pair_features_v23(
    s1_rec: dict, 
    cand_rec: dict, 
    rank: int = 0
) -> list[float]:
    """Compute dense 25-feature vector with acronyms, truncations, and co-location defense."""
    c_name1 = s1_rec.get('clean_name', '')
    c_name2 = cand_rec.get('clean_name', '')
    tokens1 = s1_rec.get('name_tokens', [])
    tokens2 = cand_rec.get('name_tokens', [])
    set_tok1 = set(tokens1)
    set_tok2 = set(tokens2)
    
    # 1. Name Exact Match
    name_exact = 1.0 if c_name1 and c_name1 == c_name2 else 0.0
    
    # 2. Token-Sort Jaccard
    name_sort_jaccard = token_sort_jaccard(tokens1, tokens2)
    
    # 3. Standard Token Jaccard
    name_tok_jaccard = fast_jaccard(tokens1, tokens2)
    
    # 4. Name Char 2-gram Dice
    name_dice_2g = char_dice_2gram(c_name1, c_name2)
    
    # 5. Name Char 3-gram Jaccard
    name_char_jaccard = fast_jaccard(char_ngrams(c_name1, n=3), char_ngrams(c_name2, n=3))
    
    # 6. Token Overlap Ratio
    token_overlap_ratio = (len(set_tok1 & set_tok2) / len(set_tok1)) if set_tok1 else 0.0
    
    # 7. Name Length Difference Ratio
    l1, l2 = len(c_name1), len(c_name2)
    max_l = max(l1, l2)
    name_len_diff = abs(l1 - l2) / max_l if max_l > 0 else 0.0

    # 8. Address Numbers Match (House numbers)
    d1 = set(s1_rec.get('addr_digits', []))
    d2 = set(cand_rec.get('addr_digits', []))
    if d1 and d2:
        num_match = 1.0 if len(d1 & d2) > 0 else 0.0
        house_num_mismatch_veto = 1.0 if len(d1 & d2) == 0 else 0.0
    else:
        num_match = 0.5
        house_num_mismatch_veto = 0.0

    # 9. Postal Code Match & Veto
    p1 = s1_rec.get('postal_code', '')
    p2 = cand_rec.get('postal_code', '')
    if p1 and p2:
        postal_match = 1.0 if p1 == p2 else 0.0
        postal_mismatch_veto = 1.0 if p1 != p2 else 0.0
    else:
        postal_match = 0.5
        postal_mismatch_veto = 0.0

    # 10. Address Word Jaccard
    c_addr1 = s1_rec.get('clean_addr', '')
    c_addr2 = cand_rec.get('clean_addr', '')
    addr_tok1 = c_addr1.split()
    addr_tok2 = c_addr2.split()
    addr_tok_jaccard = fast_jaccard(addr_tok1, addr_tok2)

    # 11. Address Char 3-gram Jaccard
    addr_char_jaccard = fast_jaccard(char_ngrams(c_addr1, n=3), char_ngrams(c_addr2, n=3))

    # 12. Pure Street Words Jaccard
    st1 = [w for w in addr_tok1 if not w.isdigit()]
    st2 = [w for w in addr_tok2 if not w.isdigit()]
    street_words_jaccard = fast_jaccard(st1, st2)

    # 13. ANTI-CO-LOCATION PENALTY
    co_location_conflict = 1.0 if (num_match == 1.0 and name_sort_jaccard < 0.45 and name_exact == 0.0) else 0.0

    # 14. First Token Match
    first_tok_match = 1.0 if (tokens1 and tokens2 and tokens1[0] == tokens2[0]) else 0.0

    # 15. Contact Phone Match
    ph1 = s1_rec.get('phone', '')
    ph2 = cand_rec.get('phone', '')
    phone_match = 1.0 if (ph1 and ph2 and ph1 == ph2) else (0.5 if not ph1 or not ph2 else 0.0)

    # 16. Complex Plot Code Match
    codes1 = set(s1_rec.get('complex_codes', []))
    codes2 = set(cand_rec.get('complex_codes', []))
    code_match = 1.0 if (codes1 and codes2 and len(codes1 & codes2) > 0) else 0.0

    # 17. Source Indicators
    cand_id = cand_rec.get('entity_id', '')
    is_s2 = 1.0 if cand_id.startswith('S2-') else 0.0
    is_s3 = 1.0 if cand_id.startswith('S3-') else 0.0

    # 18. Retrieval Rank
    rank_feat = float(rank)

    # 19. Joint Name-Address Interaction
    name_addr_interaction = name_dice_2g * addr_char_jaccard

    # 20. Acronym Match (NEW in v2.3)
    acr1 = s1_rec.get('acronym', '')
    acr2 = cand_rec.get('acronym', '')
    acronym_match = 0.0
    if acr1 and (acr1 == c_name2 or acr1 == acr2):
        acronym_match = 1.0
    elif acr2 and (acr2 == c_name1 or acr2 == acr1):
        acronym_match = 1.0

    # 21. Longest Common Prefix (LCP) Ratio for Truncation Detection (NEW in v2.3)
    lcp_ratio = get_lcp_ratio(c_name1, c_name2)

    # 22. Distinctiveness Score of Entity (NEW in v2.3)
    distinctiveness_score = s1_rec.get('distinctiveness', 0.5)

    return [
        name_exact,
        name_sort_jaccard,
        name_tok_jaccard,
        name_dice_2g,
        name_char_jaccard,
        token_overlap_ratio,
        name_len_diff,
        num_match,
        postal_match,
        addr_tok_jaccard,
        addr_char_jaccard,
        street_words_jaccard,
        co_location_conflict,
        house_num_mismatch_veto,
        postal_mismatch_veto,
        first_tok_match,
        phone_match,
        code_match,
        is_s2,
        is_s3,
        rank_feat,
        name_addr_interaction,
        acronym_match,
        lcp_ratio,
        distinctiveness_score
    ]

FEATURE_NAMES_V23 = [
    'name_exact',
    'name_sort_jaccard',
    'name_tok_jaccard',
    'name_dice_2g',
    'name_char_jaccard',
    'token_overlap_ratio',
    'name_len_diff',
    'num_match',
    'postal_match',
    'addr_tok_jaccard',
    'addr_char_jaccard',
    'street_words_jaccard',
    'co_location_conflict',
    'house_num_mismatch_veto',
    'postal_mismatch_veto',
    'first_tok_match',
    'phone_match',
    'code_match',
    'is_s2',
    'is_s3',
    'rank_feat',
    'name_addr_interaction',
    'acronym_match',
    'lcp_ratio',
    'distinctiveness_score'
]
