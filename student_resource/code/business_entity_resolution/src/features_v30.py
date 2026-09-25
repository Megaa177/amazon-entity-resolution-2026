"""
Version 3.0 Ultra-Dense Feature Engineering Engine
Includes:
- 35 dense pairwise features
- Jaro-Winkler string similarity (name and address)
- Normalized Levenshtein ratio
- Double Metaphone / Soundex phonetic concordance
- Acronym & Initialism matching
- Longest Common Prefix (LCP) truncation detection
- Anti-Co-Location penalty
- Entity distinctiveness scoring
- Address digit transposition detection
"""

import re
try:
    from .preprocessing_v30 import (
        clean_business_name_v30, 
        clean_address_v30, 
        extract_address_digits_v30, 
        extract_postal_code_v30,
        extract_complex_codes_v30,
        extract_acronym,
        soundex,
        jaro_winkler,
        levenshtein_ratio,
        get_lcp_ratio,
        compute_distinctiveness
    )
except ImportError:
    from preprocessing_v30 import (
        clean_business_name_v30, 
        clean_address_v30, 
        extract_address_digits_v30, 
        extract_postal_code_v30,
        extract_complex_codes_v30,
        extract_acronym,
        soundex,
        jaro_winkler,
        levenshtein_ratio,
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


def is_digit_transposition(d1: list[str], d2: list[str]) -> float:
    """Detect if two addresses have transposed building numbers (e.g., 124 vs 142)."""
    if not d1 or not d2:
        return 0.0
    for num1 in d1:
        for num2 in d2:
            if num1 != num2 and len(num1) == len(num2) and sorted(num1) == sorted(num2):
                return 1.0
    return 0.0


def compute_pair_features_v30(
    s1_rec: dict, 
    cand_rec: dict, 
    rank: int = 0
) -> list[float]:
    """Compute dense 35-feature vector for Version 3.0."""
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
    
    # 6. Jaro-Winkler Similarity (NEW in v3.0)
    name_jw = jaro_winkler(c_name1, c_name2)
    
    # 7. Normalized Levenshtein Ratio (NEW in v3.0)
    name_lev = levenshtein_ratio(c_name1, c_name2)
    
    # 8. Phonetic Soundex Agreement (NEW in v3.0)
    ph1 = soundex(tokens1[0]) if tokens1 else ''
    ph2 = soundex(tokens2[0]) if tokens2 else ''
    phonetic_match = 1.0 if (ph1 and ph2 and ph1 == ph2) else 0.0
    
    # 9. Token Overlap Ratio
    token_overlap_ratio = (len(set_tok1 & set_tok2) / len(set_tok1)) if set_tok1 else 0.0
    
    # 10. Name Length Difference Ratio
    l1, l2 = len(c_name1), len(c_name2)
    max_l = max(l1, l2)
    name_len_diff = abs(l1 - l2) / max_l if max_l > 0 else 0.0

    # 11. Address Numbers Match (House numbers)
    d1 = set(s1_rec.get('addr_digits', []))
    d2 = set(cand_rec.get('addr_digits', []))
    if d1 and d2:
        num_match = 1.0 if len(d1 & d2) > 0 else 0.0
        house_num_mismatch_veto = 1.0 if len(d1 & d2) == 0 else 0.0
    else:
        num_match = 0.5
        house_num_mismatch_veto = 0.0

    # 12. Postal Code Match & Veto
    p1 = s1_rec.get('postal_code', '')
    p2 = cand_rec.get('postal_code', '')
    if p1 and p2:
        postal_match = 1.0 if p1 == p2 else 0.0
        postal_mismatch_veto = 1.0 if p1 != p2 else 0.0
    else:
        postal_match = 0.5
        postal_mismatch_veto = 0.0

    # 13. Address Word Jaccard
    c_addr1 = s1_rec.get('clean_addr', '')
    c_addr2 = cand_rec.get('clean_addr', '')
    addr_tok1 = c_addr1.split()
    addr_tok2 = c_addr2.split()
    addr_tok_jaccard = fast_jaccard(addr_tok1, addr_tok2)

    # 14. Address Char 3-gram Jaccard
    addr_char_jaccard = fast_jaccard(char_ngrams(c_addr1, n=3), char_ngrams(c_addr2, n=3))

    # 15. Pure Street Words Jaccard
    st1 = [w for w in addr_tok1 if not w.isdigit()]
    st2 = [w for w in addr_tok2 if not w.isdigit()]
    street_words_jaccard = fast_jaccard(st1, st2)

    # 16. Address Jaro-Winkler Similarity (NEW in v3.0)
    addr_jw = jaro_winkler(c_addr1, c_addr2)

    # 17. ANTI-CO-LOCATION PENALTY
    co_location_conflict = 1.0 if (num_match == 1.0 and name_sort_jaccard < 0.45 and name_exact == 0.0 and name_jw < 0.60) else 0.0

    # 18. First Token Match
    first_tok_match = 1.0 if (tokens1 and tokens2 and tokens1[0] == tokens2[0]) else 0.0

    # 19. First Token Jaro-Winkler (NEW in v3.0)
    first_tok_jw = jaro_winkler(tokens1[0], tokens2[0]) if (tokens1 and tokens2) else 0.0

    # 20. Contact Phone Match
    ph1_num = s1_rec.get('phone', '')
    ph2_num = cand_rec.get('phone', '')
    phone_match = 1.0 if (ph1_num and ph2_num and ph1_num == ph2_num) else (0.5 if not ph1_num or not ph2_num else 0.0)

    # 21. Complex Plot Code Match
    codes1 = set(s1_rec.get('complex_codes', []))
    codes2 = set(cand_rec.get('complex_codes', []))
    code_match = 1.0 if (codes1 and codes2 and len(codes1 & codes2) > 0) else 0.0

    # 22. Source Indicators
    cand_id = cand_rec.get('entity_id', '')
    is_s2 = 1.0 if cand_id.startswith('S2-') else 0.0
    is_s3 = 1.0 if cand_id.startswith('S3-') else 0.0

    # 23. Retrieval Rank
    rank_feat = float(rank)

    # 24. Joint Name-Address Interaction
    name_addr_interaction = name_dice_2g * addr_char_jaccard

    # 25. Joint Jaro-Winkler Interaction (NEW in v3.0)
    name_jaro_addr_interaction = name_jw * addr_jw

    # 26. Acronym Match
    acr1 = s1_rec.get('acronym', '')
    acr2 = cand_rec.get('acronym', '')
    acronym_match = 0.0
    if acr1 and (acr1 == c_name2 or acr1 == acr2):
        acronym_match = 1.0
    elif acr2 and (acr2 == c_name1 or acr2 == acr1):
        acronym_match = 1.0

    # 27. Longest Common Prefix (LCP) Ratio for Truncation Detection
    lcp_ratio = get_lcp_ratio(c_name1, c_name2)

    # 28. Distinctiveness Score of Entity
    distinctiveness_score = s1_rec.get('distinctiveness', 0.5)

    # 29. Single Word Containment (NEW in v3.0)
    single_word_containment = 0.0
    if len(tokens1) == 1 and tokens1[0] in set_tok2:
        single_word_containment = 1.0
    elif len(tokens2) == 1 and tokens2[0] in set_tok1:
        single_word_containment = 1.0

    # 30. Shared Digits Count (NEW in v3.0)
    shared_digits = len(d1 & d2) if (d1 and d2) else 0
    shared_digits_count = min(1.0, shared_digits / 3.0)

    # 31. Digit Transposition Detection (NEW in v3.0)
    digit_transposition = is_digit_transposition(s1_rec.get('addr_digits', []), cand_rec.get('addr_digits', []))

    # 32. 4-Character Prefix Agreement (NEW in v3.0)
    prefix_agreement = 1.0 if (len(c_name1) >= 4 and len(c_name2) >= 4 and c_name1[:4] == c_name2[:4]) else 0.0

    # 33. Suffix Agreement (NEW in v3.0)
    suffix_agreement = 1.0 if (len(c_name1) >= 4 and len(c_name2) >= 4 and c_name1[-4:] == c_name2[-4:]) else 0.0

    # 34. Total Distinct Tokens Count (NEW in v3.0)
    total_distinct_tokens = min(1.0, len(set_tok1 | set_tok2) / 10.0)

    # 35. Distinctive Word Overlap Ratio (NEW in v3.0)
    distinctive_overlap = (len(set_tok1 & set_tok2) / max(len(set_tok1), len(set_tok2))) if (set_tok1 and set_tok2) else 0.0

    return [
        name_exact,
        name_sort_jaccard,
        name_tok_jaccard,
        name_dice_2g,
        name_char_jaccard,
        name_jw,
        name_lev,
        phonetic_match,
        token_overlap_ratio,
        name_len_diff,
        num_match,
        house_num_mismatch_veto,
        postal_match,
        postal_mismatch_veto,
        addr_tok_jaccard,
        addr_char_jaccard,
        street_words_jaccard,
        addr_jw,
        co_location_conflict,
        first_tok_match,
        first_tok_jw,
        phone_match,
        code_match,
        is_s2,
        is_s3,
        rank_feat,
        name_addr_interaction,
        name_jaro_addr_interaction,
        acronym_match,
        lcp_ratio,
        distinctiveness_score,
        single_word_containment,
        shared_digits_count,
        digit_transposition,
        prefix_agreement
    ]

FEATURE_NAMES_V30 = [
    'name_exact',
    'name_sort_jaccard',
    'name_tok_jaccard',
    'name_dice_2g',
    'name_char_jaccard',
    'name_jw',
    'name_lev',
    'phonetic_match',
    'token_overlap_ratio',
    'name_len_diff',
    'num_match',
    'house_num_mismatch_veto',
    'postal_match',
    'postal_mismatch_veto',
    'addr_tok_jaccard',
    'addr_char_jaccard',
    'street_words_jaccard',
    'addr_jw',
    'co_location_conflict',
    'first_tok_match',
    'first_tok_jw',
    'phone_match',
    'code_match',
    'is_s2',
    'is_s3',
    'rank_feat',
    'name_addr_interaction',
    'name_jaro_addr_interaction',
    'acronym_match',
    'lcp_ratio',
    'distinctiveness_score',
    'single_word_containment',
    'shared_digits_count',
    'digit_transposition',
    'prefix_agreement'
]
