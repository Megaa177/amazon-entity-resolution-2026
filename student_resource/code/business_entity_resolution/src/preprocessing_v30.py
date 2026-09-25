"""
Version 3.0 Universal Preprocessing, Phonetics & String Metric Engine
Includes:
- Mojibake repair & universal Unicode NFD normalization
- Indic phonetic transliteration (Devanagari, Tamil, Telugu, Gujarati, Bengali)
- Double Metaphone / Soundex phonetic hash encoder
- Fast Jaro-Winkler string distance
- Normalized Levenshtein ratio
- Bi-directional acronym & initialism extraction
- Longest Common Prefix (LCP) truncation detection
- Entity distinctiveness scoring
"""

import re
import unicodedata
from collections import Counter

MOJIBAKE_MAP = {
    'PrAcsident': 'President',
    'A%cole': 'Ecole',
    'SantAc': 'Sante',
    'RenaudiA"re': 'Renaudiere',
    'ClAcment': 'Clement',
    'SuA"de': 'Suede',
    'ThAcnard': 'Thenard',
    'RAcunis': 'Reunis',
}

INDIC_CHAR_MAP = {
    0x0905: 'a', 0x0906: 'aa', 0x0907: 'i', 0x0908: 'ee', 0x0909: 'u', 0x090a: 'oo',
    0x090f: 'e', 0x0910: 'ai', 0x0913: 'o', 0x0914: 'au',
    0x0915: 'k', 0x0916: 'kh', 0x0917: 'g', 0x0918: 'gh', 0x0919: 'n',
    0x091a: 'ch', 0x091b: 'chh', 0x091c: 'j', 0x091d: 'jh', 0x091e: 'n',
    0x091f: 't', 0x0920: 'th', 0x0921: 'd', 0x0922: 'dh', 0x0923: 'n',
    0x0924: 't', 0x0925: 'th', 0x0926: 'd', 0x0927: 'dh', 0x0928: 'n',
    0x092a: 'p', 0x092b: 'ph', 0x092c: 'b', 0x092d: 'bh', 0x092e: 'm',
    0x092f: 'y', 0x0930: 'r', 0x0932: 'l', 0x0935: 'v', 0x0936: 'sh',
    0x0937: 'sh', 0x0938: 's', 0x0939: 'h',
    0x093e: 'aa', 0x093f: 'i', 0x0940: 'ee', 0x0941: 'u', 0x0942: 'oo',
    0x0947: 'e', 0x0948: 'ai', 0x094b: 'o', 0x094c: 'au',
    0x0902: 'n', 0x094d: ''
}

UNIVERSAL_LEGAL_SUFFIXES = {
    'inc', 'incorporated', 'corp', 'corporation', 'llc', 'ltd', 'limited',
    'pvt', 'private', 'co', 'company', 'group', 'holdings', 'enterprise', 'enterprises',
    'pllc', 'llp', 'lp', 'associates', 'association', 'services', 'solutions', 'partners',
    'opc', 'sarl', 'sas', 'sasu', 'eurl', 'sci', 'snc', 'gie', 'societe', 'sa',
    'and', 'or', 'of', 'in', 'at', 'the', 'a', 'an', '&'
}

UNIVERSAL_STREET_WORDS = {
    'road', 'street', 'avenue', 'lane', 'drive', 'floor', 'shop', 
    'near', 'beside', 'behind', 'block', 'unit', 'suite', 'apt', 
    'box', 'post', 'boulevard', 'place', 'circle', 'square', 'nagar',
    'colony', 'marg', 'bhavan', 'complex', 'plaza', 'building', 'center',
    'house', 'tower', 'phase', 'sector', 'point', 'cross',
    # French & European street types & modifiers
    'rue', 'chemin', 'allee', 'impasse', 'quai', 'cours', 'passage',
    'route', 'bis', 'ter', 'quater', 'bd', 'av', 'che', 'imp', 'all', 'r'
}


def clean_mojibake(text: str) -> str:
    """Repair corrupted diacritic encodings."""
    if not text:
        return ""
    s = str(text)
    for k, v in MOJIBAKE_MAP.items():
        if k in s:
            s = s.replace(k, v)
    s = re.sub(r'A[c%](?=[a-z])', 'e', s)
    s = re.sub(r'A"', 'e', s)
    s = re.sub(r'\ufffd', '', s)
    return s


def transliterate_indic(text: str) -> str:
    """Phonetically romanize Indic scripts (Devanagari, Tamil, Telugu, Gujarati, Bengali)."""
    if not text:
        return ""
    res = []
    for c in str(text):
        o = ord(c)
        if 0x0900 <= o <= 0x097F:
            res.append(INDIC_CHAR_MAP.get(o, ''))
        elif 0x0980 <= o <= 0x0D7F:
            script_base = (o // 0x80) * 0x80
            dev_equiv = 0x0900 + (o - script_base)
            res.append(INDIC_CHAR_MAP.get(dev_equiv, ''))
        else:
            res.append(c)
    return ''.join(res)


def normalize_universal(text: str) -> str:
    """Universal string normalizer: Mojibake repair + Indic transliteration + NFKD diacritic removal."""
    if not text:
        return ""
    s = clean_mojibake(str(text))
    s = transliterate_indic(s)
    nfd = unicodedata.normalize('NFD', s)
    ascii_clean = ''.join(c for c in nfd if unicodedata.category(c) != 'Mn')
    return ascii_clean.lower()


def clean_business_name_v30(name: str, extra_suffixes: set = None) -> str:
    """Universal business name normalizer."""
    if not name or str(name).strip() == '':
        return ""
    s = normalize_universal(str(name))
    
    if ' trading as ' in s:
        s = s.split(' trading as ')[-1]
    elif ' t/a ' in s:
        s = s.split(' t/a ')[-1]
    s = re.sub(r'\b(d/?b/?a|formerly|f/?k/?a|a/?k/?a)\b', ' ', s)
    s = re.sub(r'\.(com|in|org|net|fr|co|us|gov|edu)\b', '', s)
    s = re.sub(r'[^\w\s]+', ' ', s)
    tokens = s.split()
    
    active_suffixes = UNIVERSAL_LEGAL_SUFFIXES.union(extra_suffixes or set())
    filtered = [t for t in tokens if t not in active_suffixes]
    return " ".join(filtered) if filtered else " ".join(tokens)


def clean_address_v30(address: str, extra_stopwords: set = None) -> str:
    """Universal address normalizer."""
    if not address or str(address).strip() == '':
        return ""
    s = normalize_universal(str(address))
    s = re.sub(r'[^\w\s]+', ' ', s)
    tokens = s.split()
    active_stopwords = UNIVERSAL_STREET_WORDS.union(extra_stopwords or set())
    meaningful = [t for t in tokens if t.isdigit() or (len(t) >= 3 and t not in active_stopwords)]
    return " ".join(meaningful)


def soundex(token: str) -> str:
    """Compute 4-character American Soundex phonetic hash."""
    if not token or not token[0].isalpha():
        return ""
    token = token.upper()
    first = token[0]
    mapping = {
        'B': '1', 'F': '1', 'P': '1', 'V': '1',
        'C': '2', 'G': '2', 'J': '2', 'K': '2', 'Q': '2', 'S': '2', 'X': '2', 'Z': '2',
        'D': '3', 'T': '3', 'L': '4', 'M': '5', 'N': '5', 'R': '6'
    }
    encoded = [first]
    prev = mapping.get(first, '0')
    for c in token[1:]:
        curr = mapping.get(c, '0')
        if curr != '0' and curr != prev:
            encoded.append(curr)
        prev = curr
    res = ''.join(encoded)[:4]
    return res.ljust(4, '0')


def jaro_winkler(s1: str, s2: str, p: float = 0.1, max_l: int = 4) -> float:
    """High-speed Jaro-Winkler string similarity with prefix scaling."""
    if s1 == s2:
        return 1.0
    len1, len2 = len(s1), len(s2)
    if len1 == 0 or len2 == 0:
        return 0.0
        
    match_distance = max(len1, len2) // 2 - 1
    s1_matches = [False] * len1
    s2_matches = [False] * len2
    matches = 0
    transpositions = 0
    
    for i in range(len1):
        start = max(0, i - match_distance)
        end = min(i + match_distance + 1, len2)
        for j in range(start, end):
            if s2_matches[j]: continue
            if s1[i] != s2[j]: continue
            s1_matches[i] = True
            s2_matches[j] = True
            matches += 1
            break
            
    if matches == 0:
        return 0.0
        
    k = 0
    for i in range(len1):
        if not s1_matches[i]: continue
        while not s2_matches[k]: k += 1
        if s1[i] != s2[k]: transpositions += 1
        k += 1
        
    transpositions //= 2
    jaro = (matches / len1 + matches / len2 + (matches - transpositions) / matches) / 3.0
    
    prefix = 0
    for c1, c2 in zip(s1[:max_l], s2[:max_l]):
        if c1 == c2: prefix += 1
        else: break
        
    return min(1.0, jaro + prefix * p * (1.0 - jaro))


def levenshtein_ratio(s1: str, s2: str) -> float:
    """Normalized Levenshtein edit distance ratio in [0.0, 1.0]."""
    if s1 == s2: return 1.0
    l1, l2 = len(s1), len(s2)
    if l1 == 0 or l2 == 0: return 0.0
    
    # DP with single-row optimization
    prev_row = list(range(l2 + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1] * (l2 + 1)
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row[j + 1] = min(insertions, deletions, substitutions)
        prev_row = curr_row
        
    dist = prev_row[l2]
    return 1.0 - (dist / float(max(l1, l2)))


def extract_acronym(name: str) -> str:
    """Extract 2-6 character acronym from business name tokens."""
    c = clean_business_name_v30(name)
    tokens = [t for t in c.split() if len(t) > 0 and t[0].isalnum()]
    if 2 <= len(tokens) <= 6:
        return "".join(t[0] for t in tokens).lower()
    return ""


def get_lcp_ratio(s1: str, s2: str) -> float:
    """Compute Longest Common Prefix (LCP) ratio to detect truncations."""
    if not s1 or not s2:
        return 0.0
    min_len = min(len(s1), len(s2))
    if min_len == 0:
        return 0.0
    prefix_len = 0
    for c1, c2 in zip(s1, s2):
        if c1 == c2:
            prefix_len += 1
        else:
            break
    return prefix_len / float(min_len)


def compute_distinctiveness(name: str) -> float:
    """Score entity distinctiveness in [0.0, 1.0]."""
    clean = clean_business_name_v30(name)
    tokens = clean.split()
    if not tokens:
        return 0.0
    num_tokens = len(tokens)
    total_len = len(clean)
    score = 0.2
    if num_tokens >= 2: score += 0.25
    if num_tokens >= 4: score += 0.20
    if total_len >= 12: score += 0.20
    if total_len >= 22: score += 0.15
    return min(1.0, score)


def extract_address_digits_v30(address: str) -> list[str]:
    """Extract numeric building/house numbers without leading zeros."""
    if not address:
        return []
    raw_digits = re.findall(r'\b\d+\b', str(address))
    res = []
    for d in raw_digits:
        try:
            val = str(int(d))
            if len(val) <= 7:
                res.append(val)
        except ValueError:
            pass
    return res


def extract_postal_code_v30(address: str, country: str) -> str:
    """Universal postal code extractor."""
    if not address:
        return ""
    s = str(address)
    if country == 'India':
        m = re.findall(r'\b[1-9]\d{5}\b', s)
        return m[-1] if m else ""
    elif country in ('US', 'France'):
        m = re.findall(r'\b\d{5}\b', s)
        return m[-1] if m else ""
    else:
        m = re.findall(r'\b\d{4,6}\b', s)
        return m[-1] if m else ""


def extract_complex_codes_v30(address: str) -> list[str]:
    """Extract plot/shop codes."""
    if not address:
        return []
    s = str(address).lower()
    codes = re.findall(r'\b[a-z]{1,3}-\d{2,4}[a-z]?\b|\b\d+/\d+\b|\b\d+-\d+-\d+\b', s)
    return [c.replace('-', '') for c in codes]
