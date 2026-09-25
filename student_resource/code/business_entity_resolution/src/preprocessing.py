import re
import unicodedata

# Multilingual corporate/legal suffixes (US, India, France, Global)
LEGAL_SUFFIXES = {
    # US / UK / Global
    'inc', 'incorporated', 'corp', 'corporation', 'llc', 'ltd', 'limited',
    'pvt', 'private', 'co', 'company', 'group', 'holdings', 'enterprise', 'enterprises',
    'pllc', 'llp', 'lp', 'associates', 'association', 'services', 'solutions',
    # India
    'opc',
    # France
    'sa', 'sarl', 'sas', 'sasu', 'eurl', 'sci', 'snc', 'gie', 'societe',
    # Common English noise words
    'and', 'or', 'of', 'in', 'at', 'the', 'a', 'an', '&'
}

FRENCH_CITIES = {
    'paris', 'lyon', 'marseille', 'bordeaux', 'nantes', 'toulouse',
    'lille', 'nice', 'strasbourg', 'rennes', 'montpellier'
}

ADDRESS_ABBREVIATIONS = {
    'rd': 'road',
    'st': 'street',
    'ave': 'avenue',
    'blvd': 'boulevard',
    'dr': 'drive',
    'ln': 'lane',
    'ct': 'court',
    'pl': 'place',
    'sq': 'square',
    'pkwy': 'parkway',
    'hwy': 'highway',
    'apt': 'apartment',
    'ste': 'suite',
    'fl': 'floor',
    'bldg': 'building',
    'dept': 'department',
    'no': 'number',
    'sec': 'sector',
    'po': 'post',
    # French address terms
    'bd': 'boulevard',
    'av': 'avenue',
    'che': 'chemin',
    'imp': 'impasse',
    'all': 'allee',
    'r': 'rue',
}

ACCENTS_MAP = str.maketrans(
    'éèêëàâäôöûüùîïçñÉÈÊËÀÂÄÔÖÛÜÙÎÏÇÑ',
    'eeeeaaaoouuuiicnEEEEAAAOOUUUIICN'
)


def remove_accents(text: str) -> str:
    """Normalize Latin diacritics while preserving Indic scripts."""
    if not text:
        return ""
    s = unicodedata.normalize('NFC', str(text))
    return s.translate(ACCENTS_MAP)


def clean_business_name(name: str) -> str:
    """Preprocess business name into a standardized string preserving Unicode."""
    if not name or str(name).strip() == '':
        return ""
    s = remove_accents(str(name)).lower()
    
    # Strip trading as / dba prefixes to extract core entity name
    if ' trading as ' in s:
        s = s.split(' trading as ')[-1]
    elif ' t/a ' in s:
        s = s.split(' t/a ')[-1]
    s = re.sub(r'\b(d/?b/?a|formerly|f/?k/?a|a/?k/?a)\b', ' ', s)
    
    # Remove web domain extensions if present (.com, .in, .org, .fr)
    s = re.sub(r'\.(com|in|org|net|fr|co|us|gov|edu)\b', '', s)
    
    # Replace punctuation characters with spaces while keeping word characters (letters, digits, Indic scripts)
    s = re.sub(r'[^\w\s]+', ' ', s)
    tokens = s.split()
    
    # Filter corporate legal suffixes
    filtered = [t for t in tokens if t not in LEGAL_SUFFIXES]
    
    # If filtered tokens have multiple words, remove generic French city tokens
    if len(filtered) > 1:
        non_city = [t for t in filtered if t not in FRENCH_CITIES]
        if non_city:
            filtered = non_city
            
    return " ".join(filtered) if filtered else " ".join(tokens)


def get_name_tokens(name: str) -> list[str]:
    """Get list of informative cleaned tokens for a business name."""
    cleaned = clean_business_name(name)
    return [t for t in cleaned.split() if len(t) > 1]


def clean_address(address: str) -> str:
    """Preprocess and standardize address."""
    if not address or str(address).strip() == '':
        return ""
    s = remove_accents(str(address)).lower()
    s = re.sub(r'[^\w\s/-]+', ' ', s)
    tokens = s.split()
    expanded = [ADDRESS_ABBREVIATIONS.get(t, t) for t in tokens]
    return " ".join(expanded)


def extract_address_digits(address: str) -> list[str]:
    """Extract numeric tokens without leading zeros."""
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


def extract_complex_codes(address: str) -> list[str]:
    """Extract plot/shop/block codes like wz-187c, af-684, 6-2-101, 59/101."""
    if not address:
        return []
    s = str(address).lower()
    codes = re.findall(r'\b[a-z]{1,3}-\d{2,4}[a-z]?\b|\b\d+/\d+\b|\b\d+-\d+-\d+\b', s)
    return [c.replace('-', '') for c in codes]


def extract_postal_code(address: str, country: str) -> str:
    """Extract standard postal code by country format."""
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
