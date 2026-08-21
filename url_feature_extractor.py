"""
URL feature extraction for the phishing detection model.

Extracts 38 features directly from the URL string - the original 45-feature
set minus the 2 WHOIS-based features (Domain_Age_Flag, Domain_Expiry_Flag)
and the 3 live-page-fetch features (Has_Iframe, Has_Mouseover,
Excessive_Redirects), which required network calls and were dropped for
speed and reliability, and minus 2 further features that the shipped model
was NOT trained on:

  - Has_Unicode_Domain - dropped during training
  - Brand_Similarity   - was a placeholder that always returned 0, carrying
                         no information, and was dropped during training

Their helper functions are retained below (unused by extract_url_features) in
case either is ever implemented properly and the model retrained; adding a
column back here requires retraining, since the model validates column names.

Column order matches url_random_forest_model.pkl's model.feature_names_in_
exactly - verified directly against the model file.
"""

import re
import math
from collections import Counter
from urllib.parse import urlparse


# Known URL-shortening services. A match here is a mild phishing signal, since
# shorteners hide the real destination domain from the person clicking the link.
SHORTENING_SERVICES = [
    "bit.ly", "goo.gl", "tinyurl.com", "t.co", "ow.ly", "is.gd", "buff.ly",
    "adf.ly", "bit.do", "cutt.ly", "shorte.st", "rebrand.ly", "tiny.cc",
    "lnkd.in", "db.tt", "qr.ae", "v.gd", "x.co", "po.st", "u.to", "j.mp",
    "s.id", "rb.gy", "shorturl.at", "clck.ru", "soo.gd",
]
SHORTENING_PATTERN = re.compile("|".join(re.escape(s) for s in SHORTENING_SERVICES))

# Words that commonly show up in phishing domains/paths as social-engineering bait
# (e.g. "verify-account-support.com"). A match is a mild phishing signal, not
# proof on its own, which is why it is kept as a separate raw feature rather
# than a hard rule.
SENSITIVE_WORDS = [
    "account", "confirm", "banking", "secure", "signin", "login", "verify",
    "update", "password", "username", "billing", "security", "payment",
    "customer", "service", "verification", "limited", "access", "urgent",
    "suspend", "unlock", "recover", "wallet", "support", "identity",
    "validate", "authentication", "alert", "important",
]

IP_PATTERN = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")


def _calculate_entropy(text):
    if not text:
        return 0.0
    char_counts = Counter(text)
    total_chars = len(text)
    entropy = 0.0
    for count in char_counts.values():
        probability = count / total_chars
        entropy -= probability * math.log2(probability)
    return entropy


def url_length(url):
    return len(url)

def is_shortened_url(url):
    return 1 if SHORTENING_PATTERN.search(url) else 0

def has_prefix_suffix(url):
    return 1 if "-" in urlparse(url).netloc else 0

def count_dots(url):
    return url.count(".")

def has_sensitive_word(url):
    domain = urlparse(url).netloc.lower()
    return 1 if any(word in domain for word in SENSITIVE_WORDS) else 0

def has_unicode_domain(url):
    domain = urlparse(url).netloc
    return 1 if any(ord(ch) > 127 for ch in domain) else 0

def get_domain_length(url):
    return len(urlparse(url).netloc)

def get_path_length(url):
    return len(urlparse(url).path)

def get_query_length(url):
    return len(urlparse(url).query)

def contains_login(url):
    return 1 if 'login' in url.lower() else 0

def contains_verify(url):
    return 1 if 'verify' in url.lower() else 0

def contains_account(url):
    return 1 if 'account' in url.lower() else 0

def contains_security(url):
    return 1 if 'security' in url.lower() else 0

def contains_password(url):
    return 1 if 'password' in url.lower() else 0

def contains_payment(url):
    return 1 if 'payment' in url.lower() else 0

def has_percent_encoding(url):
    return 1 if '%' in url else 0

def has_punycode(url):
    return 1 if 'xn--' in urlparse(url).netloc else 0

def url_entropy(url):
    return _calculate_entropy(url)

def domain_entropy(url):
    return _calculate_entropy(urlparse(url).netloc)

def digit_ratio(url):
    domain = urlparse(url).netloc
    if not domain:
        return 0.0
    return sum(c.isdigit() for c in domain) / len(domain)

def special_char_ratio(url):
    domain = urlparse(url).netloc
    if not domain:
        return 0.0
    return sum(1 for c in domain if not c.isalnum() and c != '.') / len(domain)

def domain_hyphen_count(url):
    return urlparse(url).netloc.count('-')

def brand_similarity(url, known_brands=None):
    # Placeholder - always 0. A real implementation would compare the domain
    # against a list of known brand domains using a string-similarity metric.
    return 0

def digit_count(url):
    return sum(c.isdigit() for c in url)

def letter_count(url):
    return sum(c.isalpha() for c in url)

def hyphen_count(url):
    return url.count('-')

def slash_count(url):
    return url.count('/')

def underscore_count(url):
    return url.count('_')

def question_count(url):
    return url.count('?')

def equal_count(url):
    return url.count('=')

def ampersand_count(url):
    return url.count('&')

def percent_count(url):
    return url.count('%')

def at_count(url):
    return url.count('@')

def has_https(url):
    return 1 if urlparse(url).scheme == 'https' else 0

def has_ip(url):
    return 1 if IP_PATTERN.search(url) else 0

def subdomain_count(url):
    netloc = urlparse(url).netloc
    if re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", netloc) or netloc == 'localhost':
        return 0
    if ':' in netloc:
        netloc = netloc.split(':')[0]
    parts = netloc.split('.')
    return max(0, len(parts) - 2)

def path_depth(url):
    segments = urlparse(url).path.split('/')
    return sum(1 for s in segments if s)

def query_parameter_count(url):
    query = urlparse(url).query
    if not query:
        return 0
    return sum(1 for p in query.split('&') if p)

def has_port(url):
    return 1 if urlparse(url).port is not None else 0

def has_fragment(url):
    return 1 if urlparse(url).fragment else 0


def ensure_trailing_slash(url: str) -> str:
    """Add a trailing slash to the URL if it doesn't already end with one."""
    if not url.endswith('/'):
        url += '/'
    return url


# Order matches url_random_forest_model.pkl's model.feature_names_in_ exactly.
FEATURE_COLUMNS = [
    "URL_Length", "Path_Depth", "Has_IP", "Is_Shortened",
    "Has_Prefix_Suffix", "Dot_Count", "Has_Sensitive_Word",
    "Domain_Length", "Path_Length", "Query_Length",
    "Digit_Count", "Letter_Count", "Hyphen_Count", "Slash_Count",
    "Underscore_Count", "Question_Count", "Equal_Count", "Ampersand_Count",
    "Percent_Count", "At_Count",
    "Has_HTTPS", "Subdomain_Count", "Query_Parameter_Count", "Has_Port", "Has_Fragment",
    "Contains_Login", "Contains_Verify", "Contains_Account",
    "Contains_Security", "Contains_Password", "Contains_Payment",
    "Has_Percent_Encoding", "Has_Punycode", "URL_Entropy", "Domain_Entropy",
    "Domain_Digit_Ratio", "Domain_Special_Char_Ratio", "Domain_Hyphen_Count",
]


def extract_url_features(url):
    """Extract the 38-feature vector for a single URL, in FEATURE_COLUMNS order."""
    url = ensure_trailing_slash(url)

    lexical = [
        url_length(url), path_depth(url), has_ip(url), is_shortened_url(url),
        has_prefix_suffix(url), count_dots(url), has_sensitive_word(url),
        get_domain_length(url), get_path_length(url), get_query_length(url),
    ]
    characters = [
        digit_count(url), letter_count(url), hyphen_count(url), slash_count(url),
        underscore_count(url), question_count(url), equal_count(url), ampersand_count(url),
        percent_count(url), at_count(url),
    ]
    structure = [
        has_https(url), subdomain_count(url), query_parameter_count(url), has_port(url), has_fragment(url),
    ]
    suspicious_patterns = [
        contains_login(url), contains_verify(url), contains_account(url),
        contains_security(url), contains_password(url), contains_payment(url),
    ]
    obfuscation = [
        has_percent_encoding(url), has_punycode(url), url_entropy(url), domain_entropy(url),
    ]
    domain = [
        digit_ratio(url), special_char_ratio(url), domain_hyphen_count(url),
    ]

    values = lexical + characters + structure + suspicious_patterns + obfuscation + domain
    return dict(zip(FEATURE_COLUMNS, values))
