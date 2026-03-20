"""
Feature Extractor v2.0 — Production-grade scam signal extraction
24 engineered features covering text, URL, and statistical signals.
"""
import re
import math
from collections import Counter

HIGH_RISK_TLDS = [
    'tk', 'ml', 'ga', 'cf', 'gq', 'pw', 'click', 'loan', 'win',
    'bid', 'racing', 'kim', 'xyz', 'top', 'cc', 'ru', 'cn'
]

URL_SHORTENERS = [
    'bit.ly', 'tinyurl', 't.co', 'goo.gl', 'is.gd', 'cutt.ly', 'ow.ly', 'rb.gy'
]

VERIFIED_DOMAINS = [
    'google.com', 'apple.com', 'amazon.com', 'microsoft.com',
    'paypal.com', 'github.com', 'youtube.com', 'linkedin.com',
    'twitter.com', 'facebook.com', 'instagram.com', 'zoom.us'
]

URGENCY_KEYWORDS = [
    'urgent', 'immediately', 'verify now', 'act now', 'suspended',
    'account locked', 'limited time', 'expires', 'asap', 'right now',
    'last chance', 'final notice', 'action required', 'response required'
]

MONEY_KEYWORDS = [
    'free money', 'lottery', 'winner', 'prize', 'earn', 'income',
    'profit', 'investment', 'guaranteed', 'cash', 'reward', 'bonus',
    'crypto', 'bitcoin', 'forex', 'job offer', 'work from home',
    'make money', 'financial', 'loan', 'credit'
]

SENSITIVE_KEYWORDS = [
    'password', 'cvv', 'pin', 'otp', 'login', 'social security',
    'ssn', 'bank account', 'credit card', 'debit card', 'routing number',
    'date of birth', 'mother maiden', 'secret question', 'passcode',
    'verify your identity', 'confirm your details', 'update your info'
]

OFF_PLATFORM_KEYWORDS = [
    'telegram', 'whatsapp', 'signal', 'dm me', 'text me',
    'call this number', 'contact us at', 'reach us on', 'inbox me'
]

THREAT_KEYWORDS = [
    'your account will be', 'will be suspended', 'will be deleted',
    'blocked', 'compromised', 'unauthorized access', 'unusual activity',
    'suspicious activity', 'we detected', 'security alert', 'fraud alert'
]

LEGITIMACY_MARKERS = [
    'documentation', 'meeting', 'schedule', 'report', 'attached',
    'please find', 'regards', 'sincerely', 'team', 'department',
    'office', 'conference', 'presentation', 'project', 'update'
]


def extract_urls(text):
    pattern = r'(https?://[^\s<>"]+|www\.[^\s<>"]+|\b[a-zA-Z0-9.-]+\.[a-zA-Z]{2,6}(?:/[^\s]*)?)'
    return re.findall(pattern, text.lower())


def char_entropy(text):
    if len(text) < 2:
        return 0.0
    counts = Counter(text)
    total = len(text)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def avg_word_length(text):
    words = re.findall(r'[a-zA-Z]+', text)
    if not words:
        return 0.0
    return sum(len(w) for w in words) / len(words)


def keyword_density(text_lower, keywords):
    hits = sum(1 for kw in keywords if kw in text_lower)
    return hits / len(keywords)


def extract_features(text, manual_score=None):
    text_lower = text.lower()
    words = text_lower.split()
    n_words = max(len(words), 1)
    urls = extract_urls(text)

    f1  = int(any(kw in text_lower for kw in URGENCY_KEYWORDS))
    f2  = int(any(kw in text_lower for kw in MONEY_KEYWORDS))
    f3  = int(any(kw in text_lower for kw in SENSITIVE_KEYWORDS))
    f4  = int(any(kw in text_lower for kw in OFF_PLATFORM_KEYWORDS))
    f5  = int(any(kw in text_lower for kw in THREAT_KEYWORDS))
    f6  = int(any(kw in text_lower for kw in LEGITIMACY_MARKERS))

    f7  = len(text)
    f8  = text.count('!')
    f9  = text.count('?')
    f10 = sum(c.isupper() for c in text) / max(len(text), 1)
    f11 = sum(c.isdigit() for c in text) / max(len(text), 1)
    f12 = round(char_entropy(text_lower), 4)
    f13 = round(avg_word_length(text), 4)
    f14 = sum(1 for c in text if c in '!?@#$%^&*()[]{}') / max(len(text), 1)

    f15 = round(keyword_density(text_lower, URGENCY_KEYWORDS), 4)
    f16 = round(keyword_density(text_lower, MONEY_KEYWORDS), 4)
    f17 = round(keyword_density(text_lower, SENSITIVE_KEYWORDS), 4)

    f18 = len(urls)
    f19 = len(urls) / n_words

    ip_url, shortener, risky_tld, spoofing, verified = 0, 0, 0, 0, 0
    for url in urls:
        if re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', url):
            ip_url = 1
        if any(s in url for s in URL_SHORTENERS):
            shortener = 1
        if any(url.endswith('.' + tld) or ('.' + tld + '/') in url for tld in HIGH_RISK_TLDS):
            risky_tld = 1
        brand_spoof = any(b in url for b in ['paypal', 'amazon', 'google', 'apple', 'microsoft'])
        is_verified = any(v in url for v in VERIFIED_DOMAINS)
        if brand_spoof and not is_verified:
            spoofing = 1
        if is_verified:
            verified = 1

    f20 = ip_url
    f21 = shortener
    f22 = risky_tld
    f23 = spoofing
    f24 = verified

    return [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11,
            f12, f13, f14, f15, f16, f17, f18, f19, f20,
            f21, f22, f23, f24]


FEATURE_NAMES = [
    'has_urgency', 'has_money', 'has_sensitive', 'has_off_platform',
    'has_threat', 'has_legitimacy_marker',
    'text_length', 'exclamation_count', 'question_count',
    'uppercase_ratio', 'digit_ratio', 'char_entropy', 'avg_word_length',
    'punctuation_density',
    'urgency_density', 'money_density', 'sensitive_density',
    'num_urls', 'url_density',
    'ip_url', 'url_shortener', 'risky_tld', 'domain_spoof', 'verified_domain'
]