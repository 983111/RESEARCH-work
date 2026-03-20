"""
multilingual_feature_extractor.py
==================================
Extends the original 24-feature ScamShield feature vector with 8 multilingual
features (f25–f32). The original 24 features remain unchanged and are computed
via the existing feature_extractor.py logic (copied inline to avoid import
path issues when deploying to Android).

Feature vector layout:
  f1–f24  : original ScamShield features (see feature_extractor.py)
  f25     : detected_lang_int   (0=en,1=hi,2=mr,3=te,4=kn,5=other/mixed)
  f26     : has_urgency_ml      (urgency keyword in detected language)
  f27     : has_money_ml        (money keyword in detected language)
  f28     : has_sensitive_ml    (sensitive keyword in detected language)
  f29     : has_off_platform_ml (off-platform keyword in detected language)
  f30     : has_threat_ml       (threat keyword in detected language)
  f31     : script_mismatch     (Roman chars injected into native-script msg)
  f32     : char_ngram_scam_score (float [0,1] from lightweight char-ngram LR)

Design decisions:
  - f25–f30 mirror the structure of f1–f5 so the GBM can learn joint signals
  - f31 catches a common obfuscation tactic: "verify your ఖాతా at bit.ly/x"
  - f32 is a meta-feature: output of a separate lightweight char-ngram model
    that captures subword scam patterns invisible to keyword matching.
    Set to 0.0 when the ngram model is not loaded (graceful degradation).
"""

from __future__ import annotations

import re
import math
from collections import Counter
from typing import Optional

from lang_detect import detect_language, script_mismatch_score, lang_to_int
from multilingual_lexicons import get_keywords, get_cross_language_keywords

# ─────────────────────────────────────────────────────────────────────────────
# Reproduce original feature extractor constants inline
# (copied from feature_extractor.py to keep this file self-contained)
# ─────────────────────────────────────────────────────────────────────────────

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


def _extract_urls(text: str) -> list[str]:
    pattern = r'(https?://[^\s<>"]+|www\.[^\s<>"]+|\b[a-zA-Z0-9.-]+\.[a-zA-Z]{2,6}(?:/[^\s]*)?)'
    return re.findall(pattern, text.lower())


def _char_entropy(text: str) -> float:
    if len(text) < 2:
        return 0.0
    counts = Counter(text)
    total = len(text)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def _avg_word_length(text: str) -> float:
    words = re.findall(r'[a-zA-Z\u0900-\u0CFF]+', text)
    if not words:
        return 0.0
    return sum(len(w) for w in words) / len(words)


def _keyword_density(text_lower: str, keywords: list[str]) -> float:
    hits = sum(1 for kw in keywords if kw in text_lower)
    return hits / len(keywords)


# ─────────────────────────────────────────────────────────────────────────────
# Original 24-feature extraction (unchanged logic from feature_extractor.py)
# ─────────────────────────────────────────────────────────────────────────────

def _extract_original_24(text: str) -> list[float]:
    """Extract the original 24 ScamShield features."""
    text_lower = text.lower()
    words = text_lower.split()
    n_words = max(len(words), 1)
    urls = _extract_urls(text)

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
    f12 = round(_char_entropy(text_lower), 4)
    f13 = round(_avg_word_length(text), 4)
    f14 = sum(1 for c in text if c in '!?@#$%^&*()[]{}') / max(len(text), 1)
    f15 = round(_keyword_density(text_lower, URGENCY_KEYWORDS), 4)
    f16 = round(_keyword_density(text_lower, MONEY_KEYWORDS), 4)
    f17 = round(_keyword_density(text_lower, SENSITIVE_KEYWORDS), 4)
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

    return [
        f1,  f2,  f3,  f4,  f5,  f6,  f7,  f8,  f9,  f10,
        f11, f12, f13, f14, f15, f16, f17, f18, f19, ip_url,
        shortener, risky_tld, spoofing, verified
    ]


# ─────────────────────────────────────────────────────────────────────────────
# New 8 multilingual features (f25–f32)
# ─────────────────────────────────────────────────────────────────────────────

def _extract_multilingual_8(
    text: str,
    ngram_model: Optional[object] = None,
) -> list[float]:
    """
    Extract 8 multilingual features.

    Args:
        text        : raw message text
        ngram_model : optional scikit-learn model with predict_proba().
                      If provided, its output forms f32 (char_ngram_scam_score).
                      If None, f32 = 0.0 (graceful degradation).
    """
    text_lower = text.lower()

    # f25: language integer
    lang = detect_language(text)
    f25  = float(lang_to_int(lang))

    # f26–f30: multilingual keyword signals
    # Strategy: check language-specific lexicon first, then cross-language fallback
    # This is important because many Indian scam messages mix scripts.
    def _has_keyword(category: str) -> int:
        # Language-specific check
        lang_kws = get_keywords(lang, category)
        if any(kw in text_lower for kw in lang_kws):
            return 1
        # Fallback: cross-language (catches Romanized variants from all 4 languages)
        cross_kws = get_cross_language_keywords(category)
        if any(kw in text_lower for kw in cross_kws):
            return 1
        return 0

    f26 = float(_has_keyword("URGENCY"))
    f27 = float(_has_keyword("MONEY"))
    f28 = float(_has_keyword("SENSITIVE"))
    f29 = float(_has_keyword("OFF_PLATFORM"))
    f30 = float(_has_keyword("THREAT"))

    # f31: script mismatch score
    # High when Roman chars appear in a native-script dominated message
    # (e.g. "bit.ly/xyz" injected into a Hindi message)
    f31 = round(script_mismatch_score(text), 4)

    # f32: char-ngram model scam probability (optional)
    if ngram_model is not None:
        try:
            import numpy as np
            prob = float(ngram_model.predict_proba([text])[0][1])
            f32 = round(prob, 4)
        except Exception:
            f32 = 0.0
    else:
        f32 = 0.0

    return [f25, f26, f27, f28, f29, f30, f31, f32]


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_NAMES_ORIGINAL = [
    'has_urgency', 'has_money', 'has_sensitive', 'has_off_platform',
    'has_threat', 'has_legitimacy_marker',
    'text_length', 'exclamation_count', 'question_count',
    'uppercase_ratio', 'digit_ratio', 'char_entropy', 'avg_word_length',
    'punctuation_density',
    'urgency_density', 'money_density', 'sensitive_density',
    'num_urls', 'url_density',
    'ip_url', 'url_shortener', 'risky_tld', 'domain_spoof', 'verified_domain',
]

FEATURE_NAMES_MULTILINGUAL = [
    'detected_lang_int',
    'has_urgency_ml',
    'has_money_ml',
    'has_sensitive_ml',
    'has_off_platform_ml',
    'has_threat_ml',
    'script_mismatch',
    'char_ngram_scam_score',
]

FEATURE_NAMES_EXTENDED = FEATURE_NAMES_ORIGINAL + FEATURE_NAMES_MULTILINGUAL  # 32 total


def extract_features_extended(
    text: str,
    ngram_model: Optional[object] = None,
) -> list[float]:
    """
    Extract the full 32-feature vector (24 original + 8 multilingual).

    Args:
        text        : raw message text (any language)
        ngram_model : optional pre-loaded char-ngram model for f32

    Returns:
        List of 32 floats in the order defined by FEATURE_NAMES_EXTENDED.
    """
    original_24    = _extract_original_24(text)
    multilingual_8 = _extract_multilingual_8(text, ngram_model=ngram_model)
    return original_24 + multilingual_8


def extract_features_original_24(text: str) -> list[float]:
    """Extract only the original 24 features (for backward compatibility)."""
    return _extract_original_24(text)
