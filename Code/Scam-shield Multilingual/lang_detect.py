"""
lang_detect.py
==============
Zero-dependency language/script detection using Unicode block ranges.
Works fully offline — suitable for Android edge deployment.

Supported scripts and their languages:
  Devanagari (U+0900–U+097F) → Hindi (hi) or Marathi (mr)
  Telugu     (U+0C00–U+0C7F) → Telugu (te)
  Kannada    (U+0C80–U+0CFF) → Kannada (kn)
  Latin      (U+0041–U+007A) → English (en) or Romanized Indian

For Hindi vs Marathi disambiguation we check for Marathi-specific conjuncts
and common function words. If ambiguous, we default to Hindi.
"""

from __future__ import annotations


# ─────────────────────────────────────────────────────────────────────────────
# Unicode block boundaries
# ─────────────────────────────────────────────────────────────────────────────

_DEVANAGARI_START  = 0x0900
_DEVANAGARI_END    = 0x097F
_TELUGU_START      = 0x0C00
_TELUGU_END        = 0x0C7F
_KANNADA_START     = 0x0C80
_KANNADA_END       = 0x0CFF
_LATIN_UPPER_START = 0x0041
_LATIN_UPPER_END   = 0x005A
_LATIN_LOWER_START = 0x0061
_LATIN_LOWER_END   = 0x007A
_DIGIT_START       = 0x0030
_DIGIT_END         = 0x0039

# Marathi-specific words (cannot appear in Hindi)
_MARATHI_MARKERS = {
    "आहे", "नाही", "करा", "आहेत", "आणि", "मला", "तुम्ही",
    "आपला", "असे", "होते", "म्हणजे", "सांगा", "करणे",
}

# Hindi-specific particles / postpositions rarely used in Marathi
_HINDI_MARKERS = {
    "है", "हैं", "नहीं", "करें", "आपका", "मुझे", "कीजिए",
    "जाएगा", "होगा", "लेकिन", "यहाँ", "वहाँ",
}


# ─────────────────────────────────────────────────────────────────────────────
# Core detection
# ─────────────────────────────────────────────────────────────────────────────

def _count_script_chars(text: str) -> dict[str, int]:
    """Count characters belonging to each script block."""
    counts = {"devanagari": 0, "telugu": 0, "kannada": 0, "latin": 0, "other": 0}
    for ch in text:
        cp = ord(ch)
        if _DEVANAGARI_START <= cp <= _DEVANAGARI_END:
            counts["devanagari"] += 1
        elif _TELUGU_START <= cp <= _TELUGU_END:
            counts["telugu"] += 1
        elif _KANNADA_START <= cp <= _KANNADA_END:
            counts["kannada"] += 1
        elif (_LATIN_UPPER_START <= cp <= _LATIN_UPPER_END or
              _LATIN_LOWER_START <= cp <= _LATIN_LOWER_END):
            counts["latin"] += 1
        elif _DIGIT_START <= cp <= _DIGIT_END:
            pass  # digits are neutral
        else:
            counts["other"] += 1
    return counts


def _devanagari_to_hindi_or_marathi(text: str) -> str:
    """Disambiguate Hindi vs Marathi for a Devanagari text."""
    words = set(text.split())
    marathi_hits = len(words & _MARATHI_MARKERS)
    hindi_hits   = len(words & _HINDI_MARKERS)
    if marathi_hits > hindi_hits:
        return "mr"
    return "hi"


def detect_language(text: str) -> str:
    """
    Detect the primary language of a message.

    Strategy:
      1. Strip URLs and common ASCII brand names before computing script counts.
         This prevents a Kannada/Hindi message with an injected English URL
         (a common scam pattern like "bit.ly/verify") from being misclassified
         as English.
      2. Determine dominant Unicode script block.
      3. If Devanagari, disambiguate Hindi vs Marathi via marker words.

    Returns:
        "en" – English / primarily Latin
        "hi" – Hindi (Devanagari)
        "mr" – Marathi (Devanagari)
        "te" – Telugu
        "kn" – Kannada
        "mixed" – significant mix of two or more non-Latin scripts
        "other" – unrecognised
    """
    import re as _re

    if not text or not text.strip():
        return "en"

    # Strip URLs (http/https/www + domain patterns) before script analysis.
    # We want to detect the *human-written* part of the message language.
    _url_pat = _re.compile(
        r'https?://\S+|www\.\S+|[a-zA-Z0-9.-]+\.[a-zA-Z]{2,6}(?:/\S*)?'
    )
    stripped = _url_pat.sub(" ", text)

    counts = _count_script_chars(stripped)
    total_script = sum(counts.values()) - counts["other"]
    if total_script == 0:
        # Entire message was URLs/numbers — fall back to full text
        counts = _count_script_chars(text)
        total_script = sum(counts.values()) - counts["other"]
        if total_script == 0:
            return "en"

    # Determine dominant script (ignoring 'other')
    script_counts = {k: v for k, v in counts.items() if k != "other"}
    dominant = max(script_counts, key=script_counts.get)
    dominant_ratio = script_counts[dominant] / max(total_script, 1)

    if dominant_ratio < 0.40:
        return "mixed"

    if dominant == "latin":
        return "en"
    elif dominant == "devanagari":
        return _devanagari_to_hindi_or_marathi(text)
    elif dominant == "telugu":
        return "te"
    elif dominant == "kannada":
        return "kn"

    return "other"


def get_script_dominance(text: str) -> float:
    """
    Return the fraction of non-digit, non-punctuation characters that belong
    to a non-Latin script. High value in an otherwise-English context = suspicious.
    """
    counts = _count_script_chars(text)
    non_latin_script = counts["devanagari"] + counts["telugu"] + counts["kannada"]
    all_script = non_latin_script + counts["latin"]
    if all_script == 0:
        return 0.0
    return non_latin_script / all_script


def script_mismatch_score(text: str) -> float:
    """
    Detect Roman-script words injected into a native-script message.
    Scammers often inject English brand names / URLs into native-script messages
    to bypass keyword filters while maintaining social engineering effectiveness.

    Returns a float in [0, 1]: 0 = no mismatch, 1 = extreme mismatch.
    """
    counts = _count_script_chars(text)
    native = counts["devanagari"] + counts["telugu"] + counts["kannada"]
    latin  = counts["latin"]

    if native == 0 and latin == 0:
        return 0.0
    if native == 0:
        return 0.0   # Purely Latin — expected for English
    if latin == 0:
        return 0.0   # Purely native — no mismatch

    # Mismatch is high when both native and Latin chars are present
    total = native + latin
    minority = min(native, latin)
    return minority / total


# ─────────────────────────────────────────────────────────────────────────────
# Language code → integer mapping (for use as ML feature)
# ─────────────────────────────────────────────────────────────────────────────

LANG_TO_INT = {
    "en":    0,
    "hi":    1,
    "mr":    2,
    "te":    3,
    "kn":    4,
    "mixed": 5,
    "other": 5,
}


def lang_to_int(lang_code: str) -> int:
    return LANG_TO_INT.get(lang_code, 5)
