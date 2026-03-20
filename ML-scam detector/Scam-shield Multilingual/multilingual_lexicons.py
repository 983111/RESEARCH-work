"""
multilingual_lexicons.py
========================
Scam-signal keyword lexicons for Hindi, Marathi, Telugu, and Kannada.
Each language covers:
  - URGENCY  : urgency / action-now phrases
  - MONEY    : financial bait phrases
  - SENSITIVE: credential / personal-data request phrases
  - OFF_PLATFORM: redirect-to-external-channel phrases
  - THREAT   : threat / suspension language
  - LEGIT    : legitimacy markers (safe signal, negative weight)

Script notes:
  - Hindi/Marathi: Devanagari (U+0900–U+097F)
  - Telugu       : Telugu script (U+0C00–U+0C7F)
  - Kannada      : Kannada script (U+0C80–U+0CFF)

Romanized ("Hinglish") variants are included because Indian users
frequently mix scripts in SMS/chat messages.
"""

# ─────────────────────────────────────────────────────────────────────────────
# HINDI (हिन्दी)
# ─────────────────────────────────────────────────────────────────────────────

HINDI = {
    "URGENCY": [
        # Native Devanagari
        "तुरंत",        # immediately
        "अभी",          # right now
        "जल्दी करें",  # hurry up
        "सावधान",       # beware / alert
        "अंतिम सूचना", # final notice
        "तत्काल",       # urgent
        "अंतिम मौका",  # last chance
        "समय सीमा",     # deadline
        "अभी वेरीफाई करें",  # verify now
        "खाता बंद होगा",     # account will close
        # Romanized Hinglish
        "turant",
        "abhi verify karo",
        "jaldi karo",
        "last chance",
        "account band ho jayega",
        "seedha reply karo",
        "abhi call karo",
    ],
    "MONEY": [
        "लॉटरी",        # lottery
        "इनाम",         # prize
        "मुफ्त",        # free
        "पैसे कमाएं",  # earn money
        "बड़ा मुनाफा",  # big profit
        "निवेश",        # investment
        "रोज कमाएं",   # earn daily
        "घर बैठे कमाई", # earn from home
        "क्रिप्टो",     # crypto
        "बिटकॉइन",      # bitcoin
        "रिवॉर्ड",      # reward
        "कैशबैक",       # cashback
        "गिफ्ट कार्ड",  # gift card
        # Romanized
        "lottery jeeta",
        "free paisa",
        "ghar baithe kamao",
        "bitcoin invest karo",
        "guaranteed profit",
        "daily earning",
        "reward milega",
    ],
    "SENSITIVE": [
        "पासवर्ड",      # password
        "ओटीपी",        # OTP
        "पिन",          # PIN
        "आधार नंबर",    # Aadhaar number
        "बैंक खाता",    # bank account
        "क्रेडिट कार्ड", # credit card
        "सीवीवी",       # CVV
        "जन्मतिथि",     # date of birth
        "पैन नंबर",     # PAN number
        # Romanized
        "otp bhejo",
        "password batao",
        "aadhar number do",
        "bank account number",
        "cvv batao",
        "pin share karo",
    ],
    "OFF_PLATFORM": [
        "व्हाट्सएप",    # WhatsApp
        "टेलीग्राम",    # Telegram
        "मुझे कॉल करें", # call me
        "इस नंबर पर",   # on this number
        "डीएम करें",    # DM me
        # Romanized
        "whatsapp karo",
        "telegram join karo",
        "is number pe call karo",
        "dm karo",
        "seedha contact karo",
    ],
    "THREAT": [
        "खाता बंद",     # account closed
        "गिरफ्तारी",    # arrest
        "कानूनी कार्रवाई", # legal action
        "ब्लॉक कर दिया", # blocked
        "निलंबित",      # suspended
        "जुर्माना",     # fine
        "FIR दर्ज",     # FIR filed
        # Romanized
        "account block ho gaya",
        "arrest ho sakta hai",
        "legal notice aaya hai",
        "fine bharno",
        "suspended kar diya",
    ],
    "LEGIT": [
        "बैठक",         # meeting
        "रिपोर्ट",      # report
        "दस्तावेज",     # document
        "नमस्ते",       # hello (formal)
        "धन्यवाद",      # thank you
        "कृपया",        # please
        "विभाग",        # department
        "अनुसूची",      # schedule
        # Romanized
        "meeting hai",
        "report bhejo",
        "regards",
        "department se",
    ],
}

# ─────────────────────────────────────────────────────────────────────────────
# MARATHI (मराठी)
# ─────────────────────────────────────────────────────────────────────────────

MARATHI = {
    "URGENCY": [
        "ताबडतोब",      # immediately
        "त्वरित",       # urgent
        "आत्ताच",       # right now
        "शेवटची संधी",  # last chance
        "अंतिम सूचना",  # final notice
        "खाते बंद होईल", # account will close
        "लवकर करा",     # do it quickly
        "मुदत संपेल",   # deadline approaching
        # Romanized
        "taabadtob verify kara",
        "aattach kara",
        "khate band hoil",
        "last chance ahe",
    ],
    "MONEY": [
        "लॉटरी",        # lottery
        "बक्षीस",       # prize
        "मोफत",         # free
        "पैसे मिळवा",   # earn money
        "गुंतवणूक",     # investment
        "क्रिप्टो",     # crypto
        "कमाई करा",     # earn
        "घरबसल्या कमाई", # earn from home
        "इनाम",         # reward
        # Romanized
        "lottery jinkla",
        "free paise",
        "ghar baslya kamva",
        "crypto invest kara",
        "guaranteed napha",
    ],
    "SENSITIVE": [
        "पासवर्ड",      # password
        "ओटीपी",        # OTP
        "पिन",          # PIN
        "आधार क्रमांक", # Aadhaar number
        "बँक खाते",     # bank account
        "क्रेडिट कार्ड", # credit card
        "पॅन क्रमांक",  # PAN number
        "जन्मतारीख",    # date of birth
        # Romanized
        "otp sanga",
        "password sanga",
        "aadhar number sanga",
        "bank account number sanga",
    ],
    "OFF_PLATFORM": [
        "व्हाट्सअॅप",   # WhatsApp
        "टेलिग्राम",    # Telegram
        "मला कॉल करा",  # call me
        "या नंबरवर",    # on this number
        # Romanized
        "whatsapp kara",
        "telegram join kara",
        "ya numbervara call kara",
        "dm kara",
    ],
    "THREAT": [
        "खाते बंद",     # account closed
        "अटक",          # arrest
        "कायदेशीर कारवाई", # legal action
        "ब्लॉक केले",   # blocked
        "निलंबित",      # suspended
        "दंड",          # fine / penalty
        # Romanized
        "account block jhale",
        "atak hoil",
        "legal notice aale",
        "dand bhara",
    ],
    "LEGIT": [
        "बैठक",         # meeting
        "अहवाल",        # report
        "कागदपत्रे",    # documents
        "नमस्कार",      # hello (formal)
        "धन्यवाद",      # thank you
        "कृपया",        # please
        "विभाग",        # department
        # Romanized
        "meeting ahe",
        "report pathva",
        "regards",
        "vibhagakadun",
    ],
}

# ─────────────────────────────────────────────────────────────────────────────
# TELUGU (తెలుగు)
# ─────────────────────────────────────────────────────────────────────────────

TELUGU = {
    "URGENCY": [
        "వెంటనే",       # immediately
        "అత్యవసరం",     # urgent
        "ఇప్పుడే",      # right now
        "చివరి అవకాశం", # last chance
        "ఖాతా మూసివేయబడుతుంది", # account will be closed
        "గడువు తీరుతోంది", # deadline approaching
        "వెంటనే వెరిఫై చేయండి", # verify immediately
        # Romanized
        "ventane verify cheyandi",
        "ippude cheyandi",
        "account close avutundi",
        "last chance undi",
    ],
    "MONEY": [
        "లాటరీ",        # lottery
        "బహుమతి",       # prize
        "ఉచితంగా",      # free
        "డబ్బు సంపాదించండి", # earn money
        "పెట్టుబడి",    # investment
        "క్రిప్టో",     # crypto
        "రోజూ సంపాదించండి", # earn daily
        "ఇంట్లో కూర్చుని సంపాదించండి", # earn from home
        "రివార్డ్",     # reward
        # Romanized
        "lottery gettinav",
        "free money vastundi",
        "inti nundi sapadinchu",
        "crypto invest cheyyandi",
        "guaranteed profit",
    ],
    "SENSITIVE": [
        "పాస్వర్డ్",    # password
        "ఓటీపీ",        # OTP
        "పిన్",         # PIN
        "ఆధార్ నంబర్",  # Aadhaar number
        "బ్యాంక్ ఖాతా", # bank account
        "క్రెడిట్ కార్డ్", # credit card
        "సీవీవీ",       # CVV
        "పాన్ నంబర్",   # PAN number
        # Romanized
        "otp cheppandi",
        "password cheppandi",
        "aadhar number ivvandi",
        "bank account number ivvandi",
    ],
    "OFF_PLATFORM": [
        "వాట్సాప్",     # WhatsApp
        "టెలిగ్రామ్",   # Telegram
        "నాకు కాల్ చేయండి", # call me
        "ఈ నంబర్‌కు",   # to this number
        # Romanized
        "whatsapp cheyandi",
        "telegram join avvandi",
        "ee numberu ki call cheyandi",
        "dm cheyandi",
    ],
    "THREAT": [
        "ఖాతా మూసివేయబడింది", # account closed
        "అరెస్ట్",      # arrest
        "చట్టపరమైన చర్య", # legal action
        "బ్లాక్ చేశారు", # blocked
        "నిలిపివేశారు",  # suspended
        "జరిమానా",      # fine
        # Romanized
        "account block chesaru",
        "arrest avutaru",
        "legal notice vastundi",
        "fine kattu",
    ],
    "LEGIT": [
        "సమావేశం",      # meeting
        "నివేదిక",      # report
        "పత్రాలు",      # documents
        "నమస్కారం",     # hello (formal)
        "ధన్యవాదాలు",   # thank you
        "దయచేసి",       # please
        "విభాగం",       # department
        # Romanized
        "meeting undi",
        "report pathampandi",
        "regards",
        "vibhagam nundi",
    ],
}

# ─────────────────────────────────────────────────────────────────────────────
# KANNADA (ಕನ್ನಡ)
# ─────────────────────────────────────────────────────────────────────────────

KANNADA = {
    "URGENCY": [
        "ತಕ್ಷಣ",        # immediately
        "ತುರ್ತು",        # urgent
        "ಈಗಲೇ",         # right now
        "ಕೊನೆಯ ಅವಕಾಶ",  # last chance
        "ಖಾತೆ ಮುಚ್ಚಲಾಗುವುದು", # account will close
        "ಗಡುವು ಮೀರುತ್ತಿದೆ", # deadline approaching
        "ತಕ್ಷಣ ಪರಿಶೀಲಿಸಿ", # verify immediately
        # Romanized
        "takshana verify madi",
        "igale madi",
        "account close aaguttade",
        "last chance ide",
    ],
    "MONEY": [
        "ಲಾಟರಿ",        # lottery
        "ಬಹುಮಾನ",       # prize
        "ಉಚಿತ",         # free
        "ಹಣ ಸಂಪಾದಿಸಿ",  # earn money
        "ಹೂಡಿಕೆ",       # investment
        "ಕ್ರಿಪ್ಟೋ",     # crypto
        "ಪ್ರತಿದಿನ ಸಂಪಾದಿಸಿ", # earn daily
        "ಮನೆಯಲ್ಲಿ ಕುಳಿತು ಸಂಪಾದಿಸಿ", # earn from home
        "ಬಹುಮಾನ",       # reward
        # Romanized
        "lottery geddidevi",
        "free hana bantide",
        "maneyalli koothu sampadi",
        "crypto invest madi",
        "guaranteed profit ide",
    ],
    "SENSITIVE": [
        "ಪಾಸ್‌ವರ್ಡ್",   # password
        "ಒಟಿಪಿ",         # OTP
        "ಪಿನ್",          # PIN
        "ಆಧಾರ್ ಸಂಖ್ಯೆ",  # Aadhaar number
        "ಬ್ಯಾಂಕ್ ಖಾತೆ",  # bank account
        "ಕ್ರೆಡಿಟ್ ಕಾರ್ಡ್", # credit card
        "ಸಿವಿವಿ",        # CVV
        "ಪ್ಯಾನ್ ನಂಬರ್",  # PAN number
        # Romanized
        "otp heli",
        "password heli",
        "aadhar number kodi",
        "bank account number kodi",
    ],
    "OFF_PLATFORM": [
        "ವಾಟ್ಸಾಪ್",     # WhatsApp
        "ಟೆಲಿಗ್ರಾಮ್",   # Telegram
        "ನನಗೆ ಕರೆ ಮಾಡಿ",  # call me
        "ಈ ನಂಬರ್‌ಗೆ",    # to this number
        # Romanized
        "whatsapp madi",
        "telegram join aagi",
        "ee numbarge call madi",
        "dm madi",
    ],
    "THREAT": [
        "ಖಾತೆ ಮುಚ್ಚಲಾಗಿದೆ", # account closed
        "ಬಂಧನ",         # arrest
        "ಕಾನೂನು ಕ್ರಮ",  # legal action
        "ಬ್ಲಾಕ್ ಮಾಡಲಾಗಿದೆ", # blocked
        "ಅಮಾನತ್ತು",     # suspended
        "ದಂಡ",          # fine
        # Romanized
        "account block maadidare",
        "bandhan aaguttare",
        "legal notice bantide",
        "danda kattu",
    ],
    "LEGIT": [
        "ಸಭೆ",          # meeting
        "ವರದಿ",         # report
        "ದಾಖಲೆಗಳು",     # documents
        "ನಮಸ್ಕಾರ",      # hello (formal)
        "ಧನ್ಯವಾದ",       # thank you
        "ದಯವಿಟ್ಟು",     # please
        "ವಿಭಾಗ",        # department
        # Romanized
        "meeting ide",
        "report kali",
        "regards",
        "vibhagadinda",
    ],
}

# ─────────────────────────────────────────────────────────────────────────────
# Combined lookup — used by the feature extractor
# ─────────────────────────────────────────────────────────────────────────────

ALL_LANGUAGES = {
    "hi": HINDI,
    "mr": MARATHI,
    "te": TELUGU,
    "kn": KANNADA,
}

# Flat sets for fast O(1) membership testing — built once at import time
_FLAT: dict[str, set[str]] = {}
for _lang_code, _lang_dict in ALL_LANGUAGES.items():
    for _category, _kw_list in _lang_dict.items():
        _key = f"{_lang_code}_{_category}"
        _FLAT[_key] = {kw.lower() for kw in _kw_list}

# Cross-language flat sets (used when language is unknown / mixed)
_CROSS: dict[str, set[str]] = {}
for _category in ["URGENCY", "MONEY", "SENSITIVE", "OFF_PLATFORM", "THREAT", "LEGIT"]:
    _CROSS[_category] = set()
    for _lang_code, _lang_dict in ALL_LANGUAGES.items():
        _CROSS[_category].update(kw.lower() for kw in _lang_dict.get(_category, []))


def get_keywords(lang_code: str, category: str) -> set[str]:
    """Return keyword set for a specific language and category.
    Falls back to cross-language set if lang_code is unknown.
    """
    key = f"{lang_code}_{category}"
    if key in _FLAT:
        return _FLAT[key]
    return _CROSS.get(category, set())


def get_cross_language_keywords(category: str) -> set[str]:
    """Return merged keyword set for a category across all 4 languages."""
    return _CROSS.get(category, set())
