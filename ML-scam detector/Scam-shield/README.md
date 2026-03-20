# ScamShield — Interpretable Multi-Signal Scam Detection

**ML + Adversarial Evaluation + Statistical Significance | End-to-End AI Pipeline | 5 Languages**

Live demo: https://innterpretablescamdetection.vercel.app  
Research paper DOI: [10.5281/zenodo.18988170](https://doi.org/10.5281/zenodo.18988170)

A production-oriented scam detection system combining interpretable machine learning, adversarial robustness testing, and statistical significance evaluation — extended to detect scams in **Hindi, Marathi, Telugu, and Kannada** via a 844 KB Android-compatible model bundle.

> English: 19,992 samples · 17 scam categories · 24 engineered features · McNemar significance testing  
> Multilingual: +7,975 samples · 5 languages · 32 features (24 + 8 new) · 844 KB Android bundle

---

## Table of Contents

- [Quick Start](#quick-start)
- [Training Results — English](#training-results--english)
- [Multilingual Extension](#multilingual-extension)
- [System Overview](#system-overview)
- [Repository Structure](#repository-structure)
- [Dataset](#dataset)
- [Feature Engineering](#feature-engineering)
- [Model Architecture](#model-architecture)
- [Adversarial Robustness](#adversarial-robustness)
- [API Reference](#api-reference)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [Author](#author)

---

## Quick Start

```bash
# Install dependencies
pip install pandas scikit-learn scipy flask joblib

# ── English pipeline ──────────────────────────────────────────────────
# Step 1 — Generate dataset (20k samples)
python build_realistic_dataset.py

# Step 2 — Train all models + run full evaluation
python train_and_evaluate.py

# Step 3 — Start inference API (port 5000)
python ml_api.py

# ── Multilingual extension ────────────────────────────────────────────
# Step 4 — Generate multilingual dataset (7,975 samples, 5 languages)
cd scamshield_multilingual
python build_multilingual_dataset.py

# Step 5 — Train multilingual models + evaluation
python train_multilingual.py

# Step 6 — Multilingual inference
python multilingual_inference_api.py
```

English results → `eval_results/full_evaluation.json` · model → `models/scam_detector_final.pkl`  
Multilingual results → `scamshield_multilingual/eval_results/multilingual_evaluation.json`

---

## Training Results — English

All results from `train_and_evaluate.py` on 19,992 samples, stratified 80/20 split.

### Model Comparison — Held-Out Test Set (n = 3,999)

| Model | F1 | AUC | Recall | Precision | MCC |
|---|---|---|---|---|---|
| Naive Bayes (TF-IDF) | 0.8136 | 0.8857 | 0.7904 | 0.8382 | 0.639 |
| LinearSVC (TF-IDF) | 0.8284 | 0.8879 | 0.7474 | 0.9291 | 0.704 |
| Logistic Regression | 0.9158 | 0.9740 | 0.8999 | 0.9321 | 0.835 |
| Random Forest | 0.9826 | 0.9988 | 0.9770 | 0.9884 | 0.966 |
| **ScamShield GBM** ← proposed | **0.9969** | **1.0000** | **0.9938** | **1.0000** | **0.994** |

### 3-Fold Cross-Validation

| Model | F1 Mean | F1 Std | Recall | AUC |
|---|---|---|---|---|
| Naive Bayes | 0.8262 | ±0.0014 | 0.7759 | 0.8856 |
| LinearSVC | 0.8278 | ±0.0015 | 0.7470 | 0.8879 |
| Logistic Regression | 0.9140 | ±0.0066 | 0.8957 | 0.9739 |
| Random Forest | 0.9778 | ±0.0075 | 0.9651 | 0.9985 |
| **ScamShield GBM** | **0.9969** | **±0.0004** | **0.9938** | **1.0000** |

### Statistical Significance — McNemar's Test vs ScamShield

| Baseline | χ² | p-value | Significant |
|---|---|---|---|
| Naive Bayes | 722.0 | < 0.001 | Yes *** |
| LinearSVC | 617.0 | < 0.001 | Yes *** |
| Logistic Regression | 329.0 | < 0.001 | Yes *** |
| Random Forest | 67.0 | 0.0022 | Yes ** |

### Adversarial Robustness — English (scam-only recall under attack)

| Scenario | Recall | Δ Recall |
|---|---|---|
| Clean (no attack) | 1.0000 | — |
| Synonym substitution (`verify → confirm`) | 0.2906 | −0.71 |
| Homoglyph attack (`a → а, e → е`) | 0.1846 | −0.82 |
| URL obfuscation (redirect wrapping) | 0.2026 | −0.80 |

The model is **fragile to obfuscation attacks** on English. This is the primary limitation and the motivation for the char n-gram meta-model (f32) in the multilingual extension.

### Top Features by Importance (GBM, English)

| Feature | Importance | Direction |
|---|---|---|
| url_density | 0.253 | ↑ scam |
| text_length | 0.173 | ↑ scam |
| punctuation_density | 0.158 | ↑ scam |
| digit_ratio | 0.095 | ↑ scam |
| char_entropy | 0.090 | ↑ scam |
| sensitive_density | 0.070 | ↑ scam |
| uppercase_ratio | 0.049 | ↑ scam |
| has_money | 0.019 | ↑ scam |

### Honest Interpretation

The near-perfect GBM test score (F1 = 1.000) is partly a synthetic dataset artifact — `url_density` and `char_entropy` jointly create near-perfect class separation. The **CV F1 = 0.9969** is the robust estimate. For a conservative real-world baseline, the **Naive Bayes / LinearSVC scores (F1 ≈ 0.83)** are more honest.

---

## Multilingual Extension

Extends ScamShield to detect scams in **Hindi, Marathi, Telugu, and Kannada** — the four largest South Asian language groups by smartphone penetration. The extension is **additive**: no English pipeline file is modified.

### What's New

| | Original ScamShield | Multilingual Extension |
|---|---|---|
| Languages | English only | English + Hindi + Marathi + Telugu + Kannada |
| Features | 24 | 32 (24 original + 8 new, f25–f32) |
| Language detection | — | Zero-dependency Unicode block detection |
| Android bundle | ~2 MB | **842 KB** (281 KB GBM + 561 KB ngram) |
| Script mismatch detection | — | ✓ Catches injected Roman URLs in native-script messages |
| Char n-gram model | — | ✓ Script-agnostic subword pattern matching (f32) |
| Adversarial attacks tested | 3 | 4 (+script-swap attack) |

### Multilingual Dataset (Synthetic)

| Split | Total | Scam | Safe |
|---|---|---|---|
| Train (80%) | 6,380 | 3,187 | 3,193 |
| Test (20%) | 1,595 | 797 | 798 |
| **Total** | **7,975** | **3,984** | **3,991** |

### Per-Language Results (Synthetic Benchmark)

| Language | F1 | AUC | n Samples | Note |
|---|---|---|---|---|
| Hindi (hi) | 1.0000* | 1.0000 | 3,680 | Largest pool |
| Telugu (te) | 1.0000* | 1.0000 | 1,690 | — |
| Kannada (kn) | 1.0000* | 1.0000 | 1,610 | — |
| Marathi (mr) | — | — | — | Insufficient class balance in test split |

*Synthetic dataset artifact — same caveat as English.

### Adversarial Robustness — Multilingual (4 attacks, scam recall)

| Attack | Recall | Δ Recall | Note |
|---|---|---|---|
| Clean (no attack) | 1.0000 | — | — |
| Synonym substitution | 1.0000 | 0.0000 | Romanized variants in training lexicons |
| Homoglyph attack | 1.0000 | 0.0000 | f32 char n-gram provides fallback |
| URL obfuscation | 1.0000 | 0.0000 | script_mismatch (f31) catches injected URLs |
| **Script swap (new)** | 1.0000 | 0.0000 | Both native-script + Romanized in training |

### New Features (f25–f32)

| Feature | Description | Type | Signal |
|---|---|---|---|
| `detected_lang_int` | 0=en, 1=hi, 2=mr, 3=te, 4=kn, 5=other | Int | Context |
| `has_urgency_ml` | Urgency keyword in detected language (native + Romanized) | Binary | Scam ▲ |
| `has_money_ml` | Money/lottery keyword in detected language | Binary | Scam ▲ |
| `has_sensitive_ml` | Credential request keyword in detected language | Binary | Scam ▲ |
| `has_off_platform_ml` | Off-platform redirect in detected language | Binary | Scam ▲ |
| `has_threat_ml` | Threat/suspension keyword in detected language | Binary | Scam ▲ |
| `script_mismatch` | Roman chars injected into native-script message | Float [0,1] | Scam ▲ |
| `char_ngram_scam_score` | Output probability of char 3–5gram LR model | Float [0,1] | Scam ▲ |

### Android Bundle Size

| File | Size | Purpose |
|---|---|---|
| `multilingual_scam_detector.pkl` | 281 KB | 32-feature GBM + isotonic calibration |
| `multilingual_ngram_model.pkl` | 561 KB | Char 3–5gram TF-IDF + LogisticRegression |
| **Total** | **842 KB** | Well within low-RAM Android budget |

**Why not IndicBERT or mBERT?** IndicBERT is ~300 MB; mBERT ~700 MB. Neither fits on low-RAM Android without significant quantisation infrastructure. Character n-gram TF-IDF requires no tokenizer, no vocab, and no GPU. The total bundle here is 842 KB — 350× smaller than IndicBERT.

---

## System Overview

```
Input Text (any of: EN / HI / MR / TE / KN)
        │
        ▼
[Language Detection]          ← zero-dependency Unicode, strips URLs first
        │
        ├──► [Char N-gram Model] ──► f32 (char_ngram_scam_score)
        │    TF-IDF 3–5gram, 8k features, LogReg — 561 KB
        │    Script-agnostic, works on any Unicode
        │
        ├──► [Multilingual Lexicons] ──► f25–f31
        │    Native-script + Romanized keywords
        │    6 categories × 4 languages
        │    Script mismatch score
        │
        └──► [Original 24 Features] ──► f1–f24
             URL signals · English keywords · stats
                    │
                    ▼
        [32-Feature GBM Ensemble] — 281 KB
        CalibratedClassifierCV / isotonic
        Language-specific threshold
                    │
                    ▼
         safe / suspicious / scam
```

---

## Repository Structure

```
scam-detection/
│
├── feature_extractor.py              # 24-feature extraction pipeline (English)
├── build_realistic_dataset.py        # English dataset generator (20k samples)
├── train_and_evaluate.py             # English training + evaluation pipeline
├── ml_api.py                         # English Flask inference API (port 5000)
├── model_metadata.json               # English feature importances + thresholds
│
├── scamshield_multilingual/
│   ├── multilingual_lexicons.py      # Scam keyword lexicons (HI/MR/TE/KN)
│   ├── lang_detect.py                # Zero-dependency Unicode language detection
│   ├── multilingual_feature_extractor.py  # 32-feature extraction pipeline
│   ├── build_multilingual_dataset.py # Multilingual dataset generator
│   ├── train_multilingual.py         # Multilingual training + evaluation
│   ├── multilingual_inference_api.py # Multilingual Flask API + CLI test
│   ├── multilingual_model_metadata.json  # Feature names, thresholds, sizes
│   │
│   ├── models/
│   │   ├── multilingual_scam_detector.pkl   # 32-feat GBM (281 KB)
│   │   └── multilingual_ngram_model.pkl     # Char n-gram LR (561 KB)
│   │
│   └── eval_results/
│       └── multilingual_evaluation.json     # Full metrics, per-language, adversarial
│
├── eval_results/
│   └── full_evaluation.json          # English evaluation results
│
├── models/
│   └── scam_detector_final.pkl       # English trained model
│
├── .gitignore
└── README.md
```

---

## Dataset

### English Dataset

Generated by `build_realistic_dataset.py`. No template slots — every message is natural language.

**Scale:** 19,992 samples · 9,996 scam · 9,996 safe · balanced

**17 scam categories:**

| Category | Examples |
|---|---|
| Phishing | Account suspension, login alerts |
| Lottery | Prize wins, gift cards |
| Job scam | Work from home, crypto trading |
| Tech support | Virus alerts, fake helplines |
| Advance fee | Nigerian prince, inheritance |
| Delivery scam | Customs fee, redelivery |
| Fake government | IRS, SSA, FBI threats |
| Credential theft | Direct password/PIN requests |
| Romance scam | Emotional manipulation, money requests |
| Crypto scam | Giveaways, pump-and-dump |
| Fake charity | Disaster relief, donation fraud |
| Investment scam | Guaranteed returns, forex |
| Emergency scam | Grandparent scam, stranded abroad |
| Impersonation | Celebrity, CEO, bank manager |
| Fake loan | Instant approval, upfront fee |
| Fake product | Counterfeit goods, miracle pills |
| Refund scam | Tax refund, insurance overpayment |

**7 legitimate categories:** casual SMS, work communications, real notifications, personal messages, banking alerts, e-commerce, educational.

### Multilingual Dataset

Generated by `scamshield_multilingual/build_multilingual_dataset.py`. Follows the same 17-category taxonomy in each of the 4 languages, with both native-script and Romanized variants.

**Scale:** 7,975 samples · 3,984 scam · 3,991 safe · balanced  
**Languages:** Hindi (HI) · Marathi (MR) · Telugu (TE) · Kannada (KN)

---

## Feature Engineering

### Original 24 Features (f1–f24, English — unchanged)

#### Text Features (f1–f14)

| Feature | Description | Type |
|---|---|---|
| `has_urgency` | Urgency cues: "urgent", "act now", "expires" | Binary |
| `has_money` | Financial keywords: "lottery", "earn", "crypto" | Binary |
| `has_sensitive` | Credential requests: password, CVV, OTP, SSN | Binary |
| `has_off_platform` | Redirect: Telegram, WhatsApp, DM me | Binary |
| `has_threat` | Threat language: "will be suspended", "blocked" | Binary |
| `has_legitimacy_marker` | Professional cues: "regards", "meeting", "report" | Binary |
| `text_length` | Character count | Integer |
| `exclamation_count` | Number of `!` characters | Integer |
| `question_count` | Number of `?` characters | Integer |
| `uppercase_ratio` | Fraction of uppercase letters | Float |
| `digit_ratio` | Fraction of digit characters | Float |
| `char_entropy` | Shannon entropy of character distribution | Float |
| `avg_word_length` | Mean word length in characters | Float |
| `punctuation_density` | Special character density | Float |

#### Keyword Density Features (f15–f17)

| Feature | Description |
|---|---|
| `urgency_density` | Fraction of urgency keywords present (0–1) |
| `money_density` | Fraction of money keywords present (0–1) |
| `sensitive_density` | Fraction of sensitive keywords present (0–1) |

#### URL Features (f18–f24)

| Feature | Description | Type |
|---|---|---|
| `num_urls` | Number of URLs detected | Integer |
| `url_density` | URL-to-word ratio | Float |
| `ip_url` | IP-address based URL present | Binary |
| `url_shortener` | bit.ly, tinyurl, goo.gl detected | Binary |
| `risky_tld` | High-risk TLD: .tk, .ml, .xyz, .pw | Binary |
| `domain_spoof` | Brand name in non-verified domain | Binary |
| `verified_domain` | Trusted domain: google.com, github.com | Binary |

### New 8 Multilingual Features (f25–f32)

| Feature | Description | Type | Signal |
|---|---|---|---|
| `detected_lang_int` | 0=en, 1=hi, 2=mr, 3=te, 4=kn, 5=other/mixed | Int | Context |
| `has_urgency_ml` | Urgency keyword in detected language (native script + Romanized) | Binary | Scam ▲ |
| `has_money_ml` | Money/lottery keyword in detected language | Binary | Scam ▲ |
| `has_sensitive_ml` | Credential request keyword in detected language | Binary | Scam ▲ |
| `has_off_platform_ml` | Off-platform redirect keyword in detected language | Binary | Scam ▲ |
| `has_threat_ml` | Threat/suspension keyword in detected language | Binary | Scam ▲ |
| `script_mismatch` | Roman chars injected into native-script message (float 0–1) | Float | Scam ▲ |
| `char_ngram_scam_score` | Output probability of char 3–5gram LR model | Float | Scam ▲ |

---

## Model Architecture

### English Model

**Final model:** `CalibratedClassifierCV(GradientBoostingClassifier, method='isotonic', cv=3)`

| Parameter | Value |
|---|---|
| n_estimators | 150 |
| max_depth | 4 |
| learning_rate | 0.05 |
| subsample | 0.8 |
| calibration | isotonic regression |

### Multilingual Models

Two models loaded together at startup:

**Model 1 — 32-feature GBM (281 KB):**  
Same hyperparameters as English GBM. Trained on the 32-feature vector including f32 from Model 2.

**Model 2 — Char N-gram LR (561 KB):**
```python
Pipeline([
    TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), max_features=8000),
    StandardScaler(with_mean=False),
    LogisticRegression(C=1.0, solver='liblinear', class_weight='balanced'),
])
```
Script-agnostic — works identically on Devanagari, Telugu, Kannada, and Latin. Requires no tokenizer or language-specific preprocessing.

### Training Pipeline

```
English:
  Load dataset (19,992) → Stratified split (80/20) → 3-fold CV across 5 families
  → Train GBM + isotonic calibration → Adversarial evaluation → McNemar tests → Save

Multilingual:
  Generate dataset (7,975) → Train char n-gram model → Re-extract 32 features (f32 populated)
  → Train 32-feat GBM + isotonic calibration → Per-language eval → 4-attack adversarial eval → Save bundle
```

---

## Adversarial Robustness

### English (3 attacks)

```
Clean              1.0000
Synonym attack     0.2906   (−71%)
Homoglyph attack   0.1846   (−82%)
URL obfuscation    0.2026   (−80%)
```

### Multilingual (4 attacks — includes script-swap)

The **script-swap attack** replaces native-script scam keywords with their Romanized equivalents (`तुरंत` → `turant`, `వెంటనే` → `ventane`) to test whether the model relies purely on script-specific character patterns.

```
Clean              1.0000
Synonym attack     1.0000   (0.00  — Romanized variants in training)
Homoglyph attack   1.0000   (0.00  — f32 char n-gram fallback)
URL obfuscation    1.0000   (0.00  — script_mismatch (f31) catches injected URLs)
Script swap        1.0000   (0.00  — native + Romanized both in training lexicons)
```

Note: 0.0 delta on synthetic data because training lexicons include both script variants. Real-world adversarial robustness against novel out-of-lexicon obfuscation remains an open question.

---

## API Reference

### English API — `ml_api.py` (port 5000)

```bash
python ml_api.py
```

**POST /predict**

```json
{ "content": "URGENT! Verify your account now at bit.ly/fake-link" }
```

```json
{
  "verdict": "scam",
  "probability": 0.9412,
  "risk_score": 94,
  "threshold_used": 0.999,
  "top_signals": [
    { "feature": "url_shortener", "value": 1, "importance": 0.1128 },
    { "feature": "has_urgency",   "value": 1, "importance": 0.1940 }
  ]
}
```

### Multilingual API — `multilingual_inference_api.py` (port 5001)

```bash
cd scamshield_multilingual
python multilingual_inference_api.py
```

**POST /predict** — same request format, works for all 5 languages

```json
{
  "content": "आपके बैंक खाते में संदिग्ध गतिविधि। bit.ly/verify-now"
}
```

```json
{
  "verdict":     "scam",
  "probability": 1.0,
  "risk_score":  100,
  "language":    "hi",
  "threshold":   0.85,
  "top_signals": [
    { "feature": "char_ngram_scam_score", "value": 1.0, "importance": 1.0 }
  ]
}
```

**Verdict thresholds:**

| Language | Scam Threshold |
|---|---|
| English (en) | 0.90 |
| Hindi (hi) | 0.85 |
| Telugu (te) | 0.85 |
| Kannada (kn) | 0.85 |
| Marathi (mr) | 0.80 |

---

## Limitations

**Synthetic data (both pipelines).** All samples are generated. Real-world performance will be lower. For production use, supplement with:

**English:**
- [UCI SMS Spam Collection](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection) — 5,574 real SMS
- [Nazario Phishing Corpus](https://www.monkey.org/~jose/phishing/) — 6,000+ phishing emails
- [Enron-Spam Dataset](http://www2.aueb.gr/users/ion/data/enron-spam/) — 33,000+ emails

**Multilingual:**
- [AI4Bharat IndicNLP](https://indicnlp.ai4bharat.org/) — monolingual corpora for Indian languages
- [IndicGLUE](https://indicnlp.ai4bharat.org/indic-glue/) — evaluation benchmarks

**Adversarial fragility (English).** Recall drops to 0.18–0.29 under synonym substitution and homoglyph attacks. The char n-gram meta-feature (f32) provides a partial non-keyword-based fallback.

**Marathi pool size.** Smaller training pool than Hindi/Telugu/Kannada. Production deployment needs more Marathi training data.

**Language coverage.** 5 of 22 scheduled Indian languages. No support yet for Bengali, Tamil, Malayalam, Punjabi, Gujarati.

---

## Future Work

**English:**
- [ ] Integrate real-world datasets (UCI SMS Spam, Enron, Nazario)
- [ ] Character-level CNN features for homoglyph robustness
- [ ] Full SHAP value visualization
- [ ] Adversarial training with synonym/homoglyph augmentation

**Multilingual:**
- [ ] Integrate AI4Bharat IndicNLP and IndicGLUE corpora
- [ ] Expand Marathi training pool
- [ ] Add Bengali (U+0980–U+09FF) and Tamil (U+0B80–U+0BFF)
- [ ] Quantised IndicSBERT sentence embeddings as additional features
- [ ] Per-language adversarial training with script-swap augmentation
- [ ] Domain reputation API integration for Indian financial domains

**Both:**
- [ ] LLM safety layer (multilingual Llama Guard / GPT-4o for non-English)
- [ ] Online learning pipeline
- [ ] Docker containerisation + Kubernetes deployment
- [ ] Behavioral signals: sender patterns, timing, contact-graph structure

---

## Installation

```bash
git clone <repository-url>
cd scam-detection

pip install pandas scikit-learn scipy flask joblib

# English
python build_realistic_dataset.py
python train_and_evaluate.py
python ml_api.py

# Multilingual
cd scamshield_multilingual
python build_multilingual_dataset.py
python train_multilingual.py
python multilingual_inference_api.py
```

Python 3.8+ required.

---

## Author

**Vishwajeet Adkine**  
AI / ML · Applied Systems · Security AI  
Portfolio: [https://portfolio-vishwajeetadkine.vercel.app/](https://portfolio-vishwajeetadkine.vercel.app/)

---

## References

1. Fette et al. (2007) — Learning to detect phishing emails. *WWW 2007*
2. Lundberg & Lee (2017) — SHAP. *NeurIPS 2017*
3. Ribeiro et al. (2016) — "Why should I trust you?". *KDD 2016*
4. Iyer et al. (2023) — Llama Guard. *arXiv:2312.06674*
5. Friedman (2001) — Gradient boosting machine. *Annals of Statistics*
6. Pedregosa et al. (2011) — Scikit-learn. *JMLR 12*
7. Kakwani et al. (2020) — IndicNLPSuite. *EMNLP Findings 2020*
8. Kunchukuttan et al. (2020) — AI4Bharat-IndicNLP Corpus. *arXiv:2005.00085*

---

## License

Provided for educational and research purposes. Production deployment requires validation on real-world data.
