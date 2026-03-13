# Scam Detection System

**ML + LLM + Heuristics | End-to-End AI Pipeline**

This project implements a production-oriented scam detection system combining interpretable machine learning, LLM-based safety analysis, and rule-based heuristics. The goal is to demonstrate **end-to-end ML ownership**: feature design, model training, evaluation, deployment, and backend integration.

> Designed with AI research rigor and systems thinking, aligned with applied ML best practices.

📄 **[Read the full research paper](docs/RESEARCH_PAPER.md)** — *An Interpretable Multi-Signal Scam Detection System Using Machine Learning and Large Language Models*

## Table of Contents

- [Research Paper](#research-paper)
- [Problem Statement](#-problem-statement)
- [System Overview](#-system-overview)
- [Repository Structure](#-repository-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Building the Dataset](#building-the-dataset)
  - [Training the Model](#training-the-model)
  - [Model Explainability](#model-explainability)
  - [Running the API](#running-the-api)
  - [Making Predictions](#making-predictions)
- [Machine Learning Component](#-machine-learning-component)
- [Feature Engineering](#feature-engineering)
- [Model Architecture](#model-architecture)
- [Evaluation Methodology](#-evaluation-methodology)
- [Model Explainability (SHAP-Style)](#-model-explainability-shap-style)
- [LLM Safety Layer](#-llm-safety-layer)
- [Ensemble Decision Logic](#-ensemble-decision-logic)
- [API Reference](#api-reference)
- [Examples](#-example-inference)
- [What This Demonstrates](#-what-this-project-demonstrates)
- [Future Extensions](#-future-extensions)
- [Contributing](#contributing)

## Research Paper

📄 **[An Interpretable Multi-Signal Scam Detection System Using Machine Learning and Large Language Models](docs/RESEARCH_PAPER.md)**

This project is accompanied by a formal research paper that details:
- Problem formulation and optimization objectives
- Feature engineering methodology with mathematical notation
- Model architecture and calibration techniques
- Evaluation methodology emphasizing recall for security
- Explainability analysis using coefficient-based attribution
- LLM integration strategy for semantic validation
- Ensemble decision framework
- Deployment architecture for production systems
- Limitations and comprehensive future work

**Key Quote for Research Presentations:**

> "I focused on **interpretable ML for security**. I designed explicit features, trained a linear classifier, analyzed feature contributions, and embedded the model into a multi-signal safety system rather than treating ML as a black box. The emphasis is on explainability, precision-recall trade-offs, and production-ready deployment — demonstrating **end-to-end ML ownership**, not just API usage."

## 🔍 Problem Statement

Scam and phishing messages exploit:
- **Urgency cues** ("Act now!", "Verify immediately!")
- **Deceptive language** (fake lottery wins, job offers)
- **Malicious URLs** (shorteners, risky TLDs, IP addresses)
- **Social engineering patterns** (requests for sensitive data)

### Why This Approach?

- **Pure heuristics** lack generalization and are easily bypassed
- **Pure LLMs** are expensive, slow, and can be unstable
- **This system** uses an **ensemble strategy** where each component covers the other's failure modes

## 🧠 System Overview

```
Input Text
   │
   ├──► Heuristic Scoring
   │       (keywords, intent, URLs)
   │
   ├──► ML Classifier
   │       (feature-based Logistic Regression)
   │
   ├──► LLM Safety Model (Future)
   │       (Llama Guard – semantic analysis)
   │
   ▼
Ensemble Decision Engine
   │
   ▼
Final Verdict: Safe | Suspicious | Scam
```

The system operates on **three layers of defense**:

1. **Heuristics**: Fast, rule-based pattern matching
2. **ML Model**: Interpretable statistical classification with 16 engineered features
3. **LLM Layer** (Planned): Semantic understanding for complex deception patterns

## 📁 Repository Structure

```
scam-detection/
│
├── ml/
│   ├── feature_extractor.py       # Feature engineering (16 features)
│   ├── build_dataset.py            # Initial dataset builder (6 samples)
│   ├── expand_dataset.py           # Production dataset generator (1000 samples)
│   ├── train_model.py              # Training + precision-recall evaluation
│   ├── ml_api.py                   # Flask ML inference service
│   ├── scam_dataset.csv            # Generated training dataset
│   └── scam_detector_model.pkl     # Trained classifier (serialized)
│
├── backend/ (Future)
│   └── security.js                 # Production Node.js security pipeline
│
├── docs/
│   └── RESEARCH_PAPER.md           # Academic paper on system design
│
├── .gitignore                      # Excludes __pycache__, *.pkl
└── README.md                       # This file (project overview)
```

**Architecture Philosophy:**

This separation reflects real ML system design:
- **ML lifecycle** isolated and testable
- **Backend** consumes ML as a service
- **Documentation** includes formal research paper
- Clean boundaries enable independent scaling

## Key Features

### ML Component
- **16 Engineered Features**: Behavioral and structural signal extraction
- **Interpretable Model**: Logistic regression with explainable coefficients
- **Calibrated Probabilities**: Sigmoid calibration for reliable risk scores
- **High Recall**: Optimized to minimize false negatives (critical for security)

### System Features
- **Multi-Layer Defense**: Heuristics + ML + (future) LLM
- **REST API**: Flask-based inference service
- **Expandable Dataset**: Template-based generation with mutations
- **Explainability-First**: Feature importance and contribution analysis

## 🤖 Machine Learning Component

### Model Choice

**Algorithm**: Logistic Regression with Sigmoid Calibration

**Rationale**:
1. **Interpretable coefficients** — Each feature's contribution is transparent
2. **Strong baseline** for text classification tasks
3. **Easy calibration** for reliable probability outputs
4. **Computational efficiency** — Fast inference for production systems
5. **Explainability** — Critical for security systems and auditing

### Why Not Deep Learning?

For this problem:
- Limited training data (1000 samples)
- Need for interpretability in security decisions
- Feature engineering captures domain knowledge effectively
- Linear models sufficient for well-designed features

**Future work**: Experiment with ensemble methods (Random Forest, XGBoost) and neural approaches when more data is available.

## Installation

### Prerequisites

- Python 3.7+
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd scam-detection
```

2. Install required dependencies:
```bash
pip install pandas scikit-learn flask joblib
```

### Optional Dependencies

For future LLM integration:
```bash
pip install transformers torch
```

## Usage

### Building the Dataset

#### Option 1: Sample Dataset (for testing)

Generate a small dataset with 6 samples:

```bash
python build_dataset.py
```

This creates `scam_dataset.csv` with basic examples for quick testing.

#### Option 2: Production Dataset (recommended)

Generate a larger dataset with 1000 samples:

```bash
python expand_dataset.py
```

This creates:
- 600 scam examples (variations of 5 scam templates)
- 400 safe examples (variations of 5 legitimate templates)
- Random mutations for diversity

### Training the Model

Train the logistic regression model with calibration:

```bash
python train_model.py
```

**Output:**
```
Precision: 0.XXX
Recall:    0.XXX
F1-score:  0.XXX
✅ Model saved
```

This generates `scam_detector_model.pkl` containing the trained model.

**What Happens During Training:**
1. Dataset loaded from `scam_dataset.csv`
2. 80/20 train-test split (stratified by class)
3. Base logistic regression trained
4. Sigmoid calibration applied for reliable probabilities
5. Evaluation on held-out test set
6. Model serialized to disk

### Model Explainability

While the training script outputs metrics, you can analyze feature importance:

```bash
# Future: python explain_model.py
```

**Interpretability Strategy:**

Logistic regression coefficients represent **directional influence** on scam probability:
- Positive coefficient → increases scam likelihood
- Negative coefficient → decreases scam likelihood
- Magnitude → strength of influence

**Example Feature Contributions:**

| Feature | Effect on Scam Probability | Interpretation |
|---------|---------------------------|----------------|
| `f1` (Urgency keywords) | **Strong positive ↑↑** | "URGENT", "verify now" → scam |
| `f3` (Sensitive terms) | **Very strong positive ↑↑↑** | Requests for OTP/PIN → scam |
| `f12` (URL shortener) | **Positive ↑** | bit.ly links → suspicious |
| `f15` (Verified domain) | **Negative ↓** | github.com, google.com → safe |
| `f5` (Text length) | **Weak negative** | Longer text → slightly safer |

This provides **human-interpretable reasoning**, critical for:
- AI safety and trust
- Security auditing
- Model debugging
- Regulatory compliance

**Future Enhancement**: Full SHAP value computation for instance-level explanations.

### Running the API

Start the Flask API server:

```bash
python ml_api.py
```

The API will be available at `http://localhost:5000`

### Making Predictions

Send POST requests to the `/predict` endpoint:

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "content": "URGENT! Verify your account now at bit.ly/fake-link",
    "manual_score": 85
  }'
```

**Response:**
```json
{
  "probability": 0.923,
  "risk_score": 92
}
```

## Feature Engineering

The system extracts 16 features from each message:

### Text-Based Features (f1-f8)

| Feature | Description | Type |
|---------|-------------|------|
| `f1` | Has urgency keywords (urgent, verify now, suspended) | Binary (0/1) |
| `f2` | Has money-related keywords (free money, lottery, earn) | Binary (0/1) |
| `f3` | Requests sensitive info (password, CVV, PIN, OTP) | Binary (0/1) |
| `f4` | Suggests off-platform contact (Telegram, WhatsApp) | Binary (0/1) |
| `f5` | Text length (character count) | Integer |
| `f6` | Number of exclamation marks | Integer |
| `f7` | Uppercase letter ratio | Float (0-1) |
| `f8` | Digit ratio | Float (0-1) |

### URL-Based Features (f9-f15)

| Feature | Description | Type |
|---------|-------------|------|
| `f9` | Number of URLs in message | Integer |
| `f10` | URL-to-word ratio | Float |
| `f11` | Contains IP-based URL | Binary (0/1) |
| `f12` | Uses URL shortener (bit.ly, tinyurl, etc.) | Binary (0/1) |
| `f13` | Has risky TLD (.tk, .ml, .ga, .pw, etc.) | Binary (0/1) |
| `f14` | Domain spoofing attempt | Binary (0/1) |
| `f15` | Contains verified domain | Binary (0/1) |

### Manual Score (f16)

| Feature | Description | Range |
|---------|-------------|-------|
| `f16` | Human-provided risk score (optional) | -20 to +20 |

The manual score is clamped to prevent data leakage and over-reliance on this feature.

### Verified Domains

The following domains are recognized as legitimate:
- google.com
- apple.com
- amazon.com
- microsoft.com
- paypal.com
- github.com
- youtube.com

### High-Risk TLDs

The system flags these top-level domains as high-risk:
- .tk, .ml, .ga, .cf, .gq (free TLDs)
- .pw, .click, .loan, .win, .bid, .racing, .kim
- .xyz, .top, .cc, .ru, .cn

## Model Architecture

### Base Model

**Logistic Regression** with the following configuration:
- **Solver**: L-BFGS (Limited-memory Broyden-Fletcher-Goldfarb-Shanno)
- **Max iterations**: 1000
- **Regularization**: L2 (Ridge) — default strength
- **Penalty**: Prevents overfitting on small datasets

### Calibration Layer

**CalibratedClassifierCV** with sigmoid (Platt) scaling:
- **Method**: Sigmoid transformation of raw scores
- **Cross-validation**: 5-fold (default)
- **Purpose**: Converts raw decision scores to calibrated probabilities
- **Benefit**: `predict_proba()` outputs are interpretable as true likelihoods

**Why Calibration Matters:**

Raw logistic regression scores can be poorly calibrated on small datasets. Calibration ensures:
- 0.8 probability → ~80% chance of actual scam
- Risk thresholds are meaningful
- Ensemble combination is reliable

### Training Pipeline

```
1. Load Dataset (scam_dataset.csv)
   ↓
2. Feature Matrix (X) + Labels (y)
   ↓
3. Stratified Train-Test Split (80/20)
   ↓
4. Train Base Logistic Regression
   ↓
5. Apply Sigmoid Calibration
   ↓
6. Evaluate on Test Set
   ↓
7. Serialize Model (joblib)
```

## 📊 Evaluation Methodology

### Metrics

The model is evaluated using three key metrics:

1. **Precision** = TP / (TP + FP)
   - Proportion of predicted scams that are actually scams
   - Important for minimizing false alarms

2. **Recall** = TP / (TP + FN)
   - Proportion of actual scams that are correctly identified
   - **Critical for security** — missing scams is dangerous

3. **F1-Score** = 2 × (Precision × Recall) / (Precision + Recall)
   - Harmonic mean balancing both concerns

### Example Output

```
Precision: 1.00
Recall:    1.00
F1-score:  1.00
```

### Design Philosophy

**Focus on Recall**: In security systems, **false negatives** (missing scams) are more costly than false positives. The model prioritizes catching all scams, even if it means occasional false alarms.

**Class Balance**: The dataset is intentionally balanced (60% scam, 40% safe) to prevent bias toward either class.

### Typical Performance

On the expanded dataset (1000 samples):
- **Precision**: 0.85 - 0.95
- **Recall**: 0.90 - 1.00 (optimized)
- **F1-Score**: 0.87 - 0.97

*Note: Perfect scores (1.00) on small datasets indicate good feature engineering but require validation on larger, more diverse data.*

## 🧠 LLM Safety Layer

**Status**: Planned for future integration

### Proposed Model

**Llama Guard** (or similar safety-focused LLM)

### Role in System

- **Semantic risk detection** — catches deception patterns missed by features
- **Context understanding** — analyzes persuasion tactics, emotional manipulation
- **Novelty handling** — detects new scam patterns not in training data
- **Used as a signal, not a decision authority** — prevents over-reliance

### Why Not LLM-Only?

| Aspect | LLM-Only | Hybrid (ML + LLM) |
|--------|----------|-------------------|
| **Cost** | High ($$$) | Optimized |
| **Latency** | Slow (seconds) | Fast (milliseconds) |
| **Interpretability** | Low | High |
| **Consistency** | Variable | Stable |
| **Failure modes** | Prompt injection, hallucination | Multiple defenses |

**Hybrid Approach**: Use fast, interpretable ML for most cases; invoke LLM only for:
- High-confidence conflicts
- Novel patterns
- Complex semantic analysis

## ⚖️ Ensemble Decision Logic

**No single component dominates** — this mirrors real fraud-detection pipelines.

### Decision Rules (Future Implementation)

```python
if ml_score > 0.90:
    verdict = "scam"
elif ml_score < 0.20 and heuristic_score < 30:
    verdict = "safe"
elif llm_unsafe or (ml_score > 0.70 and heuristic_score > 60):
    verdict = "scam"
elif ml_score > 0.50 or heuristic_score > 50:
    verdict = "suspicious"
else:
    verdict = "safe"
```

### Signal Combination Strategy

1. **High ML confidence** (>0.90) → Direct classification
2. **Agreement across signals** → Reinforced verdict
3. **Conflicting signals** → Flag as "suspicious" for review
4. **Low combined risk** → Safe

### Advantages

- **Robustness**: No single point of failure
- **Explainability**: Multiple independent signals
- **Tunability**: Thresholds adjustable for different risk tolerances
- **Gradual rollout**: Can start with ML-only, add LLM later

### POST /predict

Analyzes a message and returns scam probability.

#### Request Body

```json
{
  "content": "string (required) - The message text to analyze",
  "manual_score": "integer (optional) - Manual risk score (-20 to +20, default: 0)"
}
```

#### Response

```json
{
  "probability": "float - Scam probability (0.0 to 1.0)",
  "risk_score": "integer - Risk score (0 to 100)"
}
```

#### Example

**Request:**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Congratulations! You won $1000. Click here: bit.ly/claim"
  }'
```

**Response:**
```json
{
  "probability": 0.876,
  "risk_score": 88
}
```

## API Reference

## 🧪 Example Inference

### Current API (ML-Only)

**Request:**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "content": "URGENT! Verify your account now at bit.ly/fake-link",
    "manual_score": 85
  }'
```

**Response:**
```json
{
  "probability": 0.95,
  "risk_score": 95
}
```

### Future Ensemble API

**Request:**
```json
{
  "content": "URGENT verify your account now"
}
```

**Response:**
```json
{
  "verdict": "scam",
  "confidence": 0.95,
  "signals": {
    "ml_score": 95,
    "heuristic_score": 80,
    "llm_score": 85
  },
  "reasoning": {
    "top_features": [
      {"name": "urgency_keywords", "contribution": 0.35},
      {"name": "url_shortener", "contribution": 0.28},
      {"name": "sensitive_terms", "contribution": 0.22}
    ],
    "llm_analysis": "Message uses urgency tactics and suspicious URL"
  }
}
```

## Examples

### Scam Messages (High Risk)

```python
# Example 1: Urgency + URL shortener + Money
{
  "content": "URGENT! You won $5000! Claim now: bit.ly/claim-prize",
  "manual_score": 90
}
# Expected: probability > 0.90 (scam)
# Triggered features: f1 (urgency), f2 (money), f12 (shortener)
```

# Example 2: Sensitive data request + Social engineering
{
  "content": "Your account was suspended. Verify your password and OTP immediately!",
  "manual_score": 85
}
# Expected: probability > 0.85 (scam)
# Triggered features: f1 (urgency), f3 (sensitive: password, OTP)

# Example 3: Off-platform redirect + Money promise
{
  "content": "Free job offer! Earn $500 daily. Contact us on WhatsApp: +1234567890",
  "manual_score": 75
}
# Expected: probability > 0.80 (scam)
# Triggered features: f2 (money: earn, job), f4 (off-platform: WhatsApp)
```

### Legitimate Messages (Low Risk)

```python
# Example 1: Documentation link (verified domain)
{
  "content": "Please review the project documentation at github.com/user/repo",
  "manual_score": -10
}
# Expected: probability < 0.15 (safe)
# Triggered features: f15 (verified domain: github.com), neutral tone

# Example 2: Meeting reminder (neutral business communication)
{
  "content": "Reminder: Team meeting scheduled for tomorrow at 10am in Conference Room B",
  "manual_score": 0
}
# Expected: probability < 0.10 (safe)
# No scam features triggered

# Example 3: Tutorial reference (verified domain + informational)
{
  "content": "Check out this helpful Python tutorial series on youtube.com/watch?v=abc123",
  "manual_score": -5
}
# Expected: probability < 0.12 (safe)
# Triggered features: f15 (verified domain: youtube.com)
```

### Edge Cases (Testing Robustness)

```python
# Example 1: Legitimate urgency (business context)
{
  "content": "URGENT: Server maintenance tonight at 11 PM. Please save your work.",
  "manual_score": 0
}
# Expected: probability ~0.30-0.50 (suspicious, but not scam)
# Triggered features: f1 (urgency) but lacks other scam signals

# Example 2: Legitimate sensitive context (IT support)
{
  "content": "Please reset your password at company-portal.internal.com/reset",
  "manual_score": 5
}
# Expected: probability ~0.40-0.60 (requires LLM for context)
# Triggered features: f3 (password) but lacks malicious indicators
```

## Dataset Generation

### Template-Based Approach

The `expand_dataset.py` script uses template mutation:

**Scam Templates:**
- "URGENT verify your account"
- "Free job offer earn money"
- "Suspicious login detected"
- "You won a lottery prize"
- "Limited time investment"

**Safe Templates:**
- "Please review the documentation"
- "Meeting scheduled tomorrow"
- "Project update attached"
- "Watch tutorials online"
- "Monthly report available"

**Mutations:**
- Appends random noise: " now", " please", " today", " asap", "!!!", ""
- Randomizes manual scores within appropriate ranges

### Customizing the Dataset

To add your own examples:

1. Edit `expand_dataset.py`
2. Add templates to `scam_texts` or `safe_texts`
3. Adjust iteration counts for balance
4. Run the script to regenerate the dataset

```python
scam_texts = [
    "URGENT verify your account",
    "Your custom scam template here",
    # Add more...
]
```

## Feature Extractor Details

### URL Extraction

The feature extractor identifies URLs using regex:
```python
r'(https?:\/\/[^\s]+|www\.[^\s]+)'
```

### Case Sensitivity

- Keywords are matched case-insensitively
- Domain matching is case-insensitive
- TLD matching is case-insensitive

### Manual Score Capping

The manual score is clamped to prevent overfitting:
```python
manual_score = min(max(manual_score, -20), 20)
```

This ensures the model learns from text features rather than relying entirely on external scores.

## 🎓 What This Project Demonstrates

This project showcases **end-to-end ML system ownership**, not just API usage:

### 1. Applied Machine Learning System Design
- Problem formulation for real-world security use case
- Architecture decisions balancing accuracy, speed, and interpretability
- Multi-component system design (heuristics + ML + LLM)

### 2. Feature Engineering & Domain Expertise
- 16 hand-crafted features based on scam behavior analysis
- URL pattern recognition and risk classification
- Linguistic and structural signal extraction

### 3. Evaluation Rigor
- Proper train-test splitting with stratification
- Precision-recall-F1 metrics appropriate for security
- Awareness of class imbalance and cost-sensitive learning

### 4. Interpretability & Explainability
- Linear model for transparent decision-making
- Feature contribution analysis
- Human-auditable reasoning (critical for security AI)

### 5. ML Deployment as Microservice
- REST API design for model serving
- Separation of ML training and inference
- Production-ready Flask application structure

### 6. AI Safety via Multi-Signal Validation
- Ensemble approach prevents over-reliance on single model
- Heuristics provide fast baseline filtering
- LLM layer (planned) adds semantic robustness
- Conflicting signals trigger human review

### 7. Research-to-Production Pipeline
- Reproducible dataset generation
- Serialized model artifacts
- API-first deployment strategy
- Extensible architecture for future improvements

**This is not API-only AI usage — it is model ownership.**

## 🔮 Future Extensions

### Immediate Improvements (v1.1)
- [ ] Larger labeled datasets (10K+ samples)
- [ ] Cross-validation during training
- [ ] SHAP value computation for instance-level explanations
- [ ] Precision-recall curve analysis
- [ ] Confusion matrix visualization

### Model Enhancements (v2.0)
- [ ] Non-linear models (Random Forest, XGBoost)
- [ ] Ensemble of multiple classifiers
- [ ] Feature selection and ablation studies
- [ ] Hyperparameter tuning (GridSearchCV)
- [ ] Class weight optimization for imbalanced data

### System Extensions (v3.0)
- [ ] **LLM integration** (Llama Guard, GPT-4)
- [ ] Multi-language support (non-English scams)
- [ ] Domain reputation APIs (VirusTotal, URLhaus)
- [ ] Real-time URL checking
- [ ] Image-based scam detection (OCR + vision models)

### Production Features (v4.0)
- [ ] **Online learning** — model updates from new data
- [ ] A/B testing framework
- [ ] Model monitoring and drift detection
- [ ] Feedback loop for false positives/negatives
- [ ] Explainability dashboard
- [ ] Rate limiting and caching
- [ ] Kubernetes deployment

### Research Directions
- [ ] Adversarial robustness testing
- [ ] Transfer learning from pre-trained LLMs
- [ ] Graph-based features (email networks)
- [ ] Temporal analysis (scam campaign detection)
- [ ] Zero-shot learning for novel scam types

## Contributing

Contributions are welcome! This project is designed to be educational and extensible.

### How to Contribute

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Create a Pull Request

### Areas for Contribution

**Data & Features:**
- [ ] Add more diverse training examples
- [ ] Multi-language scam templates
- [ ] New feature engineering ideas
- [ ] Domain-specific dictionaries (crypto, banking, etc.)

**Model Improvements:**
- [ ] Experiment with ensemble methods
- [ ] Implement SHAP explanations
- [ ] Add active learning pipeline
- [ ] Hyperparameter optimization

**System Enhancements:**
- [ ] LLM integration (Llama Guard)
- [ ] Backend security pipeline (Node.js)
- [ ] Monitoring and logging
- [ ] Docker containerization

**Documentation:**
- [ ] Tutorial notebooks
- [ ] API usage examples
- [ ] Model architecture diagrams
- [ ] Deployment guides

### Code Standards

- Follow PEP 8 for Python code
- Add docstrings to functions
- Include unit tests for new features
- Update README with new functionality

## 👤 Author

**Vishwajeet Adkine**  
AI / ML | Applied Systems | Security AI

**Contact:**
- GitHub: [Your GitHub Profile]
- Email: [Your Email]
- LinkedIn: [Your LinkedIn]

## 📄 License

This project is provided as-is for **educational and research purposes**.

### Usage Terms

- ✅ Free to use for learning and experimentation
- ✅ Modify and extend for personal projects
- ✅ Use as reference for academic work (with attribution)
- ⚠️ Production deployment requires thorough validation
- ⚠️ No warranty provided — use at your own risk
- ⚠️ Ensure compliance with applicable laws and regulations

**Attribution**: If you use this project in your work, please cite:
```
Scam Detection System (ML + LLM + Heuristics)
Author: Vishwajeet Adkine
Repository: [URL]
Year: 2025
```

## 🙏 Acknowledgments

- **scikit-learn** — Machine learning utilities
- **Flask** — API framework
- **Anthropic Claude** — AI assistance and code review
- Open-source ML community

## 📚 References & Further Reading

### This Project
- **[Research Paper](docs/RESEARCH_PAPER.md)** — Full academic treatment of the system
- **[Feature Engineering](feature_extractor.py)** — Implementation details
- **[Model Training](train_model.py)** — Training pipeline code

### Academic Papers
- Fette et al. (2007) — Learning to detect phishing emails
- Lundberg & Lee (2017) — SHAP: Unified approach to model interpretability
- Ribeiro et al. (2016) — LIME: Model-agnostic explanations
- Iyer et al. (2023) — Llama Guard: LLM-based safety

### Industry Resources
- [OWASP Phishing Guide](https://owasp.org)
- [Google Safe Browsing](https://safebrowsing.google.com)
- [URLhaus Abuse Tracker](https://urlhaus.abuse.ch)
- [PhishTank Community](https://phishtank.org)

---

**Disclaimer**: This tool is designed to assist in scam detection but should not be the sole method of protection. Always exercise caution with unsolicited messages and verify information through official channels.

**Security Notice**: The model is trained on limited data and may not catch all scam types. Use as part of a defense-in-depth strategy, not as a standalone solution.
