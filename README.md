# Stremini — implement

# Research Work Ecosystem

Welcome to the **Research Work Ecosystem** repository. This monorepo serves as a dedicated workspace for advanced machine learning research, combining theoretical explorations of fundamental AI algorithms with practical, end-to-end machine learning pipelines.

Currently, this repository houses three primary projects: an end-to-end **ML Scam Detector**, a theoretical deep-dive into **Gradient Descent as a Dynamical System**, and mathematical research on the **Spectral Properties of Attention Matrices in Transformers**.

---

## Table of Contents

- [Projects Overview](#projects-overview)
  - [1. ML Scam Detector](#1-ml-scam-detector)
  - [2. Gradient Descent as a Dynamical System](#2-gradient-descent-as-a-dynamical-system)
  - [3. Spectral Properties of Attention Matrices](#3-spectral-properties-of-attention-matrices)
- [Project Structure](#project-structure)
- [Technology Stack](#technology-stack)
- [Detailed Features](#detailed-features)
  - [ML Scam Detector Pipeline](#ml-scam-detector-pipeline)
  - [Theoretical Research Papers](#theoretical-research-papers)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Running the ML Scam Detector](#running-the-ml-scam-detector)
  - [Viewing the Research Papers](#viewing-the-research-papers)
- [Contributing](#contributing)
- [License](#license)

---

## Projects Overview

### 1. ML Scam Detector
A comprehensive, end-to-end machine learning pipeline designed to detect scam/phishing URLs and malicious web content. This project covers everything from dataset generation and feature extraction to model training, evaluation, and API deployment.

### 2. Gradient Descent as a Dynamical System
A theoretical research paper/interactive document that analyzes the convergence properties and behaviors of Gradient Descent through the mathematical lens of dynamical systems.

### 3. Spectral Properties of Attention Matrices
A deep mathematical exploration into the core of Transformer architectures. This research investigates the eigenvalues, eigenvectors, and spectral gaps of attention matrices to better understand information flow and rank collapse in Large Language Models (LLMs).

---

## Project Structure

```text
.
├── ML_-Scam detector/
│   ├── api.py                  # API endpoints for model inference
│   ├── ml_api.py               # Core ML serving logic
│   ├── build_dataset.py        # Scripts for initial dataset construction
│   ├── expand_dataset.py       # Data augmentation and expansion utilities
│   ├── feature_extractor.py    # URL and HTML feature engineering logic
│   ├── train_model.py          # Model training pipeline
│   ├── evaluate_model.py       # Metrics and model evaluation script
│   ├── write_trainer.py        # Utility for model serialization
│   ├── scam_detector_model.pkl # Serialized trained model
│   └── model_metadata.json     # Metadata and evaluation results (eval_results.json)
│
├── Gradient Descent as a Dynamical System/
│   └── gradient_descent_dynamical.html  # Interactive research paper
│
└── Spectral Properties of Attention Matrices in Transformers/
    └── spectral_attention_matrices.html  # Interactive research paper
```

---

## Technology Stack

### Machine Learning & Data Processing
- **Language:** Python 3.8+
- **Data Processing:** Pandas, NumPy
- **Model Serialization:** Pickle (.pkl)

### API Serving
- FastAPI / Flask (via `api.py` and `ml_api.py`)

### Research & Documentation
- **Format:** HTML5, CSS3, JavaScript
- **Mathematics Rendering:** MathJax / LaTeX (embedded in HTML)

---

## Detailed Features

### ML Scam Detector Pipeline
- **Data Engineering** (`build_dataset.py`, `expand_dataset.py`): Tools to aggregate, clean, and augment datasets containing benign and malicious URLs/HTML content.
- **Feature Extraction** (`feature_extractor.py`): Custom parsers that extract lexical properties from URLs (e.g., length, entropy, presence of IP addresses) and structural features from HTML payloads.
- **Model Training & Evaluation** (`train_model.py`, `evaluate_model.py`): Scripts to train the classifier, tune hyperparameters, and output performance metrics (Precision, Recall, F1‑Score) into JSON format (`eval_results.json`).
- **Inference API** (`api.py`, `ml_api.py`): RESTful API wrappers that load the serialized .pkl model and provide real‑time scam probability scoring for incoming requests.

### Theoretical Research Papers
- **Dynamical Systems Analysis:** Bridges the gap between continuous‑time differential equations and discrete‑time optimization steps in neural networks.
- **Attention Mechanism Insights:** Provides rigorous proofs and visualizations regarding how attention weights distribute, helping to explain phenomena like over‑smoothing in deep transformers.
- **Accessible Format:** Packaged as standalone, rich HTML documents that require no local LaTeX environments to view.

---

## Getting Started

### Prerequisites
For the machine learning project, ensure you have Python installed along with the required dependencies:

```bash
pip install -r "ML_-Scam detector/requirements.txt"
# (Or manually install pandas, scikit-learn, flask/fastapi)
```

### Running the ML Scam Detector
1. **Train the Model (Optional)**
   If you wish to retrain the model on new data:

   ```bash
   cd "ML_-Scam detector"
   python build_dataset.py
   python feature_extractor.py
   python train_model.py
   python evaluate_model.py
   ```

2. **Start the Inference API**
   To run the pre‑trained model (`scam_detector_model.pkl`) locally:

   ```bash
   cd "ML_-Scam detector"
   python api.py
   # The API will typically start on http://localhost:5000 or http://localhost:8000
   ```

3. **Test the API**
   Send a POST request to the API with a target URL:

   ```bash
   curl -X POST http://localhost:5000/predict \
        -H "Content-Type: application/json" \
        -d '{"url": "http://suspicious-domain-login.com"}'
   ```

### Viewing the Research Papers
No special setup is required to read the theoretical papers. Simply open the HTML files in any modern web browser:

- **Mac:** `open "Gradient Descent as a Dynamical System/gradient_descent_dynamical.html"`
- **Linux:** `xdg-open "Spectral Properties of Attention Matrices in Transformers/spectral_attention_matrices.html"`
- **Windows:** Double‑click the respective `.html` files in File Explorer.

---

## Contributing
1. **Fork the Project**
2. **Create your Feature Branch**  
   ```bash
   git checkout -b feature/NewResearch
   ```
3. **Commit your Changes**  
   ```bash
   git commit -m 'Add new dataset expansion logic'
   ```
4. **Push to the Branch**  
   ```bash
   git push origin feature/NewResearch
   ```
5. **Open a Pull Request**

---

## License
Distributed under the MIT License. See `LICENSE` for more information.
