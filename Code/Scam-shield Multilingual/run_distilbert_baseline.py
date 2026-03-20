"""
run_distilbert_baseline.py
===========================
PLACE THIS FILE IN: H:\google\STREMINI AI\ML- scam detector\
(the ROOT folder, NOT inside Scamshield multilingual)

BEFORE RUNNING, install dependencies:
    pip install transformers datasets torch scikit-learn requests

Then run:
    cd "H:\google\STREMINI AI\ML- scam detector"
    py run_distilbert_baseline.py
"""

import os
import io
import re
import json
import math
import zipfile
import warnings
import requests
import numpy as np

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Check all imports before doing anything else
# ─────────────────────────────────────────────────────────────────────────────

print("Checking imports...")

missing = []
try:
    import torch
    print(f"  torch         OK  (version {torch.__version__})")
except ImportError:
    missing.append("torch")
    print("  torch         MISSING")

try:
    import transformers
    print(f"  transformers  OK  (version {transformers.__version__})")
except ImportError:
    missing.append("transformers")
    print("  transformers  MISSING")

try:
    from datasets import Dataset
    print("  datasets      OK")
except ImportError:
    missing.append("datasets")
    print("  datasets      MISSING")

try:
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import f1_score, roc_auc_score, precision_recall_fscore_support
    print("  scikit-learn  OK")
except ImportError:
    missing.append("scikit-learn")
    print("  scikit-learn  MISSING")

if missing:
    print("\n" + "="*60)
    print("YOU ARE MISSING THESE PACKAGES:")
    for m in missing:
        print(f"  {m}")
    print("\nRUN THIS COMMAND FIRST:")
    print(f"  pip install {' '.join(missing)}")
    print("\nThen re-run this script.")
    print("="*60)
    exit(1)

print("\nAll imports OK. Starting...\n")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Download UCI SMS Spam dataset
# ─────────────────────────────────────────────────────────────────────────────

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_fscore_support

print("="*60)
print("Downloading UCI SMS Spam Collection (real data)...")
print("="*60)

texts, labels = [], []

# Try primary source
try:
    resp = requests.get(
        "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip",
        timeout=30
    )
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        with zf.open('SMSSpamCollection') as f:
            content = f.read().decode('utf-8', errors='replace')
    for line in content.strip().split('\n'):
        parts = line.split('\t', 1)
        if len(parts) == 2:
            labels.append(1 if parts[0].strip() == 'spam' else 0)
            texts.append(parts[1].strip())
    print(f"Downloaded from UCI: {len(texts)} messages")

except Exception as e:
    print(f"Primary source failed: {e}")
    print("Trying mirror...")
    try:
        import csv
        resp2 = requests.get(
            "https://raw.githubusercontent.com/mohitgupta-omg/Kaggle-SMS-Spam-Collection-Dataset-/master/spam.csv",
            timeout=30
        )
        resp2.raise_for_status()
        lines = resp2.text.strip().split('\n')
        reader = csv.reader(lines)
        next(reader)  # skip header
        for row in reader:
            if len(row) >= 2:
                labels.append(1 if row[0].strip() == 'spam' else 0)
                texts.append(row[1].strip())
        print(f"Downloaded from mirror: {len(texts)} messages")
    except Exception as e2:
        print(f"Mirror also failed: {e2}")

        # Last resort: check for local file
        local_path = 'SMSSpamCollection'
        if os.path.exists(local_path):
            print(f"Found local file: {local_path}")
            with open(local_path, 'r', encoding='utf-8', errors='replace') as f:
                for line in f:
                    parts = line.strip().split('\t', 1)
                    if len(parts) == 2:
                        labels.append(1 if parts[0] == 'spam' else 0)
                        texts.append(parts[1])
            print(f"Loaded from local file: {len(texts)} messages")
        else:
            print("\nCould not download dataset and no local file found.")
            print("Please download manually:")
            print("  https://archive.ics.uci.edu/dataset/228/sms+spam+collection")
            print("  Extract 'SMSSpamCollection' file to this folder and re-run.")
            exit(1)

if len(texts) == 0:
    print("No data loaded. Exiting.")
    exit(1)

spam_count = sum(labels)
ham_count = len(labels) - spam_count
print(f"\nDataset: {len(texts)} messages")
print(f"  Spam: {spam_count} ({spam_count/len(texts)*100:.1f}%)")
print(f"  Ham:  {ham_count}  ({ham_count/len(texts)*100:.1f}%)")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Split into train / test (same split as load_real_data.py uses)
# ─────────────────────────────────────────────────────────────────────────────

X_tr, X_te, y_tr, y_te = train_test_split(
    texts, labels,
    test_size=0.20,
    stratify=labels,
    random_state=42
)

print(f"\nSplit: Train={len(X_tr)}, Test={len(X_te)}")
print(f"  Train spam: {sum(y_tr)}, Test spam: {sum(y_te)}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: Tokenize with DistilBERT
# ─────────────────────────────────────────────────────────────────────────────

print("\nLoading DistilBERT tokenizer...")
print("(This downloads ~250MB the first time — please wait)")

tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

def tokenize_batch(batch):
    return tokenizer(
        batch['text'],
        truncation=True,
        padding='max_length',
        max_length=128
    )

print("Tokenizing training data...")
train_ds = Dataset.from_dict({'text': X_tr, 'label': y_tr})
train_ds = train_ds.map(tokenize_batch, batched=True, batch_size=256)
train_ds = train_ds.remove_columns(['text'])
train_ds.set_format('torch')

print("Tokenizing test data...")
test_ds = Dataset.from_dict({'text': X_te, 'label': y_te})
test_ds = test_ds.map(tokenize_batch, batched=True, batch_size=256)
test_ds = test_ds.remove_columns(['text'])
test_ds.set_format('torch')

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: Fine-tune DistilBERT
# ─────────────────────────────────────────────────────────────────────────────

print("\nLoading DistilBERT model...")
model = AutoModelForSequenceClassification.from_pretrained(
    'distilbert-base-uncased',
    num_labels=2
)

# Check if GPU is available
device_info = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
print(f"Training on: {device_info}")
if not torch.cuda.is_available():
    print("  No GPU detected. Training on CPU will take ~15-30 minutes.")
    print("  If you have a GPU, make sure you installed: pip install torch --index-url https://download.pytorch.org/whl/cu118")

training_args = TrainingArguments(
    output_dir='./distilbert_results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir='./distilbert_logs',
    logging_steps=50,
    evaluation_strategy='epoch',
    save_strategy='no',
    load_best_model_at_end=False,
    seed=42,
    report_to='none',           # disables wandb/tensorboard noise
    no_cuda=not torch.cuda.is_available(),
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    f1 = f1_score(labels, preds)
    return {'f1': f1}

print("\nStarting fine-tuning (3 epochs)...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    compute_metrics=compute_metrics,
)

trainer.train()

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: Evaluate and print results
# ─────────────────────────────────────────────────────────────────────────────

print("\nEvaluating on test set...")
preds_output = trainer.predict(test_ds)
logits = preds_output.predictions
pred_labels = np.argmax(logits, axis=1)
pred_probs = torch.softmax(torch.tensor(logits), dim=1).numpy()[:, 1]

f1  = f1_score(y_te, pred_labels)
auc = roc_auc_score(y_te, pred_probs)
p, r, _, _ = precision_recall_fscore_support(y_te, pred_labels, average='binary', zero_division=0)

print("\n" + "="*60)
print("DISTILBERT RESULTS (add these to Table III in your paper)")
print("="*60)
print(f"  F1        = {f1:.4f}")
print(f"  AUC       = {auc:.4f}")
print(f"  Recall    = {r:.4f}")
print(f"  Precision = {p:.4f}")
print(f"  Test set  = {len(y_te)} messages ({sum(y_te)} spam)")
print()
print("Model size comparison for your paper:")
print("  DistilBERT fine-tuned  = ~250 MB")
print("  ScamShield GBM         = ~2 MB")
print("  Ratio                  = ~125x smaller")
print()

# Save results to JSON
results = {
    'model': 'distilbert-base-uncased',
    'dataset': 'UCI SMS Spam Collection',
    'test_size': len(y_te),
    'test_spam': int(sum(y_te)),
    'f1':        round(float(f1),  4),
    'roc_auc':   round(float(auc), 4),
    'recall':    round(float(r),   4),
    'precision': round(float(p),   4),
    'model_size_mb': 250,
    'epochs': 3,
    'seed': 42,
}

os.makedirs('eval_results', exist_ok=True)
with open('eval_results/distilbert_evaluation.json', 'w') as f:
    json.dump(results, f, indent=2)

print("Saved to eval_results/distilbert_evaluation.json")
print()
print("="*60)
print("COPY THESE NUMBERS INTO YOUR PAPER (Table III):")
print("="*60)
print()
print(f"  | DistilBERT (fine-tuned, UCI SMS) | {f1:.4f} | {auc:.4f} | {r:.4f} | {p:.4f} |")
print()
print("Your ScamShield numbers for comparison (from Table III):")
print("  | ScamShield GBM (synthetic CV)    | 0.9969 | 0.9812 | 0.9938 | 1.0000 |")
print("  | ScamShield GBM (UCI SMS CV)      | [see load_real_data.py output]    |")
print()
print("DONE.")