import sys, os, io, csv, json, zipfile, warnings, requests
import numpy as np
warnings.filterwarnings('ignore')
os.environ["TOKENIZERS_PARALLELISM"] = "false"

print("Python:", sys.version[:6], "|", sys.executable)
print()

if "3.13" in sys.version:
    print("WRONG PYTHON. Run this command instead:")
    print()
    print("C:\\Users\\lenovo\\AppData\\Local\\Programs\\Python\\Python310\\python.exe run_distilbert_baseline.py")
    print()
    sys.exit(1)

missing = []
try:
    import torch
    print("torch        OK", torch.__version__)
except:
    missing.append("torch"); print("torch        MISSING")

try:
    import transformers
    print("transformers OK", transformers.__version__)
except:
    missing.append("transformers"); print("transformers MISSING")

try:
    from datasets import Dataset
    print("datasets     OK")
except:
    missing.append("datasets"); print("datasets     MISSING")

try:
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import f1_score, roc_auc_score, precision_recall_fscore_support
    print("sklearn      OK")
except:
    missing.append("scikit-learn"); print("sklearn      MISSING")

if missing:
    print()
    print("Install missing packages with this command:")
    print()
    print('"' + sys.executable + '" -m pip install ' + " ".join(missing))
    print()
    sys.exit(1)

print()
print("All OK. Starting...\n")

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_fscore_support

# ── Step 1: Download data ──────────────────────────────────────────────────
print("=" * 55)
print("Step 1 of 5 — Downloading UCI SMS Spam dataset")
print("=" * 55)

texts, labels = [], []

try:
    print("  Connecting to UCI archive...")
    resp = requests.get(
        "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip",
        timeout=30
    )
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        content = zf.open('SMSSpamCollection').read().decode('utf-8', errors='replace')
    for line in content.strip().split('\n'):
        parts = line.split('\t', 1)
        if len(parts) == 2:
            labels.append(1 if parts[0].strip() == 'spam' else 0)
            texts.append(parts[1].strip())
    print("  Downloaded from UCI:", len(texts), "messages")

except Exception as e:
    print("  UCI failed:", str(e)[:60])
    print("  Trying GitHub mirror...")
    try:
        resp2 = requests.get(
            "https://raw.githubusercontent.com/mohitgupta-omg/Kaggle-SMS-Spam-Collection-Dataset-/master/spam.csv",
            timeout=30
        )
        resp2.raise_for_status()
        reader = csv.reader(resp2.text.strip().split('\n'))
        next(reader)
        for row in reader:
            if len(row) >= 2:
                labels.append(1 if row[0].strip() == 'spam' else 0)
                texts.append(row[1].strip())
        print("  Downloaded from mirror:", len(texts), "messages")
    except Exception as e2:
        if os.path.exists('SMSSpamCollection'):
            with open('SMSSpamCollection', 'r', encoding='utf-8', errors='replace') as f:
                for line in f:
                    parts = line.strip().split('\t', 1)
                    if len(parts) == 2:
                        labels.append(1 if parts[0] == 'spam' else 0)
                        texts.append(parts[1])
            print("  Loaded local file:", len(texts), "messages")
        else:
            print()
            print("CANNOT DOWNLOAD. Options:")
            print("  1. Check your internet connection")
            print("  2. Download manually: https://archive.ics.uci.edu/dataset/228/sms+spam+collection")
            print("     Extract 'SMSSpamCollection' file to this folder, then re-run")
            sys.exit(1)

print()
print("  Total:", len(texts), "messages")
print("  Spam: ", sum(labels), "(" + str(round(sum(labels)/len(labels)*100, 1)) + "%)")
print("  Ham:  ", len(labels) - sum(labels))

# ── Step 2: Split ──────────────────────────────────────────────────────────
print()
print("=" * 55)
print("Step 2 of 5 — Splitting data (80% train / 20% test)")
print("=" * 55)

X_tr, X_te, y_tr, y_te = train_test_split(
    texts, labels, test_size=0.20, stratify=labels, random_state=42
)
print("  Train:", len(X_tr), "| Test:", len(X_te))

# ── Step 3: Tokenize ───────────────────────────────────────────────────────
print()
print("=" * 55)
print("Step 3 of 5 — Loading tokenizer (downloads ~67MB once)")
print("=" * 55)

tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

def tok(batch):
    return tokenizer(batch['text'], truncation=True, padding='max_length', max_length=128)

print("  Tokenizing training set...")
train_ds = Dataset.from_dict({'text': X_tr, 'label': y_tr})
train_ds = train_ds.map(tok, batched=True, batch_size=256)
train_ds = train_ds.remove_columns(['text'])
train_ds.set_format('torch')

print("  Tokenizing test set...")
test_ds = Dataset.from_dict({'text': X_te, 'label': y_te})
test_ds = test_ds.map(tok, batched=True, batch_size=256)
test_ds = test_ds.remove_columns(['text'])
test_ds.set_format('torch')

print("  Done")

# ── Step 4: Train ──────────────────────────────────────────────────────────
print()
print("=" * 55)
device_str = "GPU" if torch.cuda.is_available() else "CPU (no GPU — expect 20-40 min)"
print("Step 4 of 5 — Training DistilBERT   Device:", device_str)
print("=" * 55)

model = AutoModelForSequenceClassification.from_pretrained(
    'distilbert-base-uncased', num_labels=2
)

training_args = TrainingArguments(
    output_dir='./distilbert_results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    warmup_steps=100,
    weight_decay=0.01,
    logging_steps=50,
    eval_strategy='epoch',
    save_strategy='no',
    seed=42,
    report_to='none',
    use_cpu=not torch.cuda.is_available(),
)

def metrics_fn(ep):
    logits, lbls = ep
    return {'f1': f1_score(lbls, np.argmax(logits, axis=1))}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    compute_metrics=metrics_fn,
)

trainer.train()

# ── Step 5: Evaluate ───────────────────────────────────────────────────────
print()
print("=" * 55)
print("Step 5 of 5 — Evaluating on test set")
print("=" * 55)

out = trainer.predict(test_ds)
pred_labels = np.argmax(out.predictions, axis=1)
pred_probs  = torch.softmax(torch.tensor(out.predictions, dtype=torch.float32), dim=1).numpy()[:, 1]

f1  = f1_score(y_te, pred_labels)
auc = roc_auc_score(y_te, pred_probs)
p, r, _, _ = precision_recall_fscore_support(y_te, pred_labels, average='binary', zero_division=0)

print()
print("=" * 55)
print("RESULTS — copy these into your paper (Table III)")
print("=" * 55)
print()
print("  F1        =", round(f1,  4))
print("  AUC       =", round(auc, 4))
print("  Recall    =", round(r,   4))
print("  Precision =", round(p,   4))
print()
print("  DistilBERT model size  = ~250 MB")
print("  ScamShield GBM size    = ~2 MB  (125x smaller)")
print()

os.makedirs('eval_results', exist_ok=True)
with open('eval_results/distilbert_evaluation.json', 'w') as f:
    json.dump({
        'f1':        round(float(f1),  4),
        'roc_auc':   round(float(auc), 4),
        'recall':    round(float(r),   4),
        'precision': round(float(p),   4),
    }, f, indent=2)

print("  Saved to eval_results/distilbert_evaluation.json")
print()
print("DONE. Come back to Claude and paste these numbers.")