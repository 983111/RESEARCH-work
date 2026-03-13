import csv
from feature_extractor import extract_features

# -------------------------------
# SAMPLE DATA (TEMP)
# Replace later with real dataset
# -------------------------------

samples = [
    ("URGENT! Verify your account now at bit.ly/fake-paypal", 85, 1),
    ("Congratulations you won a lottery prize", 80, 1),
    ("Free job offer earn daily income", 75, 1),
    ("Read the documentation on github.com", -50, 0),
    ("Watch tutorials on youtube.com", -40, 0),
    ("Meeting scheduled tomorrow at 10am", 0, 0)
]

with open("scam_dataset.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)

    # Header
    writer.writerow([f"f{i}" for i in range(1, 15)] + ["label"])

    for text, manual_score, label in samples:
        features = extract_features(text, manual_score)
        writer.writerow(features + [label])

print("✅ Dataset created: scam_dataset.csv")
