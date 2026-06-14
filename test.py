"""
test.py
Loads savedmodel.pth and evaluates its accuracy on the held-out test set.
"""

import joblib
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ── 1. Reload the same split (same random_state) ─────────────────────────────
print("Loading Olivetti faces dataset...")
data = fetch_olivetti_faces(shuffle=True, random_state=42)
X, y = data.data, data.target

_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

# ── 2. Load model ────────────────────────────────────────────────────────────
MODEL_PATH = "savedmodel.pth"
print(f"Loading model from {MODEL_PATH}...")
clf = joblib.load(MODEL_PATH)

# ── 3. Evaluate ──────────────────────────────────────────────────────────────
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nTest Accuracy: {accuracy * 100:.2f}%")
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred))
