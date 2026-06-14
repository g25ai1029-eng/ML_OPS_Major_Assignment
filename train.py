"""
train.py
Loads the Olivetti faces dataset, splits it 70/30, trains a
DecisionTreeClassifier, and saves the model to savedmodel.pth using joblib.
"""

import joblib
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# ── 1. Load dataset ──────────────────────────────────────────────────────────
print("Loading Olivetti faces dataset...")
data = fetch_olivetti_faces(shuffle=True, random_state=42)
X, y = data.data, data.target
print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features, "
      f"{len(set(y))} classes")

# ── 2. Split ─────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
print(f"Train size : {len(X_train)} samples")
print(f"Test  size : {len(X_test)}  samples")

# ── 3. Train ─────────────────────────────────────────────────────────────────
print("Training DecisionTreeClassifier...")
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)
print("Training complete.")

# ── 4. Save ──────────────────────────────────────────────────────────────────
MODEL_PATH = "savedmodel.pth"
joblib.dump(clf, MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")
