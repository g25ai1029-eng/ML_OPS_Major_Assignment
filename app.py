"""
app.py
Flask web application that accepts a face image upload,
pre-processes it to match the Olivetti faces format,
and returns the predicted person class.
"""

import io
import os

import joblib
import numpy as np
from flask import Flask, render_template, request, jsonify
from PIL import Image

app = Flask(__name__)

MODEL_PATH = os.getenv("MODEL_PATH", "savedmodel.pth")
model = joblib.load(MODEL_PATH)

IMAGE_SIZE = (64, 64)   # Olivetti faces are 64×64 pixels


def preprocess_image(file_storage) -> np.ndarray:
    """Convert an uploaded image to the 1-D feature vector the model expects."""
    img = Image.open(io.BytesIO(file_storage.read()))
    img = img.convert("L")                  # grayscale
    img = img.resize(IMAGE_SIZE)            # 64×64
    arr = np.array(img, dtype=np.float64)
    arr = arr / 255.0                       # normalise to [0, 1]
    return arr.flatten().reshape(1, -1)     # shape (1, 4096)


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    try:
        features = preprocess_image(file)
        predicted_class = int(model.predict(features)[0])
        probabilities = model.predict_proba(features)[0]
        confidence = float(probabilities[predicted_class]) * 100

        return jsonify({
            "predicted_class": predicted_class,
            "label": f"Person {predicted_class}",
            "confidence": f"{confidence:.1f}%"
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
