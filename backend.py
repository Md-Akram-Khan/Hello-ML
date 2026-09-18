"""Minimal Flask API for the existing cat versus non-cat classifier."""

from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image, UnidentifiedImageError

from lr_utils import load_dataset

BASE_DIR = Path(__file__).resolve().parent
IMAGE_SIZE = 64
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "avif"}

app = Flask(__name__)
CORS(app)


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -500, 500)
    return 1 / (1 + np.exp(-values))


def train_classifier() -> tuple[np.ndarray, float]:
    train_images, train_labels, _, _, _ = load_dataset()
    features = train_images.reshape(train_images.shape[0], -1).T / 255.0
    labels = train_labels
    weights = np.zeros((features.shape[0], 1))
    bias = 0.0
    sample_count = features.shape[1]

    for _ in range(2000):
        probabilities = sigmoid(np.dot(weights.T, features) + bias)
        error = probabilities - labels
        weights -= 0.005 * np.dot(features, error.T) / sample_count
        bias -= 0.005 * np.sum(error) / sample_count

    return weights, bias


WEIGHTS, BIAS = train_classifier()


def classify_image(image: Image.Image) -> tuple[str, float]:
    resized = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    pixels = np.asarray(resized, dtype=np.float32).reshape(-1, 1) / 255.0
    probability = float(sigmoid(np.dot(WEIGHTS.T, pixels) + BIAS).item())
    prediction = "cat" if probability >= 0.5 else "not_cat"
    confidence = probability if prediction == "cat" else 1 - probability
    return prediction, confidence


@app.get("/health")
def health() -> tuple[dict, int]:
    return {"status": "ok"}, 200


@app.post("/predict")
def predict() -> tuple[dict, int]:
    uploaded_file = request.files.get("file")
    if uploaded_file is None or not uploaded_file.filename:
        return jsonify({"error": "Please upload an image in the file field."}), 400

    extension = Path(uploaded_file.filename).suffix.lower().lstrip(".")
    if extension not in ALLOWED_EXTENSIONS:
        return jsonify({"error": "Supported formats are JPG, JPEG, PNG, and AVIF."}), 400

    try:
        image = Image.open(uploaded_file.stream)
        prediction, confidence = classify_image(image)
    except (UnidentifiedImageError, OSError):
        return jsonify({"error": "The uploaded file is not a valid image."}), 400

    return jsonify({"prediction": prediction, "confidence": confidence}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
