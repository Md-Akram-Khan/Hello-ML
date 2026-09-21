from pathlib import Path
import os
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image, UnidentifiedImageError
from linear_regression import (
    model,
    num_px,
    predict as predict_model,
    sigmoid,
    train_set_x,
    train_set_y_data,
)
BASE_DIR = Path(__file__).resolve().parent
IMAGE_SIZE = num_px
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "avif"}

app = Flask(__name__)
CORS(app)


MODEL = model(train_set_x, train_set_y_data, train_set_x, train_set_y_data,
              iterations=2000, alpha=0.01)


def classify_image(image: Image.Image) -> tuple[str, float]:
    resized = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    pixels = np.asarray(resized, dtype=np.float32).reshape(-1, 1) / 255.0
    prediction_value = predict_model(MODEL["w"], MODEL["b"], pixels)
    probability = float(
        sigmoid(np.dot(MODEL["w"].T, pixels) + MODEL["b"]).item()
    )
    prediction = "cat" if int(prediction_value.item()) == 1 else "not_cat"
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
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=os.environ.get("FLASK_DEBUG", "0") == "1",
    )
