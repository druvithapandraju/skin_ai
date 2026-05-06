from flask import Flask, request, jsonify, render_template
import os
import numpy as np
from datetime import datetime
from PIL import Image

# TensorFlow
import tensorflow as tf
from tensorflow.keras.models import load_model

app = Flask(__name__)

# 📁 Upload folder
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# 📦 Load trained model
MODEL_PATH = "model.h5"

if not os.path.exists(MODEL_PATH):
    raise Exception("❌ model.h5 not found. Train model first.")

model = load_model(MODEL_PATH)

# 🏷️ Class labels (match your training)
CLASS_NAMES = [
    "Melanoma ⚠️",
    "Benign ✅",
    "Basal Cell ⚠️"
]


# 🌐 Home route
@app.route("/")
def home():
    return render_template("index.html")


# 🧠 Image preprocessing
def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((224, 224))  # MobileNetV2 size
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img


# 🤖 Analyze API
@app.route("/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    try:
        # Preprocess
        img = preprocess_image(filepath)

        # Prediction
        preds = model.predict(img)
        class_index = np.argmax(preds)
        confidence = float(np.max(preds))

        result = CLASS_NAMES[class_index]

        return jsonify({
            "result": result,
            "confidence": round(confidence, 2),
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# 📄 Report API
@app.route("/report", methods=["POST"])
def report():
    data = request.get_json()

    return jsonify({
        "name": data.get("name"),
        "age": data.get("age"),
        "gender": data.get("gender"),
        "result": data.get("result"),
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })


# ▶️ Run
if __name__ == "__main__":
    app.run(debug=True)
