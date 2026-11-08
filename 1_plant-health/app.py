import io
from flask import Flask, request, jsonify, render_template
from PIL import Image, UnidentifiedImageError
from transformers import pipeline

app = Flask(__name__)

# Pre-trained model (MobileNetV2 fine-tuned on PlantVillage, 38 classes).
# Downloaded automatically on the first run (~10 MB), no training needed.
MODEL_ID = "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"
classifier = pipeline("image-classification", model=MODEL_ID)

TIPS = {
    True: "This leaf looks healthy. Keep up regular watering, good sunlight and airflow.",
    False: "Signs of disease detected. Isolate the plant, remove affected leaves, "
           "avoid wetting the foliage and consider a suitable treatment or a local expert's advice.",
}


def clean(text):
    return text.replace("_", " ").replace(",", "").strip()


def split_label(label):
    """'Tomato___Late_blight' -> ('Tomato', 'Late blight')"""
    plant, sep, condition = label.partition("___")
    if not sep:
        return "", clean(label)
    return clean(plant), clean(condition)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("image")
    if file is None:
        return jsonify(error="No image uploaded."), 400
    try:
        img = Image.open(io.BytesIO(file.read())).convert("RGB")
    except (UnidentifiedImageError, OSError):
        return jsonify(error="That file is not a valid image."), 400

    results = classifier(img, top_k=3)
    top = results[0]
    plant, condition = split_label(top["label"])
    healthy = "healthy" in top["label"].lower()

    return jsonify(
        plant=plant,
        condition=condition,
        healthy=healthy,
        confidence=round(top["score"] * 100, 1),
        tip=TIPS[healthy],
        others=[
            {"name": " - ".join(p for p in split_label(r["label"]) if p),
             "score": round(r["score"] * 100, 1)}
            for r in results
        ],
    )


if __name__ == "__main__":
    app.run(debug=False)