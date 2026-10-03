import base64
import io

from flask import Flask, jsonify, render_template, request
from PIL import Image, UnidentifiedImageError

from captioner import Captioner
from segmenter import Segmenter
from nlp import keywords, scene_summary

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB

print("Loading models (first run downloads them, please wait)...")
captioner = Captioner()
segmenter = Segmenter()
print("Models ready.")


def to_b64(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    file = request.files.get("image")
    if not file:
        return jsonify(error="No image uploaded."), 400
    try:
        image = Image.open(file.stream).convert("RGB")
    except (UnidentifiedImageError, OSError):
        return jsonify(error="That file is not a valid image."), 400

    image.thumbnail((1024, 1024))
    caption = captioner.caption(image)
    overlay, mask, objects = segmenter.segment(image)
    return jsonify(
        caption=caption,
        keywords=keywords(caption),
        objects=objects,
        summary=scene_summary(caption, objects),
        original=to_b64(image),
        overlay=to_b64(overlay),
        mask=to_b64(mask),
    )


if __name__ == "__main__":
    app.run(debug=False, port=5000)
