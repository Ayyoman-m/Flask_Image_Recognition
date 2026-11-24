"""Flask application routes for Hand Digit Image Recognition."""
import os

from flask import Flask, render_template, request
from flask_wtf.csrf import CSRFProtect

from model import preprocess_img, predict_result

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY") or os.urandom(32)
csrf = CSRFProtect(app)


@app.route("/")
def main():
    """Render the home page with the image upload form."""
    return render_template("index.html")


@app.route("/prediction", methods=["POST"])
def predict_image_file():
    """Process the uploaded image and render the prediction or an error message."""
    if "file" not in request.files or request.files["file"].filename == "":
        return render_template("result.html", err="File cannot be processed.")
    try:
        img = preprocess_img(request.files["file"].stream)
        pred = predict_result(img)
        return render_template("result.html", predictions=str(pred))
    except Exception:
        return render_template("result.html", err="File cannot be processed.")


if __name__ == "__main__":
    app.run(port=9000, debug=os.environ.get("FLASK_DEBUG") == "1")
