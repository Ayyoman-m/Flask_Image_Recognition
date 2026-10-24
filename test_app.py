"""Automated pytest suite covering routes, edge cases, and model utilities."""
import io
from unittest.mock import patch
import numpy as np
import pytest
from PIL import Image
from app import app
from model import preprocess_img, predict_result


@pytest.fixture(name="client")
def fixture_client():
    """Configure Flask test client."""
    app.config["TESTING"] = True
    with app.test_client() as test_client:
        yield test_client


def make_image_bytes():
    """Generate an in-memory PNG image."""
    buf = io.BytesIO()
    Image.new("RGB", (300, 300), color=(120, 150, 200)).save(buf, format="PNG")
    buf.seek(0)
    return buf


def test_home_page_loads(client):
    """Home page returns 200 and renders the upload form."""
    response = client.get("/")
    assert response.status_code == 200
    assert b"Hand Sign Digit" in response.data


@patch("app.predict_result", return_value=7)
def test_valid_image_prediction(mock_predict, client):
    """Valid image upload returns the (mocked) prediction."""
    data = {"file": (make_image_bytes(), "digit.png")}
    response = client.post("/prediction", data=data, content_type="multipart/form-data")
    assert response.status_code == 200
    assert b"7" in response.data
    mock_predict.assert_called_once()


def test_missing_file_handled(client):
    """Missing file shows an error instead of crashing."""
    response = client.post("/prediction", data={}, content_type="multipart/form-data")
    assert b"File cannot be processed" in response.data


def test_invalid_file_handled(client):
    """Non-image file shows an error instead of crashing."""
    data = {"file": (io.BytesIO(b"not an image"), "bad.txt")}
    response = client.post("/prediction", data=data, content_type="multipart/form-data")
    assert b"File cannot be processed" in response.data


def test_preprocess_shape_and_range():
    """Preprocessing returns a (1,224,224,3) array normalized to [0,1]."""
    out = preprocess_img(make_image_bytes())
    assert out.shape == (1, 224, 224, 3)
    assert out.min() >= 0.0 and out.max() <= 1.0


@patch("model.model.predict")
def test_predict_result_argmax(mock_predict):
    """predict_result returns the index of the highest probability."""
    probs = np.zeros((1, 10), dtype=np.float32)
    probs[0, 4] = 0.96
    mock_predict.return_value = probs
    assert predict_result(np.zeros((1, 224, 224, 3))) == 4
