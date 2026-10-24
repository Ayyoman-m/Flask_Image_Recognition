"""Module for image preprocessing and hand-digit sign prediction."""
import os
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import DepthwiseConv2D
from tensorflow.keras.utils import img_to_array

MODEL_PATH = os.path.join(os.path.dirname(__file__), "digit_model.h5")


class CompatDepthwiseConv2D(DepthwiseConv2D):  # pylint: disable=too-few-public-methods
    """DepthwiseConv2D that ignores the legacy 'groups' argument Keras 3 rejects."""

    def __init__(self, **kwargs):
        kwargs.pop("groups", None)
        super().__init__(**kwargs)


model = load_model(
    MODEL_PATH,
    compile=False,
    custom_objects={"DepthwiseConv2D": CompatDepthwiseConv2D},
)


def preprocess_img(img_path):
    """Load an image, convert to RGB, resize to 224x224, normalize to [0, 1], and reshape."""
    op_img = Image.open(img_path).convert("RGB")
    img_resize = op_img.resize((224, 224))
    img2arr = img_to_array(img_resize) / 255.0
    return img2arr.reshape(1, 224, 224, 3)


def predict_result(predict):
    """Return the predicted class index (0-9) for a preprocessed image array."""
    pred = model.predict(predict)
    return int(np.argmax(pred[0], axis=-1))
