import os
import re
import io
import base64

import numpy as np
from PIL import Image
from flask import Flask, render_template, request, jsonify

import tensorflow as tf


app = Flask(__name__)

MODEL_PATH = os.path.join("model", "mnist_cnn.keras")

model = None


# ============================================================
# LOAD CNN MODEL
# ============================================================

def load_digit_model():

    global model

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"CNN model not found: {MODEL_PATH}"
        )

    print(f"Loading CNN model from {MODEL_PATH}...")

    model = tf.keras.models.load_model(MODEL_PATH)

    print("CNN model loaded successfully.")
    print("Input shape:", model.input_shape)
    print("Output shape:", model.output_shape)


# Load model when Flask starts
load_digit_model()


# ============================================================
# PREPROCESS DRAWING
# ============================================================

def preprocess_canvas_image(pil_image):
    """
    Convert browser canvas drawing into MNIST-style 28x28 image.

    Steps:
        1. Convert to grayscale
        2. Detect handwritten pixels
        3. Find bounding box
        4. Crop digit
        5. Add padding
        6. Resize while preserving aspect ratio
        7. Center digit in 28x28 canvas
        8. Normalize to [0, 1]
    """

    # --------------------------------------------------------
    # 1. Convert to grayscale
    # --------------------------------------------------------

    gray = pil_image.convert("L")

    img = np.array(gray, dtype=np.uint8)

    # --------------------------------------------------------
    # 2. Detect handwriting
    #
    # Canvas:
    #   black background = 0
    #   white drawing    = 255
    # --------------------------------------------------------

    mask = img > 30

    # Empty canvas
    if not np.any(mask):

        blank = np.zeros(
            (28, 28),
            dtype=np.float32
        )

        preview = Image.fromarray(
            np.uint8(blank * 255)
        )

        return blank, preview

    # --------------------------------------------------------
    # 3. Find bounding box
    # --------------------------------------------------------

    ys, xs = np.where(mask)

    x_min = int(xs.min())
    x_max = int(xs.max())
    y_min = int(ys.min())
    y_max = int(ys.max())

    digit_width = x_max - x_min + 1
    digit_height = y_max - y_min + 1

    # --------------------------------------------------------
    # 4. Add padding
    # --------------------------------------------------------

    padding = int(
        max(digit_width, digit_height) * 0.20
    )

    x_min = max(
        0,
        x_min - padding
    )

    y_min = max(
        0,
        y_min - padding
    )

    x_max = min(
        img.shape[1] - 1,
        x_max + padding
    )

    y_max = min(
        img.shape[0] - 1,
        y_max + padding
    )

    cropped = img[
        y_min:y_max + 1,
        x_min:x_max + 1
    ]

    # --------------------------------------------------------
    # 5. Resize preserving aspect ratio
    #
    # Keep the digit around 20 pixels in its largest dimension.
    # This gives it space inside the 28x28 MNIST frame.
    # --------------------------------------------------------

    cropped_pil = Image.fromarray(cropped)

    width, height = cropped_pil.size

    scale = 20.0 / max(width, height)

    new_width = max(
        1,
        int(round(width * scale))
    )

    new_height = max(
        1,
        int(round(height * scale))
    )

    resized = cropped_pil.resize(
        (new_width, new_height),
        Image.Resampling.LANCZOS
    )

    # --------------------------------------------------------
    # 6. Create 28x28 black canvas
    # --------------------------------------------------------

    canvas = Image.new(
        "L",
        (28, 28),
        0
    )

    # Center digit
    left = (28 - new_width) // 2
    top = (28 - new_height) // 2

    canvas.paste(
        resized,
        (left, top)
    )

    # --------------------------------------------------------
    # 7. Normalize
    # --------------------------------------------------------

    img_28x28 = (
        np.array(
            canvas,
            dtype=np.float32
        ) / 255.0
    )

    return img_28x28, canvas


# Backward compatibility for test files
preprocess_image = preprocess_canvas_image


# ============================================================
# CNN PREDICTION
# ============================================================

def forward_predict(img_28x28):
    """
    Run the image through the CNN.

    Input:
        28x28 normalized image

    CNN input:
        (1, 28, 28, 1)

    Output:
        10 class probabilities
    """

    global model

    if model is None:
        load_digit_model()

    # Add batch dimension and channel dimension
    input_tensor = img_28x28.reshape(
        1,
        28,
        28,
        1
    )

    probabilities = model.predict(
        input_tensor,
        verbose=0
    )[0]

    return probabilities.astype(float)


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def index():

    return render_template(
        "index.html"
    )


# ============================================================
# PREDICTION API
# ============================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    try:

        # ----------------------------------------------------
        # Validate request
        # ----------------------------------------------------

        if not request.is_json:

            return jsonify({
                "error": "JSON request required."
            }), 400

        if "image" not in request.json:

            return jsonify({
                "error": "Canvas image required."
            }), 400

        # ----------------------------------------------------
        # Decode Base64 image
        # ----------------------------------------------------

        image_data = request.json["image"]

        image_data = re.sub(
            r"^data:image/.+;base64,",
            "",
            image_data
        )

        img_bytes = base64.b64decode(
            image_data
        )

        pil_img = Image.open(
            io.BytesIO(img_bytes)
        )

        # ----------------------------------------------------
        # Preprocess
        # ----------------------------------------------------

        img_28x28, processed_pil = (
            preprocess_canvas_image(
                pil_img
            )
        )

        # ----------------------------------------------------
        # CNN prediction
        # ----------------------------------------------------

        probabilities = forward_predict(
            img_28x28
        )

        # ----------------------------------------------------
        # Predicted digit
        # ----------------------------------------------------

        predicted_digit = int(
            np.argmax(probabilities)
        )

        confidence = float(
            probabilities[predicted_digit] * 100.0
        )

        # ----------------------------------------------------
        # All probabilities
        # ----------------------------------------------------

        prob_list = [
            round(
                float(p * 100.0),
                2
            )
            for p in probabilities
        ]

        # ----------------------------------------------------
        # Top 3 predictions
        # ----------------------------------------------------

        top3_indices = np.argsort(
            probabilities
        )[::-1][:3]

        top3 = [
            {
                "digit": int(idx),
                "probability": round(
                    float(
                        probabilities[idx] * 100.0
                    ),
                    2
                )
            }
            for idx in top3_indices
        ]

        # ----------------------------------------------------
        # Convert processed 28x28 image to Base64
        # ----------------------------------------------------

        buffered = io.BytesIO()

        processed_pil.save(
            buffered,
            format="PNG"
        )

        processed_b64 = (
            "data:image/png;base64,"
            + base64.b64encode(
                buffered.getvalue()
            ).decode("utf-8")
        )

        # ----------------------------------------------------
        # Return JSON
        # ----------------------------------------------------

        return jsonify({
            "success": True,
            "digit": predicted_digit,
            "confidence": round(
                confidence,
                2
            ),
            "probabilities": prob_list,
            "top3": top3,
            "processed_image": processed_b64
        })

    except Exception as e:

        print(
            f"Prediction Error: {e}"
        )

        return jsonify({
            "error": (
                f"Prediction Error: {str(e)}"
            )
        }), 500


# ============================================================
# START FLASK
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    print(
        f"Starting Flask server on "
        f"http://127.0.0.1:{port}"
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=True
    )