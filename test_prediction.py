import numpy as np
from PIL import Image, ImageDraw
import app as flask_app


def create_digit_7():

    img = Image.new(
        "L",
        (280, 280),
        color=0
    )

    draw = ImageDraw.Draw(img)

    draw.line(
        [(60, 60), (220, 60)],
        fill=255,
        width=24
    )

    draw.line(
        [(220, 60), (100, 220)],
        fill=255,
        width=24
    )

    return img


def test_cnn():

    print("================================")
    print("CNN MODEL TEST")
    print("================================")

    # Model must exist
    assert flask_app.model is not None

    print("Model input shape:",
          flask_app.model.input_shape)

    print("Model output shape:",
          flask_app.model.output_shape)

    # Must be CNN-compatible
    assert flask_app.model.input_shape[-3:] == (
        28,
        28,
        1
    )

    assert flask_app.model.output_shape[-1] == 10

    # Create test digit
    img = create_digit_7()

    # Preprocess
    img_28, preview = (
        flask_app.preprocess_image(img)
        if hasattr(flask_app, "preprocess_image")
        else flask_app.preprocess_canvas_image(img)
    )

    assert img_28.shape == (28, 28)

    # Predict
    probs = flask_app.forward_predict(img_28)

    print(
        "Probabilities:",
        [round(float(p), 4) for p in probs]
    )

    # Basic checks
    assert probs.shape == (10,)
    assert np.all(probs >= 0)
    assert np.isclose(
        np.sum(probs),
        1.0,
        atol=0.01
    )

    pred = int(np.argmax(probs))

    print("Prediction:", pred)
    print(
        "Confidence:",
        f"{probs[pred] * 100:.2f}%"
    )

    assert 0 <= pred <= 9

    print("\nCNN test passed!")


if __name__ == "__main__":
    test_cnn()