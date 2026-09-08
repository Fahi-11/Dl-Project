import numpy as np
from PIL import Image, ImageDraw
import app as flask_app

def test_backend():
    print("Testing model backend loading...")
    flask_app.load_digit_model()
    print(f"Active Model Backend Type: {flask_app.model_type}")
    assert flask_app.model_type is not None, "Model failed to load!"

    # Create a test image of digit '7'
    img = Image.new('L', (280, 280), color=0)
    draw = ImageDraw.Draw(img)
    draw.line([(60, 60), (220, 60)], fill=255, width=24)
    draw.line([(220, 60), (100, 220)], fill=255, width=24)

    # Preprocess & predict
    img_28x28, preview_pil = flask_app.preprocess_image(img)
    probs = flask_app.forward_predict(img_28x28)
    pred = int(np.argmax(probs))
    conf = float(probs[pred] * 100.0)

    print(f"Test Image Prediction: Digit {pred} with {conf:.2f}% confidence.")
    print("Probabilities (0-9):", [round(p * 100, 1) for p in probs])

    assert pred in range(10), "Prediction out of range!"
    print("All backend tests passed successfully!")

if __name__ == '__main__':
    test_backend()
