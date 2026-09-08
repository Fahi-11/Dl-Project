import os
import re
import io
import base64
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Model file path
MODEL_PATH = os.path.join("model", "mnist_cnn.keras")
model = None
weights_and_biases = None
model_type = None

def load_digit_model():
    """
    Load trained CNN model from model/mnist_cnn.keras.
    Supports TensorFlow/Keras and fallback NumPy HDF5 weight parsing.
    """
    global model, weights_and_biases, model_type
    
    if not os.path.exists(MODEL_PATH):
        print(f"Warning: Model file '{MODEL_PATH}' not found.")
        return

    print(f"Attempting to load model from {MODEL_PATH}...")

    # Strategy 1: Load using TensorFlow Keras API
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(MODEL_PATH)
        model_type = 'tf_keras'
        print("Success: Loaded model using TensorFlow Keras API!")
        return
    except Exception as e:
        print(f"TF Keras loader: {e}")

    # Strategy 2: Load using Standalone Keras API
    try:
        import keras
        model = keras.models.load_model(MODEL_PATH)
        model_type = 'keras'
        print("Success: Loaded model using Keras API!")
        return
    except Exception as e:
        print(f"Keras loader: {e}")

    # Strategy 3: Load directly from HDF5 file (NumPy fallback)
    try:
        import h5py
        with h5py.File(MODEL_PATH, 'r') as f:
            mw = f['model_weights']
            w1 = mw['dense_1']['kernel:0'][:]
            b1 = mw['dense_1']['bias:0'][:]
            w2 = mw['dense_2']['kernel:0'][:]
            b2 = mw['dense_2']['bias:0'][:]
            w3 = mw['dense_3']['kernel:0'][:]
            b3 = mw['dense_3']['bias:0'][:]
        weights_and_biases = {
            'W1': w1, 'b1': b1,
            'W2': w2, 'b2': b2,
            'W3': w3, 'b3': b3
        }
        model_type = 'numpy_h5'
        print("Success: Loaded model using HDF5 NumPy weights!")
        return
    except Exception as e:
        print(f"HDF5 loader: {e}")

    print("Error: Could not initialize model backend.")

# Load model on startup
load_digit_model()

def softmax(x):
    """Compute softmax probabilities for output array."""
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

def forward_predict(img_28x28):
    """
    Run prediction through loaded model regardless of backend type.
    img_28x28 is a 28x28 numpy array normalized to [0.0, 1.0].
    """
    global model, weights_and_biases, model_type
    
    if model_type is None:
        load_digit_model()

    if model_type in ('tf_keras', 'keras'):
        try:
            # Reshape tensor to (1, 28, 28, 1) for CNN input
            input_tensor = img_28x28.reshape(1, 28, 28, 1)
            raw_preds = model.predict(input_tensor, verbose=0)[0]
        except Exception:
            input_tensor = img_28x28.reshape(1, 784)
            raw_preds = model.predict(input_tensor, verbose=0)[0]
        return raw_preds.astype(float)

    elif model_type == 'numpy_h5':
        # NumPy forward pass through neural network layers
        x_vec = img_28x28.reshape(1, 784)
        wb = weights_and_biases
        z1 = np.dot(x_vec, wb['W1']) + wb['b1']
        a1 = np.maximum(0, z1)
        z2 = np.dot(a1, wb['W2']) + wb['b2']
        a2 = np.maximum(0, z2)
        logits = np.dot(a2, wb['W3']) + wb['b3']
        probs = softmax(logits)[0]
        return probs.astype(float)

    else:
        # Uniform fallback distribution if model missing
        return np.ones(10) / 10.0

def preprocess_canvas_image(pil_image):
    """
    Simplified Canvas Processing:
    1. Convert canvas image to grayscale ('L')
    2. Resize directly to 28x28 pixels matching MNIST standard
    3. Normalize pixel values to [0.0, 1.0]
    """
    # 1. Convert to grayscale
    gray = pil_image.convert('L')
    
    # 2. Resize directly to 28x28 pixels
    resized_28x28 = gray.resize((28, 28), Image.Resampling.BILINEAR)
    
    # 3. Convert to numpy array and normalize to [0.0, 1.0]
    img_np = np.array(resized_28x28, dtype=np.float32) / 255.0
    
    return img_np, resized_28x28

# Alias for backward compatibility
preprocess_image = preprocess_canvas_image

@app.route('/')
def index():
    """Render main page index.html"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle drawing canvas prediction:
    - Receive base64 PNG data from frontend canvas
    - Decode base64 -> PIL Image -> 28x28 grayscale array
    - Run prediction model
    - Return JSON response with predicted digit, confidence %, and top probabilities
    """
    try:
        if not request.is_json or 'image' not in request.json:
            return jsonify({'error': 'Invalid payload. Base64 canvas image required.'}), 400

        # Extract & decode Base64 canvas data
        image_data = request.json['image']
        image_data = re.sub('^data:image/.+;base64,', '', image_data)
        img_bytes = base64.b64decode(image_data)
        pil_img = Image.open(io.BytesIO(img_bytes))

        # Preprocess canvas drawing (Grayscale -> 28x28 -> Normalize)
        img_28x28, processed_pil = preprocess_canvas_image(pil_img)

        # Run model inference
        probabilities = forward_predict(img_28x28)

        # Extract predicted digit and confidence
        predicted_digit = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_digit] * 100.0)
        prob_list = [round(float(p * 100.0), 2) for p in probabilities]

        # Top 3 candidates
        top3_indices = np.argsort(probabilities)[::-1][:3]
        top3 = [{'digit': int(idx), 'probability': round(float(probabilities[idx] * 100.0), 2)} for idx in top3_indices]

        # Encode 28x28 thumbnail image for frontend display
        buffered = io.BytesIO()
        processed_pil.save(buffered, format="PNG")
        processed_b64 = "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode('utf-8')

        return jsonify({
            'success': True,
            'digit': predicted_digit,
            'confidence': round(confidence, 2),
            'probabilities': prob_list,
            'top3': top3,
            'processed_image': processed_b64
        })

    except Exception as e:
        print(f"Prediction Error: {e}")
        return jsonify({'error': f'Prediction Error: {str(e)}'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting Flask server on http://127.0.0.1:{port}")
    app.run(host='0.0.0.0', port=port, debug=True)
