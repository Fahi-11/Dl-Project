# Handwritten Digit Recognition Web App

An interactive Flask web application for handwritten digit recognition trained on the MNIST dataset using a Convolutional Neural Network (CNN).

## Features
- **Interactive Canvas**: Draw numbers directly on the browser canvas with brush size & eraser controls.
- **Real-Time Prediction**: Instant neural network inference with confidence scores and top-3 predictions.
- **Model Training**: Includes full training script (`train_model.py`) using TensorFlow/Keras.
- **Multiple Backend Fallbacks**: Support for TensorFlow/Keras and standalone NumPy HDF5 weight parsing.

## Project Structure
```
digit-upload-webapp/
├── app.py                 # Main Flask server & prediction endpoint
├── train_model.py         # CNN model training script
├── test_prediction.py     # Verification & prediction tests
├── requirements.txt       # Python dependencies
├── model/
│   └── mnist_cnn.keras    # Trained CNN model weights
├── static/                # CSS styles & JS client logic
└── templates/
    └── index.html         # Web frontend UI
```

## Getting Started

### Prerequisites
- Python 3.8+
- pip

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Fahi-11/-digit-upload-webapp.git
   cd -digit-upload-webapp
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   python app.py
   ```

4. Open your browser and navigate to `http://127.0.0.1:5000`.

## License
MIT License
