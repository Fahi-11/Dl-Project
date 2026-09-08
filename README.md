# Handwritten Digit Recognition Web App

A Flask web application that recognizes handwritten digits from an interactive browser canvas. The application uses a TensorFlow/Keras convolutional neural network trained on the MNIST dataset.

## What It Does

- Draw a single digit from `0` to `9` on the canvas.
- Adjust the brush size or enable the eraser.
- Predict automatically while drawing, or click **Predict Canvas**.
- View the predicted digit, confidence score, probability for each class, and the top three candidates.
- Load preset test drawings for digits `0` through `9`.

## How Prediction Works

When a drawing is submitted, the Flask server:

1. Decodes the canvas image.
2. Converts it to grayscale.
3. Detects the foreground strokes and finds their bounding box.
4. Adds padding, resizes the digit while preserving its proportions, and centers it in a `28 x 28` image.
5. Normalizes pixel values to the range `0.0` to `1.0`.
6. Sends the image to the CNN, which returns probabilities for the ten digits.

The displayed confidence is the model's probability for its top prediction. It is useful as an indication of certainty, but it is not a guarantee that the prediction is correct. Unusual handwriting, ambiguous shapes, or digits that differ significantly from MNIST examples can reduce accuracy.

## Requirements

- Python 3.8 or newer
- pip
- A modern web browser

The required Python packages are listed in `requirements.txt`. TensorFlow can take several minutes to install and requires additional disk space.

## Installation and Usage

Run these commands from the project directory, the folder containing `app.py`.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python app.py
```

If PowerShell blocks activation, use Command Prompt instead:

```bat
python -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python app.py
```

### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python app.py
```

When the server starts, open [http://127.0.0.1:5000](http://127.0.0.1:5000) in a browser. Keep the terminal running while using the application. Press `Ctrl+C` in that terminal to stop the server.

To leave the virtual environment after stopping the server:

```bash
deactivate
```

To use the project again later, open a terminal in the project directory and activate the existing environment before starting the server:

```powershell
.\.venv\Scripts\Activate.ps1
python app.py
```

On macOS or Linux, use `source .venv/bin/activate` instead.

The application expects the trained model at:

```text
model/mnist_cnn.keras
```

The server loads this model when it starts. If the file is missing, the application will not start successfully.

## Testing

Run the model and preprocessing checks from the project directory:

```bash
python test_prediction.py
python scratch/test_preprocessing.py
python scratch/test_offcenter.py
```

These checks verify the model input/output shape, probability output, canvas preprocessing, and centering of off-center images. They are smoke tests; they do not measure the model's accuracy across the complete MNIST test dataset.

## Retraining the Model

To train a new model with the MNIST dataset:

```bash
python train_model.py
```

The training script normalizes the MNIST images, applies rotation/translation/zoom augmentation, evaluates the model on the MNIST test set, and saves the resulting model to `model/mnist_cnn.keras`.

## Project Structure

```text
digit-upload-webapp/
├── app.py                     # Flask server, preprocessing, and prediction API
├── train_model.py             # CNN training and evaluation script
├── test_prediction.py         # Basic model prediction test
├── scratch/
│   ├── test_preprocessing.py  # Canvas preprocessing tests
│   └── test_offcenter.py      # Off-center digit test
├── requirements.txt           # Python dependencies
├── model/
│   └── mnist_cnn.keras        # Saved TensorFlow/Keras model
├── static/
│   └── style.css              # Application styles
└── templates/
    └── index.html             # Web interface and canvas logic
```

## License

MIT License
