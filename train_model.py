import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


MODEL_PATH = os.path.join("model", "mnist_cnn.keras")


def build_cnn():
    model = keras.Sequential([
        layers.Input(shape=(28, 28, 1)),

        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),

        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),

        layers.Dense(10, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def train_mnist_model():

    print("=" * 60)
    print("TRAINING MNIST CNN")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. Load MNIST
    # ---------------------------------------------------------

    print("\nLoading MNIST dataset...")

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    print("Training images:", x_train.shape)
    print("Test images:", x_test.shape)

    # ---------------------------------------------------------
    # 2. Normalize images
    # ---------------------------------------------------------

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Add channel dimension
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    print("CNN input shape:", x_train.shape)

    # ---------------------------------------------------------
    # 3. Data augmentation
    # ---------------------------------------------------------

    data_augmentation = keras.Sequential([
        layers.RandomRotation(0.08),
        layers.RandomTranslation(0.10, 0.10),
        layers.RandomZoom(0.10)
    ])

    # ---------------------------------------------------------
    # 4. Build CNN
    # ---------------------------------------------------------

    model = keras.Sequential([
        layers.Input(shape=(28, 28, 1)),

        data_augmentation,

        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),

        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),

        layers.Dense(10, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    print("\nCNN Architecture:")
    model.summary()

    # ---------------------------------------------------------
    # 5. Train
    # ---------------------------------------------------------

    print("\nStarting CNN training...")

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=3,
            restore_best_weights=True
        )
    ]

    model.fit(
        x_train,
        y_train,
        validation_split=0.1,
        epochs=15,
        batch_size=128,
        callbacks=callbacks,
        verbose=1
    )

    # ---------------------------------------------------------
    # 6. Evaluate
    # ---------------------------------------------------------

    print("\nEvaluating CNN on MNIST test set...")

    test_loss, test_accuracy = model.evaluate(
        x_test,
        y_test,
        verbose=0
    )

    print("\n" + "=" * 60)
    print("CNN TRAINING COMPLETE")
    print("=" * 60)

    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    print(f"Test Loss:     {test_loss:.4f}")

    # ---------------------------------------------------------
    # 7. Save REAL CNN
    # ---------------------------------------------------------

    os.makedirs("model", exist_ok=True)

    model.save(MODEL_PATH)

    print("\nSaved CNN model to:")
    print(MODEL_PATH)

    print("\nModel input shape:", model.input_shape)
    print("Model output shape:", model.output_shape)


if __name__ == "__main__":
    train_mnist_model()