import os
import numpy as np
import h5py
import cv2
from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

def train_mnist_model():
    print("Fetching full MNIST 70,000 dataset...")
    mnist = fetch_openml('mnist_784', version=1, parser='auto', as_frame=False)
    X = mnist.data.astype(np.float32) / 255.0
    y = mnist.target.astype(np.int64)

    print(f"Loaded {X.shape[0]} raw MNIST images.")

    # Data augmentation: generate augmented samples for robustness
    X_list = [X]
    y_list = [y]

    print("Generating augmented dataset (rotations, shifts, stroke thickness variations)...")
    # Sample a subset of training data to augment for speed & variety
    aug_indices = np.random.choice(len(X), size=30000, replace=False)
    X_aug = []
    y_aug = []

    for idx in aug_indices:
        img = X[idx].reshape(28, 28)
        lbl = y[idx]

        # 1. Random small rotation (-10 to +10 degrees)
        angle = np.random.uniform(-10, 10)
        M = cv2.getRotationMatrix2D((14, 14), angle, 1.0)
        rotated = cv2.warpAffine(img, M, (28, 28))
        X_aug.append(rotated.flatten())
        y_aug.append(lbl)

        # 2. Random shift (1 pixel up/down/left/right)
        dx, dy = np.random.randint(-2, 3, size=2)
        M_shift = np.float32([[1, 0, dx], [0, 1, dy]])
        shifted = cv2.warpAffine(img, M_shift, (28, 28))
        X_aug.append(shifted.flatten())
        y_aug.append(lbl)

        # 3. Dilation / Erosion for stroke thickness variety
        uint_img = (img * 255).astype(np.uint8)
        kernel = np.ones((2, 2), np.uint8)
        if np.random.rand() > 0.5:
            thick = cv2.dilate(uint_img, kernel, iterations=1).astype(np.float32) / 255.0
            X_aug.append(thick.flatten())
        else:
            thin = cv2.erode(uint_img, kernel, iterations=1).astype(np.float32) / 255.0
            X_aug.append(thin.flatten())
        y_aug.append(lbl)

    X_full = np.vstack(X_list + [np.array(X_aug, dtype=np.float32)])
    y_full = np.concatenate(y_list + [np.array(y_aug, dtype=np.int64)])

    print(f"Total Augmented Dataset: {X_full.shape[0]} samples.")

    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y_full, test_size=10000, random_state=42, stratify=y_full
    )

    print("Training Deep Neural Network (784 -> 512 -> 256 -> 10)...")
    mlp = MLPClassifier(
        hidden_layer_sizes=(512, 256),
        activation='relu',
        solver='adam',
        alpha=0.0001,
        batch_size=256,
        learning_rate_init=0.001,
        max_iter=30,
        early_stopping=True,
        n_iter_no_change=5,
        random_state=42,
        verbose=True
    )

    mlp.fit(X_train, y_train)

    test_acc = mlp.score(X_test, y_test) * 100.0
    print(f"\n==========================================")
    print(f"MODEL TRAINING COMPLETE!")
    print(f"Test Set Accuracy: {test_acc:.2f}%")
    print(f"==========================================\n")

    # Save to HDF5 format compatible with app.py
    os.makedirs("model", exist_ok=True)
    model_path = os.path.join("model", "mnist_cnn.keras")
    tmp_path = model_path + ".tmp"

    weights = mlp.coefs_
    biases = mlp.intercepts_

    with h5py.File(tmp_path, 'w') as f:
        f.attrs['keras_version'] = '3.0.0'
        f.attrs['backend'] = 'numpy'
        f.attrs['accuracy'] = test_acc

        model_weights = f.create_group('model_weights')

        layer0 = model_weights.create_group('dense_1')
        layer0.create_dataset('kernel:0', data=weights[0])
        layer0.create_dataset('bias:0', data=biases[0])

        layer1 = model_weights.create_group('dense_2')
        layer1.create_dataset('kernel:0', data=weights[1])
        layer1.create_dataset('bias:0', data=biases[1])

        layer2 = model_weights.create_group('dense_3')
        layer2.create_dataset('kernel:0', data=weights[2])
        layer2.create_dataset('bias:0', data=biases[2])

    if os.path.exists(model_path):
        os.remove(model_path)
    os.rename(tmp_path, model_path)
    print(f"Saved trained model to {model_path} successfully!")

if __name__ == '__main__':
    train_mnist_model()
