"""
DeepSign AI - Model Training Script
=====================================
Run after data_collection.py has produced data/landmark_data.csv

Usage:
    python train_model.py

Trains a 1D-CNN over the 126-dim hand landmark feature vector
(reshaped to 2 hands x 21 landmarks x 3 coords), saves:
  - model/deepsign_model.h5
  - model/label_encoder.pkl
  - model/training_report.png (accuracy/loss curves)
"""

import os
import numpy as np
import pandas as pd
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras import layers, models

DATA_PATH = "data/landmark_data.csv"
MODEL_DIR = "model"


def build_model(input_shape, num_classes):
    """Small 1D-CNN over the landmark feature sequence."""
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv1D(64, kernel_size=3, activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Conv1D(128, kernel_size=3, activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling1D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    if not os.path.isfile(DATA_PATH):
        print(f"ERROR: {DATA_PATH} not found. Run data_collection.py first.")
        return

    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} samples across {df['label'].nunique()} signs.")
    print(df['label'].value_counts())

    X = df.drop(columns=["label"]).values.astype(np.float32)
    y_raw = df["label"].values

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)
    num_classes = len(encoder.classes_)

    # reshape to (samples, 42 landmark-points, 3 coords) for Conv1D over landmark sequence
    X = X.reshape(-1, 42, 3)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = build_model(input_shape=(42, 3), num_classes=num_classes)
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=60,
        batch_size=16,
        callbacks=callbacks,
        verbose=1,
    )

    # Evaluate
    y_pred = np.argmax(model.predict(X_test), axis=1)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=encoder.classes_))

    # Save training curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["accuracy"], label="train")
    axes[0].plot(history.history["val_accuracy"], label="val")
    axes[0].set_title("Accuracy")
    axes[0].legend()
    axes[1].plot(history.history["loss"], label="train")
    axes[1].plot(history.history["val_loss"], label="val")
    axes[1].set_title("Loss")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "training_report.png"))
    print(f"Saved training curves to {MODEL_DIR}/training_report.png")

    # Save model + encoder
    model.save(os.path.join(MODEL_DIR, "deepsign_model.h5"))
    with open(os.path.join(MODEL_DIR, "label_encoder.pkl"), "wb") as f:
        pickle.dump(encoder, f)

    print(f"\nModel saved to {MODEL_DIR}/deepsign_model.h5")
    print(f"Label encoder saved to {MODEL_DIR}/label_encoder.pkl")
    print("\nNext step: run `python app.py`")


if __name__ == "__main__":
    main()
