import base64
from pathlib import Path

import numpy as np
import tensorflow as tf


PROJECT_DIR = Path(__file__).resolve().parent
HTML_PATH = PROJECT_DIR / "index.html"


def build_model():
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(28, 28)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    )
    return model


def main():
    print("Loading the MNIST handwritten digit dataset...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    model = build_model()
    model.summary()
    model.fit(
        x_train,
        y_train,
        epochs=5,
        validation_data=(x_test, y_test),
    )

    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.2%}")

    weights = np.concatenate(
        [weight.astype("<f4", copy=False).ravel() for weight in model.get_weights()]
    )
    encoded_weights = base64.b64encode(weights.tobytes()).decode("ascii")
    html = HTML_PATH.read_text(encoding="utf-8")
    start_marker = "<!-- MODEL_WEIGHTS_START -->"
    end_marker = "<!-- MODEL_WEIGHTS_END -->"
    if html.count(start_marker) != 1 or html.count(end_marker) != 1:
        raise RuntimeError("Could not find the model-weight markers in index.html.")

    start = html.index(start_marker) + len(start_marker)
    end = html.index(end_marker)
    if start >= end:
        raise RuntimeError("The model-weight markers are in the wrong order.")
    html = html[:start] + encoded_weights + html[end:]
    HTML_PATH.write_text(html, encoding="utf-8")
    print(f"Embedded {len(weights):,} model weights into {HTML_PATH.name}.")


if __name__ == "__main__":
    main()
