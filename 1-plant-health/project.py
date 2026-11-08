!pip install -q importlib_resources


import os

# Required so TensorFlow can import with the installed protobuf version.
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds

keras = tf.keras
layers = tf.keras.layers

IMG_SIZE = 64

dataset, info = tfds.load("plant_village", split="train", as_supervised=True, with_info=True
)

# Mark each class as healthy (1) or unhealthy (0).
class_names = info.features["label"].names
healthy_classes = np.array(["healthy" in name.lower() for name in class_names])

def resize_image(image, label):
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    return tf.cast(image, tf.uint8), label

dataset = dataset.map(resize_image).batch(1000)

image_batches = []
health_batches = []
for images, class_ids in tfds.as_numpy(dataset):
    image_batches.append(images)
    health_batches.append(healthy_classes[class_ids])

x_all = np.concatenate(image_batches)
y_all = np.concatenate(health_batches).astype("float32")
print(x_all.shape, y_all.shape)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x_all, y_all, train_size=0.8, random_state=42
)
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

print("healthy in train:", int(y_train.sum()), "| unhealthy in train:", int(len(y_train) - y_train.sum()))

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

model = keras.Sequential([
    layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),

    layers.Conv2D(16, (3, 3), activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(1, activation="sigmoid"),   # near 1 = healthy, near 0 = unhealthy
])
model.summary()

model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=["accuracy"],
)

history = model.fit(
    x_train,
    y_train,
    epochs=5,
    validation_data=(x_test, y_test),
)

test_loss, test_accuracy = model.evaluate(x_test, y_test)
print("loss", test_loss)
print("accuracy", test_accuracy)

# Display only. Training vs validation accuracy.
plt.plot(history.history["accuracy"], label="train")
plt.plot(history.history["val_accuracy"], label="test")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.legend()
plt.show()

# Display only. Shows the prediction for the same sample image. Not used for training.
sample = x_test[sample_index : sample_index + 1]
predicted = int(model.predict(sample)[0][0] > 0.5)
actual = int(y_test[sample_index])
print("predicted:", label_names[predicted], "| actual:", label_names[actual])
plt.figure()
plt.imshow(sample[0])
plt.title(f"predicted: {label_names[predicted]}, actual: {label_names[actual]}")
plt.axis("off")
plt.show()