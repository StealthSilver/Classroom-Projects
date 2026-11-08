import os

# Required so TensorFlow can import with the installed protobuf version.
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import matplotlib.pyplot as plt
import tensorflow as tf

keras = tf.keras
layers = tf.keras.layers

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

# Display only. Close the window to continue. Not used for training.
sample_index = 0
print("sample label:", int(y_test[sample_index]))
plt.imshow(x_test[sample_index], cmap="gray")
plt.title(f"label: {int(y_test[sample_index])}")
plt.axis("off")
plt.show()

x_train = x_train / 255.0
x_test = x_test / 255.0

model = keras.Sequential([
    layers.Input(shape=(28, 28)),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(64, activation="relu"),
    layers.Dense(10, activation="softmax"),
])
model.summary()

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"],
)

model.fit(
    x_train,
    y_train,
    epochs=5,
    validation_data=(x_test, y_test),
)

test_loss, test_accuracy = model.evaluate(x_test, y_test)
print("loss", test_loss)
print("accuracy", test_accuracy)

# Display only. Shows the prediction for the same image. Not used for training.
sample = x_test[sample_index : sample_index + 1]
predicted = int(model.predict(sample).argmax())
actual = int(y_test[sample_index])
print("predicted:", predicted, "actual:", actual)
plt.figure()
plt.imshow(sample[0], cmap="gray")
plt.title(f"predicted: {predicted}, actual: {actual}")
plt.axis("off")
plt.show()
