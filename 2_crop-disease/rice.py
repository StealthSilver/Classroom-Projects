import numpy as np
from datasets import load_dataset
from tensorflow import keras
from tensorflow.keras import layers

IMG_SIZE = 128

# 1. Download the dataset automatically
ds = load_dataset("Project-AgML/uci_rice_leaf_diseases_classification")
data = ds[list(ds.keys())[0]]              # use the first available split
class_names = data.features["label"].names
print("Classes:", class_names)

# 2. Convert images to numpy arrays
X = np.array([np.array(img.convert("RGB").resize((IMG_SIZE, IMG_SIZE)))
              for img in data["image"]], dtype="float32") / 255.0
y = np.array(data["label"])

# 3. Shuffle and split into train / test (80% / 20%)
idx = np.random.permutation(len(X))
X, y = X[idx], y[idx]
split = int(0.8 * len(X))
X_train, y_train = X[:split], y[:split]
X_test, y_test = X[split:], y[split:]

# 4. Build a simple CNN
model = keras.Sequential([
    layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.RandomFlip("horizontal"),        # light augmentation (small dataset)
    layers.RandomRotation(0.1),

    layers.Conv2D(16, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation="relu"),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(len(class_names), activation="softmax"),
])

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# 5. Train
model.fit(X_train, y_train, epochs=30, batch_size=8,
          validation_data=(X_test, y_test))

# 6. Evaluate
loss, acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {acc:.2f}")

# 7. Predict on one test image
pred = model.predict(X_test[:1])
print("Predicted disease:", class_names[np.argmax(pred)])
print("Actual disease:   ", class_names[y_test[0]])

# 8. Save the model
model.save("rice_disease_model.keras")