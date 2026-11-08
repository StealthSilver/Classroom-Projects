import os

# Required so TensorFlow can import with the installed protobuf version.
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import numpy as np
import tensorflow as tf

keras = tf.keras
layers = tf.keras.layers

tf.random.set_seed(1)
np.random.seed(1)

# Each sentence ends with the word that fills the blank.
sentences = [
    "robots will bring peace",
    "robots will bring chaos",
    "robots will bring hope",
    "people will bring peace",
    "people will bring chaos",
    "war will bring chaos",
    "love will bring peace",
    "fear will bring chaos",
    "kindness will bring peace",
    "science will bring progress",
    "rain will bring life",
    "time will bring change",
    "music will bring joy",
    "books will bring knowledge",
    "work will bring success",
    "sleep will bring rest",
    "fire will bring warmth",
    "winter will bring cold",
    "summer will bring heat",
    "art will bring beauty",
]

words = sorted({word for sentence in sentences for word in sentence.split()})
word_to_id = {word: index + 1 for index, word in enumerate(words)}
id_to_word = {index: word for word, index in word_to_id.items()}
vocab_size = len(word_to_id) + 1
max_len = max(len(sentence.split()) for sentence in sentences) - 1

x_train = []
y_train = []
for sentence in sentences:
    tokens = sentence.split()
    ids = [word_to_id[word] for word in tokens[:-1]]
    padded = [0] * (max_len - len(ids)) + ids
    x_train.append(padded)
    y_train.append(word_to_id[tokens[-1]])

x_train = np.array(x_train)
y_train = np.array(y_train)

model = keras.Sequential([
    layers.Input(shape=(max_len,)),
    layers.Embedding(vocab_size, 16, mask_zero=True),
    layers.LSTM(32),
    layers.Dense(vocab_size, activation="softmax"),
])
model.summary()

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"],
)

model.fit(x_train, y_train, epochs=250, batch_size=4, verbose=0)

loss, accuracy = model.evaluate(x_train, y_train, verbose=0)
print("loss", loss)
print("accuracy", accuracy)

text = input("Enter a sentence with a blank (___): ").lower()
before = text.split("___")[0]
tokens = before.replace(".", "").replace(",", "").replace("?", "").split()
unknown = [word for word in tokens if word not in word_to_id]

if not tokens:
    print("Type the words before the blank, for example: robots will bring ___")
elif unknown:
    print("unknown words:", ", ".join(unknown))
else:
    ids = [word_to_id[word] for word in tokens][-max_len:]
    padded = np.array([[0] * (max_len - len(ids)) + ids])
    probs = model.predict(padded, verbose=0)[0]
    top_ids = [int(index) for index in np.argsort(probs)[::-1] if int(index) in id_to_word][:3]
    guess = id_to_word[top_ids[0]]
    others = [id_to_word[index] for index in top_ids[1:]]
    print(" ".join(tokens + [guess]))
    print("other words:", ", ".join(others))
