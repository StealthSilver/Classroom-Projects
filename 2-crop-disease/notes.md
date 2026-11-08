What the project does

This is a rice leaf disease classifier. It takes a photo of a rice leaf and predicts which disease it has, using a small Convolutional Neural Network (CNN). The dataset is the UCI Rice Leaf Diseases dataset, which is small (about 120 images) and covers three diseases: bacterial leaf blight, brown spot, and leaf smut.

Dataset link
Hugging Face: https://huggingface.co/datasets/Project-AgML/uci_rice_leaf_diseases_classification
The code downloads it automatically, so you don’t need to download anything manually.
Code explanation

Imports and setup

numpy handles array operations.
load_dataset (Hugging Face datasets) downloads the data.
keras and layers are used to build and train the neural network.
IMG_SIZE = 128 means every image is resized to 128×128 pixels.

Step 1: Load the dataset

load_dataset(...) downloads the dataset from Hugging Face.
ds[list(ds.keys())[0]] picks the first available split, since the dataset may only have a single train split.
class_names holds the disease names, and the label numbers (0, 1, 2) map to them.

Step 2: Convert images to arrays

Each image is converted to RGB, resized to 128×128, and turned into a NumPy array.
Dividing by 255 scales pixel values from 0–255 down to 0–1, which helps training.
y holds the numeric class labels.

Step 3: Shuffle and split

np.random.permutation shuffles the data randomly.
80% goes to training and 20% to testing.
Because the dataset is tiny, the test set is only about 24 images, so accuracy can swing a lot between runs.

Step 4: Build the CNN

RandomFlip and RandomRotation are augmentation layers. They create slightly varied copies of the images during training, which reduces overfitting. They are only active during training, not testing.
Three Conv2D + MaxPooling2D blocks (16 → 32 → 64 filters): the convolutions detect features like edges, spots, and textures, and pooling shrinks the image while keeping the important information.
Flatten converts the 2D feature maps into a 1D vector.
Dense(64) is a fully connected layer that combines the features.
Dropout(0.3) randomly turns off 30% of neurons during training to prevent overfitting.
The final Dense(len(class_names), softmax) outputs a probability for each disease class.

Step 5: Compile

adam is the optimizer that updates the weights.
sparse_categorical_crossentropy is the loss function, used because labels are integers rather than one-hot vectors.
accuracy is the metric tracked during training.

Step 6: Train

epochs=30 means the model sees the full training set 30 times.
batch_size=8 means weights are updated after every 8 images.
validation_data evaluates the model on the test set after each epoch.

Step 7: Evaluate

model.evaluate computes loss and accuracy on the test set and prints the accuracy.

Step 8: Predict on one image

model.predict returns probabilities for the first test image.
np.argmax picks the class with the highest probability, and the code prints the predicted disease next to the actual one.

Step 9: Save the model

model.save("rice_disease_model.keras") stores the trained model so you can reload it later without retraining.
Things to be aware of
The test set doubles as the validation set, so the reported accuracy is slightly optimistic.
With only about 120 images, results vary between runs. Transfer learning (e.g., MobileNetV2) would likely give more stable and higher accuracy.
Run pip install datasets tensorflow pillow first if you haven’t already.
