What the project does
LeafCheck is a web app where you upload a leaf photo and it tells you whether the plant is healthy or diseased, which plant it is, and the confidence. It uses a pre-trained model, so no training is needed to run it.

The three parts
Colab notebook: a learning exercise that trains a small CNN on PlantVillage.
Flask backend (app.py): loads a ready-made model and serves predictions.
Frontend (index.html): the upload page and result display.
Notebook (training experiment)
Loads PlantVillage (54,303 images, 38 classes) with tensorflow_datasets.
Resizes images to 64×64.
Converts the 38 classes into 2: any class name containing “healthy” is 1, everything else is 0.
Splits 80% train and 20% test, then normalizes pixels to 0-1.
Builds a small CNN: 3 Conv + MaxPool blocks, Flatten, Dense(64), Dropout(0.5), and a single sigmoid output.
Trains for 5 epochs with binary crossentropy and plots train vs test accuracy.
Predicts on one sample image.
This notebook is not used by the app.
Backend (app.py)
Loads a MobileNetV2 model fine-tuned on PlantVillage from Hugging Face. It downloads automatically on first run.
/ serves the HTML page.
/predict receives the uploaded image and checks it is a valid image.
It runs the classifier and takes the top 3 predictions.
split_label turns labels like Tomato\_\_\_Late_blight into plant “Tomato” and condition “Late blight”.
A leaf counts as healthy if the top label contains “healthy”.
It returns JSON with plant, condition, healthy flag, confidence %, a care tip, and the top 3 matches.
Frontend (index.html)
Supports click-to-upload or drag-and-drop, with an image preview.
On “Check Plant”, it sends the image to /predict with fetch and shows a spinner.
The result view shows a Healthy or Disease pill, the name, an animated confidence bar, and the tip.
A low-confidence warning appears below 60%.
A “Top matches” list shows the 3 predictions.
Dark mode is supported via CSS variables.
“Check another plant” resets the page.
To run it
Install the packages: pip install flask transformers torch pillow.
Put index.html inside a templates/ folder next to app.py, because Flask’s render_template looks there.
Run python app.py and open http://127.0.0.1:5000.
Good to know
The app’s model has 38 classes, so it only recognizes plants in PlantVillage (tomato, potato, apple, etc.), not rice.
Photos taken in real-world conditions (messy backgrounds) usually score lower than the clean lab photos the model was trained on.
