import os
import cv2
import numpy as np
import pandas as pd
import scipy.io
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

# --------------- Step 1: Prepare dataset ----------------

def load_dataset(mat_file, image_dir):
    meta = scipy.io.loadmat(mat_file)
    full_paths = meta['full_path'][0]
    dobs = meta['dob'][0]
    photo_taken = meta['photo_taken'][0]
    data = []

    for i in range(len(full_paths)):
        if dobs[i] == 0:
            continue
        try:
            birth_year = int(1836 + (dobs[i] / 365.25))
            age = photo_taken[i] - birth_year
            if 0 < age < 100:
                img_path = os.path.join(image_dir, full_paths[i][0])
                data.append((img_path, age))
        except:
            continue
    return data

print("[INFO] Loading image paths and age labels...")
dataset = load_dataset("data/imdb.mat", "data/imdb")

# --------------- Step 2: Load and preprocess images ----------------

def preprocess_image(path, size=(64, 64)):
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.resize(img, size)
    img = img.astype("float32") / 255.0
    return img

print("[INFO] Preprocessing images (this may take a few minutes)...")
images, ages = [], []
for path, age in dataset[:5000]:  # limit to 5000 for speed
    img = preprocess_image(path)
    if img is not None:
        images.append(img)
        ages.append(age)

X = np.array(images)
y = np.array(ages)

# --------------- Step 3: Split data ----------------

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --------------- Step 4: Build CNN model ----------------

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(64,64,3)),
    MaxPooling2D(),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1)  # Regression output for age
])

model.compile(optimizer='adam', loss='mae', metrics=['mae'])

# --------------- Step 5: Train model ----------------

print("[INFO] Training model...")
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)

# --------------- Step 6: Save model ----------------

model.save("age_model.h5")
print("[DONE] Model saved as age_model.h5")

# --------------- Step 7: Predict example ----------------

def predict_age(image_path):
    img = preprocess_image(image_path)
    if img is None:
        print("Image not found.")
        return
    pred = model.predict(np.expand_dims(img, axis=0))[0][0]
    print(f"Predicted Age: {pred:.1f} years")

# Predict example image (change the path)
# predict_age("data/imdb/01/nm0000001_rm2008560640_1953-1-1_2000.jpg")
