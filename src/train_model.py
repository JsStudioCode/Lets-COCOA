import os
import cv2
import joblib
import numpy as np

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

from feature_extraction import extract_all_features

BASE_DIR = Path(__file__).resolve().parent.parent

RGB_PATH = BASE_DIR / "data" / "rgb"
THERMAL_PATH = BASE_DIR / "data" / "thermal"
MODEL_PATH = BASE_DIR / "models"

CLASSES = ["CPB", "OR", "R", "UR"]


X = []
y = []

for label, cls in enumerate(CLASSES):

    rgb_folder = RGB_PATH / cls
    thermal_folder = THERMAL_PATH / cls

    for i in range(1, 41):

        rgb = cv2.imread(str(rgb_folder / f"{cls}{i}.png"), cv2.IMREAD_UNCHANGED)
        front = cv2.imread(str(thermal_folder / f"{cls}{i}F.jpg"))
        back = cv2.imread(str(thermal_folder / f"{cls}{i}B.jpg"))

        if rgb is None or front is None or back is None:
            print(f"Skipping {cls}{i}")
            continue


        features = extract_all_features(rgb, front, back)

        X.append(features)
        y.append(label)

X = np.array(X)
y = np.array(y)

print("Dataset Shape:", X.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = SVC(kernel="rbf", probability=True)

print("Training...")

model.fit(X_train, y_train)

pred = model.predict(X_test)

acc = accuracy_score(y_test, pred)

print("Accuracy:", acc)

print(classification_report(y_test, pred))

MODEL_PATH.mkdir(exist_ok=True)

joblib.dump(model, MODEL_PATH / "svm_full.pkl")

print("Model saved successfully.")

