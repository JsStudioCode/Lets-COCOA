import cv2
import joblib
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

from feature_extraction import (
    extract_rgb_features,
    extract_hsv_features,
    extract_thermal_features
)

# ----------------------------
# Paths
# ----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

RGB_PATH = BASE_DIR / "data" / "rgb"
THERMAL_PATH = BASE_DIR / "data" / "thermal"
MODEL_PATH = BASE_DIR / "models" / "svm_single.pkl"

CLASSES = ["CPB", "OR", "R", "UR"]

X = []
y = []

for label, cls in enumerate(CLASSES):

    rgb_folder = RGB_PATH / cls
    thermal_folder = THERMAL_PATH / cls

    for i in range(1, 41):

        rgb = cv2.imread(str(rgb_folder / f"{cls}{i}.png"))

        if rgb is None:
            continue

        rgb_features = extract_rgb_features(rgb)
        hsv_features = extract_hsv_features(rgb)

        # ---------- Front ----------
        front = cv2.imread(str(thermal_folder / f"{cls}{i}F.jpg"))

        if front is not None:
            thermal_features = extract_thermal_features(front)

            X.append(rgb_features + hsv_features + thermal_features)
            y.append(label)

        # ---------- Back ----------
        back = cv2.imread(str(thermal_folder / f"{cls}{i}B.jpg"))

        if back is not None:
            thermal_features = extract_thermal_features(back)

            X.append(rgb_features + hsv_features + thermal_features)
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

model = SVC(
    kernel="rbf",
    probability=True
)

model.fit(X_train, y_train)

pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred))

joblib.dump(model, MODEL_PATH)

print("Single thermal model saved successfully!")