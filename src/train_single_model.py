import cv2
import joblib
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
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

        rgb = cv2.imread(str(rgb_folder / f"{cls}{i}.png"), cv2.IMREAD_UNCHANGED)

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

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(probability=True)),
])

param_grid = {
    "svm__C": [0.1, 1, 10, 100],
    "svm__gamma": ["scale", 0.01, 0.1, 1],
    "svm__kernel": ["rbf", "linear"],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("Tuning hyperparameters (GridSearchCV, 5-fold)...")
grid = GridSearchCV(pipeline, param_grid, cv=cv, scoring="accuracy", n_jobs=-1)
grid.fit(X_train, y_train)

print("Best params:", grid.best_params_)
print("Best CV accuracy (train folds):", round(grid.best_score_, 4))

model = grid.best_estimator_

pred = model.predict(X_test)
print("\nHeld-out test accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred))

full_cv_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
print("5-fold CV accuracy on full dataset: "
      f"{full_cv_scores.mean():.4f} +/- {full_cv_scores.std():.4f}")

joblib.dump(model, MODEL_PATH)
print("\nSingle thermal model saved successfully!")
