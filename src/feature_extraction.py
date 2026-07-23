import cv2
import numpy as np


def extract_rgb_features(image):

    image = cv2.resize(image, (224, 224))

    b = image[:, :, 0]
    g = image[:, :, 1]
    r = image[:, :, 2]

    features = [

        np.mean(r),
        np.mean(g),
        np.mean(b),

        np.std(r),
        np.std(g),
        np.std(b)

    ]

    return features


def extract_hsv_features(image):

    image = cv2.resize(image, (224,224))

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    features = [

        np.mean(h),
        np.mean(s),
        np.mean(v)

    ]

    return features


def extract_thermal_features(image):
    """
    Extract thermal image statistics
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    mean = np.mean(gray)
    std = np.std(gray)
    minimum = np.min(gray)
    maximum = np.max(gray)

    return [mean, std, minimum, maximum]


def extract_all_features(rgb_image, thermal_front, thermal_back):

    rgb = extract_rgb_features(rgb_image)

    hsv = extract_hsv_features(rgb_image)

    front = extract_thermal_features(thermal_front)

    back = extract_thermal_features(thermal_back)

    return np.array(
        rgb + hsv + front + back,
        dtype=np.float32
    )