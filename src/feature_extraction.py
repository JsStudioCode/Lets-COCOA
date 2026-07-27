import cv2
import numpy as np
from skimage.feature import local_binary_pattern

try:
    from src.segmentation import segment_rgb_pod, segment_thermal_pod
except ImportError:
    from segmentation import segment_rgb_pod, segment_thermal_pod


def extract_rgb_features(image):
    """
    RGB color stats, computed only over the cocoa pod (not the
    background). `image` can be:
      - 4-channel BGRA/RGBA (alpha used directly as the pod mask), or
      - 3-channel BGR shot on a plain light background (HSV-threshold
        segmented).
    """
    mask, bgr = segment_rgb_pod(image)
    bgr = cv2.resize(bgr, (224, 224))
    mask = cv2.resize(mask, (224, 224), interpolation=cv2.INTER_NEAREST)

    b, g, r = cv2.split(bgr)
    m = mask > 0
    if m.sum() == 0:          # segmentation safety fallback
        m = np.ones(mask.shape, dtype=bool)

    return [
        float(np.mean(r[m])), float(np.mean(g[m])), float(np.mean(b[m])),
        float(np.std(r[m])),  float(np.std(g[m])),  float(np.std(b[m])),
    ]


def extract_hsv_features(image):
    """Mean H/S/V over the segmented pod region only."""
    mask, bgr = segment_rgb_pod(image)
    bgr = cv2.resize(bgr, (224, 224))
    mask = cv2.resize(mask, (224, 224), interpolation=cv2.INTER_NEAREST)

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    m = mask > 0
    if m.sum() == 0:
        m = np.ones(mask.shape, dtype=bool)

    return [float(np.mean(h[m])), float(np.mean(s[m])), float(np.mean(v[m]))]


def extract_thermal_features(image):
    """
    Thermal (pseudo-temperature) stats, computed only over the pod region
    of a FLIR JPEG — excludes the logo/colorbar/readout-box overlay and
    the yellow/orange background so they don't skew mean/std/min/max.
    """
    mask = segment_thermal_pod(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    m = mask > 0
    if m.sum() == 0:
        m = np.ones(mask.shape, dtype=bool)

    px = gray[m]
    return [float(np.mean(px)), float(np.std(px)), float(np.min(px)), float(np.max(px))]


def extract_hue_histogram_features(image, bins=10):
    """
    Hue distribution over the pod region (10 bins across 0-180).
    Unlike a single mean-hue value, this captures the SHAPE of the color
    distribution -- important for separating Ripe vs Overripe, which
    differ more in how much of the pod has shifted color than in the
    single average hue.
    """
    mask, bgr = segment_rgb_pod(image)
    bgr = cv2.resize(bgr, (224, 224))
    mask = cv2.resize(mask, (224, 224), interpolation=cv2.INTER_NEAREST)

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    m = mask > 0
    if m.sum() == 0:
        m = np.ones(mask.shape, dtype=bool)

    hist, _ = np.histogram(hue[m], bins=bins, range=(0, 180))
    hist = hist.astype(np.float32)
    hist /= (hist.sum() + 1e-6)
    return hist.tolist()


def extract_lbp_features(image, n_points=8, radius=1, bins=10):
    """
    Local Binary Pattern texture histogram over the pod region.
    Captures surface texture (smooth vs mottled/rough) -- useful when
    color alone doesn't separate classes (e.g. Ripe vs Overripe).
    """
    mask, bgr = segment_rgb_pod(image)
    bgr = cv2.resize(bgr, (224, 224))
    mask = cv2.resize(mask, (224, 224), interpolation=cv2.INTER_NEAREST)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, n_points, radius, method="uniform")
    m = mask > 0
    if m.sum() == 0:
        m = np.ones(mask.shape, dtype=bool)

    hist, _ = np.histogram(lbp[m], bins=bins, range=(0, n_points + 2))
    hist = hist.astype(np.float32)
    hist /= (hist.sum() + 1e-6)
    return hist.tolist()


def extract_all_features(rgb_image, thermal_front, thermal_back):

    rgb = extract_rgb_features(rgb_image)
    hsv = extract_hsv_features(rgb_image)
    hue_hist = extract_hue_histogram_features(rgb_image)
    lbp = extract_lbp_features(rgb_image)
    front = extract_thermal_features(thermal_front)
    back = extract_thermal_features(thermal_back)

    return np.array(
        rgb + hsv + hue_hist + lbp + front + back,
        dtype=np.float32
    )
