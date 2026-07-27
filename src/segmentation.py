"""
segmentation.py
Pod isolation for CocoaPodAI.

Masks out background (RGB: white/transparent background; Thermal: FLIR
overlay + orange/yellow background) so that color/temperature statistics
are computed only over pod pixels, not the whole image.
"""

import cv2
import numpy as np


def _largest_contour_mask(mask):
    """Keep only the largest connected blob in a binary mask."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask
    largest = max(contours, key=cv2.contourArea)
    clean = np.zeros_like(mask)
    cv2.drawContours(clean, [largest], -1, 255, thickness=cv2.FILLED)
    return clean


def _clean_mask(mask, kernel_size=7):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return _largest_contour_mask(mask)


def segment_rgb_pod(img):
    """
    Return (mask, img_bgr) for an RGB pod image.

    Handles two cases:
    1) Image has an alpha channel (e.g. a PNG exported with background
       already removed) -> alpha itself IS the pod mask. This is the
       most reliable case, so it's used whenever available.
    2) Flat image (JPEG, or PNG with no transparency) shot against a
       plain white/light background -> segment via HSV thresholding
       (background = low saturation + high brightness).
    """
    if img.ndim == 3 and img.shape[2] == 4:
        alpha = img[:, :, 3]
        mask = (alpha > 10).astype(np.uint8) * 255
        img_bgr = img[:, :, :3]
    else:
        img_bgr = img
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        s, v = hsv[:, :, 1], hsv[:, :, 2]
        background = (s < 30) & (v > 200)
        mask = (~background).astype(np.uint8) * 255

    return _clean_mask(mask), img_bgr


def segment_thermal_pod(img_bgr):
    """
    Return a binary mask (255 = pod, 0 = background) for a FLIR thermal
    JPEG. Crops out the known overlay regions (logo, colorbar, info box,
    timestamp) and isolates the pod using hue (pod renders blue/cyan,
    background renders yellow/orange in the FLIR "Iron/Rainbow" palette).
    """
    h, w = img_bgr.shape[:2]

    # 1) Blank out the fixed FLIR overlay regions (approximate, tuned to
    #    the standard FLIR JPEG layout — adjust the fractions if your
    #    camera's overlay position differs).
    overlay_ok = np.ones((h, w), dtype=np.uint8) * 255
    overlay_ok[0:int(h * 0.12), int(w * 0.75):w] = 0             # top-right logo
    overlay_ok[:, int(w * 0.88):w] = 0                            # right colorbar
    overlay_ok[int(h * 0.70):h, 0:int(w * 0.42)] = 0              # bottom-left readouts
    overlay_ok[int(h * 0.85):h, int(w * 0.55):w] = 0              # bottom-right timestamp

    # 2) Hue-based split: pod = blue/cyan, background = yellow/orange.
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    pod_hue = ((hue >= 75) & (hue <= 150)).astype(np.uint8) * 255

    mask = cv2.bitwise_and(pod_hue, overlay_ok)
    return _clean_mask(mask)