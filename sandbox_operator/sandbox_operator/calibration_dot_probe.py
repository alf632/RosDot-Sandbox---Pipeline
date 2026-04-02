"""
Active-dot probing for projector calibration.

Generates single-dot images for projection and detects bright blob
centroids in camera frames (with optional background subtraction).
No ROS dependencies.
"""
import numpy as np
import cv2


def make_probe_positions(proj_w: int, proj_h: int,
                         nx: int = 7, ny: int = 4,
                         margin: float = 0.12) -> list[tuple[int, int]]:
    """
    Return (u, v) pixel positions for a regular grid across the projector image.
    `margin` is the fractional inset from each edge (e.g. 0.12 = 12 %).
    """
    positions = []
    for iy in range(ny):
        for ix in range(nx):
            u = int(proj_w * (margin + (1.0 - 2.0 * margin) * ix / max(nx - 1, 1)))
            v = int(proj_h * (margin + (1.0 - 2.0 * margin) * iy / max(ny - 1, 1)))
            positions.append((u, v))
    return positions


def make_dot_image(proj_w: int, proj_h: int, u: int, v: int,
                   radius: int = 18) -> np.ndarray:
    """Return a black BGR image with a single white filled circle at (u, v)."""
    img = np.zeros((proj_h, proj_w), dtype=np.uint8)
    cv2.circle(img, (u, v), radius, 255, -1)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def make_black_image(proj_w: int, proj_h: int) -> np.ndarray:
    """Return a fully black BGR image."""
    return np.zeros((proj_h, proj_w, 3), dtype=np.uint8)


def detect_blob_centroids(frame_gray: np.ndarray,
                          baseline_gray: np.ndarray | None = None,
                          min_area: int = 20,
                          max_area: int = 5000) -> list[tuple[float, float]]:
    """
    Detect bright blob centroids in `frame_gray`.

    If `baseline_gray` is provided the two frames are subtracted first
    to isolate the projected dot from ambient scene content.

    Returns a list of (u, v) float centroid positions.
    """
    if baseline_gray is not None:
        diff = cv2.absdiff(frame_gray, baseline_gray)
    else:
        diff = frame_gray.copy()

    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centroids = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area <= area <= max_area:
            M = cv2.moments(cnt)
            if M['m00'] > 0:
                centroids.append((M['m10'] / M['m00'], M['m01'] / M['m00']))
    return centroids
