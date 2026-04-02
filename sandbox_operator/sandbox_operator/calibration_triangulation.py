"""
Ray triangulation helpers for multi-camera projector calibration.
No ROS dependencies.
"""
import numpy as np


def triangulate(origins, directions):
    """
    Triangulate a 3-D point from N ≥ 2 rays (linear least-squares).

    Each ray i:  P(t) = origins[i] + t * directions[i]

    Minimises the sum of squared perpendicular distances from the rays to
    the returned point (equivalent to solving (I - dd^T) P = (I - dd^T) o
    for each ray and stacking into a normal-equation system).

    Returns
    -------
    point     : ndarray (3,)  best-fit 3-D point
    residuals : ndarray (N,)  perpendicular distance from each ray (metres)
    """
    origins    = [np.asarray(o, dtype=np.float64) for o in origins]
    directions = [np.asarray(d, dtype=np.float64) for d in directions]
    directions = [d / np.linalg.norm(d) for d in directions]
    n = len(origins)
    if n < 2:
        raise ValueError("Need at least 2 rays to triangulate")

    A = np.zeros((3, 3))
    b = np.zeros(3)
    for o, d in zip(origins, directions):
        M  = np.eye(3) - np.outer(d, d)
        A += M
        b += M @ o

    point = np.linalg.lstsq(A, b, rcond=None)[0]

    residuals = np.array([
        np.linalg.norm((point - o) - np.dot(point - o, d) * d)
        for o, d in zip(origins, directions)
    ])
    return point, residuals


def camera_agreement(origins, directions, threshold_m=0.005):
    """
    Check that all cameras agree on a single triangulated 3-D point.

    Returns a dict:
        point   : ndarray (3,)  triangulated position (metres)
        mean_m  : float         mean perpendicular residual (metres)
        max_m   : float         maximum perpendicular residual (metres)
        ok      : bool          True when max_m < threshold_m
    """
    point, residuals = triangulate(origins, directions)
    return {
        'point':  point,
        'mean_m': float(residuals.mean()),
        'max_m':  float(residuals.max()),
        'ok':     bool(residuals.max() < threshold_m),
    }
