"""
Camera-to-camera extrinsic consistency refinement.

Projects a coarse dot grid, collects per-camera rays, and optimises
camera origin positions to minimise inter-camera triangulation residuals.
No ROS dependencies.
"""
import numpy as np
from .calibration_triangulation import triangulate


def assess_consistency(dot_rays):
    """
    Compute per-camera mean perpendicular triangulation residual.

    dot_rays : list of {cam_name: (origin_np, direction_np)} — one dict per dot.
               Dicts with fewer than 2 cameras are skipped.

    Returns {cam_name: mean_residual_m}.
    """
    accum = {}
    count = {}
    for rays in dot_rays:
        if len(rays) < 2:
            continue
        origins    = [r[0] for r in rays.values()]
        directions = [r[1] for r in rays.values()]
        try:
            _, residuals = triangulate(origins, directions)
        except Exception:
            continue
        for cam_name, res in zip(rays.keys(), residuals):
            accum[cam_name] = accum.get(cam_name, 0.0) + float(res)
            count[cam_name] = count.get(cam_name, 0) + 1
    return {k: accum[k] / count[k] for k in accum if count.get(k, 0) > 0}


def refine_camera_translations(dot_rays, reference_cam=None):
    """
    Find per-camera origin translation corrections that minimise total
    triangulation residuals.  The reference camera is held fixed.

    Uses scipy Nelder-Mead when available, otherwise falls back to a
    simple mean-offset method.

    Returns {cam_name: delta_xyz_np} in world metres.
    Returns {} when fewer than 2 cameras are present or optimisation fails.
    """
    try:
        from scipy.optimize import minimize
        return _scipy_refine(dot_rays, reference_cam, minimize)
    except ImportError:
        return _mean_offset_correction(dot_rays, reference_cam)


# ── internal helpers ──────────────────────────────────────────────────────────

def _scipy_refine(dot_rays, reference_cam, minimize):
    all_cams = sorted({k for rays in dot_rays for k in rays})
    if len(all_cams) < 2:
        return {}
    if reference_cam is None or reference_cam not in all_cams:
        reference_cam = all_cams[0]
    free_cams = [c for c in all_cams if c != reference_cam]

    def objective(params):
        corrections = {c: params[i*3:(i+1)*3] for i, c in enumerate(free_cams)}
        total, n = 0.0, 0
        for rays in dot_rays:
            if len(rays) < 2:
                continue
            origins    = [rays[c][0] + corrections.get(c, np.zeros(3)) for c in rays]
            directions = [rays[c][1] for c in rays]
            try:
                _, res = triangulate(origins, directions)
                total += float(res.sum())
                n += 1
            except Exception:
                pass
        return total / max(n, 1)

    x0 = np.zeros(len(free_cams) * 3)
    result = minimize(objective, x0, method='Nelder-Mead',
                      options={'xatol': 5e-5, 'fatol': 1e-5, 'maxiter': 3000})

    out = {reference_cam: np.zeros(3)}
    for i, c in enumerate(free_cams):
        out[c] = np.array(result.x[i*3:(i+1)*3])
    return out


def _mean_offset_correction(dot_rays, reference_cam):
    """Fallback: correct each camera's origin by the mean ray-to-consensus offset."""
    all_cams = sorted({k for rays in dot_rays for k in rays})
    if len(all_cams) < 2:
        return {}
    if reference_cam is None or reference_cam not in all_cams:
        reference_cam = all_cams[0]

    offsets = {c: [] for c in all_cams if c != reference_cam}
    for rays in dot_rays:
        if len(rays) < 2:
            continue
        try:
            consensus, _ = triangulate(
                [rays[c][0] for c in rays], [rays[c][1] for c in rays])
        except Exception:
            continue
        for cam in offsets:
            if cam not in rays:
                continue
            o, d = rays[cam]
            t = np.dot(consensus - o, d)
            closest_on_ray = o + t * d
            offsets[cam].append(consensus - closest_on_ray)

    out = {reference_cam: np.zeros(3)}
    for cam, deltas in offsets.items():
        out[cam] = np.mean(deltas, axis=0) if deltas else np.zeros(3)
    return out


def estimated_residuals_after(dot_rays, corrections):
    """
    Estimate consistency residuals after applying `corrections` to the ray origins.
    Used to report expected improvement without re-probing.
    """
    corrected = []
    for rays in dot_rays:
        corrected.append({
            cam: (orig + corrections.get(cam, np.zeros(3)), direc)
            for cam, (orig, direc) in rays.items()
        })
    return assess_consistency(corrected)
