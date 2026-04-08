"""
Camera-to-camera extrinsic consistency refinement.

Projects a coarse dot grid, collects per-camera rays, and optimises
camera poses (translation + rotation) to minimise inter-camera
triangulation residuals.  No ROS dependencies.
"""
import numpy as np
from .calibration_triangulation import triangulate


def _rotvec_apply(rotvec, d):
    """Apply angle-axis rotation vector to a unit direction vector d."""
    angle = np.linalg.norm(rotvec)
    if angle < 1e-9:
        return d.copy()
    axis = rotvec / angle
    return (d * np.cos(angle)
            + np.cross(axis, d) * np.sin(angle)
            + axis * np.dot(axis, d) * (1.0 - np.cos(angle)))


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


def refine_camera_translations(dot_rays, reference_cam=None, scale_hint_m=0.0):
    """
    Find per-camera pose corrections (translation + rotation) that minimise
    total inter-camera triangulation residuals.  The reference camera is held
    fixed.

    Uses scipy Nelder-Mead when available (6-DOF: 3 translation + 3 Rodrigues
    rotation per free camera), otherwise falls back to a translation-only
    mean-offset method.

    scale_hint_m : approximate max residual (metres) — used to set the
                   Nelder-Mead initial simplex so that the optimizer explores
                   a region proportional to the observed error.

    Returns {cam_name: (delta_xyz_np, delta_rotvec_np)} in world frame.
    Returns {} when fewer than 2 cameras are present or optimisation fails.
    """
    try:
        from scipy.optimize import minimize
        return _scipy_refine(dot_rays, reference_cam, minimize, scale_hint_m)
    except ImportError:
        trans_only = _mean_offset_correction(dot_rays, reference_cam)
        return {cam: (delta, np.zeros(3)) for cam, delta in trans_only.items()}


# ── internal helpers ──────────────────────────────────────────────────────────

def _scipy_refine(dot_rays, reference_cam, minimize, scale_hint_m=0.0):
    """
    6-DOF Nelder-Mead optimisation: 3 translation + 3 Rodrigues rotation per
    free camera.  Returns {cam_name: (delta_t, delta_r)} in world frame.

    scale_hint_m scales the initial simplex so the optimizer can converge from
    large initial errors (e.g. 375 mm inter-camera disagreement).
    """
    all_cams = sorted({k for rays in dot_rays for k in rays})
    if len(all_cams) < 2:
        return {}
    if reference_cam is None or reference_cam not in all_cams:
        reference_cam = all_cams[0]
    free_cams = [c for c in all_cams if c != reference_cam]

    def objective(params):
        corrections_t = {c: params[i*6:i*6+3]   for i, c in enumerate(free_cams)}
        corrections_r = {c: params[i*6+3:i*6+6] for i, c in enumerate(free_cams)}
        total, n = 0.0, 0
        for rays in dot_rays:
            if len(rays) < 2:
                continue
            origins    = [rays[c][0] + corrections_t.get(c, np.zeros(3)) for c in rays]
            directions = [
                _rotvec_apply(corrections_r[c], rays[c][1])
                if c in corrections_r else rays[c][1]
                for c in rays
            ]
            try:
                _, res = triangulate(origins, directions)
                total += float(res.sum())
                n += 1
            except Exception:
                pass
        return total / max(n, 1)

    ndim = len(free_cams) * 6
    x0 = np.zeros(ndim)

    # Build initial simplex scaled to the observed error magnitude.
    # Default Nelder-Mead uses 0.05 step for zero-valued x0, which is far too
    # small when cameras disagree by hundreds of mm.
    t_step = max(0.05, scale_hint_m * 0.5)
    r_step = max(0.01, float(np.arctan(scale_hint_m / 1.0)))  # angle ≈ displacement at 1 m
    steps = np.array([t_step, t_step, t_step, r_step, r_step, r_step]
                     * len(free_cams))
    simplex = np.zeros((ndim + 1, ndim))
    simplex[0] = x0
    for j in range(ndim):
        simplex[j + 1] = x0.copy()
        simplex[j + 1, j] = steps[j]

    result = minimize(objective, x0, method='Nelder-Mead',
                      options={'xatol': 5e-5, 'fatol': 1e-5, 'maxiter': 6000,
                               'initial_simplex': simplex})

    out = {reference_cam: (np.zeros(3), np.zeros(3))}
    for i, c in enumerate(free_cams):
        out[c] = (np.array(result.x[i*6:i*6+3]), np.array(result.x[i*6+3:i*6+6]))
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
    Estimate consistency residuals after applying `corrections` to the rays.

    corrections : {cam_name: (delta_t, delta_r)} as returned by
                  refine_camera_translations.
    """
    _zero6 = (np.zeros(3), np.zeros(3))
    corrected = []
    for rays in dot_rays:
        corrected.append({
            cam: (
                orig + corrections.get(cam, _zero6)[0],
                _rotvec_apply(corrections.get(cam, _zero6)[1], direc),
            )
            for cam, (orig, direc) in rays.items()
        })
    return assess_consistency(corrected)
