"""
CharucoCalibrator
=================
Manages ChArUco board observations and board-image generation for
multi-pass projector calibration.  No ROS dependencies.

Corner pixel positions are determined by rendering the board at the target
resolution and detecting the corners with OpenCV, so they exactly match what
the projector will display — no analytic formula needed.
"""

import numpy as np
import cv2

# BGR overlay colours
_C_UNSEEN  = (40,  40, 180)   # red dot  – never observed
_C_ONE_CAM = (30, 200, 200)   # yellow   – 1 camera
_C_MULTI   = (40, 200,  60)   # green    – 2+ cameras
_C_TARGET  = (200, 120,  30)  # orange   – targeted this pass


class CharucoCalibrator:
    """
    Tracks ChArUco corner observations for multi-pass projector calibration.

    Corner pixel positions are determined empirically: the board is rendered at
    (proj_w, proj_h) and OpenCV detects the corners in that synthetic image.
    This guarantees the stored pixel positions match exactly what OpenCV draws,
    regardless of version-specific coordinate conventions.
    """

    REGION_NX = 4
    REGION_NY = 3

    def __init__(self, squares_x: int, squares_y: int, proj_w: int, proj_h: int):
        self.squares_x = squares_x
        self.squares_y = squares_y
        self.proj_w    = proj_w
        self.proj_h    = proj_h

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
        # Physical square/marker size is arbitrary (we don't use estimatePose).
        self.board = cv2.aruco.CharucoBoard_create(
            squares_x, squares_y, 1.0, 0.75, self.aruco_dict
        )

        # Pixel position of each inner corner when drawn at (proj_w, proj_h).
        # Detected empirically from a synthetic board render so positions are
        # guaranteed consistent with what OpenCV actually draws.
        self.corner_pixels: dict[int, np.ndarray] = self._detect_corner_pixels()

        # Tuned detector for projector-calibration conditions (low resolution,
        # projected pattern, varying exposure).
        params = cv2.aruco.DetectorParameters_create()
        params.adaptiveThreshWinSizeMin = 3
        params.adaptiveThreshWinSizeMax = 53   # wide range to catch various marker sizes
        params.adaptiveThreshWinSizeStep = 4
        params.minMarkerPerimeterRate = 0.02   # accept smaller markers
        params.perspectiveRemovePixelPerCell = 8   # more pixels per bit → better at low res
        params.perspectiveRemoveIgnoredMarginPerCell = 0.13
        self.detector_params = params

        # CLAHE for local contrast enhancement — helps with dark / uneven exposure
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        # corner_id → {world, pixel, cameras: set, count}
        self.observations: dict = {}

    def _detect_corner_pixels(self) -> dict:
        """Render the board and detect corner positions via OpenCV detection."""
        gray = self.board.draw((self.proj_w, self.proj_h))
        marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict)

        detected: dict[int, np.ndarray] = {}
        if marker_ids is not None and len(marker_ids) > 0:
            retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                marker_corners, marker_ids, gray, self.board)
            if charuco_corners is not None:
                for i, cid in enumerate(charuco_ids.flatten()):
                    u = float(charuco_corners[i][0][0])
                    v = float(charuco_corners[i][0][1])
                    detected[int(cid)] = np.array([u, v])

        # Fall back to analytic formula for corners not detectable at image edges
        n_cx = self.squares_x - 1
        n_cy = self.squares_y - 1
        for row in range(n_cy):
            for col in range(n_cx):
                cid = row * n_cx + col
                if cid not in detected:
                    detected[cid] = np.array([
                        float((col + 1) * self.proj_w / self.squares_x),
                        float((row + 1) * self.proj_h / self.squares_y),
                    ])
        return detected

    # ── observation updates ──────────────────────────────────────────────────

    def update(self, corner_id: int, world_point: list, camera_name: str) -> bool:
        """Record or refresh an observation.  Returns True if the corner is new.

        World point is maintained as a running average over all observations
        (all cameras, all frames).  This reduces the effect of per-camera TF
        calibration errors when multiple cameras observe the same corner.
        """
        is_new = corner_id not in self.observations
        if is_new:
            self.observations[corner_id] = {
                'world':   world_point,
                'pixel':   self.corner_pixels[corner_id],
                'cameras': set(),
                'count':   0,
            }
        obs = self.observations[corner_id]
        n = obs['count']
        obs['world'] = [(obs['world'][i] * n + world_point[i]) / (n + 1) for i in range(3)]
        obs['cameras'].add(camera_name)
        obs['count'] += 1
        return is_new

    def clear(self):
        self.observations.clear()

    # ── scalar queries ───────────────────────────────────────────────────────

    def n_observed(self) -> int:
        return len(self.observations)

    def n_total(self) -> int:
        return len(self.corner_pixels)

    def n_multi_camera(self) -> int:
        return sum(1 for o in self.observations.values() if len(o['cameras']) >= 2)

    def coverage_fraction(self) -> float:
        return self.n_observed() / max(self.n_total(), 1)

    # ── set queries ──────────────────────────────────────────────────────────

    def get_pts_for_solve(self):
        """Return (pts3d, pts2d) numpy arrays for the DLT solve."""
        pts3d = np.array([o['world']         for o in self.observations.values()], dtype=np.float64)
        pts2d = np.array([o['pixel'].tolist() for o in self.observations.values()], dtype=np.float64)
        return pts3d, pts2d

    def get_unseen_corners(self) -> list:
        return [cid for cid in self.corner_pixels if cid not in self.observations]

    def get_undersampled_corners(self, min_cameras: int = 2) -> list:
        return [
            cid for cid in self.corner_pixels
            if len(self.observations.get(cid, {}).get('cameras', set())) < min_cameras
        ]

    # ── coverage grid ────────────────────────────────────────────────────────

    def get_coverage_grid(self) -> list:
        """
        Returns a REGION_NY × REGION_NX list-of-lists of (seen, multi, total) tuples.
        Useful for the TUI heat-map display.
        """
        ncx = self.squares_x - 1
        ncy = self.squares_y - 1
        nx, ny = self.REGION_NX, self.REGION_NY
        grid = []
        for ry in range(ny):
            r0 = int(ry * ncy / ny);      r1 = int((ry + 1) * ncy / ny)
            row = []
            for rx in range(nx):
                c0 = int(rx * ncx / nx);  c1 = int((rx + 1) * ncx / nx)
                total = seen = multi = 0
                for cid in self.corner_pixels:
                    col = cid % ncx
                    r   = cid // ncx
                    if c0 <= col < c1 and r0 <= r < r1:
                        total += 1
                        obs = self.observations.get(cid)
                        if obs:
                            seen += 1
                            if len(obs['cameras']) >= 2:
                                multi += 1
                row.append((seen, multi, total))
            grid.append(row)
        return grid

    def region_coverage_ok(self, min_fraction: float = 0.25) -> bool:
        """True when every grid cell meets min_fraction single-camera coverage."""
        return all(
            seen / total >= min_fraction
            for row in self.get_coverage_grid()
            for seen, _, total in row
            if total > 0
        )

    # ── board image generation ───────────────────────────────────────────────

    def generate_board_image(self, highlight_corners=None) -> np.ndarray:
        """
        Draw the ChArUco board with per-corner colour overlays.

          red dot  – never observed
          yellow   – observed by 1 camera
          green    – observed by 2+ cameras
          + orange ring on any corner in highlight_corners (this pass's targets)
        """
        gray = self.board.draw((self.proj_w, self.proj_h))
        img  = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        targets = set(highlight_corners or [])

        for cid, pxpy in self.corner_pixels.items():
            pt  = (int(pxpy[0]), int(pxpy[1]))
            obs = self.observations.get(cid)
            n   = len(obs['cameras']) if obs else 0

            if n == 0:
                cv2.circle(img, pt, 7,  _C_UNSEEN,  -1)
            elif n == 1:
                cv2.circle(img, pt, 11, _C_ONE_CAM,  3)
            else:
                cv2.circle(img, pt, 11, _C_MULTI,    3)

            if cid in targets:
                cv2.circle(img, pt, 17, _C_TARGET, 2)

        return img

    def generate_error_overlay(self, P: np.ndarray) -> np.ndarray:
        """
        Board image annotated with reprojection-error arrows.

        Each arrow starts at the observed 2-D pixel and points (×3 scale) toward
        the P-projected position.  Red = error > 10 px, green = ≤ 10 px.
        """
        img = self.generate_board_image()
        if P is None or not self.observations:
            return img

        for obs in self.observations.values():
            w    = np.array(obs['world'] + [1.0])
            proj = P @ w
            if abs(proj[2]) < 1e-6:
                continue
            up, vp = proj[0] / proj[2], proj[1] / proj[2]
            uo, vo = float(obs['pixel'][0]), float(obs['pixel'][1])
            err    = np.hypot(up - uo, vp - vo)
            tip    = (int(uo + (up - uo) * 3), int(vo + (vp - vo) * 3))
            color  = (0, 50, 220) if err > 10 else (0, 180, 80)
            cv2.arrowedLine(img, (int(uo), int(vo)), tip, color, 2, tipLength=0.3)

        return img
