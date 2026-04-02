import rclpy
from rclpy.node import Node
import tf2_ros
import curses
import threading
import subprocess
import time
import json, yaml
import os
import socket
import numpy as np
import cv2
import base64

from std_msgs.msg import String
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
import message_filters
from image_geometry import PinholeCameraModel
import tf2_geometry_msgs  # Requires: sudo apt install ros-jazzy-tf2-geometry-msgs
from tf2_ros import StaticTransformBroadcaster
from geometry_msgs.msg import TransformStamped
from .calibration_charuco import CharucoCalibrator
from .calibration_triangulation import triangulate as _triangulate_rays
from .calibration_dot_probe import (make_probe_positions, make_dot_image,
                                     make_black_image, detect_blob_centroids)

# ChArUco board geometry (16×9 @ 120 px fills 1920×1080 exactly)
CHARUCO_SQUARES_X = 16
CHARUCO_SQUARES_Y = 9

# How long to wait for blob detection in each camera before giving up on a dot
PROBE_DOT_TIMEOUT_S = 6.0

class CalibrationCore(Node):
    """Background ROS 2 Node handling TF, Topics, and ChArUco processing"""
    def __init__(self):
        super().__init__('calibration_core')
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.bridge = CvBridge()

        # Projector Calibration State
        self.is_calibrating_proj = False
        self.calibrator: CharucoCalibrator | None = None
        self.subs = []   # dynamically created subscribers
        self._syncs = [] # message_filters synchronizers (prevent GC)

        # Heightmap state
        self.heightmap = None
        self.heightmap_sub = None
        self.sandbox_width = 1.0
        self.sandbox_length = 1.0
        self.heightmap_res_w = 256
        self.heightmap_res_h = 256

        # Projector discovery state
        self.discovered_projectors = {}
        self._projector_subs = {}

        self.imagecount = 0

        # Stereo-triangulation state (populated during calibration)
        self._corner_rays: dict = {}       # corner_id → {cam_name: (origin, direction)}
        self._stereo_corners: set = set()  # corners that have a confirmed stereo world point
        self._latest_frames: dict = {}     # cam_name → {gray, trans, origin, cam_model}
        self._probe_correspondences: list = []

        # Static TF broadcaster for publishing solved projector pose
        self._tf_static_broadcaster = StaticTransformBroadcaster(self)

    def get_available_cameras(self):
        cameras = set()
        for topic_name, _ in self.get_topic_names_and_types():
            if "depth/camera_info" in topic_name:
                parts = topic_name.split('/')
                if len(parts) >= 3:
                    cameras.add(f"/{parts[1]}/{parts[2]}")
        return list(cameras)

    def discover_projectors(self):
        """Subscribe to any new /projectors/* topics and return current map."""
        qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )
        for topic_name, _ in self.get_topic_names_and_types():
            if topic_name.startswith('/projectors/') and topic_name not in self._projector_subs:
                sub = self.create_subscription(
                    String, topic_name,
                    lambda msg, t=topic_name: self._on_projector(t, msg),
                    qos
                )
                self._projector_subs[topic_name] = sub
        return dict(self.discovered_projectors)

    def _on_projector(self, topic_name, msg):
        try:
            self.discovered_projectors[topic_name] = json.loads(msg.data)
        except json.JSONDecodeError:
            pass

    def get_projector_id(self, topic_name):
        """Get the stable string ID from the projector's topic payload."""
        return self.discovered_projectors[topic_name]["projector_id"]

    def get_tf(self, target_frame, source_frame):
        try:
            return self.tf_buffer.lookup_transform(target_frame, source_frame, rclpy.time.Time()).transform
        except tf2_ros.TransformException:
            return None

    def _on_heightmap(self, msg):
        hm = self.bridge.imgmsg_to_cv2(msg, "32FC1")
        if self.heightmap is None:
            finite = hm[np.isfinite(hm)]
            stats = f"shape={hm.shape}, min={finite.min():.3f}, max={finite.max():.3f}, mean={finite.mean():.3f}" if len(finite) > 0 else f"shape={hm.shape}, all NaN"
            self._debug_log(f"Heightmap received: {stats}")
            self._debug_log(f"Sandbox: {self.sandbox_width}x{self.sandbox_length}m, grid: {self.heightmap_res_w}x{self.heightmap_res_h}")
        self.heightmap = hm

    def _lookup_height(self, x, y):
        """Look up Z from the merged heightmap at world (x, y)."""
        if self.heightmap is None:
            return None
        gx = (x / self.sandbox_width + 0.5) * self.heightmap_res_w
        gy = (y / self.sandbox_length + 0.5) * self.heightmap_res_h
        gx_i, gy_i = int(round(gx)), int(round(gy))
        h, w = self.heightmap.shape[:2]
        if 0 <= gx_i < w and 0 <= gy_i < h:
            z = float(self.heightmap[gy_i, gx_i])
            if np.isfinite(z):
                return z
        return None

    def start_projector_calibration(self, cameras, sandbox_config, proj_info):
        for sub in self.subs:
            self.destroy_subscription(sub.sub)
        self.subs.clear()
        self._syncs.clear()
        self._no_marker_frames = 0
        self._few_charuco_frames = 0
        self.debug_lines = []
        self._individual_msg_count = {}
        self._corner_rays = {}
        self._stereo_corners = set()
        self._latest_frames = {}
        self._probe_correspondences = []

        # Sandbox geometry for heightmap lookup
        sandbox = sandbox_config.get('sandbox', {})
        self.sandbox_width = sandbox.get('width', 1.0)
        self.sandbox_length = sandbox.get('length', 1.0)
        output_res = sandbox_config.get('output_res', {})
        self.heightmap_res_w = output_res.get('width', 256)
        self.heightmap_res_h = output_res.get('height', 256)

        # Initialise (or reset) the calibrator for this projector's resolution
        self.calibrator = CharucoCalibrator(
            CHARUCO_SQUARES_X, CHARUCO_SQUARES_Y,
            proj_info['width'], proj_info['height'],
        )
        n_detected = sum(1 for v in self.calibrator.corner_pixels.values()
                         if v is not None)
        self._debug_log(
            f"ChArUco board {CHARUCO_SQUARES_X}x{CHARUCO_SQUARES_Y} "
            f"@ {proj_info['width']}x{proj_info['height']}: "
            f"{n_detected}/{self.calibrator.n_total()} corner pixels detected from synthetic render")
        self.is_calibrating_proj = True
        self.imagecount = 0

        # Subscribe to merged heightmap
        if self.heightmap_sub is not None:
            self.destroy_subscription(self.heightmap_sub)
        self.heightmap = None
        sensor_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE
        )
        self.heightmap_sub = self.create_subscription(
            Image, '/sandbox/heightmap', self._on_heightmap, sensor_qos)
        self._debug_log("Subscribed to /sandbox/heightmap")

        # Subscribe to RGB + CameraInfo on all cameras — activate one at a time to
        # avoid saturating the USB bus.
        for i, cam in enumerate(cameras):
            if i > 0:
                time.sleep(2.0)
            cam_short = cam.split('/')[-1]

            # Try to increase color resolution before enabling stream
            for param, val in [('color_width', '1280'), ('color_height', '720')]:
                r = subprocess.run(
                    ["ros2", "param", "set", cam, param, val],
                    capture_output=True, text=True)
                self._debug_log(
                    f"[{cam_short}] {param}={val}: "
                    f"{'OK' if r.returncode == 0 else 'FAILED – ' + r.stderr.strip()[:60]}")

            r = subprocess.run(
                ["ros2", "param", "set", cam, "enable_color", "true"],
                capture_output=True, text=True)
            if r.returncode == 0:
                self._debug_log(f"[{cam_short}] color stream enabled")
            else:
                self._debug_log(
                    f"[{cam_short}] enable_color FAILED: {r.stderr.strip()[:80]}")

            rgb_topic = f'{cam}/color/image_raw'
            info_topic = f'{cam}/color/camera_info'

            rgb_sub = message_filters.Subscriber(self, Image, rgb_topic)
            info_sub = message_filters.Subscriber(self, CameraInfo, info_topic)

            cam_name = cam.split('/')[-1]
            self._individual_msg_count[cam_name] = 0
            rgb_sub.registerCallback(lambda msg, n=cam_name: self._on_individual_msg(n))

            ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, info_sub], 10, 0.1)
            ts.registerCallback(self.cam_callback)
            self.subs.extend([rgb_sub, info_sub])
            self._syncs.append(ts)
            self._debug_log(f"Subscribed to {cam} (RGB + CameraInfo)")

    def _debug_log(self, msg):
        self.debug_lines.append(msg)
        if len(self.debug_lines) > 12:
            self.debug_lines.pop(0)
        self.get_logger().info(f"[CalibTUI] {msg}")
        with open("/tmp/calibration_debug.log", "a") as f:
            f.write(f"{msg}\n")

    def _on_individual_msg(self, name):
        if name in self._individual_msg_count:
            self._individual_msg_count[name] += 1

    def stop_projector_calibration(self, cameras):
        self.is_calibrating_proj = False
        self._syncs.clear()
        for sub in self.subs:
            self.destroy_subscription(sub.sub)
        self.subs.clear()
        if self.heightmap_sub is not None:
            self.destroy_subscription(self.heightmap_sub)
            self.heightmap_sub = None
        for i, cam in enumerate(cameras):
            if i > 0:
                time.sleep(1.0)
            cam_short = cam.split('/')[-1]
            r = subprocess.run(
                ["ros2", "param", "set", cam, "enable_color", "false"],
                capture_output=True, text=True)
            if r.returncode != 0:
                self.get_logger().warn(
                    f"[{cam_short}] enable_color=false FAILED: {r.stderr.strip()[:80]}")

    def cam_callback(self, rgb_msg, info_msg):
        self.imagecount += 1
        frame_id = rgb_msg.header.frame_id
        cam_name = frame_id.split('_color_')[0]  # e.g. cam_028522074036

        cv_rgb = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
        gray   = cv2.cvtColor(cv_rgb, cv2.COLOR_BGR2GRAY)
        if self.calibrator is not None:
            gray = self.calibrator.clahe.apply(gray)

        # Resolve TF early so _latest_frames is always populated for dot-probe phase
        cam_model = PinholeCameraModel()
        cam_model.fromCameraInfo(info_msg)
        try:
            trans = self.tf_buffer.lookup_transform(
                'sandbox_origin', frame_id, rclpy.time.Time())
        except tf2_ros.TransformException as e:
            if self.is_calibrating_proj:
                self._debug_log(f"TF lookup FAILED ({cam_name}): {e}")
            return

        # Camera optical centre in sandbox_origin
        pt_origin = PointStamped()
        pt_origin.point.x = pt_origin.point.y = pt_origin.point.z = 0.0
        origin_world = tf2_geometry_msgs.do_transform_point(pt_origin, trans)
        origin = np.array([origin_world.point.x, origin_world.point.y, origin_world.point.z])

        # Cache latest frame for dot-probe phase (always, regardless of calibration state)
        self._latest_frames[cam_name] = {
            'gray': gray.copy(), 'info': info_msg,
            'trans': trans, 'origin': origin, 'cam_model': cam_model,
            'timestamp': time.monotonic(),
        }

        if not self.is_calibrating_proj or self.calibrator is None:
            return
        if self.heightmap is None:
            self._debug_log("Waiting for heightmap...")
            return

        marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(
            gray, self.calibrator.aruco_dict,
            parameters=self.calibrator.detector_params)
        if marker_ids is None or len(marker_ids) == 0:
            self._no_marker_frames += 1
            if self._no_marker_frames % 30 == 1:
                h, w = gray.shape[:2]
                self._debug_log(
                    f"[{cam_name}] No ArUco markers in {self._no_marker_frames} frames "
                    f"(img {w}x{h})")
            return

        retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            marker_corners, marker_ids, gray, self.calibrator.board)
        n_charuco = len(charuco_corners) if charuco_corners is not None else 0
        if charuco_corners is None or n_charuco <= 6:
            self._few_charuco_frames += 1
            if self._few_charuco_frames % 10 == 1:
                self._debug_log(
                    f"[{cam_name}] {len(marker_ids)} ArUco markers but only "
                    f"{n_charuco} ChArUco corners (need >6)")
            return

        points_added = 0
        for i in range(len(charuco_corners)):
            corner_id = int(charuco_ids[i][0])
            if corner_id not in self.calibrator.corner_pixels:
                continue
            u, v = float(charuco_corners[i][0][0]), float(charuco_corners[i][0][1])

            direction = self._compute_ray_direction(u, v, cam_model, trans, origin)

            # Accumulate ray for this corner+camera
            if corner_id not in self._corner_rays:
                self._corner_rays[corner_id] = {}
            self._corner_rays[corner_id][cam_name] = (origin.copy(), direction)

            # ── Try stereo triangulation when ≥ 2 cameras have rays ──────────
            rays = self._corner_rays[corner_id]
            if len(rays) >= 2:
                try:
                    pt, residuals = _triangulate_rays(
                        [r[0] for r in rays.values()],
                        [r[1] for r in rays.values()])
                    if residuals.max() < 0.010:   # 10 mm threshold
                        # First stereo estimate for this corner: discard any
                        # previous heightmap estimate so they don't mix
                        if corner_id not in self._stereo_corners:
                            self._stereo_corners.add(corner_id)
                            self.calibrator.observations.pop(corner_id, None)
                        is_new = self.calibrator.update(
                            corner_id, pt.tolist(), cam_name)
                        if is_new:
                            points_added += 1
                        continue
                except Exception:
                    pass

            # ── Heightmap fallback (skip if stereo already used for this corner) ──
            if corner_id in self._stereo_corners:
                continue
            if abs(direction[2]) < 1e-6:
                continue

            t     = -origin[2] / direction[2]
            point = origin + t * direction
            converged = False
            for _ in range(3):
                z = self._lookup_height(point[0], point[1])
                if z is None:
                    break
                t     = (z - origin[2]) / direction[2]
                point = origin + t * direction
                converged = True

            if converged:
                is_new = self.calibrator.update(
                    corner_id, [float(point[0]), float(point[1]), float(point[2])],
                    cam_name)
                if is_new:
                    points_added += 1
                    if self.calibrator.n_observed() <= 5:
                        self._debug_log(
                            f"  pt3d=[{point[0]:.3f},{point[1]:.3f},{point[2]:.3f}] "
                            f"z={z:.3f} origin=[{origin[0]:.3f},{origin[1]:.3f},{origin[2]:.3f}]")

        if points_added > 0:
            obs    = self.calibrator.n_observed()
            multi  = self.calibrator.n_multi_camera()
            stereo = len(self._stereo_corners)
            self._debug_log(
                f"[{cam_name}] +{points_added} new  total={obs}  "
                f"multi-cam={multi}  stereo={stereo}")

    def _wait_for_fresh_frames(self, timeout=5.0):
        """
        Block until every camera in _latest_frames has published at least one
        new frame after this call was made.  Returns True on success, False on timeout.
        """
        old_ts = {k: v.get('timestamp', 0.0) for k, v in self._latest_frames.items()}
        if not old_ts:
            time.sleep(0.5)
            return False
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if all(self._latest_frames.get(k, {}).get('timestamp', 0.0) > ts
                   for k, ts in old_ts.items()):
                return True
            time.sleep(0.05)
        return False

    def _wait_for_visual_change(self, pre_snapshots, pixel_thresh=15,
                                changed_frac=0.002, timeout=8.0):
        """
        Block until every camera in pre_snapshots shows a visible change vs its
        snapshot (i.e. the projector has rendered the new image and the camera
        has captured it).

        pixel_thresh   : per-pixel absolute diff that counts as "changed"
        changed_frac   : fraction of pixels that must exceed pixel_thresh
        timeout        : give up after this many seconds
        Returns the set of camera names that confirmed a change.
        """
        deadline = time.monotonic() + timeout
        confirmed = set()
        while time.monotonic() < deadline:
            for cam_name, old_gray in pre_snapshots.items():
                if cam_name in confirmed:
                    continue
                fd = self._latest_frames.get(cam_name)
                if fd is None:
                    continue
                new_gray = fd['gray']
                if old_gray.shape != new_gray.shape:
                    confirmed.add(cam_name)
                    continue
                diff = np.abs(new_gray.astype(np.int16) - old_gray.astype(np.int16))
                if np.mean(diff > pixel_thresh) > changed_frac:
                    confirmed.add(cam_name)
            if len(confirmed) >= len(pre_snapshots):
                return confirmed
            time.sleep(0.05)
        return confirmed

    def _compute_ray_direction(self, u, v, cam_model, trans, origin):
        """Normalised ray direction in sandbox_origin for pixel (u, v)."""
        ray_cam = np.array(cam_model.projectPixelTo3dRay((u, v)))
        pt_ray  = PointStamped()
        pt_ray.point.x, pt_ray.point.y, pt_ray.point.z = \
            float(ray_cam[0]), float(ray_cam[1]), float(ray_cam[2])
        ray_world = tf2_geometry_msgs.do_transform_point(pt_ray, trans)
        direction = np.array([
            ray_world.point.x - origin[0],
            ray_world.point.y - origin[1],
            ray_world.point.z - origin[2],
        ])
        return direction / np.linalg.norm(direction)

    def n_stereo_corners(self) -> int:
        """Number of ChArUco corners whose world point came from stereo triangulation."""
        return len(self._stereo_corners)

    def run_dot_probe_phase(self, proj_info, send_board_fn, status_cb=None):
        """
        Project a grid of white dots, triangulate via camera rays.
        Populates self._probe_correspondences.
        Returns number of accepted correspondences.
        """
        self._probe_correspondences = []
        pw, ph = proj_info['width'], proj_info['height']
        positions = make_probe_positions(pw, ph)

        # Snapshot current frames, then project black and wait for the visual change
        # to confirm Godot rendered the black image before recording the baseline.
        pre_snap = {k: v['gray'].copy() for k, v in self._latest_frames.items()}
        send_board_fn(make_black_image(pw, ph), "probe baseline (black)")
        if status_cb:
            status_cb("Waiting for black baseline to render on projector...")
        self._wait_for_visual_change(pre_snap, timeout=10.0)
        baseline = {k: v['gray'].copy() for k, v in self._latest_frames.items()}

        n_ok = 0
        for i, (pu, pv) in enumerate(positions):
            # Snapshot frames BEFORE sending the new dot image.
            # _wait_for_visual_change uses this to confirm the projector switched.
            pre_snap = {k: v['gray'].copy() for k, v in self._latest_frames.items()}
            send_board_fn(make_dot_image(pw, ph, pu, pv))

            # Phase A: wait for the projector to visually update (dot appeared / moved).
            # This ensures we don't detect the previous dot as the current one.
            dot_deadline = time.monotonic() + PROBE_DOT_TIMEOUT_S
            if status_cb:
                status_cb(f"Dot {i+1}/{len(positions)} ({pu},{pv})  waiting for projector…")
            self._wait_for_visual_change(
                pre_snap,
                timeout=max(1.0, PROBE_DOT_TIMEOUT_S - 2.0))

            # Phase B: blob detection on the settled frame.
            detected = {}  # cam_name -> (centroid, frame_data)
            while time.monotonic() < dot_deadline:
                for cam_name, fd in list(self._latest_frames.items()):
                    if cam_name in detected:
                        continue
                    blobs = detect_blob_centroids(fd['gray'], baseline.get(cam_name))
                    if len(blobs) == 1:
                        detected[cam_name] = (blobs[0], fd)

                elapsed = dot_deadline - time.monotonic()
                if status_cb:
                    status_cb(
                        f"Dot {i+1}/{len(positions)} ({pu},{pv})  "
                        f"{len(detected)}/{len(self._latest_frames)} cam(s) confirmed  "
                        f"{max(0, PROBE_DOT_TIMEOUT_S - elapsed):.1f}/{PROBE_DOT_TIMEOUT_S:.0f}s")

                if len(detected) >= len(self._latest_frames):
                    break
                time.sleep(0.05)

            origins_d, directions_d = [], []
            for cam_name, (centroid, fd) in detected.items():
                cu, cv_b = centroid
                d = self._compute_ray_direction(
                    cu, cv_b, fd['cam_model'], fd['trans'], fd['origin'])
                origins_d.append(fd['origin'])
                directions_d.append(d)

            if len(origins_d) >= 2:
                try:
                    pt, residuals = _triangulate_rays(origins_d, directions_d)
                    if residuals.max() < 0.010:
                        self._probe_correspondences.append({
                            'world': pt.tolist(),
                            'pixel': [float(pu), float(pv)],
                            'residual_m': float(residuals.max()),
                            'source': 'stereo',
                        })
                        n_ok += 1
                        continue
                except Exception:
                    pass
            elif len(origins_d) == 1 and self.heightmap is not None:
                origin, direction = origins_d[0], directions_d[0]
                if abs(direction[2]) > 1e-6:
                    t     = -origin[2] / direction[2]
                    point = origin + t * direction
                    converged = False
                    for _ in range(3):
                        z = self._lookup_height(point[0], point[1])
                        if z is None:
                            break
                        t     = (z - origin[2]) / direction[2]
                        point = origin + t * direction
                        converged = True
                    if converged:
                        self._probe_correspondences.append({
                            'world': point.tolist(),
                            'pixel': [float(pu), float(pv)],
                            'residual_m': None,
                            'source': 'heightmap',
                        })
                        n_ok += 1

        return n_ok

    def publish_projector_tf(self, proj_id, result):
        """Publish projector_<id>_link TF in sandbox_origin via StaticTransformBroadcaster."""
        extri = result['extrinsics']
        pos   = extri['translation']
        basis = np.array(extri['basis'], dtype=np.float64)
        quat  = self._rot_to_quat(basis)

        t = TransformStamped()
        t.header.stamp        = self.get_clock().now().to_msg()
        t.header.frame_id     = 'sandbox_origin'
        t.child_frame_id      = f'projector_{proj_id}_link'
        t.transform.translation.x = float(pos[0])
        t.transform.translation.y = float(pos[1])
        t.transform.translation.z = float(pos[2])
        t.transform.rotation.x = float(quat[0])
        t.transform.rotation.y = float(quat[1])
        t.transform.rotation.z = float(quat[2])
        t.transform.rotation.w = float(quat[3])
        self._tf_static_broadcaster.sendTransform(t)
        self._debug_log(f"Published TF: sandbox_origin → projector_{proj_id}_link")

    @staticmethod
    def _rot_to_quat(R):
        """Convert 3×3 rotation matrix to (x, y, z, w) quaternion (normalised)."""
        R = np.asarray(R, dtype=np.float64)
        trace = R[0, 0] + R[1, 1] + R[2, 2]
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
        norm = np.sqrt(x*x + y*y + z*z + w*w)
        return np.array([x, y, z, w]) / norm

    def solve_projector_matrix(self, extra_pts3d=None, extra_pts2d=None):
        if self.calibrator is None or self.calibrator.n_observed() < 10:
            return None

        pts3d, pts2d = self.calibrator.get_pts_for_solve()
        n_charuco = len(pts3d)
        n_probe   = 0
        if extra_pts3d is not None and len(extra_pts3d) > 0:
            pts3d   = np.vstack([pts3d, np.array(extra_pts3d, dtype=np.float64)])
            pts2d   = np.vstack([pts2d, np.array(extra_pts2d, dtype=np.float64)])
            n_probe = len(extra_pts3d)
        n = len(pts3d)

        self._debug_log(f"Solving with {n} points (DLT): "
                        f"{n_charuco} ChArUco + {n_probe} probe")
        self._debug_log(f"  X: [{pts3d[:,0].min():.3f}, {pts3d[:,0].max():.3f}]"
                        f"  Y: [{pts3d[:,1].min():.3f}, {pts3d[:,1].max():.3f}]"
                        f"  Z: [{pts3d[:,2].min():.3f}, {pts3d[:,2].max():.3f}]")

        # --- Normalize points for numerical stability ---
        mean3 = pts3d.mean(axis=0)
        scale3 = np.sqrt(((pts3d - mean3) ** 2).sum(axis=1).mean())
        T3 = np.eye(4); T3[:3, :3] /= scale3; T3[:3, 3] = -mean3 / scale3

        mean2 = pts2d.mean(axis=0)
        scale2 = np.sqrt(((pts2d - mean2) ** 2).sum(axis=1).mean())
        T2 = np.eye(3); T2[:2, :2] /= scale2; T2[:2, 2] = -mean2 / scale2

        pts3d_n = (T3 @ np.hstack([pts3d, np.ones((n, 1))]).T).T
        pts2d_n = (T2 @ np.hstack([pts2d, np.ones((n, 1))]).T).T

        # --- Build DLT system: 2n equations, 12 unknowns (elements of P) ---
        A = np.zeros((2 * n, 12))
        for i in range(n):
            X, Y, Z, W = pts3d_n[i]
            u, v, _ = pts2d_n[i]
            A[2*i]   = [ X,  Y,  Z,  W,  0,  0,  0,  0, -u*X, -u*Y, -u*Z, -u*W]
            A[2*i+1] = [ 0,  0,  0,  0,  X,  Y,  Z,  W, -v*X, -v*Y, -v*Z, -v*W]

        _, s, Vt = np.linalg.svd(A)
        self._debug_log(f"  DLT singular values (last 3): {s[-3]:.4f}, {s[-2]:.4f}, {s[-1]:.6f}")
        self._debug_log(f"  DLT condition number: {s[0]/s[-1]:.1f}")

        P_n = Vt[-1].reshape(3, 4)

        # --- Denormalize: P = T2^{-1} @ P_n @ T3 ---
        P = np.linalg.inv(T2) @ P_n @ T3

        # Ensure points project with positive depth (sandbox_origin in front of projector)
        # P[2,:] @ [0,0,0,1] = P[2,3] should be positive
        if P[2, 3] < 0:
            P = -P

        # --- Decompose P = K [R | t] via RQ decomposition ---
        # decomposeProjectionMatrix returns (K, R, t_world_homogeneous)
        K, R, t_hom, _, _, _, _ = cv2.decomposeProjectionMatrix(P)
        K = K / K[2, 2]

        # Fix sign: ensure positive focal lengths
        for col in range(2):
            if K[col, col] < 0:
                K[:, col] *= -1
                R[col, :] *= -1

        # Ensure proper rotation matrix (det = +1)
        if np.linalg.det(R) < 0:
            R = -R

        # Camera (projector) position in world frame is the null-space of P
        cam_pos = (t_hom[:3] / t_hom[3]).flatten()

        w, h = self.calibrator.proj_w, self.calibrator.proj_h
        cx_off = float(K[0, 2] - w / 2)
        cy_off = float(K[1, 2] - h / 2)
        self._debug_log(
            f"  fx={K[0,0]:.1f} fy={K[1,1]:.1f} "
            f"cx={K[0,2]:.1f} (+{cx_off:+.0f} from centre)  "
            f"cy={K[1,2]:.1f} ({cy_off:+.0f} from centre)")
        self._debug_log(
            f"  cam_pos (sandbox_origin, m): "
            f"X={cam_pos[0]:.3f} Y={cam_pos[1]:.3f} Z={cam_pos[2]:.3f}")

        # Compute reprojection error
        tvec = -R @ cam_pos
        proj = (K @ (R @ pts3d.T + tvec.reshape(3, 1))).T
        proj = proj[:, :2] / proj[:, 2:3]
        errs = np.sqrt(((proj - pts2d) ** 2).sum(axis=1))
        rms     = float(np.sqrt((errs ** 2).mean()))
        max_err = float(errs.max())
        self._debug_log(f"  RMS reprojection error: {rms:.2f} px   max: {max_err:.1f} px")

        # Convert from OpenCV optical frame to ROS camera_link convention
        # optical: X-right, Y-down, Z-forward  →  link: X-forward, Y-left, Z-up
        optical_to_link = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=np.float64)
        cam_rot = R.T @ optical_to_link.T

        # P maps [X,Y,Z,1] in sandbox_origin (ROS: X-fwd, Y-left, Z-up, meters)
        # to projector pixels [u,v] (top-left origin, right+down positive).
        P_list = [[float(P[r][c]) for c in range(4)] for r in range(3)]

        return {
            # ── raw projection matrix (ground truth) ──────────────────────────
            # projection_matrix @ [X, Y, Z, 1]  →  [u*w, v*w, w]
            # World frame: sandbox_origin  (X=forward into sandbox, Y=left, Z=up)
            # Pixel frame: u=right (0=left edge), v=down (0=top edge)
            "projection_matrix": P_list,

            # ── intrinsics ────────────────────────────────────────────────────
            # IMPORTANT: cx and cy are NOT at the image centre for a tilted
            # projector.  Godot's Camera3D frustum_offset must account for
            # (cx - width/2) and (cy - height/2) or the position will be wrong.
            "intrinsics": {
                "fx": float(K[0, 0]), "fy": float(K[1, 1]),
                "cx": float(K[0, 2]), "cy": float(K[1, 2]),
                "width": w, "height": h,
                "cx_offset_from_centre_px": cx_off,
                "cy_offset_from_centre_px": cy_off,
            },

            # ── extrinsics ────────────────────────────────────────────────────
            # translation = projector optical centre (camera centre C) in
            #               sandbox_origin frame.  This is NOT the OpenCV
            #               translation vector t = -R @ C.
            # basis       = columns are camera_link axes (X=fwd, Y=left, Z=up)
            #               expressed in sandbox_origin frame.
            "extrinsics": {
                "translation": [float(cam_pos[0]), float(cam_pos[1]), float(cam_pos[2])],
                "basis": [
                    [float(cam_rot[r][c]) for c in range(3)] for r in range(3)
                ],
            },

            # ── diagnostics ───────────────────────────────────────────────────
            "reprojection": {
                "rms_px":    round(rms, 3),
                "max_px":    round(max_err, 3),
                "n_corners": n,
                "n_charuco": n_charuco,
                "n_probe":   n_probe,
            },
        }


def run_tui(stdscr, ros_node):
    curses.curs_set(0)
    stdscr.nodelay(False)
    
    while True:
        stdscr.clear()
        stdscr.addstr(1, 2, "=== AR Sandbox Calibration Menu ===", curses.A_BOLD)
        stdscr.addstr(3, 4, "1. Calibrate Camera (AprilTag)")
        stdscr.addstr(4, 4, "2. Calibrate Projector (ChArUco)")
        stdscr.addstr(6, 4, "q. Quit")
        stdscr.addstr(8, 2, "Select an option: ")
        stdscr.refresh()

        key = stdscr.getch()
        if key == ord('1'):
            camera_calibration_flow(stdscr, ros_node)
        elif key == ord('2'):
            projector_calibration_flow(stdscr, ros_node)
        elif key == ord('q'):
            break

def camera_calibration_flow(stdscr, ros_node):
    stdscr.clear()
    stdscr.addstr(1, 2, "--- Camera Calibration ---", curses.A_BOLD)
    stdscr.addstr(3, 2, "Scanning for active cameras...")
    stdscr.refresh()
    time.sleep(1) # Give ROS a moment to discover topics

    cameras = ros_node.get_available_cameras()
    if not cameras:
        stdscr.addstr(5, 2, "No cameras found! Press any key to return.")
        stdscr.getch()
        return

    stdscr.addstr(5, 2, "Available Cameras:")
    for i, cam in enumerate(cameras):
        stdscr.addstr(6 + i, 4, f"{i + 1}. {cam}")
    
    stdscr.addstr(8 + len(cameras), 2, "Select camera number (or 'q' to cancel): ")
    stdscr.refresh()

    while True:
        key = stdscr.getch()
        if key == ord('q'): return
        idx = key - ord('1')
        if 0 <= idx < len(cameras):
            selected_cam = cameras[idx]
            break

    stdscr.clear()
    stdscr.addstr(1, 2, f"Calibrating {selected_cam}", curses.A_BOLD)
    stdscr.refresh()

    # 1. Turn off IR Emitter on ALL cameras, enable IR stream on selected camera
    stdscr.addstr(3, 2, "[1/6] Disabling IR Emitters on all cameras, enabling IR Stream...")
    stdscr.refresh()
    for cam in cameras:
        subprocess.run(["ros2", "param", "set", cam, "depth_module.emitter_enabled", "0"], capture_output=True)
    subprocess.run(["ros2", "param", "set", selected_cam, "enable_infra1", "true"], capture_output=True)

    stdscr.addstr(4, 2, "[2/6] Publishing Tag static transforms...")
    stdscr.refresh()
    
    calibrations_path = "/tmp/calibrations"
    yaml_path = f"{calibrations_path}/tags.yaml"  # Make sure this matches your container mount path
    tag_tf_processes = []
    
    try:
        with open(yaml_path, 'r') as f:
            tags_config = yaml.safe_load(f)
            
        family = tags_config.get('apriltag', {}).get('ros__parameters', {}).get('family', '36h11')
        positions = tags_config.get('apriltag', {}).get('ros__parameters', {}).get('positions', {})
        
        for tag_id, coords in positions.items():
            frame_id = f"tag{family}:{tag_id}"
            
            cmd = [
                "ros2", "run", "tf2_ros", "static_transform_publisher",
                "--x", str(coords.get('X', 0.0)),
                "--y", str(coords.get('Y', 0.0)),
                "--z", str(coords.get('Z', 0.0)),
                "--roll", str(coords.get('ROLL', 0.0)),
                "--pitch", str(coords.get('PITCH', 0.0)),
                "--yaw", str(coords.get('YAW', 0.0)),
                "--frame-id", frame_id,
                "--child-frame-id", "sandbox_origin"
            ]
            # Spawn in background
            proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            tag_tf_processes.append(proc)
            
    except Exception as e:
        stdscr.addstr(5, 2, f"YAML Error: {str(e)}", curses.A_STANDOUT)
        stdscr.getch()
        return

    # 2. Start AprilTag Node
    stdscr.addstr(5, 2, "[3/6] Starting AprilTag Detector...")
    stdscr.refresh()
    
    apriltag_cmd = [
        "ros2", "run", "apriltag_ros", "apriltag_node",
        "--ros-args",
        "-r", f"image_rect:={selected_cam}/infra1/image_rect_raw",
        "-r", f"camera_info:={selected_cam}/infra1/camera_info",
        "--params-file", yaml_path 
    ]
    tag_process = subprocess.Popen(apriltag_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # 3. Collect and stability-check TF measurements

    # Extract just the camera name for the TF frame (e.g., cam_1234_link)
    cam_frame = f"{selected_cam.split('/')[-1]}_link"

    STABLE_WINDOW    = 20    # consecutive samples used to judge stability
    STABLE_THR_MM    = 2.0   # max std-dev (mm) over window to accept
    MIN_SAMPLES      = 30    # minimum total samples before accepting
    TIMEOUT_S        = 60.0  # give up after this long

    stdscr.addstr(6, 2, f"[4/6] Collecting AprilTag transforms for {cam_frame}...")
    stdscr.addstr(7, 2,  "      Waiting for transform to stabilise...")
    stdscr.refresh()

    samples   = []
    t_start   = time.time()
    transform_data = None   # will hold the final averaged dict

    while time.time() - t_start < TIMEOUT_S:
        tf = ros_node.get_tf('sandbox_origin', cam_frame)
        if tf:
            samples.append(tf)

        std_mm    = None
        stable    = False
        if len(samples) >= STABLE_WINDOW:
            recent = samples[-STABLE_WINDOW:]
            std_xyz = [
                np.std([getattr(s.translation, ax) for s in recent]) * 1000
                for ax in ('x', 'y', 'z')
            ]
            std_mm  = max(std_xyz)
            stable  = std_mm < STABLE_THR_MM and len(samples) >= MIN_SAMPLES

        elapsed   = time.time() - t_start
        remaining = max(0, TIMEOUT_S - elapsed)
        if std_mm is not None:
            status = (f"STABLE  std={std_mm:.2f} mm" if stable
                      else f"UNSTABLE  std={std_mm:.2f} mm  (need <{STABLE_THR_MM:.1f})")
        else:
            status = "gathering..."

        stdscr.addstr(7, 2,
            f"      n={len(samples):3d}  {status:<45s}  {remaining:.0f}s left   ")
        stdscr.refresh()

        if stable:
            to_avg = samples[-STABLE_WINDOW:]
            tx  = float(np.mean([s.translation.x for s in to_avg]))
            ty  = float(np.mean([s.translation.y for s in to_avg]))
            tz  = float(np.mean([s.translation.z for s in to_avg]))
            qx  = float(np.mean([s.rotation.x    for s in to_avg]))
            qy  = float(np.mean([s.rotation.y    for s in to_avg]))
            qz  = float(np.mean([s.rotation.z    for s in to_avg]))
            qw  = float(np.mean([s.rotation.w    for s in to_avg]))
            q_n = float(np.sqrt(qx**2 + qy**2 + qz**2 + qw**2))
            transform_data = {
                "x": tx, "y": ty, "z": tz,
                "qx": qx/q_n, "qy": qy/q_n, "qz": qz/q_n, "qw": qw/q_n,
                "_samples": len(to_avg), "_std_mm": round(std_mm, 3),
            }
            break

        time.sleep(0.1)

    if transform_data is None and samples:
        # Timed out but have some data — use last STABLE_WINDOW (or all)
        to_avg = samples[-STABLE_WINDOW:] if len(samples) >= STABLE_WINDOW else samples
        std_xyz = [
            np.std([getattr(s.translation, ax) for s in to_avg]) * 1000
            for ax in ('x', 'y', 'z')
        ]
        std_mm = max(std_xyz)
        tx  = float(np.mean([s.translation.x for s in to_avg]))
        ty  = float(np.mean([s.translation.y for s in to_avg]))
        tz  = float(np.mean([s.translation.z for s in to_avg]))
        qx  = float(np.mean([s.rotation.x    for s in to_avg]))
        qy  = float(np.mean([s.rotation.y    for s in to_avg]))
        qz  = float(np.mean([s.rotation.z    for s in to_avg]))
        qw  = float(np.mean([s.rotation.w    for s in to_avg]))
        q_n = float(np.sqrt(qx**2 + qy**2 + qz**2 + qw**2))
        transform_data = {
            "x": tx, "y": ty, "z": tz,
            "qx": qx/q_n, "qy": qy/q_n, "qz": qz/q_n, "qw": qw/q_n,
            "_samples": len(to_avg), "_std_mm": round(std_mm, 3), "_unstable": True,
        }
        stdscr.addstr(8, 2,
            f"WARNING: did not stabilise (std={std_mm:.2f} mm). Saving anyway.")
        stdscr.refresh()

    # 4. Clean up & Save
    stdscr.addstr(9, 2, "[5/6] Cleaning up nodes, restoring IR Emitter and disabling IR Stream...")
    stdscr.refresh()
    tag_process.terminate()
    for cam in cameras:
        subprocess.run(["ros2", "param", "set", cam, "depth_module.emitter_enabled", "1"], capture_output=True)
    subprocess.run(["ros2", "param", "set", selected_cam, "enable_infra1", "false"], capture_output=True)

    # Kill all static TF publishers
    for proc in tag_tf_processes:
        proc.terminate()

    stdscr.addstr(10, 2, "[6/6] Processing Results...")
    stdscr.refresh()

    if transform_data:
        os.makedirs(f"{calibrations_path}/tf_configs", exist_ok=True)
        out_file = f"{calibrations_path}/tf_configs/{cam_frame}.json"
        with open(out_file, 'w') as f:
            json.dump(transform_data, f, indent=4)

        std_info = f"  std={transform_data['_std_mm']:.2f} mm over {transform_data['_samples']} samples"
        warn     = "  (UNSTABLE — consider recalibrating)" if transform_data.get('_unstable') else ""
        stdscr.addstr(12, 2, f"SUCCESS! Saved to {out_file}", curses.A_BOLD)
        stdscr.addstr(13, 2, std_info + warn, curses.A_DIM)
        stdscr.refresh()
    else:
        stdscr.addstr(12, 2, "FAILED! No AprilTag transform received. Is the tag visible?",
                      curses.A_STANDOUT)
        stdscr.refresh()

    stdscr.addstr(15, 2, "Press any key to return to menu.")
    stdscr.getch()


def _safe_addstr(stdscr, row, col, text, attr=0):
    """curses.addstr but silently skip if the terminal is too small."""
    max_y, max_x = stdscr.getmaxyx()
    if row >= max_y - 1 or col >= max_x - 1:
        return
    try:
        stdscr.addstr(row, col, text[:max_x - col - 1], attr)
    except curses.error:
        pass


def _draw_coverage_grid(stdscr, top_row, col, grid):
    """
    Draw a REGION_NY × REGION_NX ASCII heat-map of corner coverage.

    Each cell shows one character:
      G (green)   – multi-camera coverage ≥ 50 %
      y (yellow)  – single-camera coverage ≥ 25 %
      . (dim)     – below threshold
    """
    use_color = curses.has_colors()
    for ry, row_data in enumerate(grid):
        parts = []
        for seen, multi, total in row_data:
            frac = seen / total if total > 0 else 0.0
            mfrac = multi / total if total > 0 else 0.0
            if mfrac >= 0.5:
                ch = 'G'
            elif frac >= 0.25:
                ch = 'y'
            else:
                ch = '.'
            parts.append(ch)
        line = '[' + ']['.join(parts) + ']'
        attr = curses.A_DIM
        _safe_addstr(stdscr, top_row + ry, col, line, attr)


# ── pass definitions ──────────────────────────────────────────────────────────

_PASSES = [
    dict(
        label='Pass 1 — Spread',
        hint='Waiting for corners to appear across all board regions.',
        timeout=45,
        stagnation=15,
        done_fn=lambda cal: cal.region_coverage_ok(0.25) and cal.n_observed() >= 15,
        highlight_fn=lambda cal: cal.get_unseen_corners(),
    ),
    dict(
        label='Pass 2 — Density',
        hint='Waiting to accumulate more corner observations.',
        timeout=40,
        stagnation=12,
        done_fn=lambda cal: cal.n_observed() >= 55,
        highlight_fn=lambda cal: cal.get_unseen_corners(),
    ),
    dict(
        label='Pass 3 — Multi-camera',
        hint='Orange corners need to be seen by a second camera.',
        timeout=40,
        stagnation=12,
        done_fn=lambda cal: cal.n_multi_camera() >= 25,
        highlight_fn=lambda cal: cal.get_undersampled_corners(min_cameras=2),
    ),
]


def _run_preflight(ros_node, proj_info, send_board_fn):
    """
    Project a center dot, detect it with all cameras, check triangulation agreement.
    Returns (ok: bool, status_message: str).
    Gracefully skips when fewer than 2 cameras have frames.
    """
    pw, ph = proj_info['width'], proj_info['height']
    pre_snap = {k: v['gray'].copy() for k, v in ros_node._latest_frames.items()}
    send_board_fn(make_dot_image(pw, ph, pw // 2, ph // 2), "pre-flight center dot")
    ros_node._wait_for_visual_change(pre_snap, timeout=8.0)

    frames = dict(ros_node._latest_frames)
    if len(frames) < 2:
        return True, f"Pre-flight skipped (<2 cameras ready, {len(frames)} available)"

    origins_pf, directions_pf = [], []
    for cam_name, fd in frames.items():
        blobs = detect_blob_centroids(fd['gray'])
        if len(blobs) == 1:
            cu, cv_b = blobs[0]
            d = ros_node._compute_ray_direction(
                cu, cv_b, fd['cam_model'], fd['trans'], fd['origin'])
            origins_pf.append(fd['origin'])
            directions_pf.append(d)

    if len(origins_pf) < 2:
        return True, (f"Pre-flight: only {len(origins_pf)}/{len(frames)} cam(s) "
                      f"detected the dot — continuing anyway")
    try:
        pt, residuals = _triangulate_rays(origins_pf, directions_pf)
        max_mm  = float(residuals.max()  * 1000)
        mean_mm = float(residuals.mean() * 1000)
        ok = max_mm < 5.0
        return ok, (f"Pre-flight: {len(origins_pf)} cams  "
                    f"mean={mean_mm:.1f} mm  max={max_mm:.1f} mm  "
                    f"({'OK' if ok else 'WARN >5 mm — consider recalibrating cameras'})")
    except Exception as e:
        return True, f"Pre-flight triangulation error: {e}"


def projector_calibration_flow(stdscr, ros_node):
    stdscr.clear()
    stdscr.addstr(1, 2, "--- Projector Calibration ---", curses.A_BOLD)

    # ── 1. Discover projectors ────────────────────────────────────────────────
    stdscr.addstr(3, 2, "Scanning for active projectors...")
    stdscr.refresh()
    ros_node.discover_projectors()
    time.sleep(1.0)
    projectors = ros_node.discover_projectors()

    if not projectors:
        stdscr.addstr(5, 2, "No projectors found on /projectors/* topics!", curses.A_STANDOUT)
        stdscr.addstr(6, 2, "Is a projector_loader running?")
        stdscr.addstr(8, 2, "Press any key to return.")
        stdscr.getch()
        return

    sorted_topics = sorted(projectors.keys())
    stdscr.addstr(5, 2, "Available Projectors:")
    for i, topic in enumerate(sorted_topics):
        info = projectors[topic]
        stdscr.addstr(6 + i, 4, f"{i + 1}. {info['projector_id']}  ({info['width']}x{info['height']})")
    stdscr.addstr(8 + len(sorted_topics), 2, "Select projector number (or 'q' to cancel): ")
    stdscr.refresh()

    while True:
        key = stdscr.getch()
        if key == ord('q'): return
        idx = key - ord('1')
        if 0 <= idx < len(sorted_topics):
            selected_topic = sorted_topics[idx]
            break

    proj_info = projectors[selected_topic]
    proj_id   = ros_node.get_projector_id(selected_topic)

    # ── 2. Find cameras ───────────────────────────────────────────────────────
    cameras = ros_node.get_available_cameras()
    if not cameras:
        stdscr.addstr(10 + len(sorted_topics), 2, "No cameras found! Press any key to return.")
        stdscr.getch()
        return

    # ── 3. Load config ────────────────────────────────────────────────────────
    stdscr.clear()
    stdscr.addstr(1, 2, f"Calibrating {proj_id} via {len(cameras)} camera(s)", curses.A_BOLD)
    stdscr.addstr(3, 2, "Loading config and connecting to Godot...")
    stdscr.refresh()

    godot_ip   = "127.0.0.1"
    godot_port = 5007
    sandbox_config = {}
    try:
        with open("/ros2_ws/config.json", "r") as f:
            main_cfg       = json.load(f)
            loader_settings = main_cfg.get("loader_settings", {})
            godot_cfg      = loader_settings.get("godot_loader", {})
            godot_ip       = godot_cfg.get("godot_ip",   godot_ip)
            godot_port     = godot_cfg.get("godot_port", godot_port)
            sandbox_config = loader_settings.get("repro_loader", {})
    except Exception:
        pass

    def send_board_to_godot(img, label=""):
        """Encode img as PNG and send calibrate_projector command to Godot."""
        _, buf  = cv2.imencode('.png', img)
        payload = {
            "command":      "calibrate_projector",
            "projector_id": str(proj_id),
            "image_b64":    base64.b64encode(buf).decode('utf-8'),
        }
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(3.0)
                s.connect((godot_ip, godot_port))
                s.sendall(json.dumps(payload).encode('utf-8'))
            if label:
                ros_node._debug_log(f"Board sent to Godot: {label}")
            return True
        except Exception as e:
            ros_node._debug_log(f"Godot board send failed: {e}")
            return False

    # ── 4. Start ROS calibration (subscribes cameras + heightmap) ─────────────
    ros_node.start_projector_calibration(cameras, sandbox_config, proj_info)
    cal = ros_node.calibrator  # convenience alias

    # ── Phase 0: Pre-flight camera agreement ──────────────────────────────────
    if len(cameras) >= 2:
        stdscr.erase()
        _safe_addstr(stdscr, 0, 2, f"Phase 0: Pre-flight  —  {proj_id}", curses.A_BOLD)
        _safe_addstr(stdscr, 2, 2, "Waiting for cameras to initialise...")
        stdscr.refresh()
        # Allow cameras time to produce frames (2 s already spent per camera in start)
        time.sleep(2.0)

        _safe_addstr(stdscr, 2, 2, "Projecting center dot — checking camera agreement...")
        stdscr.refresh()
        pf_ok, pf_msg = _run_preflight(ros_node, proj_info, send_board_to_godot)
        _safe_addstr(stdscr, 4, 2, pf_msg,
                     curses.A_NORMAL if pf_ok else curses.A_STANDOUT)
        if not pf_ok:
            _safe_addstr(stdscr, 6, 2,
                "Press 'a' to abort and recalibrate cameras, any other key to continue.")
            stdscr.refresh()
            curses.flushinp()
            key = stdscr.getch()
            if key == ord('a'):
                ros_node.stop_projector_calibration(cameras)
                stdscr.nodelay(False)
                return
        else:
            _safe_addstr(stdscr, 6, 2, "Cameras agree — starting collection.")
            stdscr.refresh()
            time.sleep(1.0)

    # ── 5. Multi-pass collection loop (Phase 1) ───────────────────────────────
    stdscr.nodelay(True)
    max_y, max_x = stdscr.getmaxyx()
    LOG_TOP    = 13
    GRID_TOP   = 8
    LOG_HEIGHT = max(3, max_y - LOG_TOP - 2)

    for pass_idx, pass_def in enumerate(_PASSES):
        pass_label = pass_def['label']
        pass_hint  = pass_def['hint']

        # Generate and send the initial board for this pass
        highlight   = pass_def['highlight_fn'](cal)
        board_img   = cal.generate_board_image(highlight)
        send_board_to_godot(board_img, pass_label)
        ros_node._debug_log(f"=== {pass_label}: {pass_hint}")

        pass_start     = time.time()
        last_count     = cal.n_observed()
        last_progress  = time.time()
        last_board_send = time.time()
        aborted        = False

        while True:
            now = time.time()

            # ── TUI redraw ──
            stdscr.erase()
            n_obs   = cal.n_observed()
            n_multi = cal.n_multi_camera()
            n_total = cal.n_total()
            elapsed = now - pass_start
            remaining = max(0, pass_def['timeout'] - elapsed)
            stale   = now - last_progress

            _safe_addstr(stdscr, 0, 2,
                f"Projector: {proj_id}  |  {len(cameras)} camera(s)", curses.A_BOLD)
            _safe_addstr(stdscr, 1, 2,
                f"{pass_label}  [{pass_idx+1}/{len(_PASSES)}]  timeout in {remaining:.0f}s")
            _safe_addstr(stdscr, 2, 2, f"Hint: {pass_hint}", curses.A_DIM)

            # Progress bar
            bar_w  = min(38, max_x - 24)
            filled = int(bar_w * min(n_obs, n_total) / n_total)
            bar    = '#' * filled + '-' * (bar_w - filled)
            _safe_addstr(stdscr, 4, 2,
                f"Corners:  [{bar}] {n_obs}/{n_total} ({100*n_obs//max(n_total,1)}%)")
            stereo = ros_node.n_stereo_corners()
            _safe_addstr(stdscr, 5, 2,
                f"Multi-cam: {n_multi}  Stereo: {stereo}  Frames: {ros_node.imagecount}  "
                f"Stagnant: {stale:.0f}s/{pass_def['stagnation']}s")

            hm = "YES" if ros_node.heightmap is not None else "waiting..."
            cam_status = "  ".join(
                f"{n}:{c}" for n, c in ros_node._individual_msg_count.items())
            _safe_addstr(stdscr, 6, 2,
                f"Heightmap: {hm}   Cams: {cam_status}   'q'=abort")

            # Coverage grid
            _safe_addstr(stdscr, GRID_TOP, 2,
                "Coverage (G=2+cams  y=1cam  .=empty):", curses.A_DIM)
            _draw_coverage_grid(stdscr, GRID_TOP + 1, 4, cal.get_coverage_grid())

            # Log
            _safe_addstr(stdscr, LOG_TOP, 2, "Log:", curses.A_DIM)
            for i, line in enumerate(ros_node.debug_lines[-LOG_HEIGHT:]):
                _safe_addstr(stdscr, LOG_TOP + 1 + i, 4,
                    line[:max_x - 6], curses.A_DIM)

            stdscr.refresh()

            # ── refresh board when new corners arrive (throttled) ──
            if n_obs > last_count:
                last_count    = n_obs
                last_progress = now
                if now - last_board_send >= 5.0:
                    highlight = pass_def['highlight_fn'](cal)
                    board_img = cal.generate_board_image(highlight)
                    send_board_to_godot(board_img)
                    last_board_send = now

            # ── exit conditions ──
            if pass_def['done_fn'](cal):
                ros_node._debug_log(
                    f"Pass {pass_idx+1} goal reached "
                    f"({n_obs} corners, {n_multi} multi-cam).")
                break
            if stale >= pass_def['stagnation']:
                ros_node._debug_log(
                    f"Pass {pass_idx+1} stagnated ({stale:.0f}s without new corners).")
                break
            if elapsed >= pass_def['timeout']:
                ros_node._debug_log(
                    f"Pass {pass_idx+1} timed out after {pass_def['timeout']}s.")
                break

            key = stdscr.getch()
            if key == ord('q'):
                aborted = True
                break

            time.sleep(0.1)

        if aborted:
            ros_node.stop_projector_calibration(cameras)
            stdscr.nodelay(False)
            return

    # ── Phase 2: Active dot probing ───────────────────────────────────────────
    stdscr.nodelay(False)
    n_charuco_obs = cal.n_observed()
    if n_charuco_obs >= 10:
        stdscr.erase()
        _safe_addstr(stdscr, 0, 2, f"Phase 2: Active Dot Probing  —  {proj_id}", curses.A_BOLD)
        _safe_addstr(stdscr, 1, 2,
            f"ChArUco: {n_charuco_obs} corners  ({ros_node.n_stereo_corners()} stereo-triangulated)")
        _safe_addstr(stdscr, 2, 2,
            "Projecting dot grid for direct pixel↔world correspondences...")
        _safe_addstr(stdscr, 3, 2, "Press 's' to skip this phase.")
        stdscr.nodelay(True)
        stdscr.refresh()

        probe_status_line = [""]
        def _probe_status(msg):
            probe_status_line[0] = msg
            _safe_addstr(stdscr, 5, 4, msg)
            stdscr.refresh()

        key = stdscr.getch()
        if key != ord('s'):
            stdscr.nodelay(False)
            n_probe = ros_node.run_dot_probe_phase(proj_info, send_board_to_godot, _probe_status)
            ros_node._debug_log(f"Dot probe complete: {n_probe} correspondences.")
            _safe_addstr(stdscr, 7, 2, f"Dot probe: {n_probe} correspondences added.")
            stdscr.refresh()
            time.sleep(1.0)
        else:
            stdscr.nodelay(False)
            ros_node._debug_log("Dot probe skipped by user.")
    else:
        ros_node._debug_log("Dot probe skipped (too few ChArUco corners).")

    # ── 6. Solve ──────────────────────────────────────────────────────────────
    ros_node.stop_projector_calibration(cameras)

    stdscr.erase()
    _safe_addstr(stdscr, 1, 2, f"Solving projection matrix for {proj_id}...", curses.A_BOLD)
    _safe_addstr(stdscr, 2, 2,
        f"ChArUco: {cal.n_observed()} corners  ({cal.n_multi_camera()} multi-cam  "
        f"{ros_node.n_stereo_corners()} stereo)  "
        f"Probe: {len(ros_node._probe_correspondences)}")
    stdscr.refresh()

    extra_pts3d = [p['world'] for p in ros_node._probe_correspondences] or None
    extra_pts2d = [p['pixel'] for p in ros_node._probe_correspondences] or None
    results = ros_node.solve_projector_matrix(extra_pts3d, extra_pts2d)

    if not results:
        _safe_addstr(stdscr, 4, 2, "FAILED! Not enough data to solve.", curses.A_STANDOUT)
        curses.flushinp()
        _safe_addstr(stdscr, 6, 2, "Press any key to return.")
        stdscr.refresh()
        stdscr.getch()
        return

    # ── 7. Verification overlay ───────────────────────────────────────────────
    P = np.array(results['projection_matrix'])
    err_img = cal.generate_error_overlay(P)
    send_board_to_godot(err_img, "Verification overlay (arrows = reprojection error)")
    ros_node._debug_log("Verification overlay sent to Godot.")

    # Publish projector TF for RViz2 verification
    ros_node.publish_projector_tf(proj_id, results)

    intr  = results['intrinsics']
    extri = results['extrinsics']
    repro = results['reprojection']
    pos   = extri['translation']
    cx_off = intr['cx_offset_from_centre_px']
    cy_off = intr['cy_offset_from_centre_px']
    rms_line = (f"RMS {repro['rms_px']:.2f} px   max {repro['max_px']:.1f} px   "
                f"n={repro['n_corners']} "
                f"({repro.get('n_charuco', repro['n_corners'])} charuco"
                f" + {repro.get('n_probe', 0)} probe)")

    stdscr.erase()
    _safe_addstr(stdscr, 0, 2, "Verification overlay on projector — check alignment visually", curses.A_BOLD)
    _safe_addstr(stdscr, 1, 2, rms_line)
    _safe_addstr(stdscr, 2, 2, "Green arrows = error ≤10px   Red arrows = error >10px   (×3 scale)")

    _safe_addstr(stdscr, 4, 2, "── Intrinsics ──", curses.A_DIM)
    _safe_addstr(stdscr, 5, 4,
        f"fx={intr['fx']:.1f}  fy={intr['fy']:.1f}  "
        f"cx={intr['cx']:.1f} ({cx_off:+.0f} from centre)  "
        f"cy={intr['cy']:.1f} ({cy_off:+.0f} from centre)")
    if abs(cx_off) > intr['width'] * 0.05 or abs(cy_off) > intr['height'] * 0.05:
        _safe_addstr(stdscr, 6, 4,
            "LARGE principal point offset — Godot frustum_offset MUST use actual cx/cy",
            curses.A_STANDOUT)

    _safe_addstr(stdscr, 8, 2, "── Projector position (sandbox_origin: X=fwd, Y=left, Z=up) ──", curses.A_DIM)
    _safe_addstr(stdscr, 9, 4,
        f"X={pos[0]:+.3f} m   Y={pos[1]:+.3f} m   Z={pos[2]:+.3f} m")
    _safe_addstr(stdscr, 10, 4,
        "(translation = optical centre C, NOT OpenCV t vector; basis columns = cam_link axes in world)")

    _safe_addstr(stdscr, 12, 2, "Press any key to save and return to menu.")
    stdscr.refresh()
    curses.flushinp()
    stdscr.nodelay(False)
    stdscr.getch()

    # ── 8. Save ───────────────────────────────────────────────────────────────
    calibrations_path = "/tmp/calibrations"
    os.makedirs(f"{calibrations_path}/tf_configs", exist_ok=True)
    out_file = f"{calibrations_path}/tf_configs/projector_{proj_id}.json"
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=4)

    stdscr.erase()
    _safe_addstr(stdscr, 1, 2, f"Saved: {out_file}", curses.A_BOLD)
    _safe_addstr(stdscr, 2, 2, rms_line, curses.A_DIM)
    _safe_addstr(stdscr, 4, 2, "Press any key to return to menu.")
    stdscr.refresh()
    curses.flushinp()
    stdscr.getch()

def main(args=None):
    rclpy.init(args=args)
    ros_node = CalibrationCore()
    spin_thread = threading.Thread(target=rclpy.spin, args=(ros_node,), daemon=True)
    spin_thread.start()

    try:
        curses.wrapper(run_tui, ros_node)
    except KeyboardInterrupt:
        pass
    finally:
        ros_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()