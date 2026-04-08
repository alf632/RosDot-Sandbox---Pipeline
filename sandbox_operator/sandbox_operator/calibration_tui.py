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
from .calibration_camera_refine import (assess_consistency,
                                        refine_camera_translations,
                                        estimated_residuals_after)

# ChArUco board geometry (16×9 @ 120 px fills 1920×1080 exactly)
CHARUCO_SQUARES_X = 16
CHARUCO_SQUARES_Y = 9

# How long to wait for blob detection in each camera before giving up on a dot
PROBE_DOT_TIMEOUT_S = 6.0

# Iterative refinement convergence settings
REFINEMENT_CONFIG = {
    'target_rms_px':      1.5,   # stop when RMS below this
    'target_max_px':      4.0,   # stop when max error below this
    'min_improvement_px': 0.05,  # stop when iteration gains less than this
    'max_iterations':     8,     # hard limit on iterations
    'reprobe_worst_frac': 0.35,  # fraction of probe points to re-probe each iteration
}

CALIBRATIONS_PATH = "/tmp/calibrations"

def _godot_tcp_send(ip, port, payload, timeout=3.0):
    """Fire-and-forget TCP JSON send to Godot. Returns True on success."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            s.connect((ip, port))
            s.sendall(json.dumps(payload).encode('utf-8'))
        return True
    except Exception:
        return False


class CalibrationCore(Node):
    """Background ROS 2 Node handling TF, Topics, and ChArUco processing"""
    def __init__(self):
        super().__init__('calibration_core')
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.bridge = CvBridge()
        self.debug_lines = []

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
        self._probe_correspondences: dict = {}   # corner_id → {world, pixel, residual_m, source, count}
        self._board_dirty: bool = False        # set when new corners added → Phase 1 resends board
        self._collecting_probes: bool = False  # Phase 2 gate: only then cam_callback writes probe corrs
        # Per-camera stats shown in CAMERA BAR (replaces log spam)
        self._cam_stats: dict = {}         # cam_name → {frames, no_marker, few_charuco}
        self._cams_ready_time: dict = {}   # cam_name → monotonic time of first frame
        # Stream health: maps short cam_name → full ros namespace (for recovery)
        self._cam_namespaces: dict = {}    # cam_name → "/host/cam_serial"
        self._recovery_attempts: dict = {} # cam_name → monotonic time of last recovery
        self._warned_frame_ids: set = set()  # frame_ids logged for non-optical warning

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
        self.debug_lines = []
        self._individual_msg_count = {}
        self._corner_rays = {}
        self._stereo_corners = set()
        self._latest_frames = {}
        self._probe_correspondences = {}
        self._board_dirty = False
        self._collecting_probes = False
        self._cam_stats = {}
        self._cams_ready_time = {}

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
            self._cam_namespaces[cam_name] = cam
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

    def check_stream_health(self, stale_threshold=10.0, recovery_cooldown=30.0):
        """
        Check all subscribed cameras for stale streams.  If a camera hasn't
        produced a frame in `stale_threshold` seconds, attempt to toggle its
        color stream off/on.  Returns list of cam_names that were recovered.

        Respects `recovery_cooldown` to avoid spamming toggle attempts.
        """
        now = time.monotonic()
        recovered = []
        for cam_name in list(self._cam_namespaces):
            fd = self._latest_frames.get(cam_name)
            last_frame = fd['timestamp'] if fd else self._cams_ready_time.get(cam_name, 0.0)
            stale_s = now - last_frame if last_frame > 0 else now
            if stale_s < stale_threshold:
                continue
            # Check cooldown
            last_recovery = self._recovery_attempts.get(cam_name, 0.0)
            if now - last_recovery < recovery_cooldown:
                continue
            cam_ns = self._cam_namespaces[cam_name]
            self._debug_log(
                f"[{cam_name}] Stream stale ({stale_s:.0f}s) — toggling color stream")
            self._recovery_attempts[cam_name] = now
            try:
                subprocess.run(
                    ["ros2", "param", "set", cam_ns, "enable_color", "false"],
                    capture_output=True, text=True, timeout=5)
                time.sleep(1.0)
                subprocess.run(
                    ["ros2", "param", "set", cam_ns, "enable_color", "true"],
                    capture_output=True, text=True, timeout=5)
                recovered.append(cam_name)
                self._debug_log(f"[{cam_name}] Recovery toggle sent")
            except Exception as e:
                self._debug_log(f"[{cam_name}] Recovery failed: {e}")
        return recovered

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

        # Only accept optical frames (Z-forward convention required for ray math)
        if not frame_id.endswith('_optical_frame'):
            if frame_id not in self._warned_frame_ids:
                self._warned_frame_ids.add(frame_id)
                self._debug_log(
                    f"WARN: '{frame_id}' is not an optical frame — skipping. "
                    f"Check camera driver TF configuration.")
            return

        cam_name = frame_id.split('_color_')[0]  # e.g. cam_028522074036

        cv_rgb   = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
        raw_gray = cv2.cvtColor(cv_rgb, cv2.COLOR_BGR2GRAY)
        # CLAHE only for ChArUco detection; dot-probe uses raw_gray
        gray = self.calibrator.clahe.apply(raw_gray) if self.calibrator is not None else raw_gray

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
        now_mono = time.monotonic()
        self._latest_frames[cam_name] = {
            'gray':     gray.copy(),      # CLAHE-enhanced — ChArUco detection
            'raw_gray': raw_gray.copy(),  # plain grayscale — dot blob detection
            'info': info_msg,
            'trans': trans, 'origin': origin, 'cam_model': cam_model,
            'timestamp': now_mono,
        }
        if cam_name not in self._cams_ready_time:
            self._cams_ready_time[cam_name] = now_mono

        # Update per-camera stats (shown in CAMERA BAR)
        if cam_name not in self._cam_stats:
            self._cam_stats[cam_name] = {'frames': 0, 'no_marker': 0, 'few_charuco': 0}
        self._cam_stats[cam_name]['frames'] += 1

        if not self.is_calibrating_proj or self.calibrator is None:
            return
        if self.heightmap is None:
            self._debug_log("Waiting for heightmap...")
            return

        marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(
            gray, self.calibrator.aruco_dict,
            parameters=self.calibrator.detector_params)
        if marker_ids is None or len(marker_ids) == 0:
            self._cam_stats[cam_name]['no_marker'] += 1
            return

        retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            marker_corners, marker_ids, gray, self.calibrator.board)
        n_charuco = len(charuco_corners) if charuco_corners is not None else 0
        if charuco_corners is None or n_charuco <= 6:
            self._cam_stats[cam_name]['few_charuco'] += 1
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
                # Wait until all participating cameras have been stable for 3 s
                # to ensure the driver has fully published their TF chains.
                all_warmed = all(
                    now_mono - self._cams_ready_time.get(c, now_mono) >= 3.0
                    for c in rays
                )
                if all_warmed:
                    try:
                        pt, residuals = _triangulate_rays(
                            [r[0] for r in rays.values()],
                            [r[1] for r in rays.values()])
                        # Sanity bounds: reject points outside sandbox (+ 60% margin)
                        in_bounds = (
                            abs(pt[0]) < self.sandbox_width  * 0.6 and
                            abs(pt[1]) < self.sandbox_length * 0.6 and
                            -0.2 < pt[2] < 0.6
                        )
                        if residuals.max() < 0.010 and in_bounds:
                            # First stereo estimate for this corner: discard any
                            # previous heightmap estimate so they don't mix
                            if corner_id not in self._stereo_corners:
                                self._stereo_corners.add(corner_id)
                                self.calibrator.observations.pop(corner_id, None)
                            is_new = self.calibrator.update(
                                corner_id, pt.tolist(), cam_name)
                            if is_new:
                                points_added += 1
                            # Phase 2 probe: running-average per corner (deduped by corner_id)
                            if self._collecting_probes and corner_id in self.calibrator.corner_pixels:
                                new_res  = float(residuals.max())
                                existing = self._probe_correspondences.get(corner_id)
                                if existing is None:
                                    pu, pv = self.calibrator.corner_pixels[corner_id]
                                    self._probe_correspondences[corner_id] = {
                                        'world':      pt.tolist(),
                                        'pixel':      [float(pu), float(pv)],
                                        'residual_m': new_res,
                                        'source':     'stereo',
                                        'count':      1,
                                    }
                                else:
                                    n = existing['count']
                                    existing['world'] = [
                                        (existing['world'][i] * n + pt[i]) / (n + 1)
                                        for i in range(3)]
                                    existing['residual_m'] = (existing['residual_m'] * n + new_res) / (n + 1)
                                    existing['count'] = n + 1
                            continue
                        elif not in_bounds:
                            self._debug_log(
                                f"Stereo pt out of bounds "
                                f"[{pt[0]:.2f},{pt[1]:.2f},{pt[2]:.2f}] — "
                                f"falling back to heightmap")
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
            self._board_dirty = True  # trigger reactive board refresh in Phase 1/2 loop

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
                new_gray = fd.get('raw_gray', fd['gray'])
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

    def _publish_camera_tf(self, cam_frame, tf_json):
        """Immediately republish a corrected camera TF via StaticTransformBroadcaster."""
        t = TransformStamped()
        t.header.stamp    = self.get_clock().now().to_msg()
        t.header.frame_id = 'sandbox_origin'
        t.child_frame_id  = cam_frame
        t.transform.translation.x = float(tf_json['x'])
        t.transform.translation.y = float(tf_json['y'])
        t.transform.translation.z = float(tf_json['z'])
        t.transform.rotation.x = float(tf_json['qx'])
        t.transform.rotation.y = float(tf_json['qy'])
        t.transform.rotation.z = float(tf_json['qz'])
        t.transform.rotation.w = float(tf_json['qw'])
        self._tf_static_broadcaster.sendTransform(t)

    def _apply_consistency_corrections(self, corrections):
        """Apply camera pose corrections to TF JSON files and republish TFs."""
        for cam_name, correction in corrections.items():
            delta_t, delta_r = correction
            t_mm  = float(np.linalg.norm(delta_t) * 1000)
            r_rad = float(np.linalg.norm(delta_r))
            if t_mm < 1.0 and r_rad < 1e-3:   # < 1 mm and < 0.06° — skip
                continue
            cam_frame = f"{cam_name}_link"
            tf_file   = f"{CALIBRATIONS_PATH}/tf_configs/{cam_frame}.json"
            try:
                with open(tf_file) as f:
                    tf_data = json.load(f)
                tf_data['x'] += float(delta_t[0])
                tf_data['y'] += float(delta_t[1])
                tf_data['z'] += float(delta_t[2])
                tf_data['_consistency_correction_mm'] = round(t_mm, 2)
                if r_rad > 1e-6:
                    # Apply rotation correction to existing quaternion.
                    # delta_r is an angle-axis in sandbox_origin frame; it
                    # pre-multiplies the existing rotation so the resulting
                    # ray directions in sandbox frame are R_delta @ d_old.
                    angle = r_rad
                    axis  = delta_r / angle
                    s     = float(np.sin(angle / 2))
                    q_delta = np.array([
                        axis[0]*s, axis[1]*s, axis[2]*s, float(np.cos(angle / 2))])
                    q_old = np.array([
                        tf_data['qx'], tf_data['qy'],
                        tf_data['qz'], tf_data['qw']])
                    # Hamilton product: q_delta * q_old  (sandbox-frame rotation first)
                    x1, y1, z1, w1 = q_delta
                    x2, y2, z2, w2 = q_old
                    q_new = np.array([
                        w1*x2 + x1*w2 + y1*z2 - z1*y2,
                        w1*y2 - x1*z2 + y1*w2 + z1*x2,
                        w1*z2 + x1*y2 - y1*x2 + z1*w2,
                        w1*w2 - x1*x2 - y1*y2 - z1*z2,
                    ])
                    q_new /= np.linalg.norm(q_new)
                    tf_data['qx'] = float(q_new[0])
                    tf_data['qy'] = float(q_new[1])
                    tf_data['qz'] = float(q_new[2])
                    tf_data['qw'] = float(q_new[3])
                    tf_data['_consistency_correction_deg'] = \
                        round(float(np.degrees(r_rad)), 3)
                with open(tf_file, 'w') as f:
                    json.dump(tf_data, f, indent=4)
                self._publish_camera_tf(cam_frame, tf_data)
                rot_str = (f"  rot={np.degrees(r_rad):+.2f}°" if r_rad > 1e-6 else "")
                self._debug_log(
                    f"  {cam_name}: corrected "
                    f"[{delta_t[0]*1000:+.1f},{delta_t[1]*1000:+.1f},"
                    f"{delta_t[2]*1000:+.1f}] mm{rot_str}")
            except FileNotFoundError:
                self._debug_log(f"  {cam_name}: TF file not found, skipping")
            except Exception as e:
                self._debug_log(f"  {cam_name}: correction failed: {e}")

    def run_camera_consistency_phase(self, proj_info, send_board_fn, status_cb=None):
        """
        Project a coarse dot grid, assess inter-camera triangulation consistency,
        and iteratively correct camera poses (translation + rotation).

        Runs up to 3 iterations.  Each iteration re-collects fresh rays using
        corrected TFs, then re-optimises.  Stops early when the max residual
        drops below 5 mm or improvement stalls (<15 %).

        Returns (corrected: bool,
                 before_mm: {cam: initial_residual_mm},
                 after_mm:  {cam: final_residual_mm})
        """
        MAX_ITERS       = 3
        TARGET_MM       = 5.0
        MIN_IMPROVEMENT = 0.15   # 15 %

        positions = make_probe_positions(
            proj_info['width'], proj_info['height'], nx=5, ny=3)

        initial_before_mm = None
        final_after_mm    = {}
        prev_max_mm       = None
        corrected_any     = False

        for iteration in range(MAX_ITERS):
            tag = f"Iter {iteration + 1}/{MAX_ITERS}"

            # 1. Capture fresh baseline + dot rays with current TFs
            if status_cb:
                status_cb(f"{tag}: capturing baseline…")
            baseline = self._capture_probe_baseline(proj_info, send_board_fn, status_cb)
            dot_rays = self._probe_dot_grid(
                proj_info, send_board_fn, positions, baseline, status_cb)
            multi = [r for r in dot_rays if len(r) >= 2]

            if not multi:
                self._debug_log(f"{tag}: no multi-camera dots detected.")
                if iteration == 0:
                    return False, {}, {}
                break   # keep results from previous iteration

            # 2. Measure consistency
            before_m   = assess_consistency(multi)
            iter_mm    = {k: v * 1000 for k, v in before_m.items()}
            max_mm     = max(iter_mm.values(), default=0.0)

            if initial_before_mm is None:
                initial_before_mm = iter_mm

            self._debug_log(
                f"{tag}: " +
                "  ".join(f"{k}={v:.1f}mm" for k, v in iter_mm.items()))

            # 3. Converged?
            if max_mm < TARGET_MM:
                self._debug_log(
                    f"{tag}: {max_mm:.1f}mm < {TARGET_MM}mm — converged.")
                final_after_mm = iter_mm
                break

            # 4. Diminishing returns?
            if prev_max_mm is not None:
                improvement = 1.0 - max_mm / prev_max_mm
                if improvement < MIN_IMPROVEMENT:
                    self._debug_log(
                        f"{tag}: improvement {improvement * 100:.0f}% "
                        f"< {MIN_IMPROVEMENT * 100:.0f}% — stopping.")
                    final_after_mm = iter_mm
                    break

            # 5. Optimise
            self._debug_log(
                f"{tag}: max {max_mm:.1f}mm — optimising camera poses…")
            scale_hint = max(before_m.values(), default=0.0)
            corrections = refine_camera_translations(
                multi, scale_hint_m=scale_hint)
            if not corrections:
                self._debug_log(
                    f"{tag}: optimisation unavailable (scipy missing?).")
                final_after_mm = iter_mm
                break

            # 6. Apply corrections to TF files + republish
            self._apply_consistency_corrections(corrections)
            time.sleep(0.5)   # let TF buffer propagate

            # 7. Estimate post-correction residuals (for logging; next
            #    iteration will measure for real with corrected TFs)
            after_m  = estimated_residuals_after(multi, corrections)
            after_mm = {k: v * 1000 for k, v in after_m.items()}
            final_after_mm = after_mm
            self._debug_log(
                f"{tag} est. after: " +
                "  ".join(f"{k}={v:.1f}mm" for k, v in after_mm.items()))

            prev_max_mm   = max_mm
            corrected_any = True

        if initial_before_mm is None:
            return False, {}, {}
        if not final_after_mm:
            final_after_mm = initial_before_mm
        return corrected_any, initial_before_mm, final_after_mm

    def run_iterative_refinement(self, proj_info, send_board_fn,
                                 send_results_fn, config, status_cb=None):
        """
        Iteratively drop the worst-error probe correspondences and re-solve until
        the reprojection error converges or a limit is hit.

        Uses ChArUco-based probe correspondences from _probe_correspondences.
        Each iteration drops the worst-performing reprobe_frac of correspondences
        (outlier rejection) and re-solves, replacing blob-based re-probing.

        Calls send_results_fn(results) after each improvement so Godot receives
        live updates throughout the process.

        Returns the best results dict (same format as solve_projector_matrix).
        """
        target_rms   = config.get('target_rms_px',      1.5)
        target_max   = config.get('target_max_px',       4.0)
        min_improve  = config.get('min_improvement_px',  0.05)
        max_iter     = config.get('max_iterations',      8)
        reprobe_frac = config.get('reprobe_worst_frac',  0.35)

        # ── Initial solve ────────────────────────────────────────────────────
        probes = list(self._probe_correspondences.values())
        extra_pts3d = [p['world'] for p in probes] or None
        extra_pts2d = [p['pixel'] for p in probes] or None
        results = self.solve_projector_matrix(extra_pts3d, extra_pts2d)
        if results is None:
            return None

        send_results_fn(results)
        best_rms = results['reprojection']['rms_px']
        best_max = results['reprojection']['max_px']
        self._debug_log(
            f"Initial solve: RMS={best_rms:.2f}px  max={best_max:.1f}px  "
            f"n={results['reprojection']['n_corners']}")

        # ── Convergence loop ─────────────────────────────────────────────────
        for iteration in range(max_iter):
            if status_cb:
                status_cb(
                    f"Refinement iter {iteration+1}/{max_iter}  "
                    f"RMS={best_rms:.2f}px  max={best_max:.1f}px")

            if best_rms <= target_rms and best_max <= target_max:
                self._debug_log(
                    f"Converged at iter {iteration+1} "
                    f"(RMS={best_rms:.2f}px ≤ {target_rms}px).")
                break

            if not self._probe_correspondences:
                self._debug_log("No probe correspondences — stopping refinement.")
                break

            # ── Rank probe points by reprojection error ───────────────────
            P = np.array(results['projection_matrix'])
            ranked = []
            for cid, c in self._probe_correspondences.items():
                w    = np.array(c['world'] + [1.0])
                proj = P @ w
                if abs(proj[2]) < 1e-9:
                    continue
                err = float(np.hypot(proj[0]/proj[2] - c['pixel'][0],
                                     proj[1]/proj[2] - c['pixel'][1]))
                ranked.append((err, cid))
            ranked.sort(key=lambda x: x[0], reverse=True)

            # ── Outlier rejection: drop worst-error correspondences ────────
            n_drop   = max(1, int(len(ranked) * reprobe_frac))
            drop_ids = {cid for _, cid in ranked[:n_drop]}
            self._probe_correspondences = {
                cid: c for cid, c in self._probe_correspondences.items()
                if cid not in drop_ids
            }
            self._debug_log(
                f"  iter {iteration+1}: dropped {n_drop} outlier correspondences "
                f"(err {ranked[0][0]:.1f}–{ranked[n_drop-1][0]:.1f}px)  "
                f"remaining {len(self._probe_correspondences)}")

            # ── Re-solve ──────────────────────────────────────────────────
            probes = list(self._probe_correspondences.values())
            extra_pts3d = [p['world'] for p in probes]
            extra_pts2d = [p['pixel'] for p in probes]
            new_results = self.solve_projector_matrix(extra_pts3d, extra_pts2d)
            if new_results is None:
                break

            new_rms = new_results['reprojection']['rms_px']
            new_max = new_results['reprojection']['max_px']
            improvement = best_rms - new_rms

            if improvement > min_improve:
                results  = new_results
                best_rms = new_rms
                best_max = new_max
                send_results_fn(results)
                self._debug_log(
                    f"  → improved: RMS {best_rms+improvement:.2f}→{best_rms:.2f}px")
            else:
                self._debug_log(
                    f"  → no improvement (Δ{improvement:.3f}px) — stopping.")
                break

        return results

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

    # ── Blob-based dot probe — used only by run_camera_consistency_check ──────

    def _capture_probe_baseline(self, proj_info, send_board_fn, status_cb=None):
        """Project black, wait for visual change, return baseline raw_gray dict."""
        pw, ph = proj_info['width'], proj_info['height']
        pre_snap = {k: v['raw_gray'].copy() for k, v in self._latest_frames.items()}
        send_board_fn(make_black_image(pw, ph), "probe baseline (black)")
        if status_cb:
            status_cb("Waiting for black baseline to render on projector…")
        self._wait_for_visual_change(pre_snap, timeout=10.0)
        return {k: v['raw_gray'].copy() for k, v in self._latest_frames.items()}

    def _probe_dot_grid(self, proj_info, send_board_fn, positions, baseline,
                        status_cb=None):
        """
        Project each (u, v) position as a white dot, detect blobs in all cameras.
        Used by the camera consistency check to sample multi-camera ray residuals.

        Returns list of {cam_name: (origin_np, direction_np)} — one dict per dot.
        """
        pw, ph = proj_info['width'], proj_info['height']
        dot_rays = []
        for i, (pu, pv) in enumerate(positions):
            pre_snap = {k: v['raw_gray'].copy() for k, v in self._latest_frames.items()}
            send_board_fn(make_dot_image(pw, ph, pu, pv))
            self._wait_for_visual_change(
                pre_snap, timeout=max(1.0, PROBE_DOT_TIMEOUT_S - 2.0))

            detected = {}
            dot_deadline = time.monotonic() + PROBE_DOT_TIMEOUT_S
            while time.monotonic() < dot_deadline:
                for cam_name, fd in list(self._latest_frames.items()):
                    if cam_name in detected:
                        continue
                    raw  = fd.get('raw_gray', fd['gray'])
                    base = baseline.get(cam_name)
                    blobs = detect_blob_centroids(raw, base)
                    if blobs:
                        detected[cam_name] = (blobs[0], fd)
                if status_cb:
                    elapsed = PROBE_DOT_TIMEOUT_S - max(0, dot_deadline - time.monotonic())
                    status_cb(
                        f"Dot {i+1}/{len(positions)} ({pu},{pv})  "
                        f"{len(detected)}/{len(self._latest_frames)} cam(s)  "
                        f"{elapsed:.1f}s")
                if len(detected) >= len(self._latest_frames):
                    break
                time.sleep(0.05)

            rays = {}
            for cam_name, (centroid, fd) in detected.items():
                cu, cv_b = centroid[0], centroid[1]
                d = self._compute_ray_direction(
                    cu, cv_b, fd['cam_model'], fd['trans'], fd['origin'])
                rays[cam_name] = (fd['origin'].copy(), d)
            dot_rays.append(rays)
        return dot_rays

    # ── ChArUco-based probe — used by Phase 2 of projector_calibration_flow ───

    def _collect_charuco_probes(self, send_board_fn, highlight_fn=None,
                                status_cb=None, timeout=120.0, stagnation=30.0):
        """
        Project the ChArUco board and let cam_callback accumulate probe
        correspondences at varied heights (objects placed on sandbox).

        Only stereo-triangulated corner observations captured while
        _collecting_probes is True are written into _probe_correspondences —
        avoiding contamination from flat-surface corners collected during the
        Phase 2 instruction screen.

        Returns number of probe correspondences added.
        """
        self._probe_correspondences = {}
        self._collecting_probes     = True
        last_n                      = 0
        last_progress_t             = time.monotonic()
        deadline                    = time.monotonic() + timeout
        last_board_t                = 0.0

        try:
            h = highlight_fn(self.calibrator) if highlight_fn else None
            send_board_fn(self.calibrator.generate_board_image(h), "ChArUco probe phase")

            while time.monotonic() < deadline:
                now     = time.monotonic()
                current = len(self._probe_correspondences)

                if current > last_n:
                    last_n          = current
                    last_progress_t = now

                # Reactive board refresh when new corners are detected
                if self._board_dirty and now - last_board_t >= 0.5:
                    self._board_dirty = False
                    h = highlight_fn(self.calibrator) if highlight_fn else None
                    send_board_fn(self.calibrator.generate_board_image(h))
                    last_board_t = now

                # Stream health: auto-recover stale cameras
                self.check_stream_health()

                if status_cb:
                    remaining = max(0.0, deadline - now)
                    stale     = now - last_progress_t
                    status_cb(
                        f"Collecting…  {current} probe correspondences  "
                        f"remaining {remaining:.0f}s  stagnant {stale:.0f}s/{stagnation:.0f}s")

                if now - last_progress_t >= stagnation:
                    self._debug_log(
                        f"ChArUco probe stagnated ({stagnation:.0f}s without new corners).")
                    break

                time.sleep(0.1)

        finally:
            self._collecting_probes = False

        return len(self._probe_correspondences)

    def run_charuco_probe_phase(self, proj_info, send_board_fn, highlight_fn=None,
                                status_cb=None):
        """
        Collect ChArUco probe correspondences while objects at varied heights
        are present on the sandbox surface.  Replaces the blob-based dot probe.

        Probe data (stereo-triangulated ChArUco corners) is collected into
        _probe_correspondences separately from Phase 1 data in cal.observations,
        allowing the solver to use both flat-surface and height-varied points.

        Returns number of probe correspondences added.
        """
        self._probe_correspondences = {}
        return self._collect_charuco_probes(send_board_fn, highlight_fn, status_cb)

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
        self._debug_log(f"  DLT reprojection error: {rms:.2f} px   max: {max_err:.1f} px")

        # --- Nonlinear (LM) refinement: DLT solution → cv2.solvePnP ---
        # Uses Levenberg-Marquardt to minimise the true nonlinear reprojection cost
        # rather than the linear DLT cost.  Only accepted if RMS improves.
        try:
            rvec_init, _ = cv2.Rodrigues(R)
            tvec_cv      = tvec.reshape(3, 1)
            ok_lm, rvec_lm, tvec_lm = cv2.solvePnP(
                pts3d.astype(np.float64),
                pts2d.astype(np.float64),
                K.astype(np.float64),
                np.zeros(4, dtype=np.float64),
                rvec_init.astype(np.float64),
                tvec_cv.astype(np.float64),
                useExtrinsicGuess=True,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if ok_lm:
                proj_lm, _ = cv2.projectPoints(
                    pts3d, rvec_lm, tvec_lm, K, np.zeros(4))
                proj_lm = proj_lm.reshape(-1, 2)
                errs_lm = np.sqrt(((proj_lm - pts2d) ** 2).sum(axis=1))
                rms_lm  = float(np.sqrt((errs_lm ** 2).mean()))
                if rms_lm < rms:
                    R_lm, _ = cv2.Rodrigues(rvec_lm)
                    tvec     = tvec_lm.flatten()
                    cam_pos  = -R_lm.T @ tvec
                    R        = R_lm
                    P        = K @ np.hstack([R_lm, tvec_lm.reshape(3, 1)])
                    if P[2, 3] < 0:
                        P = -P
                    errs    = errs_lm
                    rms     = rms_lm
                    max_err = float(errs.max())
                    self._debug_log(
                        f"  LM refined: RMS={rms:.2f}px  max={max_err:.1f}px")
                else:
                    self._debug_log(
                        f"  LM did not improve DLT "
                        f"(LM={rms_lm:.2f}px  DLT={rms:.2f}px)")
        except Exception as _lm_exc:
            self._debug_log(f"  LM refinement skipped: {_lm_exc}")

        self._debug_log(f"  Final RMS: {rms:.2f} px   max: {max_err:.1f} px")

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

        # Load previous calibration for comparison before overwriting
        _prev_cam_cal = None
        try:
            with open(out_file) as _f:
                _prev_cam_cal = json.load(_f)
        except Exception:
            pass

        with open(out_file, 'w') as f:
            json.dump(transform_data, f, indent=4)

        std_info = f"  std={transform_data['_std_mm']:.2f} mm over {transform_data['_samples']} samples"
        warn     = "  (UNSTABLE — consider recalibrating)" if transform_data.get('_unstable') else ""
        stdscr.addstr(12, 2, f"SUCCESS! Saved to {out_file}", curses.A_BOLD)
        stdscr.addstr(13, 2, std_info + warn, curses.A_DIM)

        # Delta vs. previous calibration
        if _prev_cam_cal is not None:
            try:
                d_x = (transform_data['x'] - _prev_cam_cal['x']) * 1000
                d_y = (transform_data['y'] - _prev_cam_cal['y']) * 1000
                d_z = (transform_data['z'] - _prev_cam_cal['z']) * 1000
                d_mm = float(np.sqrt(d_x**2 + d_y**2 + d_z**2))
                # Rotation delta
                q_old = np.array([_prev_cam_cal['qx'], _prev_cam_cal['qy'],
                                   _prev_cam_cal['qz'], _prev_cam_cal['qw']])
                q_new = np.array([transform_data['qx'], transform_data['qy'],
                                   transform_data['qz'], transform_data['qw']])
                dot = abs(float(np.dot(q_old, q_new)))
                d_deg = float(np.degrees(2 * np.arccos(np.clip(dot, 0.0, 1.0))))
                stdscr.addstr(14, 2,
                    f"  vs. previous: Δpos={d_mm:.1f} mm  "
                    f"(X={d_x:+.1f} Y={d_y:+.1f} Z={d_z:+.1f})  Δrot={d_deg:.2f}°",
                    curses.A_DIM)
            except Exception:
                pass

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


def _draw_calibration_tui(stdscr, ros_node, phase_label, status_msg,
                          cal=None, hint="", coverage_grid=None, footer="'q' abort"):
    """
    Unified TUI draw function used by all active calibration phases.

    Layout (rows top→bottom):
      0        HEADER  — phase label
      1        COUNTERS — corners / stereo / probe / total frames
      2        PROGRESS BAR (when cal is not None)
      3        STATUS  — current step / dot message
      4        HINT    — pass hint (optional)
      5        ─ separator ─
      6..6+grid_h   COVERAGE GRID (optional, 3 rows)
      6+grid_h+1    ─ separator ─ (only when grid shown)
      log zone      DEBUG LOG (scrolling ring buffer, all significant events)
      ─ separator ─
      cam bar zone  CAMERA BAR (one row per camera: frames / marker-hit% / warmup)
      max_y-1       FOOTER key hints
    """
    stdscr.erase()
    max_y, max_x = stdscr.getmaxyx()

    # ── Header ────────────────────────────────────────────────────────────────
    _safe_addstr(stdscr, 0, 2, phase_label, curses.A_BOLD)

    # ── Counters ──────────────────────────────────────────────────────────────
    if cal is not None:
        n_obs   = cal.n_observed()
        n_total = cal.n_total()
        n_multi = cal.n_multi_camera()
        stereo  = ros_node.n_stereo_corners()
        probe   = len(ros_node._probe_correspondences)
        _safe_addstr(stdscr, 1, 2,
            f"corners {n_obs}/{n_total}  multi-cam {n_multi}  "
            f"stereo {stereo}  probe {probe}  frames {ros_node.imagecount}")
        bar_w  = min(36, max_x - 16)
        filled = int(bar_w * min(n_obs, n_total) / max(n_total, 1))
        bar    = '█' * filled + '░' * (bar_w - filled)
        _safe_addstr(stdscr, 2, 2, f"[{bar}] {100*n_obs//max(n_total,1)}%", curses.A_DIM)

    # ── Status / hint ─────────────────────────────────────────────────────────
    _safe_addstr(stdscr, 3, 2, status_msg)
    if hint:
        _safe_addstr(stdscr, 4, 2, hint, curses.A_DIM)

    _safe_addstr(stdscr, 5, 0, '─' * (max_x - 1), curses.A_DIM)

    # ── Coverage grid (optional) ───────────────────────────────────────────────
    content_top = 6
    if coverage_grid is not None:
        _safe_addstr(stdscr, content_top, 2,
                     "Coverage (█=multi-cam  ▒=single  ░=empty):", curses.A_DIM)
        _draw_coverage_grid(stdscr, content_top + 1, 4, coverage_grid)
        grid_rows = 1 + len(coverage_grid)
        content_top += grid_rows + 1
        _safe_addstr(stdscr, content_top - 1, 0, '─' * (max_x - 1), curses.A_DIM)

    # ── Camera bar (bottom, above footer) ─────────────────────────────────────
    n_cams     = max(len(ros_node._cam_stats), 1)
    cam_top    = max_y - n_cams - 1   # one row per camera, one for separator
    _safe_addstr(stdscr, cam_top - 1, 0, '─' * (max_x - 1), curses.A_DIM)
    row = cam_top
    now_m = time.monotonic()
    for cam_name, stats in sorted(ros_node._cam_stats.items()):
        total    = max(stats['frames'], 1)
        miss_pct = 100 * stats['no_marker'] // total
        few_pct  = 100 * stats['few_charuco'] // total
        age      = now_m - ros_node._cams_ready_time.get(cam_name, now_m)
        # Stream freshness
        fd = ros_node._latest_frames.get(cam_name)
        last_ts = fd['timestamp'] if fd else 0.0
        stale_s = now_m - last_ts if last_ts > 0 else 0.0
        suffix = ""
        if age < 3.0:
            suffix = f"  warmup {max(0.0, 3.0 - age):.1f}s"
        elif stale_s > 5.0:
            suffix = f"  STALE {stale_s:.0f}s"
        attr = curses.A_STANDOUT if stale_s > 10.0 else curses.A_DIM
        _safe_addstr(stdscr, row, 2,
            f"{cam_name:<28}  f={stats['frames']}  "
            f"no-marker={miss_pct}%  few-charuco={few_pct}%{suffix}",
            attr)
        row += 1
    _safe_addstr(stdscr, max_y - 1, 2, footer, curses.A_DIM)

    # ── Debug log (middle zone) ────────────────────────────────────────────────
    log_top    = content_top
    log_bottom = cam_top - 2
    log_height = max(1, log_bottom - log_top + 1)
    for i, line in enumerate(ros_node.debug_lines[-log_height:]):
        _safe_addstr(stdscr, log_top + i, 4, line[:max_x - 6], curses.A_DIM)

    stdscr.refresh()


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


def _instruction_screen(stdscr, title, sections, footer):
    """
    Display a full-screen instruction page and wait for the user to confirm.

    sections : list of (heading, [line, ...]) tuples.
                A heading of None emits a blank line instead.
    footer   : list of (text, attr) pairs shown at the bottom.

    Returns True if the user pressed ENTER/SPACE, False if they pressed 'q'.
    """
    stdscr.nodelay(False)
    stdscr.erase()
    max_y, max_x = stdscr.getmaxyx()
    row = 0

    # ── title bar ────────────────────────────────────────────────────────────
    bar = title.center(max_x - 1)
    _safe_addstr(stdscr, row, 0, bar, curses.A_BOLD | curses.A_REVERSE)
    row += 2

    # ── sections ─────────────────────────────────────────────────────────────
    for heading, lines in sections:
        if heading is None:
            row += 1
            continue
        if heading:
            _safe_addstr(stdscr, row, 2, heading, curses.A_BOLD)
            row += 1
        for line in lines:
            _safe_addstr(stdscr, row, 4, line)
            row += 1
        row += 1

    # ── footer ───────────────────────────────────────────────────────────────
    row = max_y - len(footer) - 1
    _safe_addstr(stdscr, row - 1, 0, '─' * (max_x - 1), curses.A_DIM)
    for text, attr in footer:
        _safe_addstr(stdscr, row, 2, text, attr)
        row += 1

    stdscr.refresh()
    while True:
        key = stdscr.getch()
        if key in (ord('\n'), ord('\r'), ord(' ')):
            return True
        if key == ord('q'):
            return False
        if key == ord('s'):
            return None   # skip / alternate action


def _run_preflight(ros_node, proj_info, send_board_fn):
    """
    Project a center dot, detect it with all cameras, check triangulation agreement.
    Returns (ok: bool, status_message: str).
    Gracefully skips when fewer than 2 cameras have frames.
    """
    pw, ph = proj_info['width'], proj_info['height']
    pre_snap = {k: v['raw_gray'].copy() for k, v in ros_node._latest_frames.items()}
    send_board_fn(make_dot_image(pw, ph, pw // 2, ph // 2), "pre-flight center dot")
    ros_node._wait_for_visual_change(pre_snap, timeout=8.0)

    frames = dict(ros_node._latest_frames)
    if len(frames) < 2:
        return True, f"Pre-flight skipped (<2 cameras ready, {len(frames)} available)"

    origins_pf, directions_pf = [], []
    for cam_name, fd in frames.items():
        raw  = fd.get('raw_gray', fd['gray'])
        blobs = detect_blob_centroids(raw)
        if blobs:
            cu, cv_b = blobs[0][0], blobs[0][1]   # best candidate (largest blob)
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
    for cfg_path in ["/config/sandbox.json", "/ros2_ws/config.json"]:
        try:
            with open(cfg_path, "r") as f:
                main_cfg        = json.load(f)
                loader_settings = main_cfg.get("loader_settings", {})
                godot_cfg       = loader_settings.get("godot_loader", {})
                godot_ip        = godot_cfg.get("godot_ip",   godot_ip)
                godot_port      = godot_cfg.get("godot_port", godot_port)
                sandbox_config  = loader_settings.get("repro_loader", {})
                ros_node._debug_log(f"Config loaded from {cfg_path}")
                break
        except Exception:
            continue
    sb = sandbox_config.get('sandbox', {})
    ros_node._debug_log(
        f"Sandbox dimensions: {sb.get('width', 1.0)}×{sb.get('length', 1.0)} m"
        + ("  (DEFAULTS — config not found!)" if not sandbox_config else ""))

    def send_board_to_godot(img, label=""):
        """Encode img as PNG and send update_projector_image command to Godot."""
        _, buf  = cv2.imencode('.png', img)
        payload = {
            "command":      "update_projector_image",
            "projector_id": str(proj_id),
            "image_b64":    base64.b64encode(buf).decode('utf-8'),
        }
        ok = _godot_tcp_send(godot_ip, godot_port, payload)
        if not ok:
            ros_node._debug_log(f"Godot image send failed")
        elif label:
            ros_node._debug_log(f"→ Godot: {label}")
        return ok

    def send_calibration_event(event_name):
        """Send calibration_start or calibration_end to Godot."""
        ok = _godot_tcp_send(godot_ip, godot_port,
                             {"command": event_name, "projector_id": str(proj_id)})
        ros_node._debug_log(
            f"{'→' if ok else '✗'} Godot: {event_name}")

    # ── 4. Start ROS calibration (subscribes cameras + heightmap) ─────────────
    ros_node.start_projector_calibration(cameras, sandbox_config, proj_info)
    cal = ros_node.calibrator  # convenience alias
    send_calibration_event("calibration_start")

    _calibration_stopped = False  # track whether cameras are still running

    def _stop_cameras():
        """Stop camera subscriptions (idempotent)."""
        nonlocal _calibration_stopped
        if not _calibration_stopped:
            _calibration_stopped = True
            ros_node.stop_projector_calibration(cameras)

    def _stop_and_notify():
        """Stop cameras and immediately tell Godot calibration ended (abort paths)."""
        _stop_cameras()
        send_calibration_event("calibration_end")

    # ── Phase 0: Pre-flight camera agreement ──────────────────────────────────
    if len(cameras) >= 2:
        stdscr.erase()
        _safe_addstr(stdscr, 0, 2, f"Phase 0: Pre-flight  —  {proj_id}", curses.A_BOLD)
        _safe_addstr(stdscr, 2, 2, "Waiting for cameras to initialise...")
        stdscr.refresh()
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
                _stop_and_notify()
                stdscr.nodelay(False)
                return
        else:
            _safe_addstr(stdscr, 6, 2, "Cameras agree.")
            stdscr.refresh()
            time.sleep(0.5)

    # ── Phase 0.5: Camera consistency refinement ───────────────────────────────
    if len(cameras) >= 2:
        ok = _instruction_screen(stdscr,
            " Phase 0.5 — Camera Consistency Check ",
            [
                ("Why this matters:", [
                    "  Each camera was calibrated against the AprilTag independently.",
                    "  Small errors (2–5 mm) in different directions per camera cause",
                    "  inconsistent 3-D point estimates and degrade the projector solve.",
                    "  This phase projects a coarse dot grid, measures inter-camera",
                    "  triangulation residuals, and applies position corrections.",
                ]),
                (None, []),
                ("Requirements:", [
                    "  • Cameras must have a clear view of the sandbox.",
                    "  • The sandbox surface can be in any state — dots are triangulated",
                    "    in 3-D so no heightmap accuracy is needed.",
                    "  • This takes about 1–2 minutes.",
                ]),
                (None, []),
                ("What will happen:", [
                    "  • 15 white dots are projected one at a time (up to 3 iterations).",
                    "  • Both cameras triangulate each dot independently.",
                    "  • If inter-camera residuals exceed 5 mm, poses are optimised",
                    "    and the camera TF files are updated. Each iteration re-measures",
                    "    with corrected TFs for progressive refinement.",
                ]),
            ],
            [
                ("ENTER / SPACE  — run consistency check", curses.A_BOLD),
                ("s              — skip (continue with raw AprilTag calibration)", curses.A_DIM),
                ("q              — cancel", curses.A_DIM),
            ],
        )
        if ok is False:
            _stop_and_notify()
            return
        if ok is True:
            stdscr.erase()
            _safe_addstr(stdscr, 0, 2,
                f"Phase 0.5: Camera Consistency  —  {proj_id}", curses.A_BOLD)
            _safe_addstr(stdscr, 2, 2, "Status:")
            stdscr.refresh()

            def _cc_status(msg):
                _safe_addstr(stdscr, 4, 4, msg.ljust(80))
                stdscr.refresh()

            corrected, before_mm, after_mm = ros_node.run_camera_consistency_phase(
                proj_info, send_board_to_godot, _cc_status)

            # ── Results display ──
            stdscr.erase()
            _safe_addstr(stdscr, 0, 2,
                f"Phase 0.5: Camera Consistency  —  {proj_id}", curses.A_BOLD)

            row = 2
            for cam, mm in before_mm.items():
                after = after_mm.get(cam, mm)
                arrow = f" -> {after:.1f} mm" if corrected else ""
                _safe_addstr(stdscr, row, 4,
                    f"{cam:<30s}  initial={mm:.1f} mm{arrow}")
                row += 1

            row += 1
            if corrected:
                _safe_addstr(stdscr, row, 2,
                    "Corrections applied — TF files updated.", curses.A_BOLD)
            else:
                _safe_addstr(stdscr, row, 2,
                    "Cameras consistent — no correction needed.", curses.A_DIM)
            row += 1

            # ── Quality gate ──
            final_max_mm = max(after_mm.values(), default=0.0)
            if final_max_mm > 15.0:
                row += 1
                worst_cam = max(after_mm, key=after_mm.get)
                _safe_addstr(stdscr, row, 2,
                    f"WARNING: {worst_cam} still has {final_max_mm:.0f}mm "
                    f"residual — heightmap distortion likely.",
                    curses.A_STANDOUT)
                row += 1
                _safe_addstr(stdscr, row, 2,
                    "Consider re-running camera calibration or "
                    "checking AprilTag visibility.")
                row += 1
                _safe_addstr(stdscr, row, 2,
                    "Press 'a' to abort, any other key to continue.")
                stdscr.refresh()
                curses.flushinp()
                stdscr.nodelay(False)
                key = stdscr.getch()
                if key == ord('a'):
                    _stop_and_notify()
                    return
            else:
                row += 1
                _safe_addstr(stdscr, row, 2, "Press any key to continue.")
                stdscr.refresh()
                curses.flushinp()
                stdscr.nodelay(False)
                stdscr.getch()

            # ── Write per-camera quality weights to TF JSON ──
            for cam_name in after_mm:
                res_mm = after_mm[cam_name]
                weight = max(0.1, 1.0 - res_mm / 50.0)
                cam_frame = f"{cam_name}_link"
                tf_file = f"{CALIBRATIONS_PATH}/tf_configs/{cam_frame}.json"
                try:
                    with open(tf_file) as f:
                        tf_data = json.load(f)
                    tf_data['_quality_weight'] = round(weight, 3)
                    with open(tf_file, 'w') as f:
                        json.dump(tf_data, f, indent=4)
                except Exception:
                    pass

    # ── Phase 1 setup instructions ────────────────────────────────────────────
    ok = _instruction_screen(stdscr,
        " Phase 1 — ChArUco Corner Collection ",
        [
            ("Prepare the sandbox surface:", [
                "  • Clear the sandbox completely — remove all objects,",
                "    sand hills, and debris.",
                "  • Level the surface as flat as possible.",
                "    A flat floor gives the cleanest initial corner positions.",
                "  • The projector will display a ChArUco pattern across the",
                "    entire sandbox area. Every corner the cameras can see",
                "    contributes one 3-D correspondence point.",
                "  • Remove the AprilTag from the scene — it is no longer needed.",
            ]),
            (None, []),
            ("Camera notes:", [
                "  • Camera extrinsics were established in the camera calibration",
                "    step — no AprilTag is needed here.",
                "  • Cameras do not need a perfect top-down view; any angle that",
                "    lets them see the projected pattern on the surface works.",
                "  • The ChArUco pattern is projected by the projector itself.",
                "    Ambient light does NOT help — dim the room if possible to",
                "    maximise contrast between projected dots and the surface.",
            ]),
            (None, []),
            ("What will happen:", [
                f"  • {len(cameras)} camera(s) will be activated one by one.",
                "  • Three collection passes run automatically:",
                "      Pass 1 – spread corners across all regions",
                "      Pass 2 – increase density",
                "      Pass 3 – ensure multi-camera overlap",
                "  • Each pass ends when its goal is met or it times out.",
            ]),
        ],
        [
            ("ENTER / SPACE  — surface is ready, start collection", curses.A_BOLD),
            ("q              — cancel and return to menu",            curses.A_DIM),
        ],
    )
    if not ok:
        _stop_and_notify()
        return

    # ── 5. Multi-pass collection loop (Phase 1) ───────────────────────────────
    stdscr.nodelay(True)

    for pass_idx, pass_def in enumerate(_PASSES):
        pass_label = pass_def['label']
        pass_hint  = pass_def['hint']

        # Generate and send the initial board for this pass
        highlight   = pass_def['highlight_fn'](cal)
        board_img   = cal.generate_board_image(highlight)
        send_board_to_godot(board_img, pass_label)
        ros_node._debug_log(f"=== {pass_label}: {pass_hint}")

        pass_start      = time.time()
        last_count      = cal.n_observed()
        last_progress   = time.time()
        last_board_send = time.time()
        aborted         = False

        while True:
            now     = time.time()
            elapsed = now - pass_start
            remaining = max(0, pass_def['timeout'] - elapsed)
            stale   = now - last_progress
            n_obs   = cal.n_observed()
            n_multi = cal.n_multi_camera()
            hm      = "YES" if ros_node.heightmap is not None else "waiting…"

            _draw_calibration_tui(
                stdscr, ros_node,
                phase_label=f"Phase 1 — ChArUco  [{pass_idx+1}/{len(_PASSES)}]  {proj_id}",
                status_msg=(
                    f"{pass_label}  |  timeout {remaining:.0f}s  "
                    f"stagnant {stale:.0f}s/{pass_def['stagnation']}s  "
                    f"heightmap {hm}"),
                cal=cal,
                hint=pass_hint,
                coverage_grid=cal.get_coverage_grid(),
                footer="'q' abort",
            )

            # ── refresh board when new corners arrive (throttled to 0.5 s) ──
            if n_obs > last_count:
                last_count    = n_obs
                last_progress = now
            if ros_node._board_dirty and now - last_board_send >= 0.5:
                ros_node._board_dirty = False
                highlight = pass_def['highlight_fn'](cal)
                board_img = cal.generate_board_image(highlight)
                send_board_to_godot(board_img)
                last_board_send = now

            # ── stream health: auto-recover stale cameras ──
            ros_node.check_stream_health()

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
            _stop_and_notify()
            stdscr.nodelay(False)
            return

    # ── Phase 2 setup instructions ────────────────────────────────────────────
    stdscr.nodelay(False)
    ok = _instruction_screen(stdscr,
        " Phase 2 — ChArUco Probe (height variation required) ",
        [
            ("Why height variation matters:", [
                "  With all points on a flat surface the solver cannot",
                "  distinguish between projector tilt and principal-point offset.",
                "  Adding objects at different heights breaks this degeneracy",
                "  and gives accurate focal length and principal point (cx, cy).",
                "",
                f"  Current result without height variation:",
                f"    ChArUco corners collected: {cal.n_observed()}",
                f"    Stereo-triangulated:        {ros_node.n_stereo_corners()}",
            ]),
            (None, []),
            ("Place objects on the sandbox surface NOW:", [
                "  • 2–4 objects spread across different areas of the sandbox.",
                "  • Target heights: 10–30 cm above the floor.",
                "  • Objects with flat horizontal tops work best",
                "    (cardboard boxes, stacked boards, wooden blocks).",
                "  • Each object should cover at least 20 × 20 cm so that",
                "    several board corners land on it.",
                "  • Objects can be irregular — exact height does not need to",
                "    be known; the cameras will triangulate the true 3-D position.",
            ]),
            (None, []),
            ("What will happen:", [
                "  • The projector continues to display the ChArUco board.",
                "  • Cameras detect board corners on the object surfaces,",
                "    giving correspondences at different heights.",
                "  • The board updates in real time as new corners are found:",
                "    red = unseen  yellow = 1 camera  green = stereo-triangulated.",
                "  • Collection stops automatically when no new corners arrive",
                "    for 30 s, or after 2 min maximum.",
            ]),
        ],
        [
            ("ENTER / SPACE  — objects placed, start ChArUco probe", curses.A_BOLD),
            ("s              — skip height variation (coarser calibration)", curses.A_DIM),
            ("q              — cancel and return to menu",              curses.A_DIM),
        ],
    )
    if ok is False:   # 'q' — cancel
        _stop_and_notify()
        return

    skip_probe = (ok is None)   # 's' — skip height variation

    # ── Phase 2: ChArUco probe at varied heights ──────────────────────────────
    n_charuco_obs = cal.n_observed()
    if not skip_probe and n_charuco_obs >= 10:
        def _probe_status(msg):
            _draw_calibration_tui(
                stdscr, ros_node,
                phase_label=f"Phase 2 — ChArUco Probe  {proj_id}",
                status_msg=msg,
                cal=cal,
                coverage_grid=cal.get_coverage_grid(),
                footer="'s' skip")

        n_probe = ros_node.run_charuco_probe_phase(
            proj_info, send_board_to_godot,
            highlight_fn=lambda c: c.get_unseen_corners(),
            status_cb=_probe_status)
        ros_node._debug_log(f"ChArUco probe complete: {n_probe} correspondences.")
        _draw_calibration_tui(
            stdscr, ros_node,
            phase_label=f"Phase 2 — ChArUco Probe  {proj_id}",
            status_msg=f"Done — {n_probe} probe correspondences added.",
            cal=cal)
        time.sleep(1.0)
    else:
        if skip_probe:
            ros_node._debug_log("ChArUco probe skipped (height variation skipped by user).")
        else:
            ros_node._debug_log("ChArUco probe skipped (too few ChArUco corners).")

    # ── 6. Iterative refinement + live Godot updates ──────────────────────────
    os.makedirs(f"{CALIBRATIONS_PATH}/tf_configs", exist_ok=True)
    out_file = f"{CALIBRATIONS_PATH}/tf_configs/projector_{proj_id}.json"

    # Load previous calibration before overwriting — used for delta display
    _prev_proj_cal = None
    try:
        with open(out_file) as _f:
            _prev_proj_cal = json.load(_f)
    except Exception:
        pass

    def _save_results(res):
        with open(out_file, 'w') as f:
            json.dump(res, f, indent=4)

    def _send_calibration_to_godot(res):
        """Send reprojection error overlay and publish TF."""
        try:
            P_live = np.array(res['projection_matrix'])
            err_img = cal.generate_error_overlay(P_live)
            send_board_to_godot(err_img, "Live calibration overlay")
        except Exception as exc:
            ros_node._debug_log(f"Overlay error: {exc}")
        ros_node.publish_projector_tf(proj_id, res)

    def send_results_fn(res):
        _save_results(res)
        _send_calibration_to_godot(res)
        repro = res['reprojection']
        ros_node._debug_log(
            f"Live update → RMS={repro['rms_px']:.2f}px  "
            f"max={repro['max_px']:.1f}px  n={repro['n_corners']}")

    _draw_calibration_tui(
        stdscr, ros_node,
        phase_label=f"Phase 3 — Iterative Refinement  {proj_id}",
        status_msg="Starting initial solve…",
        cal=cal)

    def _refine_status(msg):
        _draw_calibration_tui(
            stdscr, ros_node,
            phase_label=f"Phase 3 — Iterative Refinement  {proj_id}",
            status_msg=msg,
            cal=cal)

    results = ros_node.run_iterative_refinement(
        proj_info, send_board_to_godot, send_results_fn,
        REFINEMENT_CONFIG, _refine_status)

    # Stop cameras now — Godot stays in calibration mode until user confirms.
    _stop_cameras()

    if not results:
        stdscr.erase()
        _safe_addstr(stdscr, 2, 2, "FAILED! Not enough data to solve.", curses.A_STANDOUT)
        curses.flushinp()
        stdscr.nodelay(False)
        _safe_addstr(stdscr, 4, 2, "Press any key to return.")
        stdscr.refresh()
        stdscr.getch()
        send_calibration_event("calibration_end")
        return

    # ── 7. Verification screen ────────────────────────────────────────────────
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

    # Send final overlay now — Godot is still in calibration mode so it will
    # display it. calibration_end is sent after the user confirms below.
    _send_calibration_to_godot(results)

    stdscr.erase()
    _safe_addstr(stdscr, 0, 2, "Calibration complete — check overlay on projector", curses.A_BOLD)
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

    # ── Delta vs. previous calibration ───────────────────────────────────────
    detail_row = 12
    if _prev_proj_cal is not None:
        try:
            prev_pos = _prev_proj_cal['extrinsics']['translation']
            prev_rms = _prev_proj_cal['reprojection']['rms_px']
            d_xyz_mm = [abs(pos[i] - prev_pos[i]) * 1000 for i in range(3)]
            d_rms    = repro['rms_px'] - prev_rms
            # Rotation delta: angle between old and new basis matrices
            prev_basis = np.array(_prev_proj_cal['extrinsics']['basis'], dtype=np.float64)
            new_basis  = np.array(results['extrinsics']['basis'],         dtype=np.float64)
            R_rel  = new_basis.T @ prev_basis
            cos_a  = np.clip((np.trace(R_rel) - 1) / 2, -1.0, 1.0)
            d_deg  = float(np.degrees(np.arccos(cos_a)))
            rms_sym = '↓' if d_rms < 0 else '↑'
            _safe_addstr(stdscr, detail_row, 2,
                "── vs. previous calibration ──", curses.A_DIM)
            _safe_addstr(stdscr, detail_row + 1, 4,
                f"position Δ  X={d_xyz_mm[0]:.1f} mm  "
                f"Y={d_xyz_mm[1]:.1f} mm  Z={d_xyz_mm[2]:.1f} mm  "
                f"rot={d_deg:.2f}°")
            _safe_addstr(stdscr, detail_row + 2, 4,
                f"RMS {prev_rms:.2f} px → {repro['rms_px']:.2f} px  "
                f"({rms_sym}{abs(d_rms):.2f} px)",
                curses.A_BOLD if d_rms < 0 else curses.A_DIM)
            detail_row += 4
        except Exception:
            pass

    _safe_addstr(stdscr, detail_row,     2, f"Saved: {out_file}", curses.A_DIM)
    _safe_addstr(stdscr, detail_row + 1, 2, "Press any key to return to menu.")
    stdscr.refresh()
    curses.flushinp()
    stdscr.nodelay(False)
    stdscr.getch()
    # User confirmed — tell Godot to resume normal operation
    send_calibration_event("calibration_end")

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