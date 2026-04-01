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

class CalibrationCore(Node):
    """Background ROS 2 Node handling TF, Topics, and ChArUco processing"""
    def __init__(self):
        super().__init__('calibration_core')
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.bridge = CvBridge()

        # Projector Calibration State
        self.is_calibrating_proj = False
        self.corner_observations = {}  # corner_id -> (world_point, godot_pixel)
        self.subs = [] # Hold dynamically created subscribers
        self._syncs = [] # Hold message_filters synchronizers (prevent GC)

        # Heightmap state
        self.heightmap = None  # Latest 32FC1 heightmap from /sandbox/heightmap
        self.heightmap_sub = None
        self.sandbox_width = 1.0
        self.sandbox_length = 1.0
        self.heightmap_res_w = 256
        self.heightmap_res_h = 256

        # Projector discovery state (populated from /projectors/* topics)
        self.discovered_projectors = {}  # topic_name -> {width, height, target_ip, target_port}
        self._projector_subs = {}  # topic_name -> subscription

        # Godot ChArUco Board Definition
        self.proj_width = 1920
        self.proj_height = 1080
        self.squares_x = 16
        self.squares_y = 9
        self.square_length_px = 120.0
        self.imagecount = 0
        
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
        self.charuco_board = cv2.aruco.CharucoBoard_create(
            self.squares_x, self.squares_y, self.square_length_px, 90.0, self.aruco_dict
        )
        
        # Bypass the broken OpenCV 4.6 Python bindings by mathematically generating 
        # the 2D Godot pixels. (Corner 0 is at x=120, y=120, progressing row by row).
        self.godot_2d_pixels = np.zeros(( (self.squares_x - 1) * (self.squares_y - 1), 2 ), dtype=np.float32)
        for y in range(self.squares_y - 1):
            for x in range(self.squares_x - 1):
                self.godot_2d_pixels[y * (self.squares_x - 1) + x] = [(x + 1) * self.square_length_px, (y + 1) * self.square_length_px]
        # --------------------------------------
        
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

    def start_projector_calibration(self, cameras, sandbox_config):
        for sub in self.subs:
            self.destroy_subscription(sub.sub)
        self.subs.clear()
        self._syncs.clear()

        # Sandbox geometry for heightmap lookup
        sandbox = sandbox_config.get('sandbox', {})
        self.sandbox_width = sandbox.get('width', 1.0)
        self.sandbox_length = sandbox.get('length', 1.0)
        output_res = sandbox_config.get('output_res', {})
        self.heightmap_res_w = output_res.get('width', 256)
        self.heightmap_res_h = output_res.get('height', 256)

        self.corner_observations.clear()
        self.is_calibrating_proj = True
        self.imagecount = 0
        self.debug_lines = []
        self._individual_msg_count = {}

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

        # Subscribe to RGB + CameraInfo on all cameras
        for cam in cameras:
            subprocess.run(["ros2", "param", "set", cam, "enable_color", "true"], capture_output=True)

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
        for cam in cameras:
            subprocess.run(["ros2", "param", "set", cam, "enable_color", "false"], capture_output=True)

    def cam_callback(self, rgb_msg, info_msg):
        self.imagecount += 1
        frame_id = rgb_msg.header.frame_id
        if not self.is_calibrating_proj:
            return
        if self.heightmap is None:
            self._debug_log("Waiting for heightmap...")
            return

        cv_rgb = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")

        marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(cv_rgb, self.aruco_dict)
        if marker_ids is None or len(marker_ids) == 0:
            return

        retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            marker_corners, marker_ids, cv_rgb, self.charuco_board
        )
        n_charuco = len(charuco_corners) if charuco_corners is not None else 0
        if charuco_corners is None or n_charuco <= 6:
            return

        cam_model = PinholeCameraModel()
        cam_model.fromCameraInfo(info_msg)

        try:
            trans = self.tf_buffer.lookup_transform('sandbox_origin', frame_id, rclpy.time.Time())
        except tf2_ros.TransformException as e:
            self._debug_log(f"TF lookup FAILED ({frame_id}): {e}")
            return

        # Camera origin in sandbox_origin frame
        pt_origin = PointStamped()
        pt_origin.point.x, pt_origin.point.y, pt_origin.point.z = 0.0, 0.0, 0.0
        origin_world = tf2_geometry_msgs.do_transform_point(pt_origin, trans)
        origin = np.array([origin_world.point.x, origin_world.point.y, origin_world.point.z])

        points_added = 0
        for i in range(len(charuco_corners)):
            corner_id = charuco_ids[i][0]
            u, v = float(charuco_corners[i][0][0]), float(charuco_corners[i][0][1])

            ray_cam = np.array(cam_model.projectPixelTo3dRay((u, v)))

            # Transform ray direction to sandbox_origin (rotate only, via two-point method)
            pt_ray = PointStamped()
            pt_ray.point.x, pt_ray.point.y, pt_ray.point.z = float(ray_cam[0]), float(ray_cam[1]), float(ray_cam[2])
            ray_world = tf2_geometry_msgs.do_transform_point(pt_ray, trans)
            direction = np.array([
                ray_world.point.x - origin[0],
                ray_world.point.y - origin[1],
                ray_world.point.z - origin[2]
            ])
            direction = direction / np.linalg.norm(direction)

            if abs(direction[2]) < 1e-6:
                continue

            # Iterative ray-heightmap intersection
            t = -origin[2] / direction[2]  # initial guess: Z=0 plane
            point = origin + t * direction

            converged = False
            for _ in range(3):
                z = self._lookup_height(point[0], point[1])
                if z is None:
                    break
                t = (z - origin[2]) / direction[2]
                point = origin + t * direction
                converged = True

            if converged:
                is_new = corner_id not in self.corner_observations
                self.corner_observations[corner_id] = (
                    [float(point[0]), float(point[1]), float(point[2])],
                    self.godot_2d_pixels[corner_id]
                )
                if is_new:
                    points_added += 1
                if len(self.corner_observations) <= 5 and is_new:
                    self._debug_log(f"  pt3d=[{point[0]:.3f}, {point[1]:.3f}, {point[2]:.3f}] z_hm={z:.3f} cam_origin=[{origin[0]:.3f}, {origin[1]:.3f}, {origin[2]:.3f}]")

        if points_added > 0:
            self._debug_log(f"[{frame_id}] +{points_added} new corners (total unique: {len(self.corner_observations)})")

    def solve_projector_matrix(self):
        if len(self.corner_observations) < 10:
            return None

        pts3d = np.array([obs[0] for obs in self.corner_observations.values()], dtype=np.float64)
        pts2d = np.array([obs[1].tolist() for obs in self.corner_observations.values()], dtype=np.float64)
        n = len(pts3d)

        self._debug_log(f"Solving with {n} unique corners (DLT)")
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

        self._debug_log(f"  fx={K[0,0]:.1f} fy={K[1,1]:.1f} cx={K[0,2]:.1f} cy={K[1,2]:.1f}")
        self._debug_log(f"  position: [{cam_pos[0]:.3f}, {cam_pos[1]:.3f}, {cam_pos[2]:.3f}]")

        # Compute reprojection error
        tvec = -R @ cam_pos
        proj = (K @ (R @ pts3d.T + tvec.reshape(3, 1))).T
        proj = proj[:, :2] / proj[:, 2:3]
        rms = float(np.sqrt(((proj - pts2d) ** 2).sum(axis=1).mean()))
        self._debug_log(f"  RMS reprojection error: {rms:.2f} px")

        # Convert from OpenCV optical frame to ROS camera_link convention
        # optical: X-right, Y-down, Z-forward  →  link: X-forward, Y-left, Z-up
        optical_to_link = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=np.float64)
        cam_rot = R.T @ optical_to_link.T

        # P maps [X,Y,Z,1] in sandbox_origin (ROS: X-fwd, Y-left, Z-up, meters)
        # to projector pixels [u,v] (top-left origin, right+down positive).
        P_list = [[float(P[r][c]) for c in range(4)] for r in range(3)]

        return {
            "intrinsics": {
                "fx": float(K[0, 0]), "fy": float(K[1, 1]),
                "cx": float(K[0, 2]), "cy": float(K[1, 2]),
                "width": self.proj_width, "height": self.proj_height
            },
            "extrinsics": {
                "translation": [float(cam_pos[0]), float(cam_pos[1]), float(cam_pos[2])],
                "basis": [
                    [float(cam_rot[0][0]), float(cam_rot[0][1]), float(cam_rot[0][2])],
                    [float(cam_rot[1][0]), float(cam_rot[1][1]), float(cam_rot[1][2])],
                    [float(cam_rot[2][0]), float(cam_rot[2][1]), float(cam_rot[2][2])]
                ]
            },
            "projection_matrix": P_list
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

    # 3. Wait for TF

    # Extract just the camera name for the TF frame (e.g., cam_1234_link)
    cam_frame = f"{selected_cam.split('/')[-1]}_link" 

    stdscr.addstr(6, 2, f"[4/6] Waiting for AprilTag Transform (sandbox_origin -> {cam_frame})...")
    stdscr.refresh()
    
    transform = None
    
    for _ in range(150): # 15 second timeout (150 * 0.1s)
        transform = ros_node.get_tf('sandbox_origin', cam_frame)
        if transform:
            break
        time.sleep(0.1)

    # 4. Clean up & Save
    stdscr.addstr(7, 2, "[5/6] Cleaning up nodes, restoring IR Emitter and disabling IR Stream...")
    stdscr.refresh()
    tag_process.terminate()
    for cam in cameras:
        subprocess.run(["ros2", "param", "set", cam, "depth_module.emitter_enabled", "1"], capture_output=True)
    subprocess.run(["ros2", "param", "set", selected_cam, "enable_infra1", "false"], capture_output=True)

    # Kill all static TF publishers
    for proc in tag_tf_processes:
        proc.terminate()

    stdscr.addstr(8, 2, "[6/6] Processing Results...")
    stdscr.refresh()

    if transform:
        os.makedirs(f"{calibrations_path}/tf_configs", exist_ok=True)
        out_file = f"{calibrations_path}/tf_configs/{cam_frame}.json"
        data = {
            "x": transform.translation.x, "y": transform.translation.y, "z": transform.translation.z,
            "qx": transform.rotation.x, "qy": transform.rotation.y, "qz": transform.rotation.z, "qw": transform.rotation.w
        }
        with open(out_file, 'w') as f:
            json.dump(data, f, indent=4)
            
        stdscr.addstr(10, 2, f"SUCCESS! Transform saved to {out_file}", curses.color_pair(1) if curses.has_colors() else curses.A_BOLD)
        stdscr.refresh()

    else:
        stdscr.addstr(10, 2, "FAILED! Timeout waiting for AprilTag TF. Is the tag visible?", curses.A_STANDOUT)
        stdscr.refresh()

    stdscr.addstr(12, 2, "Press any key to return to menu.")
    stdscr.getch()


def projector_calibration_flow(stdscr, ros_node):
    stdscr.clear()
    stdscr.addstr(1, 2, "--- Projector Calibration ---", curses.A_BOLD)

    # 1. Discover projectors from /projectors/* topics
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

    # 2. Pick Projector
    while True:
        key = stdscr.getch()
        if key == ord('q'): return
        idx = key - ord('1')
        if 0 <= idx < len(sorted_topics):
            selected_topic = sorted_topics[idx]
            break

    proj_info = projectors[selected_topic]
    proj_id = ros_node.get_projector_id(selected_topic)

    # 3. Find all available cameras
    cameras = ros_node.get_available_cameras()
    if not cameras:
        stdscr.addstr(10 + len(sorted_topics), 2, "No cameras found! Press any key to return.")
        stdscr.getch()
        return

    # 4. Read config
    stdscr.clear()
    stdscr.addstr(1, 2, f"Calibrating Projector {proj_id} via {len(cameras)} camera(s)", curses.A_BOLD)
    stdscr.addstr(3, 2, "[1/4] Generating ChArUco board from config...")
    stdscr.refresh()

    godot_ip = "127.0.0.1"
    godot_port = 5007
    sandbox_config = {}

    try:
        with open("/ros2_ws/config.json", "r") as f:
            main_cfg = json.load(f)
            loader_settings = main_cfg.get("loader_settings", {})
            godot_cfg = loader_settings.get("godot_loader", {})
            godot_ip = godot_cfg.get("godot_ip", godot_ip)
            godot_port = godot_cfg.get("godot_port", godot_port)
            sandbox_config = loader_settings.get("repro_loader", {})
    except Exception:
        pass

    width = proj_info['width']
    height = proj_info['height']

    board_img = ros_node.charuco_board.draw((width, height))
    _, buffer = cv2.imencode('.png', board_img)
    b64_img = base64.b64encode(buffer).decode('utf-8')

    # 5. Send ChArUco board to Godot via TCP
    stdscr.addstr(5, 2, f"[2/4] Instructing Godot ({godot_ip}:{godot_port}) to display board...")
    stdscr.refresh()

    payload = {
        "command": "calibrate_projector",
        "projector_id": str(proj_id),
        "image_b64": b64_img
    }

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(3.0)
            s.connect((godot_ip, godot_port))
            s.sendall(json.dumps(payload).encode('utf-8'))
    except Exception as e:
        stdscr.addstr(7, 2, f"FAILED! Could not reach Godot: {e}", curses.A_STANDOUT)
        stdscr.addstr(9, 2, "Press any key to return.")
        stdscr.getch()
        return

    time.sleep(1.5)

    # 6. Start Live Point Collection (all cameras + merged heightmap)
    ros_node.start_projector_calibration(cameras, sandbox_config)

    stdscr.nodelay(True)
    target_points = 50
    max_y, max_x = stdscr.getmaxyx()
    log_height = max(4, max_y - 10)

    while True:
        stdscr.erase()
        pts = len(ros_node.corner_observations)
        counts = ros_node._individual_msg_count

        stdscr.addstr(0, 2, f"Calibrating Projector {proj_id} via {len(cameras)} camera(s)", curses.A_BOLD)

        bar_width = min(40, max_x - 20)
        filled = int(bar_width * min(pts, target_points) / target_points)
        bar = "#" * filled + "-" * (bar_width - filled)
        stdscr.addstr(2, 2, f"[3/4] Points: [{bar}] {pts}/{target_points}")

        # Per-camera frame counters
        cam_status = "  ".join(f"{n}: {c}" for n, c in counts.items())
        stdscr.addstr(3, 2, f"Synced frames: {ros_node.imagecount}   |   {cam_status}")
        hm_status = "YES" if ros_node.heightmap is not None else "waiting..."
        stdscr.addstr(4, 2, f"Heightmap: {hm_status}   Press 'q' to abort")

        log_top = 6
        log_width = max_x - 6
        stdscr.addstr(log_top, 2, "Log:", curses.A_DIM)
        debug = getattr(ros_node, 'debug_lines', [])
        visible_lines = debug[-(log_height):]
        for i, line in enumerate(visible_lines):
            row = log_top + 1 + i
            if row >= max_y - 1:
                break
            stdscr.addstr(row, 4, line[:log_width], curses.A_DIM)

        stdscr.refresh()

        if pts >= target_points:
            break

        key = stdscr.getch()
        if key == ord('q'):
            ros_node.stop_projector_calibration(cameras)
            stdscr.nodelay(False)
            return

        time.sleep(0.1)

    stdscr.nodelay(False)
    ros_node.stop_projector_calibration(cameras)

    # 7. Solve Matrix
    stdscr.erase()
    stdscr.addstr(1, 2, f"Calibrating Projector {proj_id}", curses.A_BOLD)
    stdscr.addstr(3, 2, "[4/4] Solving Projector Intrinsics & Extrinsics...")
    stdscr.refresh()

    results = ros_node.solve_projector_matrix()

    if not results:
        stdscr.addstr(5, 2, "FAILED! OpenCV could not solve the matrix.", curses.A_STANDOUT)
    else:
        calibrations_path = "/tmp/calibrations"
        os.makedirs(f"{calibrations_path}/tf_configs", exist_ok=True)
        out_file = f"{calibrations_path}/tf_configs/projector_{proj_id}.json"

        with open(out_file, 'w') as f:
            json.dump(results, f, indent=4)

        stdscr.addstr(5, 2, f"SUCCESS! Saved to {out_file}", curses.A_BOLD)
        stdscr.addstr(6, 2, "(The Operator has automatically forwarded this to Godot).")

    curses.flushinp()
    stdscr.addstr(8, 2, "Press any key to return to menu.")
    stdscr.refresh()
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