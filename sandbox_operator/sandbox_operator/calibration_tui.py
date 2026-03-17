import rclpy
from rclpy.node import Node
import tf2_ros
import curses
import threading
import subprocess
import time
import json
import os
import socket
import numpy as np
import cv2

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge
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
        self.all_3d_world_points = []
        self.all_2d_godot_pixels = []
        self.subs = [] # Hold dynamically created subscribers

        # Godot ChArUco Board Definition
        self.proj_width = 1920
        self.proj_height = 1080
        self.squares_x = 16
        self.squares_y = 9
        self.square_length_px = 120.0
        
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

    def get_tf(self, target_frame, source_frame):
        try:
            return self.tf_buffer.lookup_transform(target_frame, source_frame, rclpy.time.Time()).transform
        except tf2_ros.TransformException:
            return None

    def start_projector_calibration(self, camera_prefix):
        self.all_3d_world_points.clear()
        self.all_2d_godot_pixels.clear()
        self.is_calibrating_proj = True
        
        # Dynamically subscribe to the chosen camera
        rgb_sub = message_filters.Subscriber(self, Image, f'{camera_prefix}/color/image_raw')
        depth_sub = message_filters.Subscriber(self, Image, f'{camera_prefix}/aligned_depth_to_color/image_raw')
        info_sub = message_filters.Subscriber(self, CameraInfo, f'{camera_prefix}/color/camera_info')
        
        self.ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub, info_sub], 10, 0.1)
        self.ts.registerCallback(self.cam_callback)
        self.subs = [rgb_sub, depth_sub, info_sub]

    def stop_projector_calibration(self):
        self.is_calibrating_proj = False
        self.ts = None
        for sub in self.subs:
            self.destroy_subscription(sub.sub)
        self.subs.clear()

    def cam_callback(self, rgb_msg, depth_msg, info_msg):
        if not self.is_calibrating_proj:
            return

        cv_rgb = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
        cv_depth = self.bridge.imgmsg_to_cv2(depth_msg, "16UC1")
        
        # OpenCV 4.6: Two-step ChArUco detection
        marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(cv_rgb, self.aruco_dict)
        
        charuco_corners, charuco_ids = None, None
        if marker_ids is not None and len(marker_ids) > 0:
            retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                marker_corners, marker_ids, cv_rgb, self.charuco_board
            )
        
        if charuco_corners is not None and len(charuco_corners) > 6:
            cam_model = PinholeCameraModel()
            cam_model.fromCameraInfo(info_msg)
            
            try:
                trans = self.tf_buffer.lookup_transform('sandbox_origin', rgb_msg.header.frame_id, rclpy.time.Time())
            except tf2_ros.TransformException:
                return

            for i in range(len(charuco_corners)):
                corner_id = charuco_ids[i][0]
                u, v = int(charuco_corners[i][0][0]), int(charuco_corners[i][0][1])
                
                z_depth = cv_depth[v, u] / 1000.0 
                if z_depth <= 0.0: continue
                
                ray = np.array(cam_model.projectPixelTo3dRay((u, v)))
                point_3d_cam = ray * (z_depth / ray[2])
                
                # Use standard ROS 2 TF math to rotate/translate the point
                pt_cam = PointStamped()
                pt_cam.point.x, pt_cam.point.y, pt_cam.point.z = float(point_3d_cam[0]), float(point_3d_cam[1]), float(point_3d_cam[2])
                pt_world = tf2_geometry_msgs.do_transform_point(pt_cam, trans)
                
                self.all_3d_world_points.append([pt_world.point.x, pt_world.point.y, pt_world.point.z])
                self.all_2d_godot_pixels.append(self.godot_2d_pixels[corner_id])

    def solve_projector_matrix(self):
        if len(self.all_3d_world_points) < 10:
            return None

        obj_points = np.array([self.all_3d_world_points], dtype=np.float32)
        img_points = np.array([self.all_2d_godot_pixels], dtype=np.float32)
        
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            obj_points, img_points, (self.proj_width, self.proj_height), None, None)
            
        rot_mat, _ = cv2.Rodrigues(rvecs[0])
        tvec = tvecs[0]
        
        # Package for Godot and JSON
        return {
            "intrinsics": {
                "fx": float(mtx[0, 0]), "fy": float(mtx[1, 1]),
                "cx": float(mtx[0, 2]), "cy": float(mtx[1, 2]),
                "width": self.proj_width, "height": self.proj_height
            },
            "extrinsics": {
                "translation": [float(tvec[0][0]), float(-tvec[1][0]), float(-tvec[2][0])],
                "basis": [
                    [float(rot_mat[0][0]), float(rot_mat[1][0]), float(rot_mat[2][0])],
                    [float(-rot_mat[0][1]), float(-rot_mat[1][1]), float(-rot_mat[2][1])],
                    [float(-rot_mat[0][2]), float(-rot_mat[1][2]), float(-rot_mat[2][2])]
                ]
            }
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

    # 1. Turn off IR Emitter
    stdscr.addstr(3, 2, "[1/5] Disabling IR Emitter...")
    stdscr.refresh()
    # Using ROS 2 CLI via subprocess for reliable synchronous parameter setting
    subprocess.run(["ros2", "param", "set", selected_cam, "depth_module.emitter_enabled", "0"], capture_output=True)

    # 2. Start AprilTag Node
    stdscr.addstr(4, 2, "[2/5] Starting AprilTag Detector...")
    stdscr.refresh()
    
    apriltag_cmd = [
        "ros2", "run", "apriltag_ros", "apriltag_node",
        "--ros-args",
        "-r", f"image_rect:={selected_cam}/color/image_raw",
        "-r", f"camera_info:={selected_cam}/color/camera_info",
        "--params-file", "/tmp/tags.yaml" # Ensure this is mounted in your container
    ]
    tag_process = subprocess.Popen(apriltag_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # 3. Wait for TF
    stdscr.addstr(5, 2, "[3/5] Waiting for AprilTag Transform (sandbox_origin -> camera_link)...")
    stdscr.refresh()
    
    # Extract just the camera name for the TF frame (e.g., cam_1234_link)
    cam_frame = f"{selected_cam.split('/')[-1]}_link" 
    transform = None
    
    for _ in range(150): # 15 second timeout (150 * 0.1s)
        transform = ros_node.get_tf('sandbox_origin', cam_frame)
        if transform:
            break
        time.sleep(0.1)

    # 4. Clean up & Save
    stdscr.addstr(6, 2, "[4/5] Stopping Detector and restoring IR Emitter...")
    stdscr.refresh()
    tag_process.terminate()
    subprocess.run(["ros2", "param", "set", selected_cam, "depth_module.emitter_enabled", "1"], capture_output=True)

    stdscr.addstr(7, 2, "[5/5] Processing Results...")
    stdscr.refresh()

    if transform:
        os.makedirs("tf_configs", exist_ok=True)
        out_file = f"tf_configs/{cam_frame}.json"
        data = {
            "x": transform.translation.x, "y": transform.translation.y, "z": transform.translation.z,
            "qx": transform.rotation.x, "qy": transform.rotation.y, "qz": transform.rotation.z, "qw": transform.rotation.w
        }
        with open(out_file, 'w') as f:
            json.dump(data, f, indent=4)
            
        stdscr.addstr(9, 2, f"SUCCESS! Transform saved to {out_file}", curses.color_pair(1) if curses.has_colors() else curses.A_BOLD)
    else:
        stdscr.addstr(9, 2, "FAILED! Timeout waiting for AprilTag TF. Is the tag visible?", curses.A_STANDOUT)

    stdscr.addstr(11, 2, "Press any key to return to menu.")
    stdscr.getch()

def projector_calibration_flow(stdscr, ros_node):
    stdscr.clear()
    stdscr.addstr(1, 2, "--- Projector Calibration ---", curses.A_BOLD)
    
    # 1. Pick Projector
    stdscr.addstr(3, 2, "Enter Projector ID to calibrate (1-4, or 'q' to cancel): ")
    stdscr.refresh()
    while True:
        key = stdscr.getch()
        if key == ord('q'): return
        if ord('1') <= key <= ord('4'):
            proj_id = chr(key)
            break

    # 2. Pick Camera
    cameras = ros_node.get_available_cameras()
    if not cameras:
        stdscr.addstr(5, 2, "No cameras found! Press any key to return.")
        stdscr.getch()
        return

    stdscr.addstr(5, 2, "Select observation camera:")
    for i, cam in enumerate(cameras):
        stdscr.addstr(6 + i, 4, f"{i + 1}. {cam}")
    stdscr.refresh()

    while True:
        key = stdscr.getch()
        idx = key - ord('1')
        if 0 <= idx < len(cameras):
            selected_cam = cameras[idx]
            break

    # 3. Setup UDP to Godot
    udp_ip = "127.0.0.1" # Change to your Godot IP
    udp_port = 5006
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    stdscr.clear()
    stdscr.addstr(1, 2, f"Calibrating Projector {proj_id} via {selected_cam}", curses.A_BOLD)
    stdscr.addstr(3, 2, "[1/4] Instructing Godot to display ChArUco board...")
    stdscr.refresh()
    sock.sendto(json.dumps({"command": "show_charuco", "projector": proj_id}).encode(), (udp_ip, udp_port))
    time.sleep(1.0)

    # 4. Start Point Collection
    stdscr.addstr(4, 2, "[2/4] Collecting ChArUco 3D points...")
    ros_node.start_projector_calibration(selected_cam)
    
    stdscr.nodelay(True) # Make getch non-blocking for live updates
    target_points = 50
    
    while True:
        pts = len(ros_node.all_3d_world_points)
        stdscr.addstr(5, 6, f"Points collected: {pts} / {target_points}  (Press 'q' to abort)")
        stdscr.refresh()
        
        if pts >= target_points:
            break
            
        key = stdscr.getch()
        if key == ord('q'):
            ros_node.stop_projector_calibration()
            stdscr.nodelay(False)
            return
            
        time.sleep(0.1)
        
    stdscr.nodelay(False) # Revert to blocking
    ros_node.stop_projector_calibration()

    # 5. Solve Matrix
    stdscr.addstr(7, 2, "[3/4] Solving Projector Intrinsics & Extrinsics...")
    stdscr.refresh()
    
    results = ros_node.solve_projector_matrix()
    if not results:
        stdscr.addstr(9, 2, "FAILED! OpenCV could not solve the matrix.", curses.A_STANDOUT)
    else:
        # 6. Save and Transmit
        stdscr.addstr(8, 2, "[4/4] Sending data to Godot and saving to disk...")
        
        # Send to Godot
        payload = {"command": "set_calibration", "projector": proj_id, "data": results}
        sock.sendto(json.dumps(payload).encode(), (udp_ip, udp_port))
        
        # Save to Disk
        os.makedirs("tf_configs", exist_ok=True)
        out_file = f"tf_configs/projector_{proj_id}.json"
        with open(out_file, 'w') as f:
            json.dump(results, f, indent=4)
            
        stdscr.addstr(10, 2, f"SUCCESS! Calibration saved to {out_file}", curses.A_BOLD)

    stdscr.addstr(12, 2, "Press any key to return to menu.")
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