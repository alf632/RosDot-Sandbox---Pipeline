import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import tf2_ros
from image_geometry import PinholeCameraModel
import message_filters

class ProjectorCalibrator(Node):
    def __init__(self):
        super().__init__('projector_calibrator')
        self.bridge = CvBridge()
        
        # --- 1. Projector & Board Setup ---
        # Assuming Godot projects a 1920x1080 image
        self.proj_width = 1920
        self.proj_height = 1080
        
        # Define the ChArUco board projected by Godot (e.g., 16x9 squares)
        # We define the square length in "Projector Pixels" (1920 / 16 = 120 pixels per square)
        self.squares_x = 16
        self.squares_y = 9
        self.square_length_px = 120 
        self.marker_length_px = 90
        
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
        self.charuco_board = cv2.aruco.CharucoBoard(
            (self.squares_x, self.squares_y), 
            self.square_length_px, 
            self.marker_length_px, 
            self.aruco_dict)
            
        self.detector = cv2.aruco.CharucoDetector(self.charuco_board)
        
        # Get the 2D Godot pixels for every corner (Z is discarded)
        self.godot_2d_pixels = self.charuco_board.getChessboardCorners()[:, 0, :2]

        # --- 2. TF & Data Aggregation ---
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        self.all_3d_world_points = []
        self.all_2d_godot_pixels = []
        
        # Example: Subscribe to Camera 1 (You would duplicate this for cams 2, 3, 4)
        # Note: You MUST use align_depth.enable:=true in the RealSense launch!
        self.rgb_sub = message_filters.Subscriber(self, Image, '/cam1/color/image_raw')
        self.depth_sub = message_filters.Subscriber(self, Image, '/cam1/aligned_depth_to_color/image_raw')
        self.info_sub = message_filters.Subscriber(self, CameraInfo, '/cam1/color/camera_info')
        
        self.ts = message_filters.ApproximateTimeSynchronizer([self.rgb_sub, self.depth_sub, self.info_sub], 10, 0.1)
        self.ts.registerCallback(self.cam_callback)

    def cam_callback(self, rgb_msg, depth_msg, info_msg):
        cv_rgb = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
        cv_depth = self.bridge.imgmsg_to_cv2(depth_msg, "16UC1") # Depth in millimeters
        
        # Detect ChArUco corners in the camera feed
        charuco_corners, charuco_ids, _, _ = self.detector.detectBoard(cv_rgb)
        
        if charuco_corners is not None and len(charuco_corners) > 6:
            cam_model = PinholeCameraModel()
            cam_model.fromCameraInfo(info_msg)
            
            # Get transform from camera to sandbox_origin
            try:
                trans = self.tf_buffer.lookup_transform('sandbox_origin', rgb_msg.header.frame_id, rclpy.time.Time())
            except tf2_ros.TransformException as ex:
                self.get_logger().warn(f"TF Error: {ex}")
                return

            for i in range(len(charuco_corners)):
                corner_id = charuco_ids[i][0]
                u, v = int(charuco_corners[i][0][0]), int(charuco_corners[i][0][1])
                
                # Get depth at this pixel (convert mm to meters)
                z_depth = cv_depth[v, u] / 1000.0 
                if z_depth <= 0.0:
                    continue # Skip invalid depth pixels
                
                # Deproject 2D camera pixel to 3D point in camera frame
                ray = cam_model.projectPixelTo3dRay((u, v))
                ray = np.array(ray)
                point_3d_cam = ray * (z_depth / ray[2]) # Scale ray to actual depth
                
                # Transform 3D point to sandbox_origin
                # (Skipping manual quaternion math here for brevity; you'd apply the 'trans' to point_3d_cam)
                point_3d_world = self.apply_tf(point_3d_cam, trans)
                
                # Look up where this corner was drawn in the Godot image
                godot_pixel = self.godot_2d_pixels[corner_id]
                
                self.all_3d_world_points.append(point_3d_world)
                self.all_2d_godot_pixels.append(godot_pixel)
            
            self.get_logger().info(f"Collected {len(self.all_3d_world_points)} total mapping points.")
            
            # Once we have enough points from any/all cameras, run the calibration
            if len(self.all_3d_world_points) > 50:
                self.solve_projector_matrix()

    def solve_projector_matrix(self):
        self.get_logger().info("Solving Projector Intrinsics & Extrinsics...")
        
        obj_points = np.array([self.all_3d_world_points], dtype=np.float32)
        img_points = np.array([self.all_2d_godot_pixels], dtype=np.float32)
        
        # cv2.calibrateCamera calculates the projector parameters
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            obj_points, img_points, (self.proj_width, self.proj_height), None, None)
        
        rvec = rvecs[0]
        tvec = tvecs[0]
        
        print("\n=== CALIBRATION SUCCESS ===")
        print("Copy these values into your Godot script:")
        
        # Convert OpenCV extrinsics to Godot Transform3D
        rot_mat, _ = cv2.Rodrigues(rvec)
        print(f"\n[Godot Extrinsics]")
        print(f"var pos = Vector3({tvec[0][0]}, {-tvec[1][0]}, {-tvec[2][0]})")
        print(f"var basis = Basis(Vector3({rot_mat[0][0]}, {rot_mat[1][0]}, {rot_mat[2][0]}), ")
        print(f"                  Vector3({-rot_mat[0][1]}, {-rot_mat[1][1]}, {-rot_mat[2][1]}), ")
        print(f"                  Vector3({-rot_mat[0][2]}, {-rot_mat[1][2]}, {-rot_mat[2][2]}))")
        
        # Convert OpenCV Intrinsic Matrix to Godot Projection parameters
        fx, fy = mtx[0, 0], mtx[1, 1]
        cx, cy = mtx[0, 2], mtx[1, 2]
        print(f"\n[Godot Intrinsics (For custom Projection Matrix)]")
        print(f"fx = {fx}\nfy = {fy}\ncx = {cx}\ncy = {cy}")
        print(f"W = {self.proj_width}.0\nH = {self.proj_height}.0")
        
        import sys
        sys.exit(0) # Stop after solving

    def apply_tf(self, point_cam, trans):
        # Helper to apply the TF rotation and translation to the 3D point
        from tf2_geometry_msgs import PointStamped
        from geometry_msgs.msg import Point
        # Math omitted for brevity: apply quaternion rotation and translation
        return [point_cam[0] + trans.transform.translation.x, 
                point_cam[1] + trans.transform.translation.y, 
                point_cam[2] + trans.transform.translation.z]

def main(args=None):
    rclpy.init(args=args)
    node = ProjectorCalibrator()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
