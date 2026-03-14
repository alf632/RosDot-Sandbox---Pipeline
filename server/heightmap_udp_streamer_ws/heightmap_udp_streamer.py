import rclpy
from rclpy.node import Node
from grid_map_msgs.msg import GridMap
import numpy as np
import cv2
import socket

class HeightmapStreamer(Node):
    def __init__(self):
        super().__init__('heightmap_streamer')
        
        # Subscribe to the elevation map
        self.subscription = self.create_subscription(
            GridMap, '/elevation_mapping/elevation_map', self.map_callback, 10)
        
        # UDP Socket Setup (Change IP to your Godot machine if not running on the same PC)
        self.udp_ip = os.environ.get('GODOT_IP', '127.0.0.1')
        self.udp_port = int(os.environ.get('GODOT_PORT', '4242'))
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
        # Physics bounds (in meters). We map these to the 16-bit integer range (0-65535)
        self.min_height = -0.5 # Lowest point of sand
        self.max_height = 0.5  # Highest point of sand/hands
        
        self.get_logger().info(f"Streaming UDP to {self.udp_ip}:{self.udp_port}")

    def map_callback(self, msg):
        try:
            # Find the index of the 'elevation' layer
            layer_idx = msg.layers.index('elevation')
            
            # Extract grid dimensions
            cells_x = msg.info.length_x / msg.info.resolution
            cells_y = msg.info.length_y / msg.info.resolution
            width = int(cells_y)  # Note: GridMap X/Y might be rotated depending on your TF
            height = int(cells_x)
            
            # Extract raw float data and reshape to 2D
            raw_data = np.array(msg.data[layer_idx].data, dtype=np.float32)
            grid_2d = raw_data.reshape((height, width))
            
            # Replace NaNs (unmapped areas) with the minimum height
            grid_2d = np.nan_to_num(grid_2d, nan=self.min_height)
            
            # Normalize to 0.0 - 1.0 range based on our physical bounds
            normalized = (grid_2d - self.min_height) / (self.max_height - self.min_height)
            normalized = np.clip(normalized, 0.0, 1.0)
            
            # Scale to 16-bit integer (0 - 65535)
            img_16bit = (normalized * 65535.0).astype(np.uint16)
            
            # Compress to PNG in memory
            success, encoded_img = cv2.imencode('.png', img_16bit)
            if success:
                # Blast it over UDP
                packet = encoded_img.tobytes()
                if len(packet) < 65500:
                    self.sock.sendto(packet, (self.udp_ip, self.udp_port))
                else:
                    self.get_logger().warn("Packet too large for UDP!")
                    
        except Exception as e:
            self.get_logger().error(f"Error processing map: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = HeightmapStreamer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
