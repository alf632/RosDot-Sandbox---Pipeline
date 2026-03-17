import pyrealsense2 as rs

class RealSenseLoader:
    def discover_and_load(self, operator, config):
        """Discovers physical RealSense cameras and loads a driver for each."""
        try:
            ctx = rs.context()
            devices = ctx.query_devices()
            
            if len(devices) == 0:
                operator.get_logger().warn("RealSenseLoader: No cameras found on USB bus.")
                return

            for dev in devices:
                sn = dev.get_info(rs.camera_info.serial_number)
                node_name = f"cam_{sn}"
                
                params = {
                    'serial_no': sn,
                    'depth_module.profile': '640x480x30',
                    'camera_name': node_name,
                    'enable_color': False,
                    'enable_infra1': False,
                    'enable_infra2': False,
                    'pointcloud.enable': False,
                    'align_depth.enable': False
                }
                
                # Load into the device's specific namespace
                operator.load_component(
                    package='realsense2_camera',
                    plugin='realsense2_camera::RealSenseNodeFactory',
                    name=node_name,
                    params=params,
                    namespace=operator.device_namespace,
                    use_ipc=True
                )
        except Exception as e:
            operator.get_logger().error(f"RealSense Discovery Failed: {e}")
