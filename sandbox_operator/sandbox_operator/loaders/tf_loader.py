import os
import json
import glob
import numpy as np
from geometry_msgs.msg import TransformStamped
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster

class TfLoader:
    def __init__(self):
        # Memory dictionary to track file modifications: {filepath: modification_time}
        self.known_files = {}

    def discover_and_load(self, operator, config):
        cfg = config.get('tf_loader', {})
        config_dir = cfg.get('config_dir', 'tf_configs') 
        
        # Attach a Static Broadcaster to the Operator if it doesn't have one
        if not hasattr(operator, 'static_tf_broadcaster'):
            operator.static_tf_broadcaster = StaticTransformBroadcaster(operator)
            
        search_path = os.path.join(config_dir, '*.json')
        current_files = glob.glob(search_path)
        
        # 1. Quick check: Did any files change or get added?
        needs_update = False
        for filepath in current_files:
            mtime = os.path.getmtime(filepath)
            if filepath not in self.known_files or self.known_files[filepath] != mtime:
                needs_update = True
                break
                
        # If nothing changed, exit silently. No log spam, zero CPU hit.
        if not needs_update:
            return 
            
        # 2. A change was detected! Parse and publish.
        transforms = []
        new_known_files = {}
        
        for filepath in current_files:
            filename = os.path.basename(filepath)
            child_frame = filename.replace('.json', '')
            mtime = os.path.getmtime(filepath)
            
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    
                t = TransformStamped()
                t.header.stamp = operator.get_clock().now().to_msg()
                t.header.frame_id = 'sandbox_origin'
                t.child_frame_id = child_frame

                if 'qx' in data:
                    t.transform.translation.x = float(data['x'])
                    t.transform.translation.y = float(data['y'])
                    t.transform.translation.z = float(data['z'])
                    t.transform.rotation.x = float(data['qx'])
                    t.transform.rotation.y = float(data['qy'])
                    t.transform.rotation.z = float(data['qz'])
                    t.transform.rotation.w = float(data['qw'])
                    
                elif 'extrinsics' in data:
                    ext = data['extrinsics']
                    trans = ext['translation']
                    basis = ext['basis']
                    
                    t.transform.translation.x = float(trans[0])
                    t.transform.translation.y = float(trans[1])
                    t.transform.translation.z = float(trans[2])
                    
                    qx, qy, qz, qw = self.matrix_to_quaternion(basis)
                    t.transform.rotation.x = float(qx)
                    t.transform.rotation.y = float(qy)
                    t.transform.rotation.z = float(qz)
                    t.transform.rotation.w = float(qw)
                    
                transforms.append(t)
                new_known_files[filepath] = mtime # Update our memory
                    
            except Exception as e:
                operator.get_logger().error(f"Failed to load TF from {filename}: {e}")
                
        # 3. Broadcast and update state
        if transforms:
            operator.static_tf_broadcaster.sendTransform(transforms)
            operator.get_logger().info(f"Detected TF changes! Broadcasted {len(transforms)} static transforms.")
            self.known_files = new_known_files # Lock in the new state

    def matrix_to_quaternion(self, m):
        # [Keep your existing math helper here]
        m = np.array(m)
        tr = m[0][0] + m[1][1] + m[2][2]
        if tr > 0:
            S = np.sqrt(tr + 1.0) * 2
            qw = 0.25 * S
            qx = (m[2][1] - m[1][2]) / S
            qy = (m[0][2] - m[2][0]) / S
            qz = (m[1][0] - m[0][1]) / S
        elif (m[0][0] > m[1][1]) and (m[0][0] > m[2][2]):
            S = np.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2
            qw = (m[2][1] - m[1][2]) / S
            qx = 0.25 * S
            qy = (m[0][1] + m[1][0]) / S
            qz = (m[0][2] + m[2][0]) / S
        elif m[1][1] > m[2][2]:
            S = np.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2
            qw = (m[0][2] - m[2][0]) / S
            qx = (m[0][1] + m[1][0]) / S
            qy = 0.25 * S
            qz = (m[1][2] + m[2][1]) / S
        else:
            S = np.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2
            qw = (m[1][0] - m[0][1]) / S
            qx = (m[0][2] + m[2][0]) / S
            qy = (m[1][2] + m[2][1]) / S
            qz = 0.25 * S
        return qx, qy, qz, qw