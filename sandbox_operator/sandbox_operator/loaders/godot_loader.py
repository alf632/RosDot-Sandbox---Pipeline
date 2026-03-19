import os
import yaml
import json
import socket
import glob

class GodotLoader:
    def __init__(self):
        self.known_files = {}

    def discover_and_load(self, operator, config):
        cfg = config.get('godot_loader', {})
        config_dir = cfg.get('config_dir', 'projectors') 
        self.godot_ip = cfg.get('godot_ip', '127.0.0.1')
        self.godot_port = cfg.get('godot_port', 5007)

        # 1. Watch projectors.yml (Definitions)
        self.check_projector_definitions(operator, config_dir)
        
        # 2. Watch projector_*.json (Transforms)
        self.check_projector_transforms(operator, config_dir)

    def send_to_godot(self, payload):
        """Helper to fire TCP messages to Godot silently"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1.0)
                s.connect((self.godot_ip, self.godot_port))
                s.sendall(json.dumps(payload).encode('utf-8'))
            return True
        except (ConnectionRefusedError, socket.timeout):
            return False # Godot isn't up, we'll try again next tick
            
    def check_projector_definitions(self, operator, config_dir):
        proj_file = os.path.join(config_dir, 'projectors.yml')
        if not os.path.exists(proj_file):
            operator.get_logger().warn(f"projectors definitions file not found: {proj_file}")

        mtime = os.path.getmtime(proj_file)
        if self.known_files.get(proj_file) == mtime:
            return  # No change

        try:
            with open(proj_file, 'r') as f:
                projectors = yaml.safe_load(f)

            payload = {
                "command": "update_projectors",
                "data": projectors  # Godot receives the raw parsed YAML dict
            }

            if self.send_to_godot(payload):
                self.known_files[proj_file] = mtime
                operator.get_logger().info("Pushed updated projector definitions to Godot.")
        except Exception as e:
            operator.get_logger().error(f"Failed to process projectors.yml: {e}")

    def check_projector_transforms(self, operator, config_dir):
        search_path = os.path.join(config_dir, "tf_configs", 'projector_*.json')
        
        for filepath in glob.glob(search_path):
            mtime = os.path.getmtime(filepath)
            if self.known_files.get(filepath) == mtime:
                continue # No change

            filename = os.path.basename(filepath)
            proj_id = filename.replace('projector_', '').replace('.json', '')

            try:
                with open(filepath, 'r') as f:
                    transform_data = json.load(f)
                    
                payload = {
                    "command": "update_transform",
                    "projector_id": str(proj_id),
                    "data": transform_data
                }

                if self.send_to_godot(payload):
                    self.known_files[filepath] = mtime
                    operator.get_logger().info(f"Pushed calibration transform for Projector {proj_id} to Godot.")
            except Exception as e:
                operator.get_logger().error(f"Failed to process {filename}: {e}")