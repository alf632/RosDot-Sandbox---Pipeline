import os
import json
import socket
import glob as globmod

from std_msgs.msg import String
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy


class GodotLoader:
    def __init__(self):
        self.known_files = {}
        self.subscribed_topics = set()
        self.projector_map = {}  # topic_name -> parsed JSON dict
        self.projector_map_dirty = False
        self.last_sent_snapshot = None

    def discover_and_load(self, operator, config):
        cfg = config.get('godot_loader', {})
        config_dir = cfg.get('config_dir', 'projectors')
        self.godot_ip = cfg.get('godot_ip', '127.0.0.1')
        self.godot_port = cfg.get('godot_port', 5007)

        # 1. Discover and subscribe to projector topics
        self.check_projector_topics(operator)

        # 2. Push aggregated projector definitions to Godot if changed
        if self.projector_map_dirty:
            self._push_projectors_to_godot(operator)

        # 3. Watch projector_*.json (Transforms) — unchanged
        self.check_projector_transforms(operator, config_dir)

    def check_projector_topics(self, operator):
        """Discover /projectors/*/* topics and subscribe to new ones."""
        qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )
        for topic_name, topic_types in operator.get_topic_names_and_types():
            if not topic_name.startswith('/projectors/'):
                continue
            if topic_name in self.subscribed_topics:
                continue
            operator.create_subscription(
                String, topic_name,
                lambda msg, t=topic_name: self._on_projector_msg(t, msg),
                qos
            )
            self.subscribed_topics.add(topic_name)
            operator.get_logger().info(f"Subscribed to projector topic: {topic_name}")

    def _on_projector_msg(self, topic_name, msg):
        """Handle incoming projector definition from a projector_loader."""
        try:
            data = json.loads(msg.data)
            self.projector_map[topic_name] = data
            self.projector_map_dirty = True
        except json.JSONDecodeError:
            pass

    def _push_projectors_to_godot(self, operator):
        """Aggregate all known projectors and send to Godot."""
        # Build Godot-compatible dict keyed by stable string IDs (e.g. "malf_legion_HDMI_A_1")
        godot_projectors = {}
        for topic_name in sorted(self.projector_map.keys()):
            info = self.projector_map[topic_name]
            projector_id = info["projector_id"]
            godot_projectors[projector_id] = {
                "resolution": f"{info['width']}x{info['height']}",
                "target_ip": info["target_ip"],
                "target_port": info["target_port"]
            }

        # Only send if the data actually changed
        snapshot = json.dumps(godot_projectors, sort_keys=True)
        if snapshot == self.last_sent_snapshot:
            self.projector_map_dirty = False
            return

        payload = {
            "command": "update_projectors",
            "data": godot_projectors
        }

        if self.send_to_godot(payload):
            self.last_sent_snapshot = snapshot
            self.projector_map_dirty = False
            operator.get_logger().info(f"Pushed {len(godot_projectors)} projector(s) to Godot.")

    def send_to_godot(self, payload):
        """Helper to fire TCP messages to Godot silently."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1.0)
                s.connect((self.godot_ip, self.godot_port))
                s.sendall(json.dumps(payload).encode('utf-8'))
            return True
        except (ConnectionRefusedError, socket.timeout):
            return False  # Godot isn't up, we'll try again next tick

    def check_projector_transforms(self, operator, config_dir):
        search_path = os.path.join(config_dir, "tf_configs", 'projector_*.json')

        for filepath in globmod.glob(search_path):
            mtime = os.path.getmtime(filepath)
            if self.known_files.get(filepath) == mtime:
                continue  # No change

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