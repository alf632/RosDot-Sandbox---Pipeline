import glob
import json
import os
import signal
import socket
import subprocess

from std_msgs.msg import String
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy

MPV_BACKEND_FLAGS = {
    "drm": ["--vo=drm", "--drm-connector={connector}"],
    "weston": ["--vo=gpu", "--gpu-context=wayland", "--fs", "--fs-screen-name={connector}"],
    "hyprland": ["--vo=gpu", "--gpu-context=wayland", "--fs", "--fs-screen-name={connector}"],
    "x11": ["--vo=gpu", "--gpu-context=x11egl", "--fs", "--fs-screen-name={connector}"],
}


class ProjectorLoader:
    def __init__(self):
        self.mpv_processes = {}  # connector_name -> subprocess.Popen
        self.publishers = {}    # connector_name -> Publisher
        self.shutdown_registered = False

    def discover_and_load(self, operator, config):
        cfg = config.get('projector_loader', {})
        target_displays = cfg.get('displays', [])
        operator.get_logger().debug(f"projector_loader: loaded. config={cfg}")
        if not target_displays:
            operator.get_logger().warn("projector_loader: no displays configured, skipping")
            return

        backend = cfg.get('display_backend', 'drm')
        if backend not in MPV_BACKEND_FLAGS:
            operator.get_logger().error(f"Unknown display_backend '{backend}'. Valid: {list(MPV_BACKEND_FLAGS.keys())}")
            return

        base_port = cfg.get('base_udp_port', 5004)
        host_ip = cfg.get('host_ip', '') or self._detect_host_ip()
        hostname = socket.gethostname().replace('-', '_')

        connected = self._discover_displays()
        operator.get_logger().debug(f"projector_loader: DRM connectors found: {list(connected.keys())}")
        if not connected:
            operator.get_logger().warn("projector_loader: no connected displays found in /sys/class/drm/")

        for i, connector in enumerate(target_displays):
            if connector not in connected:
                operator.get_logger().warn(f"Display {connector} not connected, skipping")
                continue

            display_info = connected[connector]
            port = base_port + i

            # Start or restart mpv if needed
            self._ensure_mpv(operator, connector, port, backend)

            # Publish projector info
            sanitized = connector.replace('-', '_')
            topic = f"/projectors/{hostname}/{sanitized}"
            if connector not in self.publishers:
                qos = QoSProfile(
                    depth=1,
                    reliability=ReliabilityPolicy.RELIABLE,
                    durability=DurabilityPolicy.TRANSIENT_LOCAL
                )
                self.publishers[connector] = operator.create_publisher(String, topic, qos)
                operator.get_logger().info(f"Publishing projector {connector} on {topic}")

            projector_id = f"{hostname}_{sanitized}"
            msg = String()
            msg.data = json.dumps({
                "projector_id": projector_id,
                "width": display_info["width"],
                "height": display_info["height"],
                "target_ip": host_ip,
                "target_port": port
            })
            self.publishers[connector].publish(msg)

        # Register shutdown cleanup once
        if not self.shutdown_registered:
            operator.context.on_shutdown(self._cleanup)
            self.shutdown_registered = True

    def _discover_displays(self):
        """Parse /sys/class/drm/ for connected displays and their preferred resolution."""
        displays = {}
        for status_path in glob.glob('/sys/class/drm/card*-*/status'):
            try:
                with open(status_path) as f:
                    if f.read().strip() != 'connected':
                        continue
            except OSError:
                continue

            # Extract connector name: /sys/class/drm/card0-HDMI-A-1/status -> HDMI-A-1
            drm_dir = os.path.dirname(status_path)
            dir_name = os.path.basename(drm_dir)
            # Strip the "cardN-" prefix
            connector = dir_name.split('-', 1)[1] if '-' in dir_name else dir_name

            modes_path = os.path.join(drm_dir, 'modes')
            width, height = 1920, 1080  # fallback
            try:
                with open(modes_path) as f:
                    first_mode = f.readline().strip()
                    if 'x' in first_mode:
                        parts = first_mode.split('x')
                        width = int(parts[0])
                        height = int(parts[1])
            except (OSError, ValueError):
                pass

            displays[connector] = {"width": width, "height": height}

        return displays

    def _ensure_mpv(self, operator, connector, port, backend):
        """Start mpv if not running, restart if dead."""
        proc = self.mpv_processes.get(connector)
        if proc is not None and proc.poll() is None:
            return  # still alive

        if proc is not None:
            operator.get_logger().warn(f"mpv for {connector} died (exit={proc.returncode}), restarting")

        backend_flags = [
            f.format(connector=connector) for f in MPV_BACKEND_FLAGS[backend]
        ]
        cmd = [
            "mpv",
            *backend_flags,
            "--no-terminal",
            "--no-input-default-bindings",
            "--profile=low-latency",
            "--untimed",
            f"udp://0.0.0.0:{port}",
        ]

        operator.get_logger().info(f"Starting mpv for {connector} on port {port}: {' '.join(cmd)}")
        try:
            self.mpv_processes[connector] = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            operator.get_logger().error("mpv not found. Install mpv to use projector_loader.")

    def _cleanup(self):
        """Terminate all mpv processes on shutdown."""
        for connector, proc in self.mpv_processes.items():
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    proc.kill()
        self.mpv_processes.clear()

    def _detect_host_ip(self):
        """Best-effort local IP detection."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                return s.getsockname()[0]
        except OSError:
            return "127.0.0.1"
