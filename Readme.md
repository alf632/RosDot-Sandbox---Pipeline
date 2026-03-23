# AR Sandbox — ROS 2 Perception Pipeline

A ROS 2 Jazzy pipeline for an augmented-reality sandbox. Multiple Intel RealSense D435 depth cameras capture the sand surface (and interacting hands), the pipeline merges and filters the individual heightmaps, and streams the result to a Godot 4 simulation that projects the virtual world back onto the real sand.
The simulation is hosted in a separate repository: https://github.com/alf632/RosDot-Sandbox---Simulation

## Architecture

The system is built around **dynamic component composition**:

- **`component_container_mt`** — a multi-threaded ROS 2 component container that hosts the C++ perception nodes.
- **`sandbox_operator`** — a Python node that discovers the local container and dynamically loads components into it based on the machine's **roles**.

Roles are defined in `config.json`. Each role lists which **loaders** to activate. A machine can have multiple roles (comma-separated): `-p role:=perception,projector`.

| Role | Loaders | Purpose |
|---|---|---|
| `perception` | `realsense_loader`, `repro_loader` | Edge device with cameras |
| `merger` | `merger_loader`, `streamer_loader`, `tf_loader`, `godot_loader` | Central processing server |
| `projector` | `projector_loader` | Display node with projectors |
| `all` | all of the above | Single-machine development setup |

### Loaders

| Loader | What it does |
|---|---|
| **realsense_loader** | Auto-discovers connected RealSense cameras via `pyrealsense2` and loads their driver nodes |
| **repro_loader** | Loads `CameraReprojector` — reprojects depth frames from each camera into a common sandbox-origin grid |
| **merger_loader** | Loads `CloudMerger` — fuses all per-device heightmaps with non-linear temporal smoothing |
| **streamer_loader** | Loads `HeightmapStreamer` — PNG-encodes the merged heightmap and sends it over UDP |
| **projector_loader** | Discovers connected displays via DRM sysfs, starts `mpv` per projector, publishes projector config to ROS topics |
| **tf_loader** | Watches `calibrations/tf_configs/*.json` and publishes static TF transforms (camera and projector poses) |
| **godot_loader** | Subscribes to projector topics and forwards definitions to Godot via TCP; watches projector calibration JSONs |

### Data flow

```
RealSense D435 (×N)
      │  depth @ 640×480, 30 Hz
      ▼
CameraReprojector          (per device, C++)
      │  reprojects depth → 256×256 heightmap grid
      │  publishes /{device}/local_heightmap (32FC2: height sum + count)
      ▼
CloudMerger                (central, C++)
      │  merges all local_heightmaps
      │  inpaints, blurs, applies non-linear temporal smoothing
      │  publishes /sandbox/heightmap (32FC1)
      ▼
HeightmapStreamer           (central, C++)
      │  converts to 8-bit grayscale PNG
      │  sends via UDP to Godot
      ▼
Godot 4 simulation          (separate workspace)
      │  receives heightmap, renders scene
      │  sends MPEG streams via UDP per projector
      ▼
ProjectorLoader             (per display node)
      │  discovers displays via /sys/class/drm/
      │  runs mpv per projector to receive and display the stream
      ▼
Physical projectors → sand
```

### Multi-machine deployment

Each machine runs its own `component_container_mt` and `sandbox_operator`. Namespace isolation is automatic — the container's namespace is derived from the sanitized hostname (e.g. `raspi-3` → `/raspi_3`). All nodes share `ROS_DOMAIN_ID=42` for discovery.

One machine acts as **controller** (`is_controller: true`) and periodically broadcasts `config.json` on `/config/sandbox_setup`. Non-controller nodes receive the config from the topic instead of reading it from disk.

## Configuration

### config.json

The main configuration file with three sections:

**`roles`** — named sets of loaders. A machine's role determines which loaders activate.

**`loader_settings`** — global settings per loader (output resolution, sandbox dimensions, smoothing parameters, network endpoints, display backend).

**`host_settings`** — per-host overrides keyed by sanitized hostname. Deep-merged on top of `loader_settings` at runtime. Use this for host-specific configuration like which display outputs are projectors:

```json
"host_settings": {
  "projector_pc_1": {
    "projector_loader": {
      "display_backend": "drm",
      "displays": ["HDMI-A-1", "HDMI-A-2"],
      "host_ip": "192.168.1.10"
    }
  }
}
```

### Projector display backends

The `projector_loader` supports multiple display backends via the `display_backend` setting:

| Backend | Use case | Requirements |
|---|---|---|
| `drm` | Dedicated projection nodes (Buildroot) | `/dev/dri` access, no compositor needed |
| `weston` | Weston-based Wayland systems | Wayland socket + `/dev/dri` |
| `hyprland` | Hyprland desktops | Wayland socket + `/dev/dri` |
| `x11` | X11 desktops (e.g. Ubuntu) | `DISPLAY` env + X11 socket |

Display discovery always uses `/sys/class/drm/card*/` regardless of backend.

### Projector identification

Each projector gets a stable string ID derived from its physical topology: `{hostname}_{connector}` (e.g. `projector_pc_1_HDMI_A_1`). This ID is used as the Godot node name, the calibration filename, and the key in all TCP commands. It remains stable across restarts and is unaffected by discovery order.

Projector config is published per-projector on `/projectors/{host}/{connector}` topics with transient_local QoS. The `godot_loader` dynamically discovers these topics and aggregates them for Godot.

### Other config files

**`calibrations/tags.yaml`** — AprilTag family, sizes, and known positions relative to `sandbox_origin`.

**`calibrations/tf_configs/`** — auto-generated transform files (camera poses as quaternions, projector poses as intrinsics + extrinsics).

## Deployment

Three Docker Compose files cover different setups:

```bash
# All-in-one (development / single machine)
docker compose -f docker-compose-all.yml up

# Distributed — on each edge device (cameras)
docker compose -f docker-compose-perception.yml up

# Distributed — on the central server
docker compose -f docker-compose-processor.yml up
```

For multi-role machines, set the role parameter in the compose file:

```yaml
command: >
  ros2 run sandbox_operator operator
  --ros-args -p role:=perception,projector -p is_controller:=false
```

## Calibration

An interactive terminal UI guides through camera and projector calibration:

```bash
./start_calibrator.sh
```

It requires the whole Pipeline (including the Godot Simulation) to be running since it discovers existing Cameras and Projectors dynamically and instructs the Simulation to display calibration images on the Projectors. Projector calibration depends on the Cameras to be calibrated first.

### Camera calibration

1. Disables the IR emitter and switches the camera to infrared mode.
2. Reads AprilTag positions from `calibrations/tags.yaml` and publishes them as static TF.
3. Runs `apriltag_ros` to detect the tags in the infrared image.
4. Solves the transform from `sandbox_origin` to the camera frame.
5. Saves the result as `calibrations/tf_configs/cam_{serial}_link.json`.

The `tf_loader` picks up changes automatically — no restart needed.

### Projector calibration

1. Discovers active projectors from `/projectors/*` ROS topics (requires a running `projector_loader`).
2. Generates a ChArUco board and sends it to Godot for display via the selected projector.
3. Detects the board corners in the camera's view, collects 3D↔2D point correspondences.
4. Solves projector intrinsics and extrinsics with OpenCV.
5. Saves as `calibrations/tf_configs/projector_{host}_{connector}.json`.

The `godot_loader` forwards the calibration to Godot automatically.

## Helper scripts

| Script | Purpose |
|---|---|
| `start_calibrator.sh` | Launch calibration TUI inside the running container |
| `start_rviz.sh` | Launch RViz in a desktop container with X11 forwarding |
| `receive_projector_streeam.sh` | View a UDP video stream with `ffplay` (for debugging) |
