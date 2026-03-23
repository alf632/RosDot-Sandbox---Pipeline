# AR Sandbox — ROS 2 Perception Pipeline

A ROS 2 Jazzy pipeline for an augmented-reality sandbox. Multiple Intel RealSense D435 depth cameras capture the sand surface (and interacting hands), the pipeline merges and filters the individual heightmaps, and streams the result to a Godot 4 simulation that projects the virtual world back onto the real sand.

## Architecture

The system is built around **dynamic component composition**:

- **`component_container_mt`** — a multi-threaded ROS 2 component container that hosts the C++ perception nodes.
- **`sandbox_operator`** — a Python node that discovers the local container and dynamically loads components into it based on the machine's **role**.

Roles are defined in `config.json`. Each role lists which **loaders** to activate:

| Role | Loaders | Purpose |
|---|---|---|
| `perception` | `realsense_loader`, `repro_loader` | Edge device with cameras |
| `merger` | `merger_loader`, `tf_loader`, `godot_loader` | Central processing server |
| `all` | all of the above | Single-machine development setup |

### Loaders

| Loader | What it does |
|---|---|
| **realsense_loader** | Auto-discovers connected RealSense cameras via `pyrealsense2` and loads their driver nodes |
| **repro_loader** | Loads `CameraReprojector` — reprojects depth frames from each camera into a common sandbox-origin grid |
| **merger_loader** | Loads `CloudMerger` (fuses all per-device heightmaps with temporal smoothing) and `HeightmapStreamer` (PNG-encodes and sends the result over UDP) |
| **tf_loader** | Watches `calibrations/tf_configs/*.json` and publishes static TF transforms (camera and projector poses) |
| **godot_loader** | Watches `calibrations/projectors.yml` and projector calibration JSONs, forwards them to Godot via TCP |

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
      │  sends via UDP
      ▼
Godot 4 simulation          (separate workspace)
      │  Camera3Ds represent real projectors
      ▼
Physical projectors → sand
```

### Multi-machine deployment

Each machine runs its own `component_container_mt` and `sandbox_operator`. Namespace isolation is automatic — the container's namespace is derived from the sanitized hostname (e.g. `raspi-3` → `/raspi_3`). All nodes share `ROS_DOMAIN_ID=42` for discovery.

One machine acts as **controller** (`is_controller: true`) and periodically broadcasts `config.json` on `/config/sandbox_setup`. Non-controller nodes receive the config from the topic instead of reading it from disk.

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

## Calibration

An interactive terminal UI guides through camera and projector calibration:

```bash
./start_calibrator.sh
```

### Camera calibration

1. Disables the IR emitter and switches the camera to infrared mode.
2. Reads AprilTag positions from `calibrations/tags.yaml` and publishes them as static TF.
3. Runs `apriltag_ros` to detect the tags in the infrared image.
4. Solves the transform from `sandbox_origin` to the camera frame.
5. Saves the result as `calibrations/tf_configs/cam_{serial}_link.json`.

The `tf_loader` picks up changes automatically — no restart needed.

### Projector calibration

1. Reads projector definitions from `calibrations/projectors.yml`.
2. Generates a ChArUco board and sends it to Godot for display via the projector.
3. Detects the board corners in the camera's view.
4. Solves projector intrinsics and extrinsics with OpenCV.
5. Saves as `calibrations/tf_configs/projector_{id}.json`.

The `godot_loader` forwards the calibration to Godot automatically.

## Helper scripts

| Script | Purpose |
|---|---|
| `start_calibrator.sh` | Launch calibration TUI inside the running container |
| `start_rviz.sh` | Launch RViz in a desktop container with X11 forwarding |
| `receive_projector_stream.sh` | View the UDP heightmap stream with `ffplay` |

## Configuration

**`config.json`** — roles, loader settings (output resolution, sandbox dimensions, smoothing parameters, network endpoints).

**`calibrations/projectors.yml`** — projector IDs, resolutions, and network addresses.

**`calibrations/tags.yaml`** — AprilTag family, sizes, and known positions relative to `sandbox_origin`.

**`calibrations/tf_configs/`** — auto-generated transform files (camera poses as quaternions, projector poses as intrinsics + extrinsics).
