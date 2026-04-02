# Godot Interface Changes — Heightmap Pipeline Overhaul

## Summary

The ROS pipeline now publishes **two separate heightmap streams** over UDP instead of one.
Both carry the same encoding as before (8-bit grayscale PNG, same resolution and Z mapping),
so any existing image-decode code can be reused unchanged.

---

## UDP Streams

| Stream   | Default Port | Source topic                     | Content                                      |
|----------|-------------|----------------------------------|----------------------------------------------|
| Visual   | **4242**    | `/sandbox/heightmap`             | Terrain + transient objects and hands        |
| Physics  | **4243**    | `/sandbox/heightmap/physics`     | Smooth terrain only — no transient objects   |

Both ports are configured in `config.json` under `streamer_loader.udp_ip/udp_port/physics_udp_port`.

---

## Encoding (unchanged)

Both streams use the same convention as the previous single stream:

```
pixel_value = clamp((height_meters + z_offset) / z_range * 255, 0, 255)
```

Default values from `config.json`:
- `z_offset = 0.1` m  (height at which pixel = 0)
- `z_range  = 0.3` m  (height span that covers 0–255)

So the height in metres for a pixel value `p` is:

```
height = (p / 255.0) * z_range - z_offset
```

Resolution: **256 × 256** pixels (configurable, same for both streams).

---

## Recommended Usage in Godot

### Water / Physics simulation
Subscribe to the **physics stream** (port 4243).

This layer has very strong temporal smoothing. It takes ~1–2 seconds to track deliberate sand
reshaping, and is completely immune to hands or objects being placed on or above the sand.
Use it as input to the water flow / fluid simulation.

### Visual heightmap overlay / rendering
Subscribe to the **visual stream** (port 4242).

This is a composite: the stable terrain layer plus a fast-responding layer that detects hands
and placed objects. Objects appear within ~165 ms and fade out within ~330 ms after removal.
Use this for the visible terrain mesh / colour map.

### If you only want one stream
If dual-stream adds complexity before it is needed, just keep using port **4242** (visual).
The physics port (4243) is optional — the pipeline only launches the physics streamer when
`physics_udp_port` is present in `config.json`.

---

## Object / Hand Detection Notes

Objects and hands on the sandbox surface appear **only in the visual stream**, not the physics
stream. The detection threshold is currently **15 mm** of height deviation above the local
terrain; anything smaller is treated as noise and ignored.

The fast layer has no object-type classification — it reports any height deviation that exceeds
the threshold. If Godot needs to distinguish "hand" from "placed rock", that inference must
happen on the Godot side (e.g., by tracking connected blobs and comparing their lifetime or
shape against expected object footprints).

---

## Parameter Tuning Reference

All parameters below live in `config.json` and take effect on the next pipeline start.

| Parameter                            | Default | Effect |
|--------------------------------------|---------|--------|
| `streamer_loader.physics_udp_port`   | 4243    | UDP port for the physics/water stream; remove key to disable |
| `merger_loader.slow_smoothing_factor`| 2.0     | Higher = slower terrain response (more noise immune) |
| `merger_loader.slow_min_alpha`       | 0.01    | Floor blend rate for slow layer; keep very small |
| `merger_loader.fast_alpha`           | 0.40    | How fast objects appear (higher = faster, ~1/fast_alpha frames) |
| `merger_loader.fast_detection_threshold` | 0.015 | Minimum height delta (m) to register as an object; raise to ignore small hands |
| `merger_loader.fast_decay_alpha`     | 0.10    | How fast objects fade after removal (~1/fast_decay_alpha frames) |
