class ReprojectorLoader:
    def discover_and_load(self, operator, config):
        """Loads a single Reprojector Manager for this device."""

        cfg = config.get('repro_loader', {})

        params = {
            'sandbox_width': cfg.get('sandbox', {}).get('width', 4.0),
            'sandbox_length': cfg.get('sandbox', {}).get('length', 6.0),
            'output_width': cfg.get('output_res', {}).get('width', 600),
            'output_height': cfg.get('output_res', {}).get('height', 400),
            # Pass the namespace so the C++ node knows its local domain
            'target_namespace': operator.device_namespace,
            # Depth range gate: reject readings outside the sandbox volume
            'depth_min_mm': cfg.get('depth_min_mm', 300),
            'depth_max_mm': cfg.get('depth_max_mm', 1500),
            # Spatial median blur on raw depth before reprojection (removes hot pixels).
            # Must be an odd integer; set to 1 to disable.
            'median_blur_kernel': cfg.get('median_blur_kernel', 3),
            # Minimum hit count per grid cell per camera (0.5 = accept any single hit; 1.5 = require ≥2).
            'sparse_hit_threshold': cfg.get('sparse_hit_threshold', 0.5),
        }

        operator.load_component(
            package='sandbox_components',
            plugin='sandbox_components::CameraReprojector',
            name='local_reprojector',
            params=params,
            namespace=operator.device_namespace,
            use_ipc=True
        )
